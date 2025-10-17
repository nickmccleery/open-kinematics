"""
Parse sweep configuration files into executable sweep specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from marshmallow.exceptions import ValidationError
from marshmallow_dataclass import class_schema

from kinematics.enums import Axis, PointID, TargetPositionMode
from kinematics.types import (
    PointTarget,
    PointTargetAxis,
    PointTargetVector,
    SweepConfig,
    WorldAxisSystem,
)


@dataclass(frozen=True)
class DirectionSpec:
    """
    Specification for a target direction.

    Attributes:
        axis (Axis | None): Principal axis (X, Y, or Z) for the direction.
        vector (Sequence[float] | None): Custom 3D vector for the direction.
    """

    axis: Axis | None = None
    vector: Sequence[float] | None = None

    def to_unit_vector(self) -> np.ndarray:
        """
        Convert this direction specification to a unit vector.

        Returns:
            Normalized 3D unit vector.

        Raises:
            ValueError: If both or neither axis and vector are specified,
                or if vector is invalid.
        """
        if (self.axis is None) == (self.vector is None):
            raise ValueError("Specify exactly one of 'axis' or 'vector'")

        if self.axis is not None:
            match self.axis:
                case Axis.X:
                    return WorldAxisSystem.X
                case Axis.Y:
                    return WorldAxisSystem.Y
                case Axis.Z:
                    return WorldAxisSystem.Z

        vec = np.asarray(self.vector, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError(f"Vector must be 3D, got shape {vec.shape}")

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ValueError("Direction vector cannot be zero")

        return vec / norm


@dataclass(frozen=True)
class TargetSpec:
    """
    Specification for a single sweep target dimension.

    Attributes:
        point (PointID): Point to be constrained.
        direction (DirectionSpec): Direction specification for the constraint.
        name (str | None): Optional human-readable name for this target.
        mode (TargetPositionMode): Whether values are relative or absolute.
        start (float | None): Starting value for linear interpolation.
        stop (float | None): Ending value for linear interpolation.
        values (Sequence[float] | None): Explicit list of values (alternative to start/stop).
    """

    point: PointID
    direction: DirectionSpec
    name: str | None = None
    mode: TargetPositionMode = TargetPositionMode.RELATIVE
    start: float | None = None
    stop: float | None = None
    values: Sequence[float] | None = None


@dataclass(frozen=True)
class SweepFileSchema:
    """
    Schema for sweep configuration files.

    Attributes:
        version (int): Schema version (currently only 1 is supported).
        steps (int | None): Number of steps for linear interpolation (used if targets
            don't specify explicit values).
        targets (list[TargetSpec]): List of target specifications for each sweep dimension.
    """

    version: int
    steps: int | None
    targets: list[TargetSpec]


def expand_target_values(spec: TargetSpec, default_steps: int | None) -> list[float]:
    """
    Expand a target specification into a list of concrete values.

    Args:
        spec: Target specification to expand.
        default_steps: Default step count if not specified in target.

    Returns:
        List of float values for this target dimension.

    Raises:
        ValueError: If values cannot be determined from the specification.
    """
    # Explicit values take precedence
    if spec.values is not None:
        return [float(v) for v in spec.values]

    # Otherwise require start/stop with step count
    if spec.start is None or spec.stop is None:
        raise ValueError(
            f"Target '{spec.name or spec.point.name}': "
            "must specify either 'values' or both 'start' and 'stop'"
        )

    steps = default_steps
    if steps is None:
        raise ValueError(
            f"Target '{spec.name or spec.point.name}': "
            "no 'steps' count available (specify at target or file level)"
        )

    return list(np.linspace(float(spec.start), float(spec.stop), int(steps)))


def direction_to_target_type(
    unit_vec: np.ndarray,
) -> PointTargetAxis | PointTargetVector:
    """
    Convert a unit vector to the appropriate target direction type.

    Args:
        unit_vec: Normalized direction vector.

    Returns:
        Either PointTargetAxis (if aligned with principal/standard basis axis) or
        PointTargetVector (for arbitrary directions).
    """
    if np.allclose(unit_vec, WorldAxisSystem.X):
        return PointTargetAxis(Axis.X)
    if np.allclose(unit_vec, WorldAxisSystem.Y):
        return PointTargetAxis(Axis.Y)
    if np.allclose(unit_vec, WorldAxisSystem.Z):
        return PointTargetAxis(Axis.Z)
    return PointTargetVector(unit_vec)


def parse_sweep_file(path: Path) -> SweepConfig:
    """
    Parse a sweep configuration file into a SweepConfig.

    Args:
        path: Path to the YAML sweep configuration file.

    Returns:
        SweepConfig ready for use with the solver.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid or contains errors.
        OSError: For file I/O errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"Sweep file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, dict):
            raise ValueError("Sweep file must contain a YAML mapping")

        # Be lenient with enum casing and missing fields in authoring-friendly YAML.
        # - Accept mode specified as lowercase values ("relative"/"absolute") by
        #   converting to enum NAMES expected by marshmallow ("RELATIVE"/"ABSOLUTE").
        # - Default missing mode to RELATIVE.
        targets_data = raw_data.get("targets")
        if isinstance(targets_data, list):
            for t in targets_data:
                if not isinstance(t, dict):
                    continue
                if "mode" not in t or t["mode"] is None:
                    t["mode"] = "RELATIVE"
                else:
                    val = t["mode"]
                    if isinstance(val, str):
                        low = val.strip().lower()
                        if low in ("relative", "abs", "absolute"):
                            # Map common shorthands and values to enum NAMES
                            t["mode"] = (
                                "ABSOLUTE" if low.startswith("abs") else "RELATIVE"
                            )
                        else:
                            # If already provided as name in various casings
                            up = val.strip().upper()
                            if up in ("RELATIVE", "ABSOLUTE"):
                                t["mode"] = up

        # Parse with marshmallow schema
        Schema = class_schema(SweepFileSchema)
        file_spec = Schema().load(raw_data)
        assert isinstance(file_spec, SweepFileSchema)

        if file_spec.version != 1:
            raise ValueError(f"Unsupported sweep version: {file_spec.version}")

        # Expand each target into its value sequence
        target_sequences: list[list[float]] = []
        for target_spec in file_spec.targets:
            values = expand_target_values(target_spec, file_spec.steps)
            target_sequences.append(values)

        # Verify all sequences have the same length
        lengths = {len(seq) for seq in target_sequences}
        if len(lengths) != 1:
            raise ValueError(
                f"All targets must have the same length, got: {sorted(lengths)}"
            )

        n_steps = next(iter(lengths))

        # Build per-dimension target lists
        sweep_dimensions: list[list[PointTarget]] = []
        for target_spec, values in zip(file_spec.targets, target_sequences):
            unit_vec = target_spec.direction.to_unit_vector()
            direction = direction_to_target_type(unit_vec)

            # We assume that direction here is generally aligned with global Y,
            # but rather than validate here, we can let the solver catch it if
            # the user tries to do something nonsensical.

            targets = [
                PointTarget(
                    point_id=target_spec.point,
                    direction=direction,
                    value=values[i],
                    mode=target_spec.mode,
                )
                for i in range(n_steps)
            ]
            sweep_dimensions.append(targets)

        return SweepConfig(sweep_dimensions)

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}") from e
    except ValidationError as e:
        raise ValueError(f"Invalid sweep specification: {e}") from e
    except (IOError, OSError) as e:
        raise OSError(f"Error reading sweep file: {e}") from e
