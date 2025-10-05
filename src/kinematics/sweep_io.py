"""
Sweep file parsing utilities.

This module defines a YAML/JSON schema for describing parametric sweeps
and parses them into the internal SweepConfig and PointTarget objects. The
preferred schema requires start/stop with a shared step count (top-level or per-target).

Schema (YAML example):

version: 1
steps: 41
targets:
    - point: TRACKROD_INBOARD
        direction: {axis: Y}
        mode: relative
        start: -10
        stop: 10
    - point: WHEEL_CENTER
        direction: {axis: Z}
        start: -20
        stop: 60
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kinematics.enums import Axis, PointID, TargetPositionMode
from kinematics.types import (
    PointTarget,
    PointTargetAxis,
    PointTargetVector,
    SweepConfig,
    make_vec3,
)


@dataclass(frozen=True)
class SweepDimension:
    point: PointID
    direction: PointTargetAxis | PointTargetVector
    mode: TargetPositionMode
    values: List[float]
    name: str | None = None


def _parse_direction(data: Dict[str, Any]) -> PointTargetAxis | PointTargetVector:
    if not isinstance(data, dict):
        raise ValueError("direction must be a mapping with 'axis' or 'vector'")
    if "axis" in data:
        axis_val = data["axis"]
        try:
            axis = Axis[axis_val] if isinstance(axis_val, str) else Axis(axis_val)
        except Exception as e:
            raise ValueError(f"Invalid axis: {axis_val}") from e
        return PointTargetAxis(axis)
    if "vector" in data:
        vec = make_vec3(data["vector"]).astype(float)
        return PointTargetVector(vec)
    raise ValueError("direction must have either 'axis' or 'vector'")


def _linspace(start: float, stop: float, steps: int) -> List[float]:
    if steps < 2:
        return [start]
    step_size = (stop - start) / (steps - 1)
    return [start + i * step_size for i in range(steps)]


def _parse_values(dim: Dict[str, Any], default_steps: int | None) -> List[float]:
    # Required schema: direct start/stop with step count from either per-target or top-level
    if "values" in dim:
        raise ValueError(
            "Explicit 'values' are not supported; use start/stop with steps"
        )
    if "linspace" in dim:
        raise ValueError("'linspace' is not supported; use start/stop with steps")

    if not ("start" in dim and "stop" in dim):
        raise ValueError("Each target must define 'start' and 'stop'")

    try:
        start = float(dim["start"])  # type: ignore[index]
        stop = float(dim["stop"])  # type: ignore[index]
    except Exception as e:
        raise ValueError("'start' and 'stop' must be numeric") from e

    steps = dim.get("steps")
    if steps is None:
        if default_steps is None:
            raise ValueError(
                "Missing 'steps' for target and no top-level 'steps' provided"
            )
        steps = default_steps
    try:
        steps = int(steps)
    except Exception as e:
        raise ValueError("'steps' must be an integer") from e
    return _linspace(start, stop, steps)


def _parse_dimension(dim: Dict[str, Any], default_steps: int | None) -> SweepDimension:
    try:
        point_raw = dim["point"]
        point = PointID[point_raw] if isinstance(point_raw, str) else PointID(point_raw)
    except Exception as e:
        raise ValueError(
            f"Invalid or missing point identifier: {dim.get('point')}"
        ) from e

    direction = _parse_direction(dim.get("direction", {"axis": "Z"}))

    mode_raw = dim.get("mode", TargetPositionMode.RELATIVE.value)
    try:
        mode = (
            TargetPositionMode(mode_raw)
            if not isinstance(mode_raw, str)
            or mode_raw in TargetPositionMode._value2member_map_
            else TargetPositionMode[mode_raw]
        )
    except Exception:
        # Accept strings 'relative'/'absolute' and enum names
        mode = TargetPositionMode.RELATIVE

    name = dim.get("name") if isinstance(dim.get("name"), str) else None

    values = _parse_values(dim, default_steps)

    # Domain-specific rule: steering rack (inner trackrod) moves along Y only
    if point == PointID.TRACKROD_INBOARD:
        if isinstance(direction, PointTargetAxis):
            if direction.axis != Axis.Y:
                raise ValueError("Steering targets must use axis Y (rack direction)")
        else:
            vec = direction.vector  # type: ignore[attr-defined]
            # Accept any vector colinear with Y (x ~ 0 and z ~ 0)
            if abs(float(vec[0])) > 1e-12 or abs(float(vec[2])) > 1e-12:
                raise ValueError("Steering vector must be aligned with Y axis")

    return SweepDimension(
        point=point, direction=direction, mode=mode, values=values, name=name
    )


def parse_sweep_file(path: Path) -> SweepConfig:
    """
    Parse a sweep YAML/JSON file into a SweepConfig.

    Args:
        path: Path to the sweep file.

    Returns:
        SweepConfig: Ready to use in solver.
    """
    if not path.exists():
        raise FileNotFoundError(f"Sweep file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None or not isinstance(data, dict):
        raise ValueError("Sweep file is empty or invalid format")

    # Accept either 'targets' (preferred) or 'sweeps' (legacy)
    items = data.get("targets")
    if items is None:
        items = data.get("sweeps")
    if not isinstance(items, list) or not items:
        raise ValueError(
            "Sweep file must contain a non-empty 'targets' (or legacy 'sweeps') list"
        )

    default_steps = data.get("steps")
    if default_steps is not None:
        try:
            default_steps = int(default_steps)
        except Exception as e:
            raise ValueError("Top-level 'steps' must be an integer") from e

    dims = [_parse_dimension(dim, default_steps) for dim in items]
    lengths = {len(d.values) for d in dims}
    if len(lengths) != 1:
        raise ValueError(
            f"All sweep dimensions must have equal lengths, got: {sorted(lengths)}"
        )

    # Construct SweepConfig target lists per dimension
    target_sweeps: List[List[PointTarget]] = []
    n_steps = next(iter(lengths))
    for d in dims:
        targets = [
            PointTarget(
                point_id=d.point, direction=d.direction, value=d.values[i], mode=d.mode
            )
            for i in range(n_steps)
        ]
        target_sweeps.append(targets)

    return SweepConfig(target_sweeps)


def describe_sweep(sweep: SweepConfig) -> List[Dict[str, Any]]:
    """
    Produce a simple, serializable description of a SweepConfig for metadata.
    """
    out: List[Dict[str, Any]] = []
    for dim in sweep.target_sweeps:
        if not dim:
            continue
        first = dim[0]
        direction: str
        if isinstance(first.direction, PointTargetAxis):
            direction = f"axis:{Axis(first.direction.axis).name}"
        else:
            vec = first.direction.vector.tolist()  # type: ignore[attr-defined]
            direction = f"vector:{vec}"
        out.append(
            {
                "point": PointID(first.point_id).name,
                "direction": direction,
                "mode": TargetPositionMode(first.mode).value,
                "values_len": len(dim),
            }
        )
    return out
