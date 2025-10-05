"""
YAML geometry loader with validation.

This module handles file I/O, schema parsing, and geometry validation for all suspension
types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml
from marshmallow.exceptions import ValidationError
from marshmallow_dataclass import class_schema

from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.suspensions.registry import build_registry

if TYPE_CHECKING:
    from kinematics.suspensions.core.geometry import SuspensionGeometry


@dataclass
class LoadedSuspension:
    """
    Result of loading a suspension geometry from a file.

    Attributes:
        geometry (SuspensionGeometry): The loaded and validated suspension geometry instance
        provider (SuspensionProvider): Instantiated provider bound to this geometry
    """

    geometry: SuspensionGeometry
    provider: SuspensionProvider


def load_geometry(file_path: Path) -> LoadedSuspension:
    """
    Load suspension geometry from a YAML file.

    Args:
        file_path: Path to the YAML geometry file.

    Returns:
        LoadedSuspension dataclass containing the geometry and provider class.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file contents are invalid or geometry type is unsupported.
        OSError: For general file handling errors.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            raise ValueError("Geometry file is empty.")

        if "type" not in yaml_data:
            raise ValueError("Geometry type not specified in file")

        geometry_type_key = yaml_data.pop("type")
        registry = build_registry()
        if geometry_type_key not in registry:
            raise ValueError(f"Unsupported geometry type: {geometry_type_key}")

        # Get model and provider classes from registry.
        model_class, provider_class = registry[geometry_type_key]

        # Create schema for the model and load the data.
        GeometrySchema = class_schema(model_class)
        geometry = cast("SuspensionGeometry", GeometrySchema().load(yaml_data))

        try:
            if not geometry.validate():
                raise ValueError("Geometry validation failed.")
        except AttributeError:
            # Geometry object doesn't have validate method, assume it's valid.
            pass

        provider = provider_class(geometry)
        return LoadedSuspension(geometry=geometry, provider=provider)

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing geometry file: {e}") from e
    except ValidationError as e:
        raise ValueError(f"Error validating geometry: {e}") from e
    except (IOError, OSError) as e:
        raise OSError(f"Error reading geometry file: {e}") from e
