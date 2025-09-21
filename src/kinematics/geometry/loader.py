"""
YAML geometry loader with validation.

Handles file I/O, schema parsing, and geometry validation.
"""

from pathlib import Path
from typing import Tuple, Type, cast

import yaml
from marshmallow.exceptions import ValidationError
from marshmallow_dataclass import class_schema

from kinematics.geometry import exceptions as exc
from kinematics.geometry.validate import GeometryType, validate_geometry
from kinematics.suspensions.base.provider_base import BaseProvider
from kinematics.suspensions.base.registry import build_registry

LoadResult = Tuple[GeometryType, Type[BaseProvider]]


def load_geometry(file_path: Path) -> LoadResult:
    """
    Load suspension geometry from a YAML file.

    Args:
        file_path: Path to the YAML geometry file.

    Returns:
        Tuple of (loaded geometry, provider class)

    Raises:
        exc.GeometryFileNotFound: If the file doesn't exist.
        exc.InvalidGeometryFileContents: If the file contents are invalid.
        exc.GeometryFileError: For general file handling errors.
        exc.UnsupportedGeometryType: If the geometry type is not recognized.
    """
    if not file_path.exists():
        raise exc.GeometryFileNotFound(f"Geometry file not found: {file_path}")

    try:
        # Initial validation.
        with open(file_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            raise exc.InvalidGeometryFileContents("Geometry file is empty.")

        if "type" not in yaml_data:
            raise exc.InvalidGeometryFileContents("Geometry type not specified in file")

        geometry_type_key = yaml_data.pop("type")
        registry = build_registry()
        if geometry_type_key not in registry:
            raise exc.UnsupportedGeometryType(
                f"Unsupported geometry type: {geometry_type_key}"
            )

        # Get model and provider classes from registry
        model_class, provider_class = registry[geometry_type_key]

        # Create schema for the model and load the data
        GeometrySchema = class_schema(model_class)
        geometry = cast(GeometryType, GeometrySchema().load(yaml_data))

        # Validate the geometry
        validate_geometry(geometry)

        return geometry, provider_class

    except yaml.YAMLError as e:
        raise exc.GeometryFileError(f"Error parsing geometry file: {e}")
    except ValidationError as e:
        raise exc.InvalidGeometryFileContents(f"Error validating geometry: {e}")
    except (IOError, OSError) as e:
        raise exc.GeometryFileError(f"Error reading geometry file: {e}")
