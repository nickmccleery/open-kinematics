from pathlib import Path
from typing import Type, TypeVar

import yaml
from marshmallow.exceptions import ValidationError
from marshmallow_dataclass import class_schema

import kinematics.geometry.exceptions as exc
from kinematics.geometry.schemas import SuspensionGeometry

T = TypeVar("T", bound=SuspensionGeometry)


def load_geometry(file_path: Path, geometry_class: Type[T] = SuspensionGeometry) -> T:
    """
    Load suspension geometry from a YAML file.

    Args:
        file_path: Path to the YAML geometry file.
        geometry_class: Class type for the geometry (defaults to SuspensionGeometry)

    Returns:
        Loaded and validated geometry object

    Raises:
        exc.GeometryFileNotFound: If the file doesn't exist.
        exc.InvalidGeometryFileContents: If the file contents are invalid.
        exc.GeometryFileError: For general file handling errors.
    """
    if not file_path.exists():
        raise exc.GeometryFileNotFound(f"Geometry file not found: {file_path}")

    GeometrySchema = class_schema(geometry_class)

    try:
        with open(file_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            raise exc.InvalidGeometryFileContents("Geometry file is empty.")

        geometry = GeometrySchema().load(yaml_data)
        if not isinstance(geometry, geometry_class):
            raise TypeError(f"Loaded data is not of type {geometry_class.__name__}")

        return geometry

    except yaml.YAMLError as e:
        raise exc.GeometryFileError(f"Error parsing geometry file: {e}")
    except ValidationError as e:
        raise exc.InvalidGeometryFileContents(f"Error validating geometry: {e}")
    except (IOError, OSError) as e:
        raise exc.GeometryFileError(f"Error reading geometry file: {e}")
