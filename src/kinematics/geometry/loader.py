from pathlib import Path
from typing import Type, Union, cast

import yaml
from marshmallow.exceptions import ValidationError
from marshmallow_dataclass import class_schema

import kinematics.geometry.exceptions as exc
from kinematics.points.main import PointID, get_all_points
from kinematics.suspensions import DoubleWishboneGeometry, MacPhersonGeometry

# Define geometry types directly here
GeometryType = Union[DoubleWishboneGeometry, MacPhersonGeometry]

GEOMETRY_TYPES: dict[str, Type[GeometryType]] = {
    "DOUBLE_WISHBONE": DoubleWishboneGeometry,
    "MACPHERSON_STRUT": MacPhersonGeometry,
}


def validate_geometry(geometry: GeometryType) -> None:
    points = get_all_points(geometry.hard_points)
    for point in points:
        if point.id is PointID.NOT_ASSIGNED:
            raise ValidationError("Found unrecognized point ID in geometry.")


def load_geometry(file_path: Path) -> GeometryType:
    """
    Load suspension geometry from a YAML file.

    Args:
        file_path: Path to the YAML geometry file.

    Returns:
        Loaded and validated geometry object

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
        if geometry_type_key not in GEOMETRY_TYPES:
            raise exc.UnsupportedGeometryType(
                f"Unsupported geometry type: {geometry_type_key}"
            )

        # Actual geometry loading.
        geometry_type = GEOMETRY_TYPES[geometry_type_key]
        GeometrySchema = class_schema(geometry_type)
        geometry = cast(GeometryType, GeometrySchema().load(yaml_data))

        # Validate the geometry.
        validate_geometry(geometry)

        return geometry

    except yaml.YAMLError as e:
        raise exc.GeometryFileError(f"Error parsing geometry file: {e}")
    except ValidationError as e:
        raise exc.InvalidGeometryFileContents(f"Error validating geometry: {e}")
    except (IOError, OSError) as e:
        raise exc.GeometryFileError(f"Error reading geometry file: {e}")
