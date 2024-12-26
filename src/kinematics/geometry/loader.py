from pathlib import Path

import yaml
from marshmallow.exceptions import ValidationError
from marshmallow_dataclass import class_schema

import kinematics.geometry.exceptions as exc
from kinematics.geometry.schemas import SuspensionGeometry


def load_geometry(file_path: Path) -> SuspensionGeometry:
    GeometrySchema = class_schema(SuspensionGeometry)

    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    try:
        geometry = GeometrySchema().load(yaml_data)
        if not isinstance(geometry, SuspensionGeometry):
            raise TypeError("Loaded data is not of type SuspensionGeometry")
        return geometry

    except ValidationError as e:
        raise exc.InvalidGeometryFileContents(f"Error loading geometry: {e}")
