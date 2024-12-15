from pathlib import Path

import yaml
from marshmallow_dataclass import class_schema

from kinematics.geometry.schemas import GeometryType, SuspensionGeometry


def load_geometry(file_path: Path) -> SuspensionGeometry:
    GeometrySchema = class_schema(SuspensionGeometry)

    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    data = GeometrySchema().load(yaml_data)
    if not isinstance(data, SuspensionGeometry):
        raise ValueError("Error loading geometry")

    return data


def validate_geometry(geometry: SuspensionGeometry) -> bool:
    if geometry.type != GeometryType.DOUBLE_WISHBONE:
        raise NotImplementedError(f"Validation for {geometry.type} not yet implemented")

    # Check that all hard points are defined.
    hp = geometry.hard_points

    return True


def example_usage():
    file_path = Path("geometry.yaml")
    try:
        geometry = load_geometry(file_path)
        print(f"Loaded {geometry.type.value} suspension: {geometry.name}")

    except Exception as e:
        print(f"Error loading geometry: {e}")


if __name__ == "__main__":
    example_usage()
