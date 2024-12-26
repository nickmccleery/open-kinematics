from pathlib import Path

import pytest
import yaml

from kinematics.geometry.exceptions import InvalidGeometryFileContents
from kinematics.geometry.loader import load_geometry
from kinematics.geometry.schemas import SuspensionGeometry


@pytest.fixture
def valid_yaml_file():
    file_path = Path(__file__).parent / "geometry.yaml"
    return file_path


@pytest.fixture
def invalid_yaml_file(tmp_path: Path):
    data = {"invalid_attribute": "value"}
    file_path = tmp_path / "invalid_geometry.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


def test_load_geometry_valid(valid_yaml_file):
    result = load_geometry(valid_yaml_file)
    assert isinstance(result, SuspensionGeometry)


def test_load_geometry_invalid(invalid_yaml_file):
    with pytest.raises(InvalidGeometryFileContents):
        load_geometry(invalid_yaml_file)
