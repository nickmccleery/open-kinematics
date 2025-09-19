from pathlib import Path

import pytest
import yaml

import kinematics.geometry.exceptions as exc
from kinematics.geometry.loader import load_geometry
from kinematics.geometry.base import SuspensionGeometry


@pytest.fixture
def empty_geometry_file(tmp_path: Path):
    empty_file = tmp_path / "empty_geometry.yaml"
    empty_file.touch()
    return empty_file


@pytest.fixture
def invalid_yaml_geometry_file(tmp_path: Path):
    file_path = tmp_path / "invalid_geometry.yaml"
    file_path.write_text('""')
    return file_path


@pytest.fixture
def invalid_geometry_file(tmp_path: Path):
    data = {"invalid_attribute": "value"}
    file_path = tmp_path / "invalid_geometry.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


def test_load_geometry_valid(double_wishbone_geometry_file):
    result = load_geometry(double_wishbone_geometry_file)
    assert isinstance(result, SuspensionGeometry)


def test_load_geometry_empty(empty_geometry_file):
    with pytest.raises(exc.InvalidGeometryFileContents):
        load_geometry(empty_geometry_file)


def test_load_geometry_not_found(tmp_path: Path):
    with pytest.raises(exc.GeometryFileNotFound):
        load_geometry(tmp_path / "file_not_found.yaml")


def test_load_geometry_invalid(invalid_geometry_file):
    with pytest.raises(exc.InvalidGeometryFileContents):
        load_geometry(invalid_geometry_file)
