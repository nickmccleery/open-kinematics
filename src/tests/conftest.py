from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def double_wishbone_geometry_file(test_data_dir: Path) -> Path:
    return test_data_dir / "geometry.yaml"
