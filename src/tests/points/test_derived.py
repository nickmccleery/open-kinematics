import numpy as np
import pytest

from kinematics.geometry.points.ids import PointID
from kinematics.points.derived import (
    get_dependencies,
    update_axle_midpoint,
    update_derived_point,
    update_wheel_center,
    update_wheel_inboard,
    update_wheel_outboard,
)


@pytest.fixture
def sample_positions():
    # Set up a simple configuration:
    # - Axle inboard at origin
    # - Axle outboard at x=2
    # - This makes the axle 2 units wide pointing along x-axis
    return {
        PointID.AXLE_INBOARD: np.array([0.0, 0.0, 0.0]),
        PointID.AXLE_OUTBOARD: np.array([2.0, 0.0, 0.0]),
    }


def test_axle_midpoint(sample_positions):
    result = update_axle_midpoint(sample_positions)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 0.0]))


def test_wheel_center(sample_positions):
    # With axle outboard at x=2.0 and offset of 0.5, wheel center should be at x=2.5
    result = update_wheel_center(sample_positions, wheel_offset=0.5)
    np.testing.assert_array_equal(result, np.array([2.5, 0.0, 0.0]))


def test_wheel_inboard_outboard(sample_positions):
    # First compute wheel center
    wheel_center = update_wheel_center(sample_positions, wheel_offset=0.5)
    positions = sample_positions.copy()
    positions[PointID.WHEEL_CENTER] = wheel_center

    # Test inboard point (should be 0.5 units inboard of wheel center)
    inboard = update_wheel_inboard(positions, wheel_width=1.0)
    np.testing.assert_array_equal(inboard, np.array([2.0, 0.0, 0.0]))

    # Test outboard point (should be 0.5 units outboard of wheel center)
    outboard = update_wheel_outboard(positions, wheel_width=1.0)
    np.testing.assert_array_equal(outboard, np.array([3.0, 0.0, 0.0]))


def test_update_derived_point(sample_positions):
    result = update_derived_point(PointID.AXLE_MIDPOINT, sample_positions)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 0.0]))


def test_dependencies():
    deps = get_dependencies(PointID.WHEEL_CENTER)
    assert deps == {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD}

    deps = get_dependencies(PointID.WHEEL_INBOARD)
    assert deps == {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD}
