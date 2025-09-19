from functools import partial

import numpy as np
import pytest

from kinematics.geometry.points.ids import PointID
from kinematics.derived_points import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.solver.manager import DerivedPointManager


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
    result = get_axle_midpoint(sample_positions)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 0.0]))


def test_wheel_center(sample_positions):
    # With axle outboard at x=2.0 and offset of 0.5, wheel center should be at x=2.5
    result = get_wheel_center(sample_positions, wheel_offset=0.5)
    np.testing.assert_array_equal(result, np.array([2.5, 0.0, 0.0]))


def test_wheel_inboard_outboard(sample_positions):
    # First compute wheel center
    wheel_center = get_wheel_center(sample_positions, wheel_offset=0.5)
    positions = sample_positions.copy()
    positions[PointID.WHEEL_CENTER] = wheel_center

    # Test inboard point (should be 0.5 units inboard of wheel center)
    inboard = get_wheel_inboard(positions, wheel_width=1.0)
    np.testing.assert_array_equal(inboard, np.array([2.0, 0.0, 0.0]))

    # Test outboard point (should be 0.5 units outboard of wheel center)
    outboard = get_wheel_outboard(positions, wheel_width=1.0)
    np.testing.assert_array_equal(outboard, np.array([3.0, 0.0, 0.0]))


def test_dependency_manager_basic(sample_positions):
    """Test that the DerivedPointManager can calculate axle midpoint"""
    definitions = {
        PointID.AXLE_MIDPOINT: (
            get_axle_midpoint,
            {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        )
    }
    manager = DerivedPointManager(definitions)

    result = manager.update(sample_positions)

    # Should have original positions plus the derived point
    assert len(result) == 3
    assert PointID.AXLE_MIDPOINT in result
    np.testing.assert_array_equal(
        result[PointID.AXLE_MIDPOINT], np.array([1.0, 0.0, 0.0])
    )


def test_dependency_manager_complex(sample_positions):
    """Test that the DerivedPointManager can handle dependencies between derived points"""
    definitions = {
        PointID.AXLE_MIDPOINT: (
            get_axle_midpoint,
            {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        ),
        PointID.WHEEL_CENTER: (
            partial(get_wheel_center, wheel_offset=0.5),
            {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        ),
        PointID.WHEEL_INBOARD: (
            partial(get_wheel_inboard, wheel_width=1.0),
            {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        ),
        PointID.WHEEL_OUTBOARD: (
            partial(get_wheel_outboard, wheel_width=1.0),
            {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        ),
    }
    manager = DerivedPointManager(definitions)

    result = manager.update(sample_positions)

    # Should have original positions plus all derived points
    assert len(result) == 6
    assert PointID.AXLE_MIDPOINT in result
    assert PointID.WHEEL_CENTER in result
    assert PointID.WHEEL_INBOARD in result
    assert PointID.WHEEL_OUTBOARD in result

    # Check values
    np.testing.assert_array_equal(
        result[PointID.AXLE_MIDPOINT], np.array([1.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        result[PointID.WHEEL_CENTER], np.array([2.5, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        result[PointID.WHEEL_INBOARD], np.array([2.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        result[PointID.WHEEL_OUTBOARD], np.array([3.0, 0.0, 0.0])
    )


def test_circular_dependency_detection():
    """Test that circular dependencies are detected"""

    def dummy_func(positions):
        return positions[PointID.AXLE_INBOARD]

    definitions = {
        PointID.WHEEL_CENTER: (dummy_func, {PointID.WHEEL_OUTBOARD}),
        PointID.WHEEL_OUTBOARD: (dummy_func, {PointID.WHEEL_CENTER}),
    }

    with pytest.raises(ValueError, match="Circular dependency detected"):
        DerivedPointManager(definitions)
        DerivedPointManager(definitions)
