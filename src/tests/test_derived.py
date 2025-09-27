"""Test the DFS-based derived point calculation system."""

import numpy as np
import pytest

from kinematics.core import PointID
from kinematics.derived import DerivedPointManager, DerivedSpec


def test_simple_derived_point_calculation():
    """Test basic derived point calculation without dependencies."""

    # Create a simple spec that calculates the midpoint of two points
    def midpoint_func(positions):
        p1 = positions[PointID.AXLE_INBOARD]
        p2 = positions[PointID.AXLE_OUTBOARD]
        return (p1 + p2) / 2

    functions = {PointID.AXLE_MIDPOINT: midpoint_func}

    dependencies = {
        PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD}
    }

    spec = DerivedSpec(functions=functions, dependencies=dependencies)
    manager = DerivedPointManager(spec)

    # Test positions
    positions = {
        PointID.AXLE_INBOARD: np.array([0.0, 0.0, 0.0]),
        PointID.AXLE_OUTBOARD: np.array([10.0, 0.0, 0.0]),
    }

    result = manager.update(positions)

    # Check that midpoint was calculated correctly
    expected_midpoint = np.array([5.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        result[PointID.AXLE_MIDPOINT], expected_midpoint
    )


def test_chained_derived_point_calculation():
    """Test derived points that depend on other derived points."""
    # Create a chain: AXLE_MIDPOINT -> WHEEL_CENTER -> WHEEL_INBOARD

    def midpoint_func(positions):
        p1 = positions[PointID.AXLE_INBOARD]
        p2 = positions[PointID.AXLE_OUTBOARD]
        return (p1 + p2) / 2

    def wheel_center_func(positions):
        midpoint = positions[PointID.AXLE_MIDPOINT]
        return midpoint + np.array([0.0, 5.0, 0.0])  # Offset by 5 in Y

    def wheel_inboard_func(positions):
        center = positions[PointID.WHEEL_CENTER]
        return center - np.array([0.0, 2.0, 0.0])  # Offset back by 2 in Y

    functions = {
        PointID.AXLE_MIDPOINT: midpoint_func,
        PointID.WHEEL_CENTER: wheel_center_func,
        PointID.WHEEL_INBOARD: wheel_inboard_func,
    }

    dependencies = {
        PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        PointID.WHEEL_CENTER: {PointID.AXLE_MIDPOINT},
        PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER},
    }

    spec = DerivedSpec(functions=functions, dependencies=dependencies)
    manager = DerivedPointManager(spec)

    # Verify the calculation order is correct (dependencies first)
    expected_order = [
        PointID.AXLE_MIDPOINT,
        PointID.WHEEL_CENTER,
        PointID.WHEEL_INBOARD,
    ]
    assert manager.update_order == expected_order

    # Test positions
    positions = {
        PointID.AXLE_INBOARD: np.array([0.0, 0.0, 0.0]),
        PointID.AXLE_OUTBOARD: np.array([10.0, 0.0, 0.0]),
    }

    result = manager.update(positions)

    # Check all derived points were calculated correctly
    expected_midpoint = np.array([5.0, 0.0, 0.0])
    expected_wheel_center = np.array([5.0, 5.0, 0.0])
    expected_wheel_inboard = np.array([5.0, 3.0, 0.0])

    np.testing.assert_array_almost_equal(
        result[PointID.AXLE_MIDPOINT], expected_midpoint
    )
    np.testing.assert_array_almost_equal(
        result[PointID.WHEEL_CENTER], expected_wheel_center
    )
    np.testing.assert_array_almost_equal(
        result[PointID.WHEEL_INBOARD], expected_wheel_inboard
    )


def test_circular_dependency_detection():
    """Test that circular dependencies are detected and raise an error."""

    def func_a(positions):
        return positions[PointID.WHEEL_CENTER] + np.array([1.0, 0.0, 0.0])

    def func_b(positions):
        return positions[PointID.WHEEL_INBOARD] + np.array([0.0, 1.0, 0.0])

    # Create a circular dependency: WHEEL_CENTER -> WHEEL_INBOARD -> WHEEL_CENTER
    functions = {PointID.WHEEL_CENTER: func_a, PointID.WHEEL_INBOARD: func_b}

    dependencies = {
        PointID.WHEEL_CENTER: {PointID.WHEEL_INBOARD},  # A depends on B
        PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER},  # B depends on A (circular!)
    }

    spec = DerivedSpec(functions=functions, dependencies=dependencies)

    # Creating the manager should detect the circular dependency and raise an error
    with pytest.raises(ValueError, match="Circular dependency detected"):
        DerivedPointManager(spec)
