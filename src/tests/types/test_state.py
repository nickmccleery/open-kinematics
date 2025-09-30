import numpy as np

from kinematics.core import GeometryDefinition, KinematicsState, Positions
from kinematics.points.ids import PointID


def test_kinematics_state_construction():
    # Test data setup.
    positions_data = {
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
    }
    positions = Positions(positions_data)
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}

    # Create state.
    state = KinematicsState(positions=positions, free_points=free_points)

    # Verify positions are stored correctly.
    np.testing.assert_array_equal(
        state.positions[PointID.LOWER_WISHBONE_OUTBOARD], np.array([1.0, 2.0, 3.0])
    )

    # Verify free points are stored.
    assert state.free_points == {PointID.LOWER_WISHBONE_OUTBOARD}

    # Verify fixed points are computed correctly.
    assert state.fixed_points == {PointID.UPPER_WISHBONE_OUTBOARD}


def test_kinematics_state_array_conversion():
    """Test the array conversion methods in KinematicsState."""
    positions_data = {
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
    }
    positions = Positions(positions_data)
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD}

    # Create state
    state = KinematicsState(positions=positions, free_points=free_points)

    # Test extraction - only testing get_free_points_as_array since update is now in Positions
    arr = state.get_free_points_as_array()
    # Should be sorted order: LOWER_WISHBONE_OUTBOARD, UPPER_WISHBONE_OUTBOARD
    expected_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(arr, expected_flat)


def test_geometry_definition_construction():
    # Test data setup.
    hard_points_data = {
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
    }
    hard_points = Positions(hard_points_data)
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}

    # Create definition.
    geom = GeometryDefinition(hard_points=hard_points, free_points=free_points)

    # Verify points are stored correctly.
    np.testing.assert_array_equal(
        geom.hard_points[PointID.LOWER_WISHBONE_OUTBOARD], np.array([1.0, 2.0, 3.0])
    )

    # Verify free points are stored.
    assert geom.free_points == {PointID.LOWER_WISHBONE_OUTBOARD}
