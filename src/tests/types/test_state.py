import numpy as np

from kinematics.geometry.points.ids import PointID
from kinematics.types.state import GeometryDefinition, KinematicsState


def test_kinematics_state_construction():
    # Test data setup.
    positions = {
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
    }
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


def test_geometry_definition_construction():
    # Test data setup.
    hard_points = {
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
    }
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}

    # Create definition.
    geom = GeometryDefinition(hard_points=hard_points, free_points=free_points)

    # Verify points are stored correctly.
    np.testing.assert_array_equal(
        geom.hard_points[PointID.LOWER_WISHBONE_OUTBOARD], np.array([1.0, 2.0, 3.0])
    )

    # Verify free points are stored.
    assert geom.free_points == {PointID.LOWER_WISHBONE_OUTBOARD}
