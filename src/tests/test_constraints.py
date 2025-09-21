import numpy as np
import pytest

from kinematics.constraints import make_point_point_distance, make_vector_angle
from kinematics.points.main import PointID


@pytest.fixture
def positions():
    return {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: np.array([1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([0.5, 1.0, 0.0]),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 1.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([0.5, 1.0, 1.0]),
    }


def test_make_point_point_distance(positions):
    constraint = make_point_point_distance(
        positions, PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD
    )

    # Check the constraint was created correctly
    assert constraint.p1 == PointID.LOWER_WISHBONE_INBOARD_FRONT
    assert constraint.p2 == PointID.LOWER_WISHBONE_OUTBOARD
    assert constraint.distance == pytest.approx(np.sqrt(1.25))  # sqrt(0.5^2 + 1.0^2)


def test_make_vector_angle():
    # Create a simpler test case with known angle
    positions = {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 0.0, 0.0]),  # Along x-axis
        PointID.UPPER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([0.0, 1.0, 0.0]),  # Along y-axis
    }

    constraint = make_vector_angle(
        positions,
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
        PointID.UPPER_WISHBONE_OUTBOARD,
    )

    # Check the constraint was created correctly
    assert constraint.v1_start == PointID.LOWER_WISHBONE_INBOARD_FRONT
    assert constraint.v1_end == PointID.LOWER_WISHBONE_OUTBOARD
    assert constraint.v2_start == PointID.UPPER_WISHBONE_INBOARD_FRONT
    assert constraint.v2_end == PointID.UPPER_WISHBONE_OUTBOARD

    # Vectors should be at right angles (pi/2 radians)
    assert constraint.angle == pytest.approx(np.pi / 2)
