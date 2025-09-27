import numpy as np
import pytest

from kinematics.constraints import AngleConstraint, DistanceConstraint
from kinematics.points.ids import PointID


@pytest.fixture
def positions():
    return {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: np.array([1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([0.5, 1.0, 0.0]),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 1.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([0.5, 1.0, 1.0]),
    }


def test_distance_constraint(positions):
    target_distance = float(
        np.linalg.norm(
            positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
            - positions[PointID.LOWER_WISHBONE_OUTBOARD]
        )
    )

    constraint = DistanceConstraint(
        PointID.LOWER_WISHBONE_INBOARD_FRONT,
        PointID.LOWER_WISHBONE_OUTBOARD,
        target_distance,
    )

    # Test that constraint is satisfied at initial positions
    assert abs(constraint.residual(positions)) < 1e-10

    # Test that constraint detects violations
    modified_positions = positions.copy()
    modified_positions[PointID.LOWER_WISHBONE_OUTBOARD] += np.array([1.0, 0.0, 0.0])

    residual = constraint.residual(modified_positions)
    assert abs(residual) > 0.1  # Should be approximately 1.0


def test_angle_constraint():
    positions = {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 0.0, 0.0]),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([0.0, 1.0, 0.0]),
    }

    constraint = AngleConstraint(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,
        v1_end=PointID.LOWER_WISHBONE_OUTBOARD,
        v2_start=PointID.UPPER_WISHBONE_INBOARD_FRONT,
        v2_end=PointID.UPPER_WISHBONE_OUTBOARD,
        target_angle=np.pi / 2,  # 90 degrees
    )

    # Should be satisfied at right angle
    assert abs(constraint.residual(positions)) < 1e-10
