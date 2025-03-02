import numpy as np
import pytest

from kinematics.constraints.ops import (
    point_fixed_axis_residual,
    point_on_line_residual,
    point_point_distance_residual,
    vector_angle_residual,
)
from kinematics.constraints.types import (
    PointFixedAxis,
    PointOnLine,
    PointPointDistance,
    VectorAngle,
)
from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.points.ids import PointID


@pytest.fixture
def simple_positions():
    return {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: np.array([1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([0.5, 1.0, 0.0]),
    }


def test_point_point_distance():
    positions = {
        PointID.AXLE_INBOARD: np.array([0.0, 0.0, 0.0]),
        PointID.AXLE_OUTBOARD: np.array([1.0, 0.0, 0.0]),
    }

    constraint = PointPointDistance(
        p1=PointID.AXLE_INBOARD, p2=PointID.AXLE_OUTBOARD, distance=1.0
    )

    # When points are at correct distance
    assert point_point_distance_residual(positions, constraint) == pytest.approx(0.0)

    # When points are too far apart
    positions[PointID.AXLE_OUTBOARD] = np.array([2.0, 0.0, 0.0])
    assert point_point_distance_residual(positions, constraint) == pytest.approx(1.0)


def test_vector_angle():
    # Set up two vectors at right angles
    positions = {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 0.0, 0.0]),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: np.array([0.0, 0.0, 0.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([0.0, 1.0, 0.0]),
    }

    constraint = VectorAngle(
        v1_start=PointID.LOWER_WISHBONE_INBOARD_FRONT,
        v1_end=PointID.LOWER_WISHBONE_OUTBOARD,
        v2_start=PointID.UPPER_WISHBONE_INBOARD_FRONT,
        v2_end=PointID.UPPER_WISHBONE_OUTBOARD,
        angle=np.pi / 2,
    )

    # When vectors are at correct angle
    assert vector_angle_residual(positions, constraint) == pytest.approx(0.0)

    # When angle is wrong
    positions[PointID.UPPER_WISHBONE_OUTBOARD] = np.array([1.0, 1.0, 0.0])
    assert vector_angle_residual(positions, constraint) != pytest.approx(0.0)


def test_point_fixed_axis():
    positions = {
        PointID.TRACKROD_INBOARD: np.array([1.0, 2.0, 3.0]),
    }

    constraint = PointFixedAxis(
        point_id=PointID.TRACKROD_INBOARD, axis=CoordinateAxis.Y, value=2.0
    )

    # When point is at correct coordinate
    assert point_fixed_axis_residual(positions, constraint) == pytest.approx(0.0)

    # When point is off target
    positions[PointID.TRACKROD_INBOARD] = np.array([1.0, 3.0, 3.0])
    assert point_fixed_axis_residual(positions, constraint) == pytest.approx(1.0)


def test_point_on_line():
    positions = {
        PointID.TRACKROD_INBOARD: np.array([0.0, 0.0, 0.0]),
        PointID.TRACKROD_OUTBOARD: np.array([0.0, 1.0, 0.0]),
    }

    constraint = PointOnLine(
        point_id=PointID.TRACKROD_OUTBOARD,
        line_point=positions[PointID.TRACKROD_INBOARD],
        line_direction=np.array([0.0, 1.0, 0.0]),  # Points along Y axis
    )

    # When point is on line
    assert point_on_line_residual(positions, constraint) == pytest.approx(0.0)

    # When point is off line
    positions[PointID.TRACKROD_OUTBOARD] = np.array([1.0, 1.0, 0.0])
    assert point_on_line_residual(positions, constraint) == pytest.approx(1.0)
