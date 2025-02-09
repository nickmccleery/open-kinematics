import numpy as np
import pytest

from kinematics.constraints.types import PointPointDistance
from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.points.ids import PointID
from kinematics.solvers.core import (
    MotionTarget,
    compute_residuals,
    compute_target_residual,
    solve_positions,
    solve_sweep,
)
from kinematics.types.state import Positions


@pytest.fixture
def simple_positions():
    return {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([-1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: np.array([1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([0.0, 1.0, 0.0]),
    }


@pytest.fixture
def length_forward_leg(simple_positions):
    x_forward_leg = np.linalg.norm(
        simple_positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        - simple_positions[PointID.LOWER_WISHBONE_OUTBOARD],
    )

    return x_forward_leg


@pytest.fixture
def length_rearward_leg(simple_positions):
    x_rearward_leg = np.linalg.norm(
        simple_positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        - simple_positions[PointID.LOWER_WISHBONE_OUTBOARD]
    )

    return x_rearward_leg


@pytest.fixture
def simple_constraints(simple_positions, length_forward_leg, length_rearward_leg):
    return [
        PointPointDistance(
            p1=PointID.LOWER_WISHBONE_INBOARD_FRONT,
            p2=PointID.LOWER_WISHBONE_OUTBOARD,
            distance=length_forward_leg,
        ),
        PointPointDistance(
            p1=PointID.LOWER_WISHBONE_INBOARD_REAR,
            p2=PointID.LOWER_WISHBONE_OUTBOARD,
            distance=length_rearward_leg,
        ),
    ]


@pytest.fixture
def simple_target(simple_positions):
    return MotionTarget(
        point_id=PointID.LOWER_WISHBONE_OUTBOARD,
        axis=CoordinateAxis.Z,
        reference_position=simple_positions[PointID.LOWER_WISHBONE_OUTBOARD],
    )


def null_derived_points(positions: Positions) -> Positions:
    return positions.copy()


def test_compute_target_residual(simple_positions, simple_target):
    # Test initial position has zero residual
    residual = compute_target_residual(simple_positions, simple_target, 0.0)
    assert residual == pytest.approx(0.0)

    # Test residual with displacement
    residual = compute_target_residual(simple_positions, simple_target, 1.0)
    assert residual == pytest.approx(-1.0)  # Current Z is 0, target is 1


def test_compute_residuals(simple_positions, simple_constraints, simple_target):
    residuals = compute_residuals(
        simple_positions, simple_constraints, simple_target, displacement=0.0
    )

    # Should be three residuals: distance constraints and target.
    assert len(residuals) == 3
    assert np.allclose(residuals, 0.0)


def test_solve_positions(
    simple_positions,
    simple_constraints,
    simple_target,
    length_forward_leg,
    length_rearward_leg,
):
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}
    displacement = 1.0

    new_positions = solve_positions(
        positions=simple_positions,
        free_points=free_points,
        constraints=simple_constraints,
        target=simple_target,
        displacement=displacement,
        compute_derived_points=null_derived_points,
    )

    # Check distance constraint maintained.
    p_front = new_positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
    p_rear = new_positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
    p_outboard = new_positions[PointID.LOWER_WISHBONE_OUTBOARD]

    assert np.linalg.norm(p_outboard - p_front) == pytest.approx(length_forward_leg)
    assert np.linalg.norm(p_outboard - p_rear) == pytest.approx(length_rearward_leg)

    # Check target displacement achieved.
    assert p_outboard[2] == pytest.approx(1.0)


def test_solve_sweep(
    simple_positions,
    simple_constraints,
    simple_target,
    length_forward_leg,
    length_rearward_leg,
):
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}
    displacements = [0.0, 0.5, 1.0]

    states = solve_sweep(
        positions=simple_positions,
        free_points=free_points,
        constraints=simple_constraints,
        target=simple_target,
        displacements=displacements,
        compute_derived_points=null_derived_points,
    )

    assert len(states) == len(displacements)

    # Check each state maintains constraints
    for i, state in enumerate(states):
        p_front = state[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        p_rear = state[PointID.LOWER_WISHBONE_INBOARD_REAR]
        p_outboard = state[PointID.LOWER_WISHBONE_OUTBOARD]

        # Distance constraints
        assert np.linalg.norm(p_outboard - p_front) == pytest.approx(length_forward_leg)
        assert np.linalg.norm(p_outboard - p_rear) == pytest.approx(length_rearward_leg)

        # Target displacement
        assert p_outboard[2] == pytest.approx(displacements[i])
