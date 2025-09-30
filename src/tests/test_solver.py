import numpy as np
import pytest

from kinematics.constraints import PointPointDistance
from kinematics.core import CoordinateAxis, KinematicsState, Positions
from kinematics.points.ids import PointID
from kinematics.solver import PointTarget, PointTargetSet, SolverConfig, solve_sweep

# Tolerance on position checks.
TOL_CHECK = 1e-4


@pytest.fixture
def simple_positions():
    positions_dict = {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([-1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: np.array([1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([0.0, 1.0, 0.0]),
    }
    return Positions(positions_dict)


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
def simple_target_set():
    displacements = [0.0, 0.5, 1.0]
    point_targets = [
        PointTarget(
            point_id=PointID.LOWER_WISHBONE_OUTBOARD,
            axis=CoordinateAxis.Z,
            value=d,
        )
        for d in displacements
    ]
    return PointTargetSet(values=point_targets)


def null_derived_points(positions: Positions) -> Positions:
    return positions.copy()


def test_solve_sweep(
    simple_positions,
    simple_constraints,
    simple_target_set,
    length_forward_leg,
    length_rearward_leg,
):
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}

    # Create KinematicsState instead of separate positions and free_points
    initial_state = KinematicsState(positions=simple_positions, free_points=free_points)

    # Extract displacement values for assertions
    displacement_values = [target.value for target in simple_target_set.values]

    states = solve_sweep(
        initial_state=initial_state,
        constraints=simple_constraints,
        targets=[simple_target_set],  # Wrapped in a list for the new API
        compute_derived_points_func=null_derived_points,
        solver_config=SolverConfig(ftol=1e-6, xtol=1e-6, verbose=0),
    )

    assert len(states) == len(displacement_values)

    # Check each state maintains constraints
    for i, state in enumerate(states):
        p_front = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        p_rear = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        p_outboard = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

        # Distance constraints
        assert np.linalg.norm(p_outboard - p_front) == pytest.approx(
            length_forward_leg, rel=TOL_CHECK
        )
        assert np.linalg.norm(p_outboard - p_rear) == pytest.approx(
            length_rearward_leg, rel=TOL_CHECK
        )

        # Target displacement
        assert p_outboard[2] == pytest.approx(displacement_values[i], rel=TOL_CHECK)
