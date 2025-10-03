import numpy as np
import pytest

from kinematics.constants import TEST_TOLERANCE
from kinematics.constraints import DistanceConstraint
from kinematics.core import PointID, SuspensionState
from kinematics.solver import PointTarget, SolverConfig, solve_sweep
from kinematics.types import Axis, PointTargetAxis, SweepConfig


@pytest.fixture
def simple_positions():
    positions_dict = {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: np.array([-1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: np.array([1.0, 0.0, 0.0]),
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([0.0, 1.0, 0.0]),
    }
    return positions_dict


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
        DistanceConstraint(
            p1=PointID.LOWER_WISHBONE_INBOARD_FRONT,
            p2=PointID.LOWER_WISHBONE_OUTBOARD,
            target_distance=length_forward_leg,
        ),
        DistanceConstraint(
            p1=PointID.LOWER_WISHBONE_INBOARD_REAR,
            p2=PointID.LOWER_WISHBONE_OUTBOARD,
            target_distance=length_rearward_leg,
        ),
    ]


@pytest.fixture
def simple_sweep_config():
    displacements = [0.0, 0.5, 1.0]
    point_targets = [
        PointTarget(
            point_id=PointID.LOWER_WISHBONE_OUTBOARD,
            direction=PointTargetAxis(Axis.Z),
            value=d,
        )
        for d in displacements
    ]
    return SweepConfig([point_targets])


def null_derived_points(positions):
    return positions.copy()


def test_solve_sweep(
    simple_positions,
    simple_constraints,
    simple_sweep_config,
    length_forward_leg,
    length_rearward_leg,
):
    free_points = {PointID.LOWER_WISHBONE_OUTBOARD}

    # Create SuspensionState instead of separate positions and free_points
    initial_state = SuspensionState(positions=simple_positions, free_points=free_points)

    # Extract displacement values for assertions
    displacement_values = [
        target.value for target in simple_sweep_config.target_sweeps[0]
    ]

    states = solve_sweep(
        initial_state=initial_state,
        constraints=simple_constraints,
        sweep_config=simple_sweep_config,
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
            length_forward_leg, rel=TEST_TOLERANCE
        )
        assert np.linalg.norm(p_outboard - p_rear) == pytest.approx(
            length_rearward_leg, rel=TEST_TOLERANCE
        )

        # Target displacement
        assert p_outboard[2] == pytest.approx(
            displacement_values[i], rel=TEST_TOLERANCE
        )
