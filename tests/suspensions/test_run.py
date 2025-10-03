from pathlib import Path

import numpy as np
import pytest

from kinematics.constants import TEST_TOLERANCE
from kinematics.constraints import DistanceConstraint
from kinematics.core import PointID
from kinematics.loader import load_geometry
from kinematics.main import solve_suspension_sweep
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import PointTarget
from kinematics.suspensions.double_wishbone import DoubleWishboneGeometry
from kinematics.types import Axis, PointTargetAxis, SweepConfig, TargetPositionMode
from kinematics.visualization.debug import create_animation
from kinematics.visualization.main import SuspensionVisualizer, WheelVisualization

# Our actual solve tolerance is a OOM tighter than this, so should be good.


@pytest.fixture
def displacements():
    n_steps = 31

    hub_range = [-30, 90]
    hub_displacements = list(np.linspace(hub_range[0], hub_range[1], n_steps))

    steer_range = [-30, 30]
    steer_displacements = list(np.linspace(steer_range[0], steer_range[1], n_steps))

    return hub_displacements, steer_displacements


@pytest.fixture
def sweep_config_fixture(displacements):
    hub_displacements, steer_displacements = displacements

    # Create hub displacement sweep.
    hub_targets = [
        PointTarget(
            point_id=PointID.WHEEL_CENTER,
            direction=PointTargetAxis(Axis.Z),
            value=x,
            mode=TargetPositionMode.RELATIVE,
        )
        for x in hub_displacements
    ]

    # Create steer sweep.
    steer_targets = [
        PointTarget(
            point_id=PointID.TRACKROD_INBOARD,
            direction=PointTargetAxis(Axis.Y),
            value=x,
            mode=TargetPositionMode.RELATIVE,
        )
        for x in steer_displacements
    ]

    # Create sweep config.
    sweep_config = SweepConfig([hub_targets, steer_targets])

    return sweep_config


def test_run_solver(
    double_wishbone_geometry_file: Path, sweep_config_fixture, displacements
) -> None:
    hub_displacements, _ = displacements

    loaded = load_geometry(double_wishbone_geometry_file)
    if not isinstance(loaded.geometry, DoubleWishboneGeometry):
        raise ValueError("Invalid geometry type")

    # Solve for all positions.
    position_states = solve_suspension_sweep(
        loaded.geometry, loaded.provider_cls, sweep_config_fixture
    )

    print("Solve complete, verifying constraints...")

    # Get initial positions for comparison using the provider.

    provider = loaded.provider_cls(loaded.geometry)  # type: ignore[call-arg]
    derived_resolver = DerivedPointsManager(provider.derived_spec())

    initial_state = provider.initial_state()
    initial_positions = derived_resolver.update(initial_state.positions)

    # Get only the length constraints for verification
    all_constraints = provider.constraints()

    length_constraints = [
        c for c in all_constraints if isinstance(c, DistanceConstraint)
    ]
    target_point_id = PointID.WHEEL_CENTER

    # Verify constraints are maintained.
    for state, displacement in zip(position_states, hub_displacements):
        # Verify length constraints.
        for constraint in length_constraints:
            p1 = state.positions[constraint.p1]
            p2 = state.positions[constraint.p2]
            current_length = np.linalg.norm(p1 - p2)

            assert (
                np.abs(current_length - constraint.target_distance) < TEST_TOLERANCE
            ), (
                f"Constraint violation at displacement {displacement}: "
                f"{constraint.p1.name} to {constraint.p2.name}"
            )

        # Verify target point z position.
        target_point_position = state.positions[target_point_id]
        initial_target_point_position = initial_positions[target_point_id]
        target_z = initial_target_point_position[2] + displacement

        assert np.abs(target_point_position[2] - target_z) < TEST_TOLERANCE, (
            f"Failed to maintain {target_point_id} at displacement {displacement}"
        )

    print("Creating animation...")

    # Extract positions from SuspensionState objects for animation
    position_states_positions = [state.positions for state in position_states]
    position_states_animate = (
        position_states_positions + position_states_positions[::-1]
    )
    output_path = Path("suspension_motion.gif")

    r_aspect = 0.55
    x_section = 270
    x_diameter = 13 * 25.4

    wheel_config = WheelVisualization(
        diameter=x_diameter + r_aspect * x_section * 2,
        width=225,
    )

    visualizer = SuspensionVisualizer(loaded.geometry, wheel_config)
    create_animation(
        position_states_animate, initial_positions, visualizer, output_path
    )
