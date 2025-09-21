from pathlib import Path

import numpy as np
import pytest

from kinematics.geometry.loader import load_geometry
from kinematics.main import solve_kinematics
from kinematics.points.ids import PointID
from kinematics.primitives import CoordinateAxis
from kinematics.solver import PointTarget, PointTargetSet
from kinematics.suspensions import DoubleWishboneGeometry
from kinematics.visualization.debug import create_animation
from kinematics.visualization.main import SuspensionVisualizer, WheelVisualization

# Our actual solve tolerance is a OOM tighter than this, so should be good.
EPSILON_CHECK = 1e-3


@pytest.fixture
def displacements():
    n_steps = 31

    hub_range = [-30, 90]
    hub_displacements = list(np.linspace(hub_range[0], hub_range[1], n_steps))

    steer_range = [-30, 30]
    steer_displacements = list(np.linspace(steer_range[0], steer_range[1], n_steps))

    return hub_displacements, steer_displacements


@pytest.fixture
def target_set(displacements):
    hub_displacements, steer_displacements = displacements

    # Create hub displacement sweep.
    hub_targets = [
        PointTarget(
            point_id=PointID.WHEEL_CENTER,
            axis=CoordinateAxis.Z,
            value=x,
        )
        for x in hub_displacements
    ]

    # Create steer sweep.
    steer_targets = [
        PointTarget(
            point_id=PointID.TRACKROD_INBOARD,
            axis=CoordinateAxis.Y,
            value=x,
        )
        for x in steer_displacements
    ]

    # Create target set.
    targets = [
        PointTargetSet(values=hub_targets),
        PointTargetSet(values=steer_targets),
    ]

    return targets


def test_run_solver(
    double_wishbone_geometry_file: Path, target_set, displacements
) -> None:
    hub_displacements, _ = displacements

    geometry = load_geometry(double_wishbone_geometry_file)
    if not isinstance(geometry, DoubleWishboneGeometry):
        raise ValueError("Invalid geometry type")

    # Solve for all positions.
    position_states = solve_kinematics(geometry, target_set)

    print("Solve complete, verifying constraints...")

    # Get initial positions for comparison using the provider.
    from kinematics.points.derived.manager import DerivedPointManager
    from kinematics.suspensions import DoubleWishboneProvider

    provider = DoubleWishboneProvider(geometry)
    derived_point_manager = DerivedPointManager(
        provider.get_derived_point_definitions()
    )

    initial_positions = provider.get_initial_positions()
    initial_positions = derived_point_manager.update(initial_positions)

    # Get only the length constraints for verification
    all_constraints = provider.get_constraints(initial_positions)
    from kinematics.constraints import PointPointDistance

    length_constraints = [
        c for c in all_constraints if isinstance(c, PointPointDistance)
    ]
    target_point_id = PointID.WHEEL_CENTER

    # Verify constraints are maintained.
    for positions, displacement in zip(position_states, hub_displacements):
        # Verify length constraints.
        for constraint in length_constraints:
            p1 = positions[constraint.p1]
            p2 = positions[constraint.p2]
            current_length = np.linalg.norm(p1 - p2)

            assert np.abs(current_length - constraint.distance) < EPSILON_CHECK, (
                f"Constraint violation at displacement {displacement}: "
                f"{constraint.p1.name} to {constraint.p2.name}"
            )

        # Verify target point z position.
        target_point_position = positions[target_point_id]
        initial_target_point_position = initial_positions[target_point_id]
        target_z = initial_target_point_position[2] + displacement

        assert (
            np.abs(target_point_position[2] - target_z) < EPSILON_CHECK
        ), f"Failed to maintain {target_point_id} at displacement {displacement}"

    print("Creating animation...")

    position_states_animate = position_states + position_states[::-1]
    output_path = Path("suspension_motion.gif")

    r_aspect = 0.55
    x_section = 270
    x_diameter = 13 * 25.4

    wheel_config = WheelVisualization(
        diameter=x_diameter + r_aspect * x_section * 2,
        width=225,
    )

    visualizer = SuspensionVisualizer(geometry, wheel_config)
    create_animation(
        position_states_animate, initial_positions, visualizer, output_path
    )
