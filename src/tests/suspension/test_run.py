from pathlib import Path

import numpy as np

from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.loader import load_geometry
from kinematics.geometry.points.ids import PointID
from kinematics.solvers.core import DisplacementTargetSet
from kinematics.suspensions.double_wishbone.geometry import DoubleWishboneGeometry
from kinematics.suspensions.double_wishbone.main import solve_suspension
from visualization.debug import create_animation
from visualization.main import SuspensionVisualizer, WheelVisualization

# Our actual solve tolerance is a OOM tighter than this, so should be good.
EPSILON_CHECK = 1e-3


def test_run_solver(double_wishbone_geometry_file: Path) -> None:
    geometry = load_geometry(double_wishbone_geometry_file)
    if not isinstance(geometry, DoubleWishboneGeometry):
        raise ValueError("Invalid geometry type")

    # Create displacement sweep.
    displacement_range = [-120, 120]
    n_steps = 25
    displacements = list(
        np.linspace(displacement_range[0], displacement_range[1], n_steps)
    )

    target = DisplacementTargetSet(
        point_id=PointID.LOWER_WISHBONE_OUTBOARD,
        axis=CoordinateAxis.Z,
        displacements=displacements,
    )

    # Solve for all positions.
    position_states = solve_suspension(geometry, [target])

    print("Solve complete, verifying constraints...")

    # Get initial positions for comparison.
    from kinematics.suspensions.double_wishbone.main import (
        WheelConfig,
        compute_derived_points,
        create_initial_positions,
        create_length_constraints,
    )

    initial_positions = create_initial_positions(geometry)
    wheel_config = WheelConfig(
        width=geometry.configuration.wheel.width,
        offset=geometry.configuration.wheel.offset,
        diameter=geometry.configuration.wheel.diameter,
    )
    initial_positions = create_initial_positions(geometry)
    initial_positions = compute_derived_points(initial_positions, wheel_config)
    length_constraints = create_length_constraints(initial_positions)
    length_constraints = create_length_constraints(initial_positions)
    target_point_id = PointID.LOWER_WISHBONE_OUTBOARD

    # Verify constraints are maintained
    for positions, displacement in zip(position_states, displacements):
        # Verify length constraints
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

    r_aspect = 0.45
    x_section = 225
    x_diameter = 17 * 25.4

    wheel_config = WheelVisualization(
        diameter=x_diameter + r_aspect * x_section * 2,
        width=225,
    )

    visualizer = SuspensionVisualizer(geometry, wheel_config)
    create_animation(position_states_animate, visualizer, output_path)
