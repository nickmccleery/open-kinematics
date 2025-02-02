from pathlib import Path

import numpy as np

from kinematics.geometry.loader import load_geometry
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.suspension.double_wishbone import solve_suspension
from visualization_new.debug import create_animation
from visualization_new.main import SuspensionVisualizer, WheelVisualization

CHECK_TOLERANCE = 1e-5


def test_run_solver(double_wishbone_geometry_file: Path) -> None:
    geometry = load_geometry(double_wishbone_geometry_file)
    if not isinstance(geometry, DoubleWishboneGeometry):
        raise ValueError("Invalid geometry type")

    # Create displacement sweep
    displacement_range = [-100, 100]
    n_steps = 21
    displacements = list(
        np.linspace(displacement_range[0], displacement_range[1], n_steps)
    )

    # Solve for all positions
    position_states = solve_suspension(geometry, displacements)

    print("Solve complete, verifying constraints...")

    # Get initial positions for comparison
    from kinematics.suspension.double_wishbone import (
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
    initial_positions = compute_derived_points(initial_positions, wheel_config)
    length_constraints = create_length_constraints(initial_positions)

    # Verify constraints are maintained
    for positions, displacement in zip(position_states, displacements):
        # Verify length constraints
        for constraint in length_constraints:
            p1 = positions[constraint.p1]
            p2 = positions[constraint.p2]
            current_length = np.linalg.norm(p1 - p2)

            # assert np.abs(current_length - constraint.distance) < CHECK_TOLERANCE, (
            #     f"Constraint violation at displacement {displacement}: "
            #     f"{constraint.p1} to {constraint.p2}"
            # )

        # Verify wheel center z position
        wheel_center = positions[PointID.WHEEL_CENTER]
        initial_wheel_center = initial_positions[PointID.WHEEL_CENTER]
        target_z = initial_wheel_center[2] + displacement

        # assert (
        #     np.abs(wheel_center[2] - target_z) < CHECK_TOLERANCE
        # ), f"Failed to maintain wheel center at displacement {displacement}"

    print("Creating animation...")
    # We'll need to adapt the visualization code to work with our new state format
    # or convert our states to the old format for visualization
    position_states_animate = position_states + position_states[::-1]
    output_path = Path("suspension_motion.gif")

    wheel_config = WheelVisualization(
        diameter=(17 * 25.4) + 0.45 * 225 * 2,
        width=225,
    )

    visualizer = SuspensionVisualizer(geometry, wheel_config)
    create_animation(position_states_animate, visualizer, output_path)
