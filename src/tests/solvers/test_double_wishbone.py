from pathlib import Path

import numpy as np

from kinematics.geometry.loader import load_geometry
from kinematics.geometry.schemas import DoubleWishboneGeometry
from kinematics.solvers.double_wishbone import DoubleWishboneSolver, SuspensionState

CHECK_TOLERANCE = 1e-6


def test_run_solver(double_wishbone_geometry_file: Path) -> None:
    """Tests the double wishbone solver through its range of motion."""
    geometry = load_geometry(double_wishbone_geometry_file)

    if not isinstance(geometry, DoubleWishboneGeometry):
        raise ValueError("Invalid geometry type")

    solver = DoubleWishboneSolver(geometry)

    # Test range of motion from -50mm to +50mm in 10mm steps.
    displacements = np.linspace(-50, 50, 11)
    states = []

    for displacement in displacements:
        state = solver.solve_positions(displacement)
        assert isinstance(state, SuspensionState)
        states.append(state)

        # Verify constraints are maintained.
        for constraint in solver.constraints:
            point_map = {
                "upper_inboard_front": geometry.hard_points.upper_wishbone.inboard_front.as_array(),
                "upper_inboard_rear": geometry.hard_points.upper_wishbone.inboard_rear.as_array(),
                "lower_inboard_front": geometry.hard_points.lower_wishbone.inboard_front.as_array(),
                "lower_inboard_rear": geometry.hard_points.lower_wishbone.inboard_rear.as_array(),
                "upper_outboard": state.upper_outboard,
                "lower_outboard": state.lower_outboard,
                "axle_inner": state.axle_inner,
                "axle_outer": state.axle_outer,
                "axle_midpoint": (state.axle_inner + state.axle_outer) / 2,
            }

            p1 = point_map[constraint.point1_name]
            p2 = point_map[constraint.point2_name]
            current_length = np.linalg.norm(p1 - p2)

            # Check that constraint length is maintained within tolerance.
            assert np.abs(current_length - constraint.length) < CHECK_TOLERANCE, (
                f"Constraint violation at displacement {displacement}: "
                f"{constraint.point1_name} to {constraint.point2_name}"
            )

        # Verify axle midpoint moves by approximately the requested displacement.
        current_target_z = solver.initial_state.axle_inner[2] + displacement
        axle_midpoint = (state.axle_inner + state.axle_outer) / 2
        axle_midpoint_z = axle_midpoint[2]

        assert (
            np.abs(axle_midpoint_z - current_target_z) < CHECK_TOLERANCE
        ), f"Failed to maintain axle midpoint at displacement {displacement}"
