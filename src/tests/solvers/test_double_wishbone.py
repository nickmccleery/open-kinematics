from pathlib import Path

import numpy as np

from kinematics.geometry.loader import load_geometry
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.solvers.constraints import PointPointDistanceConstraint
from kinematics.solvers.double_wishbone import DoubleWishboneSolver
from visualization.debug import create_suspension_animation

CHECK_TOLERANCE = 1e-2


def test_run_solver(double_wishbone_geometry_file: Path) -> None:
    """Tests the double wishbone solver through its range of motion."""
    geometry = load_geometry(double_wishbone_geometry_file)

    if not isinstance(geometry, DoubleWishboneGeometry):
        raise ValueError("Invalid geometry type")

    solver = DoubleWishboneSolver(geometry=geometry)

    # Create displacement sweep
    displacement_range = [-100, 100]
    n_steps = 21
    displacements = list(
        np.linspace(displacement_range[0], displacement_range[1], n_steps)
    )

    # Solve for all positions
    states = solver.solve_sweep(displacements)

    # Verify constraints are maintained
    for state, displacement in zip(states, displacements):
        length_constraints = [
            x for x in solver.constraints if isinstance(x, PointPointDistanceConstraint)
        ]
        # Verify length constraints
        for constraint in length_constraints:
            p1 = state.hard_points[constraint.p1].as_array()
            p2 = state.hard_points[constraint.p2].as_array()
            current_length = np.linalg.norm(p1 - p2)

            assert np.abs(current_length - constraint.distance) < CHECK_TOLERANCE, (
                f"Constraint violation at displacement {displacement}: "
                f"{constraint.p1} to {constraint.p2}"
            )

        # Verify axle midpoint z position
        axle_inner = state.hard_points[PointID.AXLE_INBOARD].as_array()
        axle_outer = state.hard_points[PointID.AXLE_OUTBOARD].as_array()
        axle_midpoint = (axle_inner + axle_outer) / 2
        initial_midpoint = (
            solver.initial_state.hard_points[PointID.AXLE_INBOARD].as_array()
            + solver.initial_state.hard_points[PointID.AXLE_OUTBOARD].as_array()
        ) / 2
        target_z = initial_midpoint[2] + displacement

        assert (
            np.abs(axle_midpoint[2] - target_z) < CHECK_TOLERANCE
        ), f"Failed to maintain axle midpoint at displacement {displacement}"

    # Create animation
    states_animate = states + states[::-1]
    output_path = Path("suspension_motion.gif")
    create_suspension_animation(geometry, states_animate, output_path)
