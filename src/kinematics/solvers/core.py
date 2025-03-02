from typing import Callable, List, NamedTuple, Sequence, Set

import numpy as np
from scipy.optimize import minimize

from kinematics.constraints.ops import (
    point_fixed_axis_residual,
    point_on_line_residual,
    point_point_distance_residual,
    vector_angle_residual,
)
from kinematics.constraints.types import (
    Constraint,
    PointFixedAxis,
    PointOnLine,
    PointPointDistance,
    VectorAngle,
)
from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.points.ids import PointID
from kinematics.types.state import Position, Positions


class DisplacementTargetSet(NamedTuple):
    point_id: PointID
    axis: CoordinateAxis
    displacements: Sequence[float]


class MotionTarget(NamedTuple):
    point_id: PointID
    axis: CoordinateAxis
    reference_position: Position


class SolverConfig(NamedTuple):
    ftol: float = 1e-6  # Function tolerance.
    residual_tolerance: float = 1e-4  # Constraint residual tolerance.
    max_iterations: int = 1000  # Maximum number of iterations.


def solve_sweep(
    positions: Positions,
    free_points: Set[PointID],
    constraints: List[Constraint],
    target: MotionTarget,
    displacements: Sequence[float],
    compute_derived_points: Callable[[Positions], Positions],
    config: SolverConfig = SolverConfig(),
) -> List[Positions]:
    """
    Solve a series of kinematic states for specified displacements.

    Args:
        positions: Initial positions of all points.
        free_points: Set of points allowed to move during optimization.
        constraints: List of constraints to satisfy.
        target: Target point and axis for the motion.
        displacements: Sequence of displacement values to solve for.
        compute_derived_points: Function to compute derived point positions.
        config: Solver configuration parameters.

    Returns:
        List of position dictionaries for each solved state.
    """
    states = []
    current_positions = dict(positions)  # Create a copy to avoid modifying original

    # Create static mappings between IDs and indices.
    all_point_ids = sorted(current_positions.keys())
    point_to_idx = {point_id: i for i, point_id in enumerate(all_point_ids)}
    i_free_points = [point_to_idx[p] for p in free_points]

    # Pre-process the target information.
    i_target_point = point_to_idx[target.point_id]
    i_target_axis = target.axis
    target_ref_value = target.reference_position[i_target_axis]

    # Function to convert from compact representation (only free points) to full
    # representation.
    def expand_positions(free_positions_flat):
        # Start with a copy of the current full positions array.
        full_positions = np.zeros((len(all_point_ids), 3))

        # Fill in fixed positions.
        for point_id, pos in current_positions.items():
            if point_id not in free_points:
                full_positions[point_to_idx[point_id]] = pos

        # Update free positions.
        for i, free_idx in enumerate(i_free_points):
            start_idx = i * 3
            full_positions[free_idx] = free_positions_flat[start_idx : start_idx + 3]

        return full_positions

    def array_to_positions_dict(positions_array):
        return {
            point_id: positions_array[point_to_idx[point_id]]
            for point_id in all_point_ids
        }

    def compute_residuals_np(full_positions_array, displacement):
        # Convert to dictionary for constraint functions.
        pos_dict = array_to_positions_dict(full_positions_array)
        pos_dict = compute_derived_points(pos_dict)

        # Update full_positions_array with derived points.
        for point_id, pos in pos_dict.items():
            if point_id in point_to_idx:
                full_positions_array[point_to_idx[point_id]] = pos

        # Compute constraint residuals.
        residuals = []
        for constraint in constraints:
            if isinstance(constraint, PointPointDistance):
                residuals.append(point_point_distance_residual(pos_dict, constraint))
            elif isinstance(constraint, VectorAngle):
                residuals.append(vector_angle_residual(pos_dict, constraint))
            elif isinstance(constraint, PointFixedAxis):
                residuals.append(point_fixed_axis_residual(pos_dict, constraint))
            elif isinstance(constraint, PointOnLine):
                residuals.append(point_on_line_residual(pos_dict, constraint))

        # Add target motion residual.
        target_point_pos = full_positions_array[i_target_point]
        target_residual = target_point_pos[i_target_axis] - (
            target_ref_value + displacement
        )
        residuals.append(target_residual)

        return np.array(residuals)

    def objective(free_positions_flat, displacement):
        # Objective function: sum of squared residuals, but sort of unnecessary.
        full_positions = expand_positions(free_positions_flat)
        residuals = compute_residuals_np(full_positions, displacement)
        return np.sum(residuals**2)

    def constraint_function(free_positions_flat, displacement, tolerance):
        # Constraint function: all residuals within tolerance.
        full_positions = expand_positions(free_positions_flat)
        residuals = compute_residuals_np(full_positions, displacement)
        return tolerance - np.abs(residuals)  # Must be >= 0 to satisfy constraint

    # Sweep.
    for displacement in displacements:
        # Create initial guess for free points.
        initial_guess = np.concatenate(
            [current_positions[p] for p in sorted(free_points)]
        )

        # Solve system.
        result = minimize(
            lambda x: objective(x, displacement),
            initial_guess,
            method="SLSQP",
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda x: constraint_function(
                        x, displacement, config.residual_tolerance
                    ),
                }
            ],
            options={
                "ftol": config.ftol,
                "maxiter": config.max_iterations,
                "disp": False,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve for displacement {displacement}: {result.message}"
            )

        # Convert result back to full position dictionary.
        full_positions = expand_positions(result.x)
        new_positions = array_to_positions_dict(full_positions)

        # Apply derived points computation.
        new_positions = compute_derived_points(new_positions)

        # Add to states and update current positions.
        states.append(new_positions)
        current_positions = new_positions

    return states
