from typing import Callable, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray
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


# Configuration types
class MotionTarget(NamedTuple):
    point_id: PointID
    axis: CoordinateAxis
    reference_position: Position


class SolverConfig(NamedTuple):
    ftol: float = 1e-6  # Constraint residual tolerance.
    xtol: float = 1e-6  # Step size tolerance.
    gtol: float = 1e-6  # Gradient tolerance.
    residual_tolerance: float = 1e-4  # Constraint residual tolerance.
    max_iterations: int = 2000  # Maximum number of iterations.


# Core solver functions
def compute_target_residual(
    positions: Positions, target: MotionTarget, displacement: float
) -> float:
    current_pos = positions[target.point_id]
    reference_pos = target.reference_position

    # Create target position by adding displacement to reference
    target_pos = reference_pos.copy()
    target_pos[target.axis] += displacement

    error = current_pos[target.axis] - target_pos[target.axis]
    return error


def compute_residuals(
    positions: Positions,
    constraints: list[Constraint],
    target: MotionTarget,
    displacement: float,
) -> NDArray:
    residuals = []

    # Constraint residuals
    for constraint in constraints:
        if isinstance(constraint, PointPointDistance):
            residuals.append(point_point_distance_residual(positions, constraint))
        elif isinstance(constraint, VectorAngle):
            residuals.append(vector_angle_residual(positions, constraint))
        elif isinstance(constraint, PointFixedAxis):
            residuals.append(point_fixed_axis_residual(positions, constraint))
        elif isinstance(constraint, PointOnLine):
            residuals.append(point_on_line_residual(positions, constraint))

    # Target residual
    target_residual = compute_target_residual(positions, target, displacement)
    residuals.append(target_residual)

    return np.array(residuals)


def solve_positions(
    positions: Positions,
    free_points: set[PointID],
    constraints: list[Constraint],
    target: MotionTarget,
    displacement: float,
    compute_derived_points: Callable[[Positions], Positions],
    config: SolverConfig = SolverConfig(),
) -> Positions:
    # Pull points into a consistently ordered array.
    free_points_ordered = sorted(free_points, key=lambda point: point.value)
    initial_guess = np.array([positions[i_point] for i_point in free_points_ordered])

    # Reshape to a vector so we can feed to the optimiser.
    initial_guess_1d = np.reshape(initial_guess, -1)

    def residual_wrapper(guess_1d: NDArray) -> NDArray:
        # Computes residuals for given free point positions.
        _positions = positions.copy()
        for i, i_point in enumerate(free_points_ordered):
            _positions[i_point] = guess_1d[i * 3 : (i + 1) * 3]

        _positions = compute_derived_points(_positions)
        return compute_residuals(_positions, constraints, target, displacement)

    def constraint_function(guess_1d: NDArray) -> NDArray:
        # COBYLA requires constraints to be g(x) >= 0, so we invert residuals.
        return -np.abs(residual_wrapper(guess_1d)) + config.residual_tolerance

    # Solve system using COBYLA.
    result = minimize(
        lambda x: 0,  # No explicit cost function; just satisfy constraints.
        initial_guess_1d,
        method="COBYLA",
        constraints={"type": "ineq", "fun": constraint_function},
        options={"tol": config.ftol, "maxiter": config.max_iterations},
    )

    if not result.success:
        raise RuntimeError(
            f"Failed to solve for displacement {displacement}m: {result.message}"
        )

    # Update positions with solution.
    new_positions = positions.copy()
    for i, pid in enumerate(free_points_ordered):
        new_positions[pid] = result.x[i * 3 : (i + 1) * 3]

    # Update derived points.
    new_positions = compute_derived_points(new_positions)

    return new_positions


def solve_sweep(
    positions: Positions,
    free_points: set[PointID],
    constraints: list[Constraint],
    target: MotionTarget,
    displacements: Sequence[float],
    compute_derived_points: Callable[[Positions], Positions],
    config: SolverConfig = SolverConfig(),
) -> list[Positions]:
    states = []
    current_positions = positions

    for displacement in displacements:
        new_positions = solve_positions(
            current_positions,
            free_points,
            constraints,
            target,
            displacement,
            compute_derived_points,
            config,
        )

        states.append(new_positions)
        current_positions = new_positions

    return states
