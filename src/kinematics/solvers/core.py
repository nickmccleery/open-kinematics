from typing import Annotated, Callable, NamedTuple, Set

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
from kinematics.types.state import Positions

AxisVector = Annotated[NDArray[np.float64], "shape=(3,)"]


class PointTarget(NamedTuple):
    point_id: PointID
    axis: CoordinateAxis | AxisVector
    value: float  # The distance along the axis to be targeted; offset from initial position.


class PointTargetSet(NamedTuple):
    values: list[PointTarget]


class SolverConfig(NamedTuple):
    ftol: float = 1e-6  # Function tolerance.
    residual_tolerance: float = 1e-4  # Constraint residual tolerance.
    max_iterations: int = 1000  # Maximum number of iterations.


def solve_positions(
    initial_positions: Positions,
    current_positions: Positions,
    constraints: list[Constraint],
    free_points: Set[PointID],
    point_targets: list[PointTarget],
    compute_derived_points: Callable[[Positions], Positions],
    residual_tolerance: float = 1e-4,
    ftol: float = 1e-6,
    max_iterations: int = 1000,
) -> Positions:
    """
    Solve a single kinematic state for the specified target positions.

    Args:
        initial_positions: Starting point for the system
        current_positions: Current state of the system (usually from previous solve)
        constraints: List of constraints to satisfy
        free_points: Set of points allowed to move
        point_targets: List of target specifications for specific points
        compute_derived_points: Function to compute derived points
        residual_tolerance: Tolerance for constraint satisfaction
        ftol: Function tolerance for optimization
        max_iterations: Maximum optimizer iterations

    Returns:
        Updated positions dictionary
    """
    # Copy so we don't mutate.
    current_positions = dict(current_positions)

    # Keep points ordered.
    point_ids = sorted(current_positions.keys())
    point_index_map = {pid: i for i, pid in enumerate(point_ids)}

    # Identify which point indices are free.
    i_free_points = [i for i, pid in enumerate(point_ids) if pid in free_points]

    def make_initial_guess() -> np.ndarray:
        guess = []
        for pid in sorted(free_points):
            guess.extend(current_positions[pid])  #
        return np.array(guess, dtype=float)

    def expand_positions(free_array: np.ndarray) -> np.ndarray:
        # Start with a copy of the current full positions array.
        full = np.zeros((len(point_ids), 3), dtype=float)

        # Fill in all positions from current state.
        for i, pid in enumerate(point_ids):
            full[i] = current_positions[pid]

        # Overwrite free positions.
        for i, free_idx in enumerate(i_free_points):
            start_idx = i * 3
            full[free_idx] = free_array[start_idx : start_idx + 3]

        return full

    def array_to_positions_dict(full_array: np.ndarray) -> Positions:
        return {pid: full_array[i].copy() for i, pid in enumerate(point_ids)}

    def compute_residuals(free_array: np.ndarray) -> np.ndarray:
        # Expand to full Nx3.
        full_positions_array = expand_positions(free_array)

        # Convert to dict, then compute derived points.
        pos_dict = array_to_positions_dict(full_positions_array)
        pos_dict = compute_derived_points(pos_dict)

        # Update full_positions_array with derived points.
        for i, pid in enumerate(point_ids):
            if pid in pos_dict:
                full_positions_array[i] = pos_dict[pid]

        residuals = []

        # (A) Geometry constraint residuals.
        for c in constraints:
            if isinstance(c, PointPointDistance):
                residuals.append(point_point_distance_residual(pos_dict, c))
            elif isinstance(c, VectorAngle):
                residuals.append(vector_angle_residual(pos_dict, c))
            elif isinstance(c, PointFixedAxis):
                residuals.append(point_fixed_axis_residual(pos_dict, c))
            elif isinstance(c, PointOnLine):
                residuals.append(point_on_line_residual(pos_dict, c))
            else:
                raise TypeError(f"Unknown constraint type: {type(c)}")

        # (B) Targets residuals.
        for target in point_targets:
            # Get target point position
            target_point_idx = point_index_map[target.point_id]
            target_point_pos = full_positions_array[target_point_idx]

            # Get reference value from initial positions.
            reference_value = initial_positions[target.point_id]

            if isinstance(target.axis, CoordinateAxis):
                # Each CoordinateAxis entry maps to a cartesian axis.
                i_axis = int(target.axis)

                # Compute the target value: reference + displacement.
                reference_value_on_axis = reference_value[i_axis]
                target_value = reference_value_on_axis + target.value

                # Residual is current position minus target position.
                residuals.append(target_point_pos[i_axis] - target_value)
            else:
                # It's a vector direction.
                direction = np.asarray(target.axis, dtype=float)

                # Project reference position onto direction.
                reference_projection = np.dot(reference_value, direction)

                # Target value is reference projection + displacement.
                target_value = reference_projection + target.value

                # Current projection.
                current_projection = np.dot(target_point_pos, direction)

                # Residual is current projection minus target projection.
                residuals.append(current_projection - target_value)

        return np.array(residuals, dtype=float)

    def objective(free_array: np.ndarray) -> float:
        r = compute_residuals(free_array)
        return float(np.sum(r**2))

    def constraint_ineq(free_array: np.ndarray) -> np.ndarray:
        r = compute_residuals(free_array)
        # SLSQP ineq: fun(x) >= 0 for feasibility.
        # We want abs(r) <= residual_tolerance => residual_tolerance - abs(r) >= 0.
        return residual_tolerance - np.abs(r)

    init_guess = make_initial_guess()

    result = minimize(
        objective,
        init_guess,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": constraint_ineq}],
        options={"ftol": ftol, "maxiter": max_iterations, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"solve_positions failed: {result.message}")

    # Convert final solution to a Positions dict & apply derived points.
    full_array_final = expand_positions(result.x)
    final_pos_dict = array_to_positions_dict(full_array_final)
    final_pos_dict = compute_derived_points(final_pos_dict)
    return final_pos_dict


def solve_sweep(
    initial_positions: Positions,
    constraints: list[Constraint],
    free_points: Set[PointID],
    targets: list[PointTargetSet],
    compute_derived_points: Callable[[Positions], Positions],
    solver_config: SolverConfig = SolverConfig(),
) -> list[Positions]:
    """
    Solve a series of kinematic states by sweeping through target values.

    Args:
        initial_positions: Starting point for the system
        constraints: List of constraints to satisfy
        free_points: Set of points allowed to move
        targets: List of target sets, each containing point targets for different variables
        compute_derived_points: Function to compute derived points
        solver_config: Configuration for the solver

    Returns:
        List of position dictionaries for each solved state
    """
    # Our targets must be aligned and the same length. Although we can target more
    # than one variable changing, i.e., we can steer while we bump, we cannot have
    # a set of targets for hub Z displacement with N elements and a set of targets
    # for rack position with M elements.

    # Validate.
    target_lengths = [len(target_set.values) for target_set in targets]
    if len(set(target_lengths)) > 1:
        raise ValueError(
            f"All target sets must have the same length. Found lengths: {target_lengths}"
        )

    # Build the actual sweep.
    sweep_targets = []
    n_steps = len(targets[0].values)

    for i in range(n_steps):
        sweep_targets.append([target_set.values[i] for target_set in targets])

    # Sweep.
    states = []
    current_positions = initial_positions.copy()

    for step_targets in sweep_targets:
        # Solve for a single step
        new_positions = solve_positions(
            initial_positions=initial_positions,
            current_positions=current_positions,
            constraints=constraints,
            free_points=free_points,
            point_targets=step_targets,
            compute_derived_points=compute_derived_points,
            residual_tolerance=solver_config.residual_tolerance,
            ftol=solver_config.ftol,
            max_iterations=solver_config.max_iterations,
        )
        states.append(new_positions)
        current_positions = new_positions

    return states
