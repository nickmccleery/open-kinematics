from typing import Annotated, Callable, List, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from kinematics.constraints import Constraint
from kinematics.core import CoordinateAxis, KinematicsState, Positions
from kinematics.points.ids import PointID

AxisVector = Annotated[NDArray[np.float64], "shape=(3,)"]


class PointTarget(NamedTuple):
    point_id: PointID
    axis: CoordinateAxis | AxisVector
    value: float


class PointTargetSet(NamedTuple):
    values: list[PointTarget]


class SolverConfig(NamedTuple):
    ftol: float = 1e-8
    xtol: float = 1e-8
    verbose: int = 0


def solve_sweep(
    initial_state: KinematicsState,
    constraints: list[Constraint],
    targets: list[PointTargetSet],
    compute_derived_points_func: Callable[[Positions], Positions],
    solver_config: SolverConfig = SolverConfig(),
) -> list[Positions]:
    """
    Solves a series of kinematic states using damped non-linear least squares.
    """
    # Validate that all target sweeps have the same number of steps
    target_lengths = [len(target_set.values) for target_set in targets]
    if len(set(target_lengths)) > 1:
        raise ValueError(
            f"All target sets must have the same length. Found: {target_lengths}"
        )

    n_steps = target_lengths[0] if target_lengths else 0
    sweep_targets = [
        [target_set.values[i] for target_set in targets] for i in range(n_steps)
    ]

    states: List[Positions] = []
    current_state = initial_state.copy()

    def compute_residuals(
        free_array: np.ndarray, step_targets: List[PointTarget]
    ) -> np.ndarray:
        """
        Calculates the error (residual) for all constraints and targets for a given state.
        The solver's goal is to drive this vector of residuals to zero.
        """
        temp_state = current_state.copy()
        temp_state.update_positions_from_array(free_array)
        all_positions = compute_derived_points_func(temp_state.positions)

        residuals = []

        # 1. Geometry constraint residuals
        for c in constraints:
            # Extend is used to handle multi-residual constraints gracefully
            residuals.extend(c.get_residual(all_positions))

        # 2. Target residuals
        for target in step_targets:
            current_pos = all_positions[target.point_id]
            initial_pos = initial_state.positions[target.point_id]

            if isinstance(target.axis, CoordinateAxis):
                axis_idx = int(target.axis)
                target_value = initial_pos[axis_idx] + target.value
                residuals.append(current_pos[axis_idx] - target_value)
            else:  # Vector axis
                direction = np.asarray(target.axis, dtype=float)
                initial_proj = np.dot(initial_pos, direction)
                target_proj = initial_proj + target.value
                current_proj = np.dot(current_pos, direction)
                residuals.append(current_proj - target_proj)

        return np.array(residuals, dtype=float)

    # Use the initial state as the first guess for the first step
    initial_guess = current_state.get_free_points_as_array()

    for step_targets in sweep_targets:
        result = least_squares(
            fun=compute_residuals,
            x0=initial_guess,
            args=(step_targets,),
            method="lm",  # Levenberg-Marquardt
            ftol=solver_config.ftol,
            xtol=solver_config.xtol,
            verbose=solver_config.verbose,
        )

        if not result.success:
            raise RuntimeError(
                f"Solver failed to converge for targets: {step_targets}."
                f"\nMessage: {result.message}"
            )

        # Update the current state with the successful solution
        current_state.update_positions_from_array(result.x)
        final_positions = compute_derived_points_func(current_state.positions)
        states.append(final_positions.copy())

        # Use the solution from this step as the initial guess for the next
        initial_guess = result.x

    return states
