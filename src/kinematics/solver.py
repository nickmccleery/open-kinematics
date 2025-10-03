from typing import Callable, NamedTuple

import numpy as np
from scipy.optimize import least_squares

from kinematics.constraints import Constraint
from kinematics.core import PointID, SuspensionState
from kinematics.targets import resolve_target
from kinematics.types import PointTarget, SweepConfig


class SolverConfig(NamedTuple):
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    verbose: int = 0


def solve_sweep(
    initial_state: SuspensionState,
    constraints: list[Constraint],
    sweep_config: SweepConfig,
    compute_derived_points_func: Callable[
        [dict[PointID, np.ndarray]], dict[PointID, np.ndarray]
    ],
    solver_config: SolverConfig = SolverConfig(),
) -> list[SuspensionState]:
    """
    Solves a series of kinematic states using damped non-linear least squares.
    """
    n_steps = sweep_config.n_steps
    sweep_targets = [
        [sweep[i] for sweep in sweep_config.target_sweeps] for i in range(n_steps)
    ]

    # Initialize state for the entire sweep - this object will only be updated after each successful step
    current_state = initial_state.copy()
    states: list[SuspensionState] = []  # Will store the final results

    # Create the single, reusable "scratchpad" state for calculations
    # This eliminates allocations inside the compute_residuals function
    scratch_state = current_state.copy()

    def compute_residuals(
        free_array: np.ndarray, step_targets: list[PointTarget]
    ) -> np.ndarray:
        """
        Calculates residuals by modifying the scratch_state in-place.
        NO NEW OBJECTS ARE CREATED HERE.
        """
        # 1. Update the scratchpad with the solver's current guess (IN-PLACE)
        scratch_state.update_from_array(free_array)

        # 2. Compute derived points using the updated scratchpad
        all_positions = compute_derived_points_func(scratch_state.positions)

        residuals = []

        # 3. Geometry constraint residuals
        for constraint in constraints:
            residual_value = constraint.residual(all_positions)

            # Handle both scalar and array residuals
            if np.isscalar(residual_value):
                residuals.append(residual_value)
            else:
                # For array residuals, extend the list
                residuals.extend(np.asarray(residual_value).flatten())

        # 4. Target residuals
        for target in step_targets:
            current_pos = all_positions[target.point_id]
            initial_pos = initial_state.positions[target.point_id]

            direction = resolve_target(target.direction)
            initial_proj = float(np.dot(initial_pos, direction))
            target_proj = initial_proj + target.value
            current_proj = float(np.dot(current_pos, direction))
            residuals.append(current_proj - target_proj)

        return np.array(residuals, dtype=float)

    # Use the initial state as the first guess for the first step
    initial_guess = current_state.get_free_array()

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

        # ---- On Success ----
        # 1. Update the main state's positions with the solution
        current_state.update_from_array(result.x)

        # 2. Re-calculate final derived points on the now-correct main state
        updated_positions = compute_derived_points_func(current_state.positions)
        current_state = SuspensionState(
            positions=updated_positions, free_points=current_state.free_points
        )

        # 3. Store a snapshot of the successful state
        states.append(current_state.copy())

        # 4. Sync the scratchpad state for the next step's calculations.
        #    This is critical.
        scratch_state = current_state.copy()

        # 5. The new guess is the successful result from this step
        initial_guess = result.x

    return states


def solve(
    initial_state: SuspensionState,
    constraints: list[Constraint],
    targets: list[PointTarget],
    compute_derived_points_func: Callable[
        [dict[PointID, np.ndarray]], dict[PointID, np.ndarray]
    ],
    solver_config: SolverConfig = SolverConfig(),
) -> SuspensionState:
    """
    Solves a single kinematic state.
    """
    sweep_config = SweepConfig([targets])
    states = solve_sweep(
        initial_state=initial_state,
        constraints=constraints,
        sweep_config=sweep_config,
        compute_derived_points_func=compute_derived_points_func,
        solver_config=solver_config,
    )
    return states[0]
