from typing import Callable, NamedTuple

import numpy as np
from scipy.optimize import least_squares

from kinematics.constants import (
    SOLVE_TOLERANCE_GRAD,
    SOLVE_TOLERANCE_STEP,
    SOLVE_TOLERANCE_VALUE,
)
from kinematics.constraints import Constraint
from kinematics.core import PointID, SuspensionState
from kinematics.targets import resolve_target
from kinematics.types import PointTarget, SweepConfig, TargetPositionMode

SOLVE_METHOD = "lm"


class SolverConfig(NamedTuple):
    ftol: float = SOLVE_TOLERANCE_VALUE
    xtol: float = SOLVE_TOLERANCE_STEP
    gtol: float = SOLVE_TOLERANCE_GRAD
    verbose: int = 0


def resolve_targets_to_absolute(
    targets: list[PointTarget],
    initial_state: SuspensionState,
) -> list[PointTarget]:
    """
    Convert all targets to absolute coordinates.

    This function implements the "convert early" pattern: all mode-specific
    logic is handled here, once, before solving begins. The solver then works
    exclusively with absolute coordinates.

    Args:
        targets: List of targets in mixed modes (RELATIVE or ABSOLUTE)
        initial_state: Reference state for resolving RELATIVE targets

    Returns:
        List of targets with all modes converted to ABSOLUTE
    """
    resolved: list[PointTarget] = []

    for target in targets:
        if target.mode == TargetPositionMode.ABSOLUTE:
            resolved.append(target)
            continue

        # Convert a relative displacement to an absolute scalar coordinate
        # along the target direction: project the initial position onto the
        # (unit) direction to get the initial coordinate, then add the displacement.
        direction = resolve_target(target.direction)
        initial_pos = initial_state.positions[target.point_id]
        initial_coord = float(np.dot(initial_pos, direction))
        absolute_value = initial_coord + target.value

        # Create new target with absolute value and mode.
        resolved.append(
            PointTarget(
                point_id=target.point_id,
                direction=target.direction,
                value=absolute_value,
                mode=TargetPositionMode.ABSOLUTE,
            )
        )

    return resolved


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
    # Convert all targets to absolute coordinates once before solving
    sweep_targets = [
        resolve_targets_to_absolute(
            [sweep[i] for sweep in sweep_config.target_sweeps], initial_state
        )
        for i in range(n_steps)
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
            residuals.append(residual_value)

        # 4. Target residuals.
        for target in step_targets:
            direction = resolve_target(target.direction)
            # Get current coordinate along the same direction (projection)
            current_pos = all_positions[target.point_id]
            current_coordinate = float(np.dot(current_pos, direction))

            # Compare to absolute target value (already resolved)
            target_coordinate = target.value
            residuals.append(current_coordinate - target_coordinate)

        return np.array(residuals, dtype=float)

    # Use the initial state as the first guess for the first step
    initial_guess = current_state.get_free_array()

    for step_targets in sweep_targets:
        # Choose solver method based on system size: 'lm' requires m >= n
        n_vars = len(current_state.free_points_order) * 3
        m_res = len(constraints) + len(step_targets)

        if n_vars > m_res:
            raise ValueError(
                f"System is underdetermined (n_vars={n_vars} > m_res={m_res}). "
                "The solve method (Levenberg-Marquardt) requires at least as many residuals as variables."
            )

        result = least_squares(
            fun=compute_residuals,
            x0=initial_guess,
            args=(step_targets,),
            method=SOLVE_METHOD,
            ftol=solver_config.ftol,
            xtol=solver_config.xtol,
            gtol=solver_config.gtol,
            verbose=solver_config.verbose,
        )

        if not result.success:
            raise RuntimeError(
                f"Solver failed to converge for targets: {step_targets}."
                f"\nMessage: {result.message}"
            )

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
