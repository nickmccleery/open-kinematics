"""
Kinematics solver using damped least squares.

This module provides functions to solve suspension kinematics by satisfying geometric
constraints and position targets using Levenberg-Marquardt.
"""

from typing import Callable, NamedTuple

import numpy as np
from scipy.optimize import least_squares

from kinematics.constants import (
    SOLVE_TOLERANCE_GRAD,
    SOLVE_TOLERANCE_STEP,
    SOLVE_TOLERANCE_VALUE,
)
from kinematics.constraints import Constraint
from kinematics.enums import PointID
from kinematics.state import SuspensionState
from kinematics.targets import resolve_target
from kinematics.types import PointTarget, SweepConfig, TargetPositionMode, Vec3
from kinematics.vector_utils.generic import project_coordinate

# Levenberg-Marquardt; damped least squares that can deal with the system being
# overdetermined (m > n), as may be the case with any redundant (but consistent)
# constraints.
SOLVE_METHOD = "lm"


class SolverConfig(NamedTuple):
    """
    Configuration parameters for the kinematic solver.

    Attributes:
        ftol (float): Tolerance for the function value convergence.
        xtol (float): Tolerance for the solution vector convergence.
        gtol (float): Tolerance for the gradient convergence.
        verbose (int): Verbosity level for the solver output.
    """

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

        # Convert a relative displacement to an absolute scalar coordinate along the
        # target direction: project the initial position onto the (unit) direction to
        # get the initial coordinate, then add the displacement.
        direction = resolve_target(target.direction)
        initial_pos = initial_state.positions[target.point_id]
        initial_coord = project_coordinate(initial_pos, direction)
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
    compute_derived_points_func: Callable[[dict[PointID, Vec3]], dict[PointID, Vec3]],
    solver_config: SolverConfig = SolverConfig(),
) -> list[SuspensionState]:
    """
    Solves a series of kinematic states by sweeping through target configurations using
    damped non-linear least squares. This function performs a sweep where each step in
    the sweep corresponds to a set of targets, solving sequentially from the initial
    state.

    Args:
        initial_state (SuspensionState): The initial suspension state to start the sweep from.
        constraints (list[Constraint]): List of geometric constraints to satisfy.
        sweep_config (SweepConfig): Configuration for the sweep, including number of steps and target sweeps.
        compute_derived_points_func (Callable): Function to compute derived points from free points.
        solver_config (SolverConfig): Configuration parameters for the solver.

    Returns:
        list[SuspensionState]: List of solved suspension states for each step in the sweep.
    """
    # Convert all targets to absolute coordinates once before solving
    sweep_targets = [
        resolve_targets_to_absolute(
            [sweep[i] for sweep in sweep_config.target_sweeps], initial_state
        )
        for i in range(sweep_config.n_steps)
    ]

    # Working state reused across the sweep; mutated in-place for performance.
    working_state = initial_state.copy()

    # For each step in our sweep, we will keep a copy of the solved state; this is
    # our result dataset.
    solution_states: list[SuspensionState] = []

    def compute_residuals(
        free_array: np.ndarray, step_targets: list[PointTarget]
    ) -> np.ndarray:
        """
        Computes the residuals for the least squares solve. This function calculates the
        residuals by evaluating geometric constraints and target deviations, modifying
        the scratch state in-place for efficiency.

        Args:
            free_array (np.ndarray): Array of free point coordinates to update the state with.
            step_targets (list[PointTarget]): List of targets for this step.

        Returns:
            np.ndarray: Array of residual values.
        """
        # In-place update of the working state with the current guess.
        working_state.update_from_array(free_array)

        # Compute derived points using the updated working state.
        all_positions = compute_derived_points_func(working_state.positions)

        # Build all of our residuals; constraints first, targets second.
        residuals = []

        for constraint in constraints:
            residual_value = constraint.residual(all_positions)
            residuals.append(residual_value)

        for target in step_targets:
            direction = resolve_target(target.direction)
            # Get current coordinate along the same direction by projection.
            current_pos = all_positions[target.point_id]
            current_coordinate = project_coordinate(current_pos, direction)

            # Compare to absolute target value.
            target_coordinate = target.value
            residuals.append(current_coordinate - target_coordinate)

        return np.array(residuals, dtype=float)

    # Initial guess built from the working state's free points.
    x_0 = working_state.get_free_array()

    for step_targets in sweep_targets:
        n_vars = len(working_state.free_points_order) * 3
        m_res = len(constraints) + len(step_targets)

        if n_vars > m_res:
            raise ValueError(
                f"System is underdetermined (n_vars={n_vars} > m_res={m_res}). "
                "The solve method (Levenberg-Marquardt) requires at least as "
                "many residuals as variables."
            )

        result = least_squares(
            fun=compute_residuals,
            x0=x_0,
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

        # Update the working state's positions with the solution.
        working_state.update_from_array(result.x)

        # Re-calculate final derived points on the now-correct working state.
        final_positions = compute_derived_points_func(working_state.positions)
        working_state.update_positions(final_positions)

        # Preserve a deep copy as the result for this step.
        solution_states.append(working_state.copy())

        # The result becomes our local first guess for the next step.
        x_0 = result.x

    return solution_states


def solve(
    initial_state: SuspensionState,
    constraints: list[Constraint],
    targets: list[PointTarget],
    compute_derived_points_func: Callable[[dict[PointID, Vec3]], dict[PointID, Vec3]],
    solver_config: SolverConfig = SolverConfig(),
) -> SuspensionState:
    """
    Solves for a single kinematic state using damped non-linear least squares. This
    function finds the suspension state that satisfies the given constraints and
    targets.

    Args:
        initial_state (SuspensionState): The initial suspension state.
        constraints (list[Constraint]): List of geometric constraints to satisfy.
        targets (list[PointTarget]): List of point targets to achieve.
        compute_derived_points_func (Callable): Function to compute derived points from free points.
        solver_config (SolverConfig): Configuration parameters for the solver.

    Returns:
        SuspensionState: The solved suspension state.
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
