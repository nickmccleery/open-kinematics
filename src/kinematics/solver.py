"""
Kinematics solver using damped least squares.

This module provides functions to solve suspension kinematics by satisfying geometric
constraints and position targets using Levenberg-Marquardt.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares

from kinematics.constants import (
    SOLVE_TOLERANCE_GRAD,
    SOLVE_TOLERANCE_STEP,
    SOLVE_TOLERANCE_VALUE,
)
from kinematics.constraints import Constraint
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.state import SuspensionState
from kinematics.targets import resolve_target
from kinematics.types import PointTarget, SweepConfig, TargetPositionMode
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


@dataclass
class SolverInfo:
    """
    Information about the solver's execution for a single solve step.

    Attributes:
        converged (bool): Whether the solver converged to a solution.
        nfev (int): Number of function evaluations performed.
        max_residual (float): Maximum residual value in the final solution.
    """

    converged: bool
    nfev: int
    max_residual: float


class ResidualComputer:
    """
    Computes residuals for kinematic constraints and targets.

    This class holds buffers that are reused across multiple solver evaluations
    to minimize allocations. A single instance is used for an entire sweep,
    with different targets passed to compute() for each step.

    Attributes:
        constraints: Geometric constraints to evaluate.
        derived_manager: Manager for computing derived points in-place.
        state_buffer: Suspension state object that is mutated during computation.
        residuals_buffer: Pre-allocated array for residual computation (reused across calls).
    """

    def __init__(
        self,
        constraints: list[Constraint],
        derived_manager: DerivedPointsManager,
        state_buffer: SuspensionState,
        n_target_variables: int,
    ):
        self.constraints = constraints
        self.derived_manager = derived_manager
        self.state_buffer = state_buffer

        # Pre-allocate residuals buffer.
        # This is our constraint residuals + target residuals. Each target adds one
        # element to the residuals vector, so if we're sweeping wheel center Z and
        # steering at the same time, we need two extra slots.
        self.n_constraints = len(constraints)
        self.residuals_buffer = np.empty(
            self.n_constraints + n_target_variables, dtype=np.float64
        )

    def compute(
        self,
        free_array: np.ndarray,
        step_targets: list[PointTarget],
    ) -> np.ndarray:
        """
        Compute residuals for the given free point positions and targets.

        This method mutates state_buffer in-place and reuses residuals_buffer
        for performance. Returns a view of the buffer containing only the
        residuals for this evaluation.

        Args:
            free_array: Flattened array of free point coordinates.
            step_targets: Target constraints for this solve step.

        Returns:
            View of residuals array containing [constraint_residuals, target_residuals].
            The view length matches the actual number of residuals for this step.

        Note:
            The returned array is a view into residuals_buffer and will be
            overwritten on the next call. The solver consumes values immediately,
            so this is safe.
        """
        # Update state buffer in-place with current guess.
        self.state_buffer.update_from_array(free_array)

        # Compute derived points in-place.
        self.derived_manager.update_in_place(self.state_buffer.positions)

        # Fill constraint residuals section: residuals[0:n_constraints].
        for i, constraint in enumerate(self.constraints):
            self.residuals_buffer[i] = constraint.residual(self.state_buffer.positions)

        # Fill target residuals section: residuals[n_constraints:n_constraints+n_targets].
        offset = self.n_constraints
        for i, target in enumerate(step_targets):
            direction = resolve_target(target.direction)
            current_pos = self.state_buffer.positions[target.point_id]
            current_coordinate = project_coordinate(current_pos, direction)
            self.residuals_buffer[offset + i] = current_coordinate - target.value

        # Return view of the used portion. Note that we must return a copy here
        # because Scipy's least sqaures keeps references to the evaluated arrays,
        # so subsequent calls would overwrite previous values.
        n_residuals = self.n_constraints + len(step_targets)
        residuals = self.residuals_buffer[:n_residuals]
        return residuals.copy()


def convert_targets_to_absolute(
    targets: list[PointTarget],
    initial_state: SuspensionState,
) -> list[PointTarget]:
    """
    Convert all targets to absolute coordinates.

    This function implements the "convert early" pattern: all mode-specific
    logic is handled here, once, before solving begins. The solver then works
    exclusively with absolute coordinates.

    Args:
        targets: List of targets in mixed modes (RELATIVE or ABSOLUTE).
        initial_state: Reference state for resolving RELATIVE targets.

    Returns:
        List of targets with all modes converted to ABSOLUTE.
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


def solve_suspension_sweep(
    initial_state: SuspensionState,
    constraints: list[Constraint],
    sweep_config: SweepConfig,
    derived_manager: DerivedPointsManager,
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[list[SuspensionState], list[SolverInfo]]:
    """
    Solves a series of kinematic states by sweeping through target configurations using
    damped non-linear least squares. This function performs a sweep where each step in
    the sweep corresponds to a set of targets, solving sequentially from the initial
    state. All state and residual buffers are reused across evaluations
    to minimize allocations.

    Args:
        initial_state (SuspensionState): The initial suspension state to start the sweep from.
        constraints (list[Constraint]): List of geometric constraints to satisfy.
        sweep_config (SweepConfig): Configuration for the sweep, including number of steps and target sweeps.
        derived_manager (DerivedPointsManager): Manager to compute derived points in-place.
        solver_config (SolverConfig): Configuration parameters for the solver.

    Returns:
        Tuple of (solved_states, solver_stats) where:
        - solved_states: List of converged suspension states for each sweep step
        - solver_stats: List of solver diagnostics for each step

    Raises:
        ValueError: If the system is underdetermined (more variables than residuals).
        RuntimeError: If the solver fails to converge at any step.
    """
    # Convert all targets to absolute coordinates once before solving.
    sweep_targets = [
        convert_targets_to_absolute(
            [sweep[i] for sweep in sweep_config.target_sweeps], initial_state
        )
        for i in range(sweep_config.n_steps)
    ]

    # Working state reused across the sweep; mutated in-place for performance.
    working_state = initial_state.copy()

    # For each step in our sweep, we will keep a copy of the solved state; this is
    # our result dataset.
    solution_states: list[SuspensionState] = []
    solver_stats: list[SolverInfo] = []

    # Create residual computer with pre-allocated buffers.
    # This single instance is reused across all solver evaluations in the sweep.
    residual_computer = ResidualComputer(
        constraints=constraints,
        derived_manager=derived_manager,
        state_buffer=working_state,
        n_target_variables=len(sweep_config.target_sweeps),
    )

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
            fun=residual_computer.compute,
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

        # We now have to synchronize working_state with the accepted solution.
        # The solver evaluates residuals at many candidate positions during the solve,
        # mutating working_state each time. When it terminates, working_state may be left
        # at a position from gradient estimation (e.g., x* + epsilon) rather than
        # the actual solution x*. We must explicitly restore it to result.x to
        # ensure the stored state matches the returned solution.
        #
        # This synchronization is necessary because we reuse working_state across all
        # residual evaluations for performance (avoiding dict allocations on each call).
        # The tradeoff is this explicit sync requirement.
        working_state.update_from_array(result.x)
        derived_manager.update_in_place(working_state.positions)

        # Store finalized state for this step.
        solution_states.append(working_state.copy())

        # Collect solver information for this step.
        max_residual = float(np.max(np.abs(result.fun))) if len(result.fun) > 0 else 0.0
        solver_info = SolverInfo(
            converged=result.success,
            nfev=result.nfev,
            max_residual=max_residual,
        )
        solver_stats.append(solver_info)

        # The result becomes our local first guess for the next step.
        x_0 = result.x

    return solution_states, solver_stats
