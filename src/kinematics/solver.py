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
) -> list[KinematicsState]:
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

    # Initialize state for the entire sweep - this object will only be updated after each successful step
    current_state = initial_state.copy()
    states: List[KinematicsState] = []  # Will store the final results

    # Create the single, reusable "scratchpad" object for calculations
    # This eliminates allocations inside the compute_residuals function
    scratch_positions = current_state.positions.copy()
    free_point_order = current_state.free_points_order  # Capture for stability

    def compute_residuals(
        free_array: np.ndarray, step_targets: List[PointTarget]
    ) -> np.ndarray:
        """
        Calculates residuals by modifying the scratch_positions in-place.
        NO NEW OBJECTS ARE CREATED HERE.
        """
        # 1. Update the scratchpad with the solver's current guess (IN-PLACE)
        scratch_positions.update_from_array(free_point_order, free_array)

        # 2. Compute derived points using the updated scratchpad
        all_positions = compute_derived_points_func(scratch_positions)

        residuals = []

        # 3. Geometry constraint residuals
        for c in constraints:
            # Extend is used to handle multi-residual constraints gracefully
            residuals.extend(c.get_residual(all_positions))

        # 4. Target residuals
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

        # ---- On Success ----
        # 1. Update the main state's positions with the solution
        current_state.positions.update_from_array(free_point_order, result.x)

        # 2. Re-calculate final derived points on the now-correct main state
        current_state.positions = compute_derived_points_func(current_state.positions)

        # 3. Store a snapshot of the successful state
        states.append(current_state.copy())

        # 4. Sync the scratchpad's fixed points for the next step's calculations.
        #    This is critical.
        scratch_positions = current_state.positions.copy()

        # 5. The new guess is the successful result from this step
        initial_guess = result.x

    return states
