from typing import List

from kinematics.core import SuspensionState
from kinematics.points.derived.manager import DerivedPointManager
from kinematics.solver import PointTargetSet, solve_sweep


def solve_kinematics(
    geometry,
    provider_class,
    point_targets: List[PointTargetSet],
) -> List[SuspensionState]:
    """
    Generic, high-level function to orchestrate the solving of any suspension geometry.

    This function performs the following steps:
    1. Looks up the correct SuspensionProvider for the given geometry type.
    2. Uses the provider to get the suspension's rules (initial state, constraints, etc.).
    3. Sets up the derived point calculation engine.
    4. Calls the core solver to run the simulation sweep.
    """
    # 1. Instantiate the provider for this specific geometry instance
    provider = provider_class(geometry)  # type: ignore[call-arg]

    # 2. Instantiate the manager for derived points using spec from the provider
    derived_spec = provider.derived_spec()
    derived_point_manager = DerivedPointManager(derived_spec)

    # 3. Get the complete initial state (including derived points) from the provider
    initial_state_with_derived = provider.initial_state()

    # 4. Create the constraints based on the complete initial state
    constraints = provider.constraints()

    # 5. Call the generic solver with all the necessary data
    kinematic_states = solve_sweep(
        initial_state=initial_state_with_derived,
        constraints=constraints,
        targets=point_targets,
        compute_derived_points_func=derived_point_manager.update,
    )

    return kinematic_states
