from typing import List

from kinematics.core import SuspensionState
from kinematics.derived import DerivedPointManager
from kinematics.solver import PointTargetSet, solve_sweep
from kinematics.suspensions import build_registry


def solve_kinematics(
    geometry,
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
    # 1. Look up the provider class from the registry based on the geometry's type
    provider_class = None
    registry = build_registry()

    # Find the geometry type in registry
    for _, (model_cls, provider_cls) in registry.items():
        if isinstance(geometry, model_cls):
            provider_class = provider_cls
            break

    if not provider_class:
        raise NotImplementedError(
            f"No provider registered for geometry type: {type(geometry).__name__}"
        )

    # 2. Instantiate the provider for this specific geometry instance
    provider = provider_class(geometry)  # type: ignore[call-arg]

    # 3. Instantiate the manager for derived points using spec from the provider
    derived_spec = provider.derived_spec()
    derived_point_manager = DerivedPointManager(derived_spec)

    # 4. Get the initial state from the provider
    initial_state = provider.initial_state()

    # 5. Calculate derived points on the initial state to create a complete reference
    initial_state_with_derived_positions = derived_point_manager.update(
        initial_state.positions
    )
    initial_state_with_derived = SuspensionState(
        positions=initial_state_with_derived_positions,
        free_points=initial_state.free_points,
    )

    # 7. Create the constraints based on the complete initial state
    constraints = provider.constraints()

    # 8. Call the generic solver with all the necessary data
    kinematic_states = solve_sweep(
        initial_state=initial_state_with_derived,
        constraints=constraints,
        targets=point_targets,
        compute_derived_points_func=derived_point_manager.update,
    )

    return kinematic_states
