from typing import List

from kinematics.core import Positions
from kinematics.points.derived.manager import DerivedPointManager
from kinematics.solver import PointTargetSet, solve_sweep
from kinematics.suspensions.models import SuspensionGeometry
from kinematics.suspensions.registry import REGISTRY


def solve_kinematics(
    geometry: SuspensionGeometry,
    point_targets: List[PointTargetSet],
) -> List[Positions]:
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

    # Find the geometry type in registry
    for type_name, (model_cls, provider_cls) in REGISTRY.items():
        if isinstance(geometry, model_cls):
            provider_class = provider_cls
            break

    if not provider_class:
        raise NotImplementedError(
            f"No provider registered for geometry type: {type(geometry).__name__}"
        )

    # 2. Instantiate the provider for this specific geometry instance
    provider = provider_class(geometry)

    # 3. Instantiate the manager for derived points using spec from the provider
    derived_spec = provider.get_derived_point_definitions()
    derived_point_manager = DerivedPointManager(derived_spec)

    # 4. Get the initial state and rules from the provider
    initial_positions = provider.get_initial_positions()
    free_points = provider.get_free_points()

    # 5. Calculate derived points on the initial state to create a complete reference
    initial_state_with_derived = derived_point_manager.update(initial_positions)

    # 6. Create the constraints based on the complete initial state
    constraints = provider.get_constraints(initial_state_with_derived)

    # 7. Call the generic solver with all the necessary data
    position_states = solve_sweep(
        initial_positions=initial_state_with_derived,
        constraints=constraints,
        free_points=free_points,
        targets=point_targets,
        compute_derived_points_func=derived_point_manager.update,
    )

    return position_states
