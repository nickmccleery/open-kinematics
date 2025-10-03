from typing import List

from kinematics.core import SuspensionState
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import solve_sweep
from kinematics.types import SweepConfig


def solve_suspension_sweep(
    geometry,
    provider_class,
    sweep_config: SweepConfig,
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
    derived_resolver = DerivedPointsManager(derived_spec)

    # 3. Get the complete initial state (including derived points) from the provider
    initial_state_with_derived = provider.initial_state()

    # 4. Create the constraints based on the complete initial state
    constraints = provider.constraints()

    # 5. Call the generic solver with all the necessary data
    kinematic_states = solve_sweep(
        initial_state=initial_state_with_derived,
        constraints=constraints,
        sweep_config=sweep_config,
        compute_derived_points_func=derived_resolver.update,
    )

    return kinematic_states
