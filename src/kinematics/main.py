"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of
suspension geometries.
"""

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
    Orchestrates the solving of suspension kinematics for a parametric sweep.

    This function coordinates the complete process of solving suspension kinematics
    by instantiating the appropriate provider, setting up derived point calculations,
    and running the solver across target configurations.

    Args:
        geometry: The suspension geometry specification.
        provider_class: The SuspensionProvider class for this geometry type.
        sweep_config: Configuration for the parametric sweep.

    Returns:
        List of solved suspension states for each step in the sweep.
    """
    provider = provider_class(geometry)
    derived_spec = provider.derived_spec()
    derived_resolver = DerivedPointsManager(derived_spec)

    kinematic_states = solve_sweep(
        initial_state=provider.initial_state(),
        constraints=provider.constraints(),
        sweep_config=sweep_config,
        compute_derived_points_func=derived_resolver.update,
    )

    return kinematic_states
