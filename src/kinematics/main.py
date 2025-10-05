"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of suspension
geometries.
"""

from typing import List

from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import solve_sweep
from kinematics.state import SuspensionState
from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.types import SweepConfig


def solve_suspension_sweep(
    provider: SuspensionProvider,
    sweep_config: SweepConfig,
) -> List[SuspensionState]:
    """
    Orchestrates the solving of suspension kinematics for a parametric sweep.

    This function coordinates the complete process of solving suspension kinematics
    by instantiating the appropriate provider, setting up derived point calculations,
    and running the solver across target configurations.

    Args:
        provider: The SuspensionProvider instance for this geometry type.
        sweep_config: Configuration for the parametric sweep.

    Returns:
        List of solved suspension states for each step in the sweep.
    """
    derived_spec = provider.derived_spec()
    derived_manager = DerivedPointsManager(derived_spec)

    kinematic_states = solve_sweep(
        initial_state=provider.initial_state(),
        constraints=provider.constraints(),
        sweep_config=sweep_config,
        derived_manager=derived_manager,
    )

    return kinematic_states
