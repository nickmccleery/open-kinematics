"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of suspension
geometries.
"""

from typing import List

from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.solver import SolverInfo, solve_suspension_sweep
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension
from kinematics.types import SweepConfig


def solve_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
) -> tuple[List[SuspensionState], List[SolverInfo]]:
    """
    Orchestrates the solving of suspension kinematics for a parametric sweep.

    This function coordinates the complete process of solving suspension kinematics
    by setting up derived point calculations and running the solver across target
    configurations.

    Args:
        suspension: The Suspension instance containing geometry and behavior.
        sweep_config: Configuration for the parametric sweep.

    Returns:
        Tuple containing the list of solved suspension states and corresponding
        solver information for each step in the sweep.
    """
    derived_spec = suspension.derived_spec()
    derived_manager = DerivedPointsManager(derived_spec)

    kinematic_states, solver_stats = solve_suspension_sweep(
        initial_state=suspension.initial_state(),
        constraints=suspension.constraints(),
        sweep_config=sweep_config,
        derived_manager=derived_manager,
    )

    return kinematic_states, solver_stats
