"""
This module contains functions for calculating "anti" geometry metrics like anti-dive
and anti-squat.
"""

from kinematics.state import SuspensionState
from kinematics.types import Vec3


def calculate_geometric_anti_dive(
    state: SuspensionState,
    svic: Vec3,
    cg_x_position: float,
    cg_height: float,
) -> float:
    # TODO
    return 0.0


def calculate_geometric_anti_squat(
    state: SuspensionState,
    svic: Vec3,
    cg_x_position: float,
    cg_height: float,
) -> float:
    # TODO
    return calculate_geometric_anti_dive(state, svic, cg_x_position, cg_height)
