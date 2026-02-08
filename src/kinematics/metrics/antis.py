"""
This module contains functions for calculating "anti" geometry metrics like anti-dive
and anti-squat.
"""

from kinematics.io.validation import Vec3Like
from kinematics.state import SuspensionState


def calculate_geometric_anti_dive(
    state: SuspensionState,
    svic: Vec3Like,
    cg_position: Vec3Like,
    wheelbase: float,
    tire_radius: float,
) -> float:
    """Calculate geometric anti-dive percentage (stub - not yet implemented)."""
    # TODO: Implement anti-dive calculation.
    return 0.0


def calculate_geometric_anti_squat(
    state: SuspensionState,
    svic: Vec3Like,
    cg_position: Vec3Like,
    wheelbase: float,
    tire_radius: float,
) -> float:
    """Calculate geometric anti-squat percentage (stub - not yet implemented)."""
    # TODO: Implement anti-squat calculation.
    return calculate_geometric_anti_dive(
        state,
        svic,
        cg_position,
        wheelbase,
        tire_radius,
    )
