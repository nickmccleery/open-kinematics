"""
This package provides functions for calculating key suspension geometry metrics from a
solved SuspensionState.
"""

from .angles import calculate_camber, calculate_caster, calculate_toe
from .antis import calculate_geometric_anti_dive, calculate_geometric_anti_squat
from .main import (
    SuspensionMetrics,
    compute_all_metrics,
    compute_all_metrics_from_geometry,
)

# __all__ defines the public API for the 'kinematics.metrics' package.
__all__ = [
    "compute_all_metrics",
    "compute_all_metrics_from_geometry",
    "SuspensionMetrics",
    "calculate_camber",
    "calculate_caster",
    "calculate_toe",
    "calculate_geometric_anti_dive",
    "calculate_geometric_anti_squat",
]
