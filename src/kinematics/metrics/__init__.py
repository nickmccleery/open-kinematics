"""
This package provides functions for calculating key suspension geometry metrics from a
solved SuspensionState.
"""

from kinematics.metrics.angles import calculate_camber, calculate_caster, calculate_toe
from kinematics.metrics.antis import (
    calculate_geometric_anti_dive,
    calculate_geometric_anti_squat,
)
from kinematics.metrics.main import SuspensionMetrics, compute_all_metrics

__all__ = [
    "compute_all_metrics",
    "SuspensionMetrics",
    "calculate_camber",
    "calculate_caster",
    "calculate_toe",
    "calculate_geometric_anti_dive",
    "calculate_geometric_anti_squat",
]
