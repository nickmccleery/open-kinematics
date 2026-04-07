"""
Anti-geometry metrics (anti-dive, anti-squat).

These metrics quantify how much the suspension geometry resists pitch
under braking or acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


def calculate_geometric_anti_dive(ctx: MetricContext) -> float | None:
    """
    Geometric anti-dive as a percentage.

    Returns None if the SVIC is undefined (parallel links).
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    # TODO: Implement anti-dive calculation using SVIC, CG, wheelbase,
    # and tire radius from ctx.
    return 0.0


def calculate_geometric_anti_squat(ctx: MetricContext) -> float | None:
    """
    Geometric anti-squat as a percentage.

    Returns None if the SVIC is undefined (parallel links).
    """
    svic = ctx.side_view_ic
    if svic is None:
        return None
    # TODO: Implement anti-squat calculation.
    return 0.0
