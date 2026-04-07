"""
Metric catalog.

Defines the ordered set of corner-level metrics and their export column names.
This is the single place to add, remove, or reorder exported metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


@dataclass(frozen=True)
class MetricDefinition:
    """
    A single metric: its export column name and computation function.
    """

    column_name: str
    compute: Callable[["MetricContext"], float | None]


def _build_default_corner_metrics() -> tuple[MetricDefinition, ...]:
    """
    Build the default corner metric catalog.

    Imports are deferred to avoid circular dependencies at module level.
    """
    from kinematics.core.enums import Axis
    from kinematics.metrics.angles import (
        calculate_camber,
        calculate_caster,
        calculate_roadwheel_angle,
        calculate_toe,
    )
    from kinematics.metrics.antis import (
        calculate_geometric_anti_dive,
        calculate_geometric_anti_squat,
    )
    from kinematics.metrics.swing_arms import (
        calculate_fvsa_length,
        calculate_svsa_length,
    )

    def _ic_coord(attr: str, axis: Axis) -> Callable[["MetricContext"], float | None]:
        def extract(ctx: "MetricContext") -> float | None:
            ic = getattr(ctx, attr)
            return None if ic is None else float(ic[axis])
        return extract

    return (
        MetricDefinition("camber_deg", calculate_camber),
        MetricDefinition("caster_deg", calculate_caster),
        MetricDefinition(
            "roadwheel_angle_deg", calculate_roadwheel_angle
        ),
        MetricDefinition("toe_deg", calculate_toe),
        MetricDefinition("svic_x_mm", _ic_coord("side_view_ic", Axis.X)),
        MetricDefinition("svic_z_mm", _ic_coord("side_view_ic", Axis.Z)),
        MetricDefinition("svsa_length_mm", calculate_svsa_length),
        MetricDefinition("fvic_y_mm", _ic_coord("front_view_ic", Axis.Y)),
        MetricDefinition("fvic_z_mm", _ic_coord("front_view_ic", Axis.Z)),
        MetricDefinition("fvsa_length_mm", calculate_fvsa_length),
        MetricDefinition(
            "anti_dive_pct", calculate_geometric_anti_dive
        ),
        MetricDefinition(
            "anti_squat_pct", calculate_geometric_anti_squat
        ),
    )


def get_default_corner_metrics() -> tuple[MetricDefinition, ...]:
    """
    Return the default ordered corner metric catalog.
    """
    return _build_default_corner_metrics()
