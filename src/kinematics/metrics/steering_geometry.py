"""
Steering axis geometry metrics.

Scrub radius and mechanical trail are measured from the point where the
steering axis (kingpin axis) intersects the local ground reference plane
through the contact patch centre to the contact patch centre itself.

Coordinate System Assumption: ISO 8855 (X-Forward, Y-Left, Z-Up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.core.enums import Axis

if TYPE_CHECKING:
    from kinematics.metrics.context import MetricContext


def calculate_scrub_radius(ctx: MetricContext) -> float | None:
    """
    Scrub radius in mm.

    The lateral (Y-axis) distance from the steering axis ground
    intersection to the contact patch centre. Positive scrub radius
    means the steering axis meets the ground inboard of the contact
    patch (the common case for a double-wishbone layout with positive
    KPI).

    The steering-axis intersection is evaluated on the horizontal plane
    at the contact patch Z-height, not at world Z = 0.

    Returns None if the steering axis is parallel to that plane.
    """
    ground_pt = ctx.steering_axis_ground_intersection
    if ground_pt is None:
        return None
    cp = ctx.contact_patch_center
    # Positive when the ground intersection is inboard of the contact
    # patch. For a left-side corner (side_sign > 0, Y > 0), inboard
    # means ground_pt_Y < cp_Y, so negate. For a right-side corner
    # it is the opposite.
    dy = float(ground_pt[Axis.Y] - cp[Axis.Y])
    return float(-ctx.side_sign * dy)


def calculate_mechanical_trail(ctx: MetricContext) -> float | None:
    """
    Mechanical trail (caster trail) in mm.

    The longitudinal (X-axis) distance from the steering axis ground
    intersection to the contact patch centre. Positive mechanical
    trail means the contact patch is behind (rearward of) the steering
    axis ground intersection, which produces a self-centring moment.

    The steering-axis intersection is evaluated on the horizontal plane
    at the contact patch Z-height, not at world Z = 0.

    Returns None if the steering axis is parallel to that plane.
    """
    ground_pt = ctx.steering_axis_ground_intersection
    if ground_pt is None:
        return None
    cp = ctx.contact_patch_center
    # Positive when the contact patch is behind the ground intersection.
    return float(ground_pt[Axis.X] - cp[Axis.X])
