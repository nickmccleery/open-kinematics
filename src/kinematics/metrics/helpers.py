"""
Internal helper functions for metric calculations.

These functions perform intermediate geometric constructions, like finding instant
centers.
"""

import numpy as np

from kinematics.constants import EPSILON
from kinematics.types import Vec3, make_vec3


def compute_wishbone_svic(
    upper_front: Vec3,
    upper_rear: Vec3,
    lower_front: Vec3,
    lower_rear: Vec3,
) -> Vec3:
    """
    Compute side view instant center from wishbone inboard mounting points.

    This function projects four mounting points onto the side-view (XZ) plane
    and finds where the upper and lower link lines intersect. This is applicable
    to any suspension with two lateral links (double wishbone, multi-link, etc.).

    The calculation solves a 2D line-line intersection problem:
    - Line A: upper_front to upper_rear (upper link)
    - Line B: lower_front to lower_rear (lower link)

    Args:
        upper_front: Upper link front inboard mounting point.
        upper_rear: Upper link rear inboard mounting point.
        lower_front: Lower link front inboard mounting point.
        lower_rear: Lower link rear inboard mounting point.

    Returns:
        The (x, y, z) coordinates of the SVIC with y=0 (centerline), or
        None if the links are parallel (no intersection).

    Note:
        This assumes the SVIC lies on the vehicle centerline (y=0).
        For asymmetric suspensions, the caller may need to adjust the Y coordinate.
    """
    # TODO: Slop generated, need to do this by hand because we don't trust clankers.
    # Project onto XZ plane (side view)
    x1, _, z1 = upper_front
    x2, _, z2 = upper_rear
    x3, _, z3 = lower_front
    x4, _, z4 = lower_rear

    # Solve 2D line-line intersection using parametric form
    # Line A: P = P1 + t(P2 - P1)
    # Line B: P = P3 + s(P4 - P3)
    den = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4)

    if abs(den) < EPSILON:
        # Lines are parallel - no intersection.
        return make_vec3([np.nan, np.nan, np.nan])

    # Calculate parameter t for line A
    t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / den

    # Find intersection point
    ic_x = x1 + t * (x2 - x1)
    ic_z = z1 + t * (z2 - z1)

    # Return with Y=0 (assumes SVIC on centerline)
    return np.array([ic_x, 0.0, ic_z])
