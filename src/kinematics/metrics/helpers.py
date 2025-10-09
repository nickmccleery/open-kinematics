"""
Internal helper functions for metric calculations.

These functions perform intermediate geometric constructions, like finding instant
centers.
"""

import numpy as np

from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.generic import compute_vector_vector_intersection


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
    # Project onto XZ plane (side view) for 2D intersection calculation
    upper_line_start = np.array([upper_front[0], upper_front[2]], dtype=np.float64)
    upper_line_end = np.array([upper_rear[0], upper_rear[2]], dtype=np.float64)
    lower_line_start = np.array([lower_front[0], lower_front[2]], dtype=np.float64)
    lower_line_end = np.array([lower_rear[0], lower_rear[2]], dtype=np.float64)

    # Compute intersection using generic vector intersection function
    # Use segments_only=False since we want infinite line intersection
    intersection = compute_vector_vector_intersection(
        upper_line_start,
        upper_line_end,
        lower_line_start,
        lower_line_end,
        segments_only=False,
    )

    if intersection is None:
        # Lines are parallel - no intersection
        return make_vec3([np.nan, np.nan, np.nan])

    # Extract intersection point and return as 3D with Y=0 (assumes SVIC on centerline)
    ic_x, ic_z = intersection.point
    return np.array([ic_x, 0.0, ic_z])
