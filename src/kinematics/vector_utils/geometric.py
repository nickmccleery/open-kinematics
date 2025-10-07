"""
Geometric computation utilities.

This module provides functions for computing geometric relationships between points and
vectors, including distances, midpoints, and angles. These utilities are fundamental to
kinematic analysis and constraint evaluation.
"""

import numpy as np

from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.generic import normalize_vector


def compute_point_point_distance(p1: Vec3, p2: Vec3) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        p1: First point in 3D space.
        p2: Second point in 3D space.

    Returns:
        The Euclidean distance between the two points.
    """
    return float(np.linalg.norm(p1 - p2))


def compute_point_point_midpoint(p1: Vec3, p2: Vec3) -> Vec3:
    """
    Compute the midpoint between two points.

    Args:
        p1: First point in 3D space.
        p2: Second point in 3D space.

    Returns:
        The midpoint vector between the two points.
    """
    return make_vec3((p1 + p2) / 2)


def compute_vector_vector_angle(v1: Vec3, v2: Vec3) -> float:
    """
    Compute the angle between two vectors in radians using atan2 for robustness.

    The vectors are automatically normalized before computing the angle,
    so input vectors do not need to be unit length. Uses atan2 of the cross
    product magnitude and dot product to avoid numerical instability near
    parallel/anti-parallel vectors.

    Args:
        v1: First vector in 3D space.
        v2: Second vector in 3D space.

    Returns:
        The angle between the vectors in radians (0 to π).

    Raises:
        ValueError: If either input vector has zero length (magnitude < EPSILON).

    References:
        Ericson, C. "Real-Time Collision Detection" (2004), Section 3.3.2, p. 39
    """
    # Will raise ValueError if either vector is zero-length.
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    dot = np.dot(v1_norm, v2_norm)
    cross = np.cross(v1_norm, v2_norm)
    cross_magnitude = np.linalg.norm(cross)

    # atan2 handles all edge cases gracefully (parallel, anti-parallel, perpendicular).
    theta = np.arctan2(cross_magnitude, dot)

    return float(theta)
