"""
Geometric computation utilities.

This module provides functions for computing geometric relationships between points and
vectors, including distances, midpoints, and angles. These utilities are fundamental to
kinematic analysis and constraint evaluation.
"""

import numpy as np

from kinematics.constants import EPSILON
from kinematics.enums import Axis
from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.generic import normalize_vector


def compute_point_point_distance(p1: Vec3, p2: Vec3) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        p1: First point in 3D space.
        p2: Second point in 3D space.

    Returns:
        The Euclidean distance between the two points (always non-negative).
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
    return make_vec3((p1 + p2) / 2.0)


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
    # normalize_vector will raise ValueError if either vector is zero-length.
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    dot = np.dot(v1_norm, v2_norm)
    cross = np.cross(v1_norm, v2_norm)
    cross_magnitude = np.linalg.norm(cross)

    # atan2 handles all edge cases gracefully (parallel, anti-parallel, perpendicular).
    theta = np.arctan2(cross_magnitude, dot)

    return float(theta)


def compute_vectors_cross_product_magnitude(v1: Vec3, v2: Vec3) -> float:
    """
    Compute the magnitude of the cross product between two normalized vectors.

    This is useful for measuring how "non-parallel" two vectors are. Returns 0
    when vectors are parallel or anti-parallel, and 1 when perpendicular.

    Args:
        v1: First vector in 3D space.
        v2: Second vector in 3D space.

    Returns:
        Magnitude of the cross product between normalized vectors (0 to 1).

    Raises:
        ValueError: If either input vector has zero length (magnitude < EPSILON).
    """
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    cross = np.cross(v1_norm, v2_norm)
    return float(np.linalg.norm(cross))


def compute_vectors_dot_product(v1: Vec3, v2: Vec3) -> float:
    """
    Compute the dot product between two normalized vectors.

    This is useful for measuring how "parallel" two vectors are. Returns 1 when
    vectors are parallel, -1 when anti-parallel, and 0 when perpendicular.

    Args:
        v1: First vector in 3D space.
        v2: Second vector in 3D space.

    Returns:
        Dot product between normalized vectors (-1 to 1).

    Raises:
        ValueError: If either input vector has zero length (magnitude < EPSILON).
    """
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    return float(np.dot(v1_norm, v2_norm))


def compute_point_to_line_distance(
    point: Vec3, line_point: Vec3, line_direction: Vec3
) -> float:
    """
    Compute the perpendicular distance from a point to a line.

    The line is defined by a point on the line and a direction vector (which
    will be normalized internally).

    Args:
        point: The point to measure distance from.
        line_point: A point on the line.
        line_direction: Direction vector of the line (will be normalized).

    Returns:
        Perpendicular distance from the point to the line (always non-negative).

    Raises:
        ValueError: If line_direction has zero length (magnitude < EPSILON).
    """
    line_dir_norm = normalize_vector(line_direction)
    point_to_line = point - line_point

    cross_product = np.cross(point_to_line, line_dir_norm)
    return float(np.linalg.norm(cross_product))


def compute_point_to_plane_distance(
    point: Vec3, plane_point: Vec3, plane_normal: Vec3
) -> float:
    """
    Compute the signed distance from a point to a plane.

    The plane is defined by a point on the plane and a normal vector (which
    will be normalized internally). Positive distance indicates the point is
    on the side of the plane in the direction of the normal.

    Args:
        point: The point to measure distance from.
        plane_point: A point on the plane.
        plane_normal: Normal vector of the plane (will be normalized).

    Returns:
        Signed distance from the point to the plane.

    Raises:
        ValueError: If plane_normal has zero length (magnitude < EPSILON).
    """
    plane_norm = normalize_vector(plane_normal)
    point_to_plane = point - plane_point

    return float(np.dot(point_to_plane, plane_norm))


def compute_scalar_triple_product(v1: Vec3, v2: Vec3, v3: Vec3) -> float:
    """
    Compute the scalar triple product v1 dot (v2 cross v3).

    The scalar triple product represents the signed volume of the parallelepiped
    defined by the three vectors. It is zero when the vectors are coplanar.

    Args:
        v1: First vector.
        v2: Second vector.
        v3: Third vector.

    Returns:
        The scalar triple product (can be positive, negative, or zero).
    """
    cross = np.cross(v2, v3)
    return float(np.dot(v1, cross))


def plane_from_three_points(
    a: Vec3, b: Vec3, c: Vec3
) -> tuple[np.ndarray, float] | None:
    """
    Construct a plane from three 3D points.

    The plane is represented in the form n·x + d = 0, where n is the unit normal
    vector and d is the distance offset.

    Args:
        a: First point defining the plane.
        b: Second point defining the plane.
        c: Third point defining the plane.

    Returns:
        A tuple containing the normal vector and distance `d`, or None if the points
        are degenerate and do not form a unique plane.
    """
    # Compute two edge vectors on the plane.
    v1 = b - a
    v2 = c - a

    # The normal is perpendicular to both edge vectors.
    normal = np.cross(v1, v2)
    normal_magnitude = np.linalg.norm(normal)

    # If the magnitude of the normal is near zero, the points are collinear.
    if normal_magnitude < EPSILON:
        return None

    # Normalize the normal vector for a consistent plane representation.
    normal /= normal_magnitude

    # Compute d for the plane equation n·x + d = 0 using point a.
    d = -float(np.dot(normal, a))

    return normal, d


def intersect_two_planes(
    n1: np.ndarray, d1: float, n2: np.ndarray, d2: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Find the 3D line of intersection between two planes.

    The line is represented by a point on the line and a direction vector. This
    function handles the case where planes may be parallel.

    Args:
        n1: The normal vector of the first plane.
        d1: The distance offset of the first plane.
        n2: The normal vector of the second plane.
        d2: The distance offset of the second plane.

    Returns:
        A tuple containing a point on the line and the line's direction vector,
        or None if the planes are parallel and have no unique intersection.
    """
    # The direction of the intersection line is perpendicular to both plane normals.
    direction = np.cross(n1, n2)
    direction_magnitude_squared = float(np.dot(direction, direction))

    # If the magnitude is near zero, the normals are parallel, so the planes are.
    if direction_magnitude_squared < EPSILON * EPSILON:
        return None

    # Find a specific point on the intersection line. This formula derives from
    # solving the system of linear equations for the two planes.
    point = np.cross(d2 * n1 - d1 * n2, direction) / direction_magnitude_squared

    return point, direction


def intersect_line_with_vertical_plane(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    plane_y: float,
) -> np.ndarray | None:
    """
    Find where a 3D line intersects a vertical plane defined by y = constant.

    This represents the intersection with a side-view plane in vehicle coordinates.

    Args:
        line_point: A point on the 3D line.
        line_direction: The direction vector of the 3D line.
        plane_y: The Y-coordinate that defines the vertical plane.

    Returns:
        The 3D intersection point, or None if the line is parallel to the plane.
    """
    direction_y = float(line_direction[Axis.Y])

    # If the line's direction in Y is near zero, it is parallel to the plane.
    if abs(direction_y) < EPSILON:
        return None

    # Solve for the parameter t where the line's Y-coordinate equals the plane's.
    # The line equation is P(t) = line_point + t * line_direction.
    # We solve P(t).y = plane_y.
    t = (plane_y - float(line_point[Axis.Y])) / direction_y

    return line_point + t * line_direction
