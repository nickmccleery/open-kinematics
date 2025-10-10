"""
Generic vector utility functions.

This module provides fundamental vector operations used throughout the kinematics
system. These operations do not use any types specific to this project, so can be used
in utility contexts without introducing circular dependencies.
"""

from typing import NamedTuple, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from kinematics.constants import EPSILON

FloatingT = TypeVar("FloatingT", bound=np.floating)


class LineIntersectionResult(NamedTuple):
    """
    Result of a line-line intersection calculation.

    Attributes:
        point: The intersection point coordinates.
        t1: Parameter t for first line (0 <= t <= 1 means point is on line segment).
        t2: Parameter t for second line (0 <= t <= 1 means point is on line segment).
    """

    point: NDArray[np.float64]
    t1: float
    t2: float


def compute_2d_vector_vector_intersection(
    line1_start: NDArray[np.float64],
    line1_end: NDArray[np.float64],
    line2_start: NDArray[np.float64],
    line2_end: NDArray[np.float64],
    *,
    segments_only: bool = True,
) -> Optional[LineIntersectionResult]:
    """
    Compute the intersection of two 2D line segments.

    This function solves for the intersection point of two lines defined by their
    start and end points. The lines are parameterized as:
    - Line 1: P = line1_start + t1 * (line1_end - line1_start)
    - Line 2: P = line2_start + t2 * (line2_end - line2_start)

    Args:
        line1_start: Start point of first line as [x, y] coordinates.
        line1_end: End point of first line as [x, y] coordinates.
        line2_start: Start point of second line as [x, y] coordinates.
        line2_end: End point of second line as [x, y] coordinates.
        segments_only: If True, only return intersection if it lies on both line
            segments (0 <= t1, t2 <= 1). If False, treat as infinite lines.

    Returns:
        LineIntersectionResult with intersection point and parameters, or None if:
        - Lines are parallel (no intersection)
        - Lines are degenerate (zero length)
        - segments_only=True and intersection is outside both segments
    """
    # Extract coordinates.
    x1, y1 = line1_start[0], line1_start[1]
    x2, y2 = line1_end[0], line1_end[1]
    x3, y3 = line2_start[0], line2_start[1]
    x4, y4 = line2_end[0], line2_end[1]

    # Calculate direction vectors.
    d1x = x2 - x1
    d1y = y2 - y1
    d2x = x4 - x3
    d2y = y4 - y3

    # Calculate denominator for parametric solution.
    denominator = d1x * d2y - d1y * d2x

    # Check for parallel or degenerate lines.
    if (
        abs(denominator) < EPSILON
        or np.hypot(d1x, d1y) < EPSILON
        or np.hypot(d2x, d2y) < EPSILON
    ):
        return None

    # Calculate relative position.
    dx = x3 - x1
    dy = y3 - y1

    # Solve for parameters.
    t1 = (dx * d2y - dy * d2x) / denominator
    t2 = (dx * d1y - dy * d1x) / denominator

    # Check if intersection is on line segments (if required).
    if segments_only and not (0 <= t1 <= 1 and 0 <= t2 <= 1):
        return None

    # Calculate intersection point.
    point_x = x1 + t1 * d1x
    point_y = y1 + t1 * d1y
    point = np.array([point_x, point_y], dtype=np.float64)

    return LineIntersectionResult(point=point, t1=t1, t2=t2)


def normalize_vector(v: NDArray[FloatingT]) -> NDArray[FloatingT]:
    """
    Normalize a vector of any dimension to a unit vector.

    Args:
        v: Input vector of any dimension.

    Returns:
        Unit vector in the same direction as the input.

    Raises:
        ValueError: If the input vector has zero length (magnitude < EPSILON).
    """
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        raise ValueError("Cannot normalize zero-length vector")
    return (v / norm).astype(v.dtype)


def project_coordinate(position: np.ndarray, direction: np.ndarray) -> float:
    """
    Computes the scalar coordinate of a position vector along a unit direction. This
    represents the signed distance of the position along the given direction.

    Args:
        position: The 3D position vector.
        direction: The unit direction vector (magnitude must be 1).

    Returns:
        The scalar projection value.

    Raises:
        ValueError: If direction is not a unit vector.
    """
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=EPSILON):
        raise ValueError(
            f"Direction vector not normalized; magnitude {np.linalg.norm(direction)}"
        )
    return float(np.dot(position, direction))


def rotate_2d_vector(vector: np.ndarray, angle_radians: float) -> np.ndarray:
    """
    Rotate a 2D vector by the specified angle.

    Args:
        vector: 2D vector [x, y] to rotate
        angle_radians: Rotation angle in radians (positive = counter-clockwise)

    Returns:
        Rotated 2D vector
    """
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    return rotation_matrix @ vector


def perpendicular_2d(vector: np.ndarray, clockwise: bool = False) -> np.ndarray:
    """
    Compute the perpendicular to a 2D vector.

    Args:
        vector: 2D vector [x, y]
        clockwise: If True, rotate clockwise (right-hand perpendicular),
                   if False, rotate anti-clockwise (left-hand perpendicular)

    Returns:
        Perpendicular 2D vector with same magnitude as input.
    """
    if clockwise:
        # 90° clockwise: [x, y] -> [y, -x].
        return np.array([vector[1], -vector[0]], dtype=np.float64)
    else:
        # 90° anti-clockwise: [x, y] -> [-y, x].
        return np.array([-vector[1], vector[0]], dtype=np.float64)
