"""
Generic vector utility functions.

This module provides fundamental vector operations used throughout the kinematics
system, including normalization, projection, and coordinate transformations. These
operations do not use any types specific to this project, so can be used in utility
contexts without introducing circular dependencies.
"""

import numpy as np
from numpy.typing import NDArray

from kinematics.constants import EPSILON


def normalize_vector(v: NDArray[np.float64]) -> NDArray[np.float64]:
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
    return (v / norm).astype(np.float64)


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
