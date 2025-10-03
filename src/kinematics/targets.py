"""
Target resolution utilities for suspension kinematics.

This module provides functions to resolve target directions into world coordinate
vectors.
"""

from kinematics.types import (
    Axis,
    PointTargetAxis,
    PointTargetDirection,
    PointTargetVector,
    Vec3,
    WorldAxisSystem,
)
from kinematics.vector_utils.generic import normalize_vector


def resolve_target(target: PointTargetDirection) -> Vec3:
    """
    Resolves a target direction specification into a unit vector in world coordinates.

    Handles both axis-based directions (X, Y, Z) and arbitrary vector directions,
    normalizing the result to ensure it's a unit vector.

    Args:
        target: The target direction specification to resolve.

    Returns:
        A unit vector in world coordinates representing the target direction.
    """
    if isinstance(target, PointTargetAxis):
        if target.axis is Axis.X:
            return WorldAxisSystem.X
        if target.axis is Axis.Y:
            return WorldAxisSystem.Y
        if target.axis is Axis.Z:
            return WorldAxisSystem.Z
        raise ValueError(f"Unsupported axis: {target.axis!r}")

    if isinstance(target, PointTargetVector):
        return normalize_vector(target.vector)

    raise TypeError(f"Unsupported target type: {type(target)!r}")
