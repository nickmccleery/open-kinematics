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
    Return a unit direction vector in world coordinates for the target.
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
