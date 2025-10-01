from kinematics.types import (
    Axis,
    AxisFrame,
    PointTargetAxis,
    PointTargetDirection,
    PointTargetVector,
    Vec3,
)
from kinematics.vector_utils.generic import normalize_vector


def resolve_target(
    target: PointTargetDirection,
    frame: AxisFrame = AxisFrame.create_standard_basis(),
) -> Vec3:
    """
    Return a unit direction vector in world coordinates for the target.
    """
    if isinstance(target, PointTargetAxis):
        if target.axis is Axis.X:
            return frame.ex
        if target.axis is Axis.Y:
            return frame.ey
        if target.axis is Axis.Z:
            return frame.ez
        raise ValueError(f"Unsupported axis: {target.axis!r}")

    if isinstance(target, PointTargetVector):
        return normalize_vector(target.vector)

    raise TypeError(f"Unsupported target type: {type(target)!r}")
