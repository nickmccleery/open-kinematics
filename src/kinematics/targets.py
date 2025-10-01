from __future__ import annotations

import numpy as np

from kinematics.types import Axis, AxisTarget, PointTargetDirection, Vec3, VectorTarget


class AxisFrame:
    """Right-handed, orthonormal basis expressed in world coordinates."""

    __slots__ = ("ex", "ey", "ez")

    def __init__(self, ex: Vec3, ey: Vec3, ez: Vec3) -> None:
        self.ex = _unit(ex)
        self.ey = _unit(ey)
        self.ez = _unit(ez)


def _unit(vector: Vec3) -> Vec3:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("Zero-length vector cannot be normalized")
    return (arr / norm).astype(np.float64, copy=False)


def resolve_target(target: PointTargetDirection, frame: AxisFrame) -> Vec3:
    """Return a unit direction vector in world coordinates for the target."""

    if isinstance(target, AxisTarget):
        if target.axis is Axis.X:
            return frame.ex
        if target.axis is Axis.Y:
            return frame.ey
        if target.axis is Axis.Z:
            return frame.ez
        raise ValueError(f"Unsupported axis: {target.axis!r}")

    if isinstance(target, VectorTarget):
        return _unit(target.vector)

    raise TypeError(f"Unsupported target type: {type(target)!r}")


DEFAULT_WORLD_FRAME = AxisFrame(
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
)

__all__ = ["AxisFrame", "DEFAULT_WORLD_FRAME", "resolve_target"]
