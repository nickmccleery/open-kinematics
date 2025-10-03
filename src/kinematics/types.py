from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated, Final, Literal, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

from kinematics.core import PointID

Vec3 = Annotated[NDArray[np.float64], Literal[3]]


def make_vec3(data) -> NDArray[np.float64]:
    """
    Create a 3-element float64 numpy array.
    """
    # Check if already correct type and shape.
    if isinstance(data, np.ndarray) and data.dtype == np.float64 and data.shape == (3,):
        return data

    arr = np.asarray(data, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Vec3 must have shape (3,), got {arr.shape}")
    return arr


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


class WorldAxisSystem:
    """
    World coordinate system unit axis vectors.

    Usage:
        WorldAxisSystem.X  # -> np.array([1.0, 0.0, 0.0])
        WorldAxisSystem.Y  # -> np.array([0.0, 1.0, 0.0])
        WorldAxisSystem.Z  # -> np.array([0.0, 0.0, 1.0])
    """

    X: Final[Vec3] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    Y: Final[Vec3] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    Z: Final[Vec3] = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class SweepConfig:
    """
    Configuration for a parametric sweep over multiple target dimensions.

    Each inner list represents one sweep dimension (e.g., bump travel, steering angle).
    All dimensions must have the same length - the sweep will iterate through
    corresponding indices across all dimensions simultaneously.

    Example:
        bump_targets = [PointTarget(..., value=-30), ..., PointTarget(..., value=30)]
        steer_targets = [PointTarget(..., value=-10), ..., PointTarget(..., value=10)]
        config = SweepConfig([bump_targets, steer_targets])
    """

    target_sweeps: list[list["PointTarget"]]

    def __post_init__(self):
        if not self.target_sweeps:
            return

        lengths = [len(sweep) for sweep in self.target_sweeps]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All sweep dimensions must have the same length. Found: {lengths}"
            )

    @property
    def n_steps(self) -> int:
        """Number of steps in the sweep."""
        if not self.target_sweeps:
            return 0
        return len(self.target_sweeps[0])


class PointTarget(NamedTuple):
    point_id: PointID
    direction: "PointTargetDirection"
    value: float


@dataclass(slots=True, frozen=True)
class PointTargetAxis:
    """
    A target direction defined by one of the principal axes.
    """

    axis: Axis


@dataclass(slots=True, frozen=True)
class PointTargetVector:
    """
    A target direction defined by an arbitrary vector.
    """

    vector: Vec3


PointTargetDirection = Union[PointTargetAxis, PointTargetVector]
