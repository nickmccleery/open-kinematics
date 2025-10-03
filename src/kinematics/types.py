"""
Type definitions and data structures for suspension kinematics.

This module provides enumerations, named tuples, and dataclasses that define
the core types used throughout the kinematics system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
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
    """
    Enumeration of the three principal axes in 3D space.
    """

    X = 0
    Y = 1
    Z = 2


class TargetPositionMode(Enum):
    """
    Specifies how a target value should be interpreted.

    RELATIVE: Value represents displacement from the initial/design position
    ABSOLUTE: Value represents an absolute coordinate in the world frame

    Note: All modes are converted to ABSOLUTE before solving begins.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


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
    """
    Defines a target constraint for a specific point during kinematic solving.

    The mode determines how the value is interpreted initially, but all targets
    are converted to absolute coordinates before solving begins.

    Attributes:
        point_id: The point to constrain
        direction: Direction along which to apply the target
        value: Target value (interpretation depends on mode)
        mode: Whether value is relative displacement or absolute coordinate
    """

    point_id: PointID
    direction: "PointTargetDirection"
    value: float
    mode: TargetPositionMode = TargetPositionMode.RELATIVE


@dataclass(slots=True, frozen=True)
class PointTargetAxis:
    """
    A target direction defined by one of the principal axes.

    Attributes:
        axis (Axis): The axis to use as the target direction.
    """

    axis: Axis


@dataclass(slots=True, frozen=True)
class PointTargetVector:
    """
    A target direction defined by an arbitrary vector.

    Attributes:
        vector (Vec3): The vector defining the target direction.
    """

    vector: Vec3


PointTargetDirection = Union[PointTargetAxis, PointTargetVector]
