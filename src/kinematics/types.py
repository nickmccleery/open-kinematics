from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated, Literal, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

from kinematics.core import PointID
from kinematics.vector_utils.generic import normalize_vector

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


@dataclass(slots=True)
class AxisFrame:
    """
    Right-handed, orthonormal basis expressed in world coordinates.
    See: https://en.wikipedia.org/wiki/Standard_basis
    """

    ex: Vec3
    ey: Vec3
    ez: Vec3

    def __post_init__(self):
        self.ex = normalize_vector(self.ex)
        self.ey = normalize_vector(self.ey)
        self.ez = normalize_vector(self.ez)

    @classmethod
    def create_standard_basis(cls) -> AxisFrame:
        return cls(
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )


class PointTarget(NamedTuple):
    point_id: PointID
    direction: PointTargetDirection
    value: float


class PointTargetSet(NamedTuple):
    values: list[PointTarget]


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
