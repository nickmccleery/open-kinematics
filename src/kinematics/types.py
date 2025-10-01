from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]


class Axis(Enum):
    """Semantic axes within a right-handed frame."""

    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass(slots=True, frozen=True)
class AxisTarget:
    """Target specified by semantic axis within a frame."""

    axis: Axis


@dataclass(slots=True, frozen=True)
class VectorTarget:
    """Target specified by an arbitrary direction vector."""

    vector: Vec3


PointTargetDirection = Union[AxisTarget, VectorTarget]

__all__ = [
    "Axis",
    "AxisTarget",
    "VectorTarget",
    "PointTargetDirection",
    "Vec3",
]
