from typing import NamedTuple, Union

from numpy.typing import NDArray

from kinematics.core.types import Position
from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.points.ids import PointID


class PointPointDistance(NamedTuple):
    # Constrain the Euclidian distance between two points.
    p1: PointID
    p2: PointID
    distance: float


class VectorAngle(NamedTuple):
    # Constrain the angle between two vectors.
    v1_start: PointID
    v1_end: PointID
    v2_start: PointID
    v2_end: PointID
    angle: float


class PointFixedAxis(NamedTuple):
    # Constrain a point to lie on a given caridnal axis.
    point_id: PointID
    axis: CoordinateAxis
    value: float


class PointOnLine(NamedTuple):
    # Constrain a point to lie on an arbitrary vector.
    point_id: PointID
    line_point: Position
    line_direction: NDArray  # Normalized direction vector.


Constraint = Union[PointPointDistance | VectorAngle | PointFixedAxis | PointOnLine]
