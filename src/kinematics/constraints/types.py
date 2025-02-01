from typing import NamedTuple

from numpy.typing import NDArray

from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.points.ids import PointID


class PointPointDistance(NamedTuple):
    p1: PointID
    p2: PointID
    distance: float


class VectorAngle(NamedTuple):
    v1_start: PointID
    v1_end: PointID
    v2_start: PointID
    v2_end: PointID
    angle: float


class PointFixedAxis(NamedTuple):
    point_id: PointID
    axis: CoordinateAxis
    value: float


class PointOnLine(NamedTuple):
    point_id: PointID
    line_point: PointID
    line_direction: NDArray  # Normalized direction vector.
