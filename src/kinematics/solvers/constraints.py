from typing import Literal, NamedTuple, Tuple

from kinematics.geometry.schemas import PointID


class PointPointDistanceConstraint(NamedTuple):
    """
    Describes a fixed point-to-point distance constraint between two points.
    """

    p1: PointID
    p2: PointID
    distance: float


class VectorOrientationConstraint(NamedTuple):
    """
    Describes a fixed vector orientation constraint between two vectors.
    """

    v1: Tuple[PointID, PointID]
    v2: Tuple[PointID, PointID]
    angle: float


class FixedAxisConstraint(NamedTuple):
    """
    Describes a constraint that fixes a point on a specified axis.
    """

    point_id: PointID
    axis: Literal["x", "y", "z"]
    value: float
