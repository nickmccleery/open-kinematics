from typing import Literal, Tuple

import numpy as np

from kinematics.geometry.schemas import Point3D, PointID


class BaseConstraint:
    """Base class for constraints."""

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        """Compute the residual for this constraint."""
        raise NotImplementedError()


class PointPointDistanceConstraint(BaseConstraint):
    """Describes a fixed point-to-point distance constraint between two points."""

    def __init__(self, p1: PointID, p2: PointID, distance: float):
        self.p1 = p1
        self.p2 = p2
        self.distance = distance

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        """Compute the distance residual between two points."""
        p1 = points[self.p1].as_array()
        p2 = points[self.p2].as_array()
        current_length = float(np.linalg.norm(p1 - p2))
        return current_length - self.distance


class VectorOrientationConstraint(BaseConstraint):
    """Describes a fixed vector orientation constraint between two vectors."""

    def __init__(
        self, v1: Tuple[PointID, PointID], v2: Tuple[PointID, PointID], angle: float
    ):
        self.v1 = v1
        self.v2 = v2
        self.angle = angle

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        """Compute the angle residual between two vectors."""
        v1_start = points[self.v1[0]].as_array()
        v1_end = points[self.v1[1]].as_array()
        v2_start = points[self.v2[0]].as_array()
        v2_end = points[self.v2[1]].as_array()

        v1 = v1_end - v1_start
        v2 = v2_end - v2_start

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        current_angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return current_angle - self.angle


class FixedAxisConstraint(BaseConstraint):
    """Describes a constraint that fixes a point on a specified axis."""

    def __init__(self, point_id: PointID, axis: Literal["x", "y", "z"], value: float):
        self.point_id = point_id
        self.axis = axis
        self.value = value

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        """Compute the residual for a fixed axis constraint."""
        point = points[self.point_id]
        point_coord = getattr(point, self.axis)
        return point_coord - self.value
