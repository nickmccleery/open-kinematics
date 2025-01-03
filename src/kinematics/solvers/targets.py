from dataclasses import dataclass
from typing import Protocol

from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.points.base import Point3D
from kinematics.geometry.points.ids import PointID


class MotionTarget(Protocol):
    point_id: PointID
    axis: CoordinateAxis
    reference_point: Point3D

    def get_current_position(self, points: dict[PointID, Point3D]) -> Point3D: ...
    def get_target_value(
        self, reference_point: Point3D, displacement: float
    ) -> Point3D: ...


@dataclass
class AxisDisplacementTarget(MotionTarget):
    point_id: PointID
    axis: CoordinateAxis
    reference_point: Point3D

    def get_current_position(self, points: dict[PointID, Point3D]) -> Point3D:
        return points[self.point_id]

    def get_target_value(
        self, reference_point: Point3D, displacement: float
    ) -> Point3D:
        coords = list(reference_point.as_array())
        coords[self.axis] += displacement
        return Point3D(*coords)
