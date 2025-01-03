from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kinematics.geometry.points.base import Point3D
from kinematics.geometry.points.ids import PointID


class MotionTarget(Protocol):
    point_id: PointID
    axis: int
    reference_position: np.ndarray

    def get_current_position(self, points: dict[PointID, Point3D]) -> np.ndarray: ...
    def get_target_value(
        self, reference_position: np.ndarray, displacement: float
    ) -> float: ...


@dataclass
class AxisDisplacementTarget(MotionTarget):
    """Target that moves along a specified axis."""

    point_id: PointID
    axis: int  # 0=x, 1=y, 2=z
    reference_position: np.ndarray

    def get_current_position(self, points: dict[PointID, Point3D]) -> np.ndarray:
        return points[self.point_id].as_array()

    def get_target_value(
        self, reference_position: np.ndarray, displacement: float
    ) -> float:
        return reference_position[self.axis] + displacement
