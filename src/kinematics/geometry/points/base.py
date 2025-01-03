from dataclasses import dataclass

import numpy as np

from kinematics.geometry.points.ids import PointID


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    fixed: bool = False
    id: PointID = PointID.NOT_ASSIGNED

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class DerivedPoint3D:
    x: float
    y: float
    z: float
    id: PointID
    deps: list[PointID]

    def update(self, points: dict[PointID, Point3D]) -> None:
        raise NotImplementedError

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
