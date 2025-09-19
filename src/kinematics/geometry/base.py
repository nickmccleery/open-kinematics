from dataclasses import dataclass

import numpy as np

from kinematics.geometry.config.setup import SuspensionConfig
from kinematics.geometry.config.units import Units
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
class SuspensionGeometry:
    """
    Base class for all suspension geometry types.
    """

    name: str
    version: str
    units: Units
    configuration: SuspensionConfig

    def validate(self) -> bool:
        raise NotImplementedError("Subclasses must implement validate()")
