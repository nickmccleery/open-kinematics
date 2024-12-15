from dataclasses import dataclass
from enum import Enum

import numpy as np


class GeometryType(Enum):
    DOUBLE_WISHBONE = "DOUBLE_WISHBONE"
    MACPHERSON_STRUT = "MACPHERSON_STRUT"


class Units(Enum):
    MILLIMETERS = "millimeters"


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class WishbonePoints:
    inboard_front: Point3D
    inboard_rear: Point3D
    outboard: Point3D


@dataclass
class RockerPoints:
    axis_front: Point3D
    axis_rear: Point3D
    pushrod_mount: Point3D

    def get_axis_vector(self) -> np.ndarray:
        front = self.axis_front.as_array()
        rear = self.axis_rear.as_array()
        axis = rear - front
        return axis / np.linalg.norm(axis)


@dataclass
class PushrodPoints:
    inboard: Point3D
    outboard: Point3D


@dataclass
class WheelAxlePoints:
    inner: Point3D
    outer: Point3D


@dataclass
class WheelConfig:
    diameter: float
    width: float


@dataclass
class AlignmentConfig:
    static_camber: float
    static_toe: float
    static_caster: float


@dataclass
class HardPoints:
    lower_wishbone: WishbonePoints
    upper_wishbone: WishbonePoints
    rocker: RockerPoints
    pushrod: PushrodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class SuspensionConfig:
    wheel: WheelConfig
    alignment: AlignmentConfig


@dataclass
class SuspensionGeometry:
    type: GeometryType
    name: str
    version: str
    units: Units
    hard_points: HardPoints
    configuration: SuspensionConfig
