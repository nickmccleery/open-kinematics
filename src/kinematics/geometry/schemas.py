from dataclasses import dataclass
from enum import Enum
from typing import Union

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
class StrutPoints:
    body_mount: Point3D  # Top mounting point to chassis.
    lower_mount: Point3D  # Lower mounting point to upright.
    guide_mount: Point3D  # Additional guiding point for strut orientation.


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
class DoubleWishboneHardPoints:
    lower_wishbone: WishbonePoints
    upper_wishbone: WishbonePoints
    rocker: RockerPoints
    pushrod: PushrodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class MacPhersonHardPoints:
    lower_wishbone: WishbonePoints
    strut: StrutPoints
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
    hard_points: Union[DoubleWishboneHardPoints, MacPhersonHardPoints]
    configuration: SuspensionConfig

    def validate(self) -> bool:
        if self.type == GeometryType.DOUBLE_WISHBONE:
            if not isinstance(self.hard_points, DoubleWishboneHardPoints):
                raise ValueError(
                    "Invalid hard points type for double wishbone suspension"
                )
        elif self.type == GeometryType.MACPHERSON_STRUT:
            if not isinstance(self.hard_points, MacPhersonHardPoints):
                raise ValueError(
                    "Invalid hard points type for MacPherson strut suspension"
                )
        return True
