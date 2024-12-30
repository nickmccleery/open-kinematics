from dataclasses import dataclass
from enum import Enum

import numpy as np


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
class StaticSetupConfig:
    static_camber: float
    static_toe: float
    static_caster: float


@dataclass
class SuspensionConfig:
    wheel: WheelConfig
    static_setup: StaticSetupConfig


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


@dataclass
class DoubleWishboneHardPoints:
    lower_wishbone: WishbonePoints
    upper_wishbone: WishbonePoints
    rocker: RockerPoints
    pushrod: PushrodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class DoubleWishboneGeometry(SuspensionGeometry):
    hard_points: DoubleWishboneHardPoints

    def validate(self) -> bool:
        return True


@dataclass
class MacPhersonHardPoints:
    lower_wishbone: WishbonePoints
    strut: StrutPoints
    rocker: RockerPoints
    pushrod: PushrodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class MacPhersonGeometry(SuspensionGeometry):
    hard_points: MacPhersonHardPoints

    def validate(self) -> bool:
        return True
