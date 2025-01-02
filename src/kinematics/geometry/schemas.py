from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Type, Union

import numpy as np


class Units(Enum):
    MILLIMETERS = "millimeters"


class PointID(IntEnum):
    NOT_ASSIGNED = 0

    LOWER_WISHBONE_INBOARD_FRONT = 1
    LOWER_WISHBONE_INBOARD_REAR = 2
    LOWER_WISHBONE_OUTBOARD = 3

    UPPER_WISHBONE_INBOARD_FRONT = 4
    UPPER_WISHBONE_INBOARD_REAR = 5
    UPPER_WISHBONE_OUTBOARD = 6

    PUSHROD_INBOARD = 7
    PUSHROD_OUTBOARD = 8

    TRACKROD_INBOARD = 9
    TRACKROD_OUTBOARD = 10

    AXLE_INBOARD = 11
    AXLE_OUTBOARD = 12
    AXLE_MIDPOINT = 13

    STRUT_INBOARD = 14
    STRUT_OUTBOARD = 15


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
class LowerWishbonePoints:
    inboard_front: Point3D
    inboard_rear: Point3D
    outboard: Point3D

    def __post_init__(self):
        self.inboard_front.fixed = True
        self.inboard_rear.fixed = True

        self.inboard_front.id = PointID.LOWER_WISHBONE_INBOARD_FRONT
        self.inboard_rear.id = PointID.LOWER_WISHBONE_INBOARD_REAR
        self.outboard.id = PointID.LOWER_WISHBONE_OUTBOARD


@dataclass
class UpperWishbonePoints:
    inboard_front: Point3D
    inboard_rear: Point3D
    outboard: Point3D

    def __post_init__(self):
        self.inboard_front.fixed = True
        self.inboard_rear.fixed = True

        self.inboard_front.id = PointID.UPPER_WISHBONE_INBOARD_FRONT
        self.inboard_rear.id = PointID.UPPER_WISHBONE_INBOARD_REAR
        self.outboard.id = PointID.UPPER_WISHBONE_OUTBOARD


@dataclass
class StrutPoints:
    inboard: Point3D
    outboard: Point3D

    def __post_init__(self):
        self.inboard.fixed = True

        self.inboard.id = PointID.STRUT_INBOARD
        self.outboard.id = PointID.STRUT_OUTBOARD


@dataclass
class WheelAxlePoints:
    inner: Point3D
    outer: Point3D

    def __post_init__(self):
        self.inner.id = PointID.AXLE_INBOARD
        self.outer.id = PointID.AXLE_OUTBOARD


@dataclass
class TrackRodPoints:
    inner: Point3D
    outer: Point3D

    def __post_init__(self):
        self.inner.fixed = False

        self.inner.id = PointID.TRACKROD_INBOARD
        self.outer.id = PointID.TRACKROD_OUTBOARD


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
    steered: bool
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
    lower_wishbone: LowerWishbonePoints
    upper_wishbone: UpperWishbonePoints
    track_rod: TrackRodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class DoubleWishboneGeometry(SuspensionGeometry):
    hard_points: DoubleWishboneHardPoints

    def validate(self) -> bool:
        return True


@dataclass
class MacPhersonHardPoints:
    lower_wishbone: LowerWishbonePoints
    strut: StrutPoints
    wheel_axle: WheelAxlePoints


@dataclass
class MacPhersonGeometry(SuspensionGeometry):
    hard_points: MacPhersonHardPoints

    def validate(self) -> bool:
        return True


GeometryType = Union[DoubleWishboneGeometry, MacPhersonGeometry]

GEOMETRY_TYPES: dict[str, Type[GeometryType]] = {
    "DOUBLE_WISHBONE": DoubleWishboneGeometry,
    "MACPHERSON_STRUT": MacPhersonGeometry,
}
