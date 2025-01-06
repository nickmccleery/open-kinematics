from dataclasses import dataclass

import numpy as np

from kinematics.geometry.points.base import DerivedPoint3D, Point3D
from kinematics.geometry.points.ids import PointID


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
        self.inner.fixed = True

        self.inner.id = PointID.TRACKROD_INBOARD
        self.outer.id = PointID.TRACKROD_OUTBOARD


class AxleMidPoint(DerivedPoint3D):
    def __init__(self):
        super().__init__(
            x=0.0,
            y=0.0,
            z=0.0,
            id=PointID.AXLE_MIDPOINT,
        )

    def get_dependencies(self) -> set[PointID]:
        return {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD}

    def update(self, points: dict[PointID, Point3D]) -> None:
        p1 = points[PointID.AXLE_INBOARD]
        p2 = points[PointID.AXLE_OUTBOARD]
        mid = (p1.as_array() + p2.as_array()) / 2
        self.x = float(mid[0])
        self.y = float(mid[1])
        self.z = float(mid[2])


class WheelCenterPoint(DerivedPoint3D):
    def __init__(self, wheel_offset: float):
        super().__init__(
            x=0.0,
            y=0.0,
            z=0.0,
            id=PointID.WHEEL_CENTER,
        )
        self.wheel_offset = wheel_offset

    def get_dependencies(self) -> set[PointID]:
        return {PointID.AXLE_OUTBOARD, PointID.AXLE_INBOARD}

    def update(self, points: dict[PointID, Point3D]) -> None:
        p1 = points[PointID.AXLE_OUTBOARD]
        p2 = points[PointID.AXLE_INBOARD]
        v = p2.as_array() - p1.as_array()
        v = v / np.linalg.norm(v)

        # Our axle outboard point is the hub face. Wheel centerline is positioned
        # along the axle centerline, offset by the wheel offset.
        pos = p2.as_array() + v * self.wheel_offset
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.z = float(pos[2])


class WheelInboardPoint(DerivedPoint3D):
    def __init__(self, wheel_width: float):
        super().__init__(
            x=0.0,
            y=0.0,
            z=0.0,
            id=PointID.WHEEL_INBOARD,
        )
        self.wheel_width = wheel_width

    def get_dependencies(self) -> set[PointID]:
        return {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD}

    def update(self, points: dict[PointID, Point3D]) -> None:
        p1 = points[PointID.WHEEL_CENTER]
        p2 = points[PointID.AXLE_INBOARD]
        v = p2.as_array() - p1.as_array()
        v = v / np.linalg.norm(v)
        # Move inward from axle point (opposite to v)
        pos = p2.as_array() - v * (self.wheel_width / 2)
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.z = float(pos[2])


class WheelOutboardPoint(DerivedPoint3D):
    def __init__(self, wheel_width: float):
        super().__init__(
            x=0.0,
            y=0.0,
            z=0.0,
            id=PointID.WHEEL_OUTBOARD,
        )
        self.wheel_width = wheel_width

    def get_dependencies(self) -> set[PointID]:
        return {PointID.WHEEL_CENTER, PointID.AXLE_OUTBOARD}

    def update(self, points: dict[PointID, Point3D]) -> None:
        p1 = points[PointID.WHEEL_CENTER]
        p2 = points[PointID.AXLE_OUTBOARD]
        v = p2.as_array() - p1.as_array()
        v = v / np.linalg.norm(v)
        # Move outward from axle point (same direction as v)
        pos = p2.as_array() + v * (self.wheel_width / 2)
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.z = float(pos[2])
