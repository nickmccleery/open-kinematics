import numpy as np

from kinematics.geometry.points.base import DerivedPoint3D, Point3D
from kinematics.geometry.points.ids import PointID


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
        pos = p1.as_array() + v * self.wheel_offset
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
        p1 = points[PointID.AXLE_INBOARD]
        p2 = points[PointID.WHEEL_CENTER]
        v = p2.as_array() - p1.as_array()
        v = v / np.linalg.norm(v)

        # Move inward from wheel center point (opposite to v).
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
        return {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD}

    def update(self, points: dict[PointID, Point3D]) -> None:
        # Use axle inboard as a zero offset wheel would have wheel center and
        # axle outboard in the same place.
        p1 = points[PointID.WHEEL_CENTER]
        p2 = points[PointID.AXLE_INBOARD]
        v = p1.as_array() - p2.as_array()
        v = v / np.linalg.norm(v)

        # Move outward from wheel center point (same direction as v).
        pos = p1.as_array() + v * (self.wheel_width / 2)
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.z = float(pos[2])
