"""
Point collection classes for different suspension components.

Each collection represents the points that define a specific suspension component
like wishbones, struts, track rods, etc.
"""

from dataclasses import dataclass

from kinematics.points.main import Point3D, PointID


@dataclass
class LowerWishbonePoints:
    """Points defining the lower wishbone/control arm geometry."""

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
    """Points defining the upper wishbone/control arm geometry."""

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
    """Points defining the strut geometry."""

    inboard: Point3D
    outboard: Point3D

    def __post_init__(self):
        self.inboard.fixed = True

        self.inboard.id = PointID.STRUT_INBOARD
        self.outboard.id = PointID.STRUT_OUTBOARD


@dataclass
class WheelAxlePoints:
    """Points defining the wheel axle geometry."""

    inner: Point3D
    outer: Point3D

    def __post_init__(self):
        self.inner.id = PointID.AXLE_INBOARD
        self.outer.id = PointID.AXLE_OUTBOARD


@dataclass
class TrackRodPoints:
    """Points defining the track rod/tie rod geometry."""

    inner: Point3D
    outer: Point3D

    def __post_init__(self):
        self.inner.fixed = True

        self.inner.id = PointID.TRACKROD_INBOARD
        self.outer.id = PointID.TRACKROD_OUTBOARD
