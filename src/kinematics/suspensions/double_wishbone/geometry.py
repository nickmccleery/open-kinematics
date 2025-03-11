from dataclasses import dataclass

from kinematics.geometry.points.collections import (
    LowerWishbonePoints,
    TrackRodPoints,
    UpperWishbonePoints,
    WheelAxlePoints,
)
from kinematics.geometry.types.base import SuspensionGeometry


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
