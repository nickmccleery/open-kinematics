from dataclasses import dataclass

from kinematics.geometry.base import SuspensionGeometry
from kinematics.geometry.points.collections import (
    LowerWishbonePoints,
    StrutPoints,
    WheelAxlePoints,
)


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
