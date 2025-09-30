"""
MacPherson strut suspension geometry models - pure data structures.

Contains dataclasses/types for MacPherson strut geometry.
No imports from solver/constraints/visualization - only configuration and point collections.
"""

from dataclasses import dataclass

from kinematics.points.collections import (
    LowerWishbonePoints,
    StrutPoints,
    WheelAxlePoints,
)
from kinematics.suspensions.base.models import SuspensionGeometry


@dataclass
class MacPhersonHardPoints:
    """Hard point collection for MacPherson strut suspension."""

    lower_wishbone: LowerWishbonePoints
    strut: StrutPoints
    wheel_axle: WheelAxlePoints


@dataclass
class MacPhersonGeometry(SuspensionGeometry):
    """MacPherson strut suspension geometry definition."""

    hard_points: MacPhersonHardPoints

    def validate(self) -> bool:
        return True
