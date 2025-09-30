"""
Double wishbone suspension geometry models - pure data structures.

Contains dataclasses/types for double wishbone geometry.
No imports from solver/constraints/visualization - only configuration and point collections.
"""

from dataclasses import dataclass

from kinematics.points.collections import (
    LowerWishbonePoints,
    TrackRodPoints,
    UpperWishbonePoints,
    WheelAxlePoints,
)
from kinematics.suspensions.base.models import SuspensionGeometry


@dataclass
class DoubleWishboneHardPoints:
    """Hard point collection for double wishbone suspension."""

    lower_wishbone: LowerWishbonePoints
    upper_wishbone: UpperWishbonePoints
    track_rod: TrackRodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class DoubleWishboneGeometry(SuspensionGeometry):
    """
    Double wishbone suspension geometry definition.
    """

    hard_points: DoubleWishboneHardPoints

    def validate(self) -> bool:
        return True
