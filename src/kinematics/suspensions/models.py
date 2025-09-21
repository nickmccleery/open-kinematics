"""
Suspension geometry models - pure data structures.

Contains dataclasses/types for each suspension geometry (e.g., DoubleWishboneGeometry).
No imports from solver/constraints/visualization - only configuration and point collections.
"""

from abc import ABC
from dataclasses import dataclass

from kinematics.configs import SuspensionConfig, Units
from kinematics.points.collections import (
    LowerWishbonePoints,
    StrutPoints,
    TrackRodPoints,
    UpperWishbonePoints,
    WheelAxlePoints,
)


@dataclass
class SuspensionGeometry(ABC):
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
    """Hard point collection for double wishbone suspension."""

    lower_wishbone: LowerWishbonePoints
    upper_wishbone: UpperWishbonePoints
    track_rod: TrackRodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class DoubleWishboneGeometry(SuspensionGeometry):
    """Double wishbone suspension geometry definition."""

    hard_points: DoubleWishboneHardPoints

    def validate(self) -> bool:
        return True


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
