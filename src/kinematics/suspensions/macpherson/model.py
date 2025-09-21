"""
MacPherson strut suspension geometry models - pure data structures.

Contains dataclasses/types for MacPherson strut geometry.
No imports from solver/constraints/visualization - only configuration and point collections.
"""

from abc import ABC
from dataclasses import dataclass

from kinematics.configs import SuspensionConfig, Units
from kinematics.points.collections import (
    LowerWishbonePoints,
    StrutPoints,
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
class MacPhersonHardPoints:
    """Hard point collection for MacPherson strut suspension."""

    lower_wishbone: LowerWishbonePoints
    strut: StrutPoints
    wheel_axle: WheelAxlePoints


@dataclass
class MacPhersonModel(SuspensionGeometry):
    """MacPherson strut suspension geometry definition."""

    hard_points: MacPhersonHardPoints

    def validate(self) -> bool:
        return True
