"""Concrete axle models composed from corner suspension models."""

from kinematics.suspensions.axle.double_wishbone import (
    DoubleWishboneAxleSuspension,
)
from kinematics.suspensions.axle.double_wishbone_pushrod_rocker import (
    DoubleWishbonePushrodRockerAxleSuspension,
)

__all__ = [
    "DoubleWishboneAxleSuspension",
    "DoubleWishbonePushrodRockerAxleSuspension",
]
