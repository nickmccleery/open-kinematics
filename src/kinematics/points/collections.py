"""
Point collection classes for different suspension components.

Each collection represents the points that define a specific suspension component
like wishbones, struts, track rods, etc.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class LowerWishbonePoints:
    """Points defining the lower wishbone geometry."""

    inboard_front: Dict[str, float]  # {'x': val, 'y': val, 'z': val}
    inboard_rear: Dict[str, float]
    outboard: Dict[str, float]


@dataclass
class UpperWishbonePoints:
    """Points defining the upper wishbone geometry."""

    inboard_front: Dict[str, float]
    inboard_rear: Dict[str, float]
    outboard: Dict[str, float]


@dataclass
class StrutPoints:
    """Points defining the strut geometry."""

    inboard: Dict[str, float]
    outboard: Dict[str, float]


@dataclass
class WheelAxlePoints:
    """Points defining the wheel axle geometry."""

    inner: Dict[str, float]
    outer: Dict[str, float]


@dataclass
class TrackRodPoints:
    """Points defining the track rod/tie rod geometry."""

    inner: Dict[str, float]
    outer: Dict[str, float]
