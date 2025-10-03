"""
Common suspension component point collections.

This module defines dataclasses for point collections which are shared across suspension
architectures.
"""

from dataclasses import dataclass


@dataclass
class LowerWishbonePoints:
    """
    Points defining the lower wishbone geometry.

    Attributes:
        inboard_front: Front inboard mounting point coordinates.
        inboard_rear: Rear inboard mounting point coordinates.
        outboard: Outboard mounting point coordinates.
    """

    inboard_front: dict[str, float]
    inboard_rear: dict[str, float]
    outboard: dict[str, float]


@dataclass
class UpperWishbonePoints:
    """
    Points defining the upper wishbone geometry.

    Attributes:
        inboard_front: Front inboard mounting point coordinates.
        inboard_rear: Rear inboard mounting point coordinates.
        outboard: Outboard mounting point coordinates.
    """

    inboard_front: dict[str, float]
    inboard_rear: dict[str, float]
    outboard: dict[str, float]


@dataclass
class WheelAxlePoints:
    """
    Points defining the wheel axle geometry.

    Attributes:
        inner: Inner axle point coordinates.
        outer: Outer axle point coordinates.
    """

    inner: dict[str, float]
    outer: dict[str, float]


@dataclass
class TrackRodPoints:
    """
    Points defining the track rod/tie rod geometry.

    Attributes:
        inner: Inner track rod mounting point coordinates.
        outer: Outer track rod mounting point coordinates.
    """

    inner: dict[str, float]
    outer: dict[str, float]
