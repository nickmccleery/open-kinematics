from dataclasses import dataclass


@dataclass
class LowerWishbonePoints:
    """
    Points defining the lower wishbone geometry.
    """

    inboard_front: dict[str, float]
    inboard_rear: dict[str, float]
    outboard: dict[str, float]


@dataclass
class UpperWishbonePoints:
    """
    Points defining the upper wishbone geometry.
    """

    inboard_front: dict[str, float]
    inboard_rear: dict[str, float]
    outboard: dict[str, float]


@dataclass
class WheelAxlePoints:
    """
    Points defining the wheel axle geometry.
    """

    inner: dict[str, float]
    outer: dict[str, float]


@dataclass
class TrackRodPoints:
    """
    Points defining the track rod/tie rod geometry.
    """

    inner: dict[str, float]
    outer: dict[str, float]
