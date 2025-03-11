from typing import Set

from kinematics.geometry.points.ids import PointID
from kinematics.points.ops import normalize_vector
from kinematics.types.state import Position, Positions

DEPENDENCIES = {
    PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
    PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
    PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
    PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
}


def get_dependencies(point_id: PointID) -> Set[PointID]:
    return DEPENDENCIES.get(point_id, set())


def update_axle_midpoint(positions: Positions) -> Position:
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.AXLE_OUTBOARD]
    return (p1 + p2) / 2


def update_wheel_center(positions: Positions, wheel_offset: float) -> Position:
    # The wheel center is offset outboard from the outboard axle point
    p1 = positions[PointID.AXLE_OUTBOARD]
    p2 = positions[PointID.AXLE_INBOARD]
    v = p1 - p2  # Vector pointing outboard
    v = normalize_vector(v)
    return p1 + v * wheel_offset


def update_wheel_inboard(positions: Positions, wheel_width: float) -> Position:
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.WHEEL_CENTER]
    v = p2 - p1
    v = normalize_vector(v)
    return p2 - v * (wheel_width / 2)


def update_wheel_outboard(positions: Positions, wheel_width: float) -> Position:
    p1 = positions[PointID.WHEEL_CENTER]
    p2 = positions[PointID.AXLE_INBOARD]
    v = p1 - p2
    v = normalize_vector(v)
    return p1 + v * (wheel_width / 2)


def update_derived_point(
    point_id: PointID,
    positions: Positions,
    wheel_width: float = 0.0,
    wheel_offset: float = 0.0,
) -> Position:
    if point_id == PointID.AXLE_MIDPOINT:
        return update_axle_midpoint(positions)
    elif point_id == PointID.WHEEL_CENTER:
        return update_wheel_center(positions, wheel_offset)
    elif point_id == PointID.WHEEL_INBOARD:
        return update_wheel_inboard(positions, wheel_width)
    elif point_id == PointID.WHEEL_OUTBOARD:
        return update_wheel_outboard(positions, wheel_width)
    else:
        raise ValueError(f"Unknown derived point type: {point_id}")
