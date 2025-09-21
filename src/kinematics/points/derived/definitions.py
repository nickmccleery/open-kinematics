"""
Derived point calculations for suspension kinematics.

Contains functions to calculate secondary points based on primary hard points,
such as wheel centers, midpoints, and offset positions.
"""

from kinematics.math import normalize_vector
from kinematics.points.main import PointID
from kinematics.primitives import Position, Positions


def get_axle_midpoint(positions: Positions) -> Position:
    """Calculates the midpoint of the axle."""
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.AXLE_OUTBOARD]
    return (p1 + p2) / 2


def get_wheel_center(positions: Positions, wheel_offset: float) -> Position:
    """Calculates the wheel center point, offset from the axle outboard face."""
    p1 = positions[PointID.AXLE_OUTBOARD]
    p2 = positions[PointID.AXLE_INBOARD]
    v = p1 - p2  # Vector pointing outboard
    v = normalize_vector(v)
    return p1 + v * wheel_offset


def get_wheel_inboard(positions: Positions, wheel_width: float) -> Position:
    """Calculates the inboard lip of the wheel."""
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.WHEEL_CENTER]
    v = p2 - p1  # Vector from axle inboard to wheel center
    v = normalize_vector(v)
    return p2 - v * (wheel_width / 2)


def get_wheel_outboard(positions: Positions, wheel_width: float) -> Position:
    """Calculates the outboard lip of the wheel."""
    p1 = positions[PointID.WHEEL_CENTER]
    p2 = positions[PointID.AXLE_INBOARD]
    v = p1 - p2  # Vector from axle inboard to wheel center
    v = normalize_vector(v)
    return p1 + v * (wheel_width / 2)
