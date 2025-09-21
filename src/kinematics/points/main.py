"""
Core point definitions and utilities.

Contains the fundamental point ID enumeration, 3D point class, and utility functions
for working with points.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class PointID(IntEnum):
    """Enumeration of all suspension system point identifiers."""

    NOT_ASSIGNED = 0

    LOWER_WISHBONE_INBOARD_FRONT = 1
    LOWER_WISHBONE_INBOARD_REAR = 2
    LOWER_WISHBONE_OUTBOARD = 3

    UPPER_WISHBONE_INBOARD_FRONT = 4
    UPPER_WISHBONE_INBOARD_REAR = 5
    UPPER_WISHBONE_OUTBOARD = 6

    PUSHROD_INBOARD = 7
    PUSHROD_OUTBOARD = 8

    TRACKROD_INBOARD = 9
    TRACKROD_OUTBOARD = 10

    AXLE_INBOARD = 11
    AXLE_OUTBOARD = 12
    AXLE_MIDPOINT = 13

    STRUT_INBOARD = 14
    STRUT_OUTBOARD = 15

    WHEEL_CENTER = 16
    WHEEL_INBOARD = 17
    WHEEL_OUTBOARD = 18


@dataclass
class Point3D:
    """Represents a 3D point in space with optional metadata."""

    x: float
    y: float
    z: float
    fixed: bool = False
    id: PointID = PointID.NOT_ASSIGNED

    def as_array(self) -> np.ndarray:
        """Convert point to numpy array."""
        return np.array([self.x, self.y, self.z])


def get_all_points(obj) -> list[Point3D]:
    """
    Recursively extracts all Point3D objects from an object or its attributes.

    Args:
        obj: Object to search for Point3D instances

    Returns:
        List of all Point3D objects found
    """
    points = []
    if isinstance(obj, Point3D):
        points.append(obj)
    elif hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            points.extend(get_all_points(value))
    return points
