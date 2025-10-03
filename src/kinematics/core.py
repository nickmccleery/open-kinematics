"""
Core primitives and data structures for suspension kinematics.

This module provides fundamental type definitions, enumerations, and state management
classes that form the foundation of the kinematics system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Set

import numpy as np


class PointID(IntEnum):
    """
    Enumeration of all point identifiers used in the suspension system.

    These identifiers represent key points in the suspension geometry including wishbone
    attachment points, pushrods, trackrods, axles, struts, and wheel centers.
    """

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
class SuspensionState:
    """
    Represents the complete state of a suspension system.

    This class manages point positions and solver metadata, providing methods
    for state manipulation, solver integration, and coordinate transformations.

    Attributes:
        positions (dict[PointID, np.ndarray]): Dictionary mapping point IDs to 3D positions.
        free_points (Set[PointID]): Set of point IDs that are free to move during solving.
        free_points_order (List[PointID]): Sorted list of free point IDs for consistent ordering.
    """

    positions: dict[PointID, np.ndarray]
    free_points: Set[PointID]
    free_points_order: List[PointID] = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize consistent ordering for free points.
        """
        self.free_points_order = sorted(list(self.free_points))

    @property
    def fixed_points(self) -> Set[PointID]:
        """
        Points that are fixed (not free to move).
        """
        return set(self.positions.keys()) - self.free_points

    def get_free_array(self) -> np.ndarray:
        """
        Convert free points to flat array for solver.
        """
        positions = [self.positions[pid] for pid in self.free_points_order]
        return np.concatenate(positions)

    def update_from_array(self, array: np.ndarray) -> None:
        """
        Update free points from solver array in-place.
        """
        n_points = len(self.free_points_order)
        if array.shape != (n_points * 3,):
            raise ValueError(
                f"Array shape {array.shape} doesn't match expected ({n_points * 3},)"
            )

        positions_2d = array.reshape(n_points, 3)
        for i, point_id in enumerate(self.free_points_order):
            self.positions[point_id] = positions_2d[i].copy()

    def copy(self) -> "SuspensionState":
        """
        Create a deep copy.
        """
        return SuspensionState(
            positions={pid: pos.copy() for pid, pos in self.positions.items()},
            free_points=self.free_points.copy(),
        )

    def get(self, point_id: PointID) -> np.ndarray:
        """
        Get position of a specific point.
        """
        return self.positions[point_id]

    def set(self, point_id: PointID, position: np.ndarray) -> None:
        """
        Set position of a specific point.
        """
        self.positions[point_id] = position.copy()

    def __getitem__(self, point_id: PointID) -> np.ndarray:
        """
        Allow dict-like access.
        """
        return self.positions[point_id]

    def __setitem__(self, point_id: PointID, position: np.ndarray) -> None:
        """
        Allow dict-like assignment.
        """
        self.positions[point_id] = position.copy()

    def __contains__(self, point_id: PointID) -> bool:
        """
        Check if point exists.
        """
        return point_id in self.positions

    def items(self):
        """
        Iterate over (point_id, position) pairs.
        """
        return self.positions.items()

    def keys(self):
        """
        Iterate over point IDs.
        """
        return self.positions.keys()

    def values(self):
        """
        Iterate over positions.
        """
        return self.positions.values()
