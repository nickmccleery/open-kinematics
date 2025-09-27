"""
Core primitive types and data structures for the kinematics module.

This module contains fundamental type definitions, coordinate systems, and basic
data structures that form the foundation of the kinematics system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Set, TypeAlias

import numpy as np

from kinematics.points.ids import PointID

Position: TypeAlias = np.ndarray


# --- Coordinate System ---
class Direction:
    """Principal direction vectors in 3D space."""

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])


class CoordinateAxis(IntEnum):
    """Enumeration of coordinate axes."""

    X = 0
    Y = 1
    Z = 2


@dataclass
class SuspensionState:
    """
    Unified state representation for suspension geometry.
    Combines positions and solver metadata in one structure.
    """

    positions: Dict[PointID, np.ndarray]
    free_points: Set[PointID]
    free_points_order: List[PointID] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize consistent ordering for free points."""
        self.free_points_order = sorted(list(self.free_points))

    @property
    def fixed_points(self) -> Set[PointID]:
        """Points that are fixed (not free to move)."""
        return set(self.positions.keys()) - self.free_points

    def get_free_array(self) -> np.ndarray:
        """Convert free points to flat array for solver."""
        positions = [self.positions[pid] for pid in self.free_points_order]
        return np.concatenate(positions)

    def update_from_array(self, array: np.ndarray) -> None:
        """Update free points from solver array in-place."""
        n_points = len(self.free_points_order)
        if array.shape != (n_points * 3,):
            raise ValueError(
                f"Array shape {array.shape} doesn't match expected ({n_points * 3},)"
            )

        positions_2d = array.reshape(n_points, 3)
        for i, point_id in enumerate(self.free_points_order):
            self.positions[point_id] = positions_2d[i].copy()

    def copy(self) -> "SuspensionState":
        """Create a deep copy."""
        return SuspensionState(
            positions={pid: pos.copy() for pid, pos in self.positions.items()},
            free_points=self.free_points.copy(),
        )

    def get(self, point_id: PointID) -> np.ndarray:
        """Get position of a specific point."""
        return self.positions[point_id]

    def set(self, point_id: PointID, position: np.ndarray) -> None:
        """Set position of a specific point."""
        self.positions[point_id] = position.copy()

    def __getitem__(self, point_id: PointID) -> np.ndarray:
        """Allow dict-like access."""
        return self.positions[point_id]

    def __setitem__(self, point_id: PointID, position: np.ndarray) -> None:
        """Allow dict-like assignment."""
        self.positions[point_id] = position.copy()

    def __contains__(self, point_id: PointID) -> bool:
        """Check if point exists."""
        return point_id in self.positions

    def items(self):
        """Iterate over (point_id, position) pairs."""
        return self.positions.items()

    def keys(self):
        """Iterate over point IDs."""
        return self.positions.keys()

    def values(self):
        """Iterate over positions."""
        return self.positions.values()


@dataclass(frozen=True)
class GeometryDefinition:
    """Defines the geometry of a suspension system."""

    hard_points: Dict[PointID, np.ndarray]
    free_points: Set[PointID]
