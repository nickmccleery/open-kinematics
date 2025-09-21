"""
Core primitive types and data structures for the kinematics module.

This module contains fundamental type definitions, coordinate systems, and basic
data structures that form the foundation of the kinematics system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, Iterator, Set, Tuple, TypeAlias

import numpy as np

from kinematics.points.ids import PointID

# --- Core Data Types ---
Position: TypeAlias = np.ndarray


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
class Positions:
    """
    Container for managing point positions with typed operations.

    Provides methods for converting between dictionary format and flat arrays
    for solver operations while maintaining a stable ordering of free points.
    """

    data: Dict[PointID, np.ndarray]

    def get(self, point_id: PointID) -> np.ndarray:
        """Get position of a specific point."""
        return self.data[point_id]

    def set(self, point_id: PointID, position: np.ndarray) -> None:
        """Set position of a specific point."""
        self.data[point_id] = position.copy()

    def array(self, free_points: Iterable[PointID]) -> np.ndarray:
        """
        Convert free points to flat array for solver.

        Args:
            free_points: Iterable of point IDs to extract (order preserved)

        Returns:
            Array of shape (n_points * 3,) containing flattened positions
        """
        free_points_list = list(free_points)  # Ensure stable order
        positions = [self.data[pid] for pid in free_points_list]
        return np.concatenate(positions)

    def update_from_array(
        self, free_points: Iterable[PointID], array: np.ndarray
    ) -> None:
        """
        Update positions from flat array returned by solver.

        Args:
            free_points: Iterable of point IDs (must match order used in array())
            array: Flat array of shape (n_points * 3,) containing new positions
        """
        free_points_list = list(free_points)  # Ensure stable order
        n_points = len(free_points_list)

        if array.shape != (n_points * 3,):
            raise ValueError(
                f"Array shape {array.shape} doesn't match expected ({n_points * 3},) "
                f"for {n_points} points"
            )

        # Reshape and update
        positions_2d = array.reshape(n_points, 3)
        for i, point_id in enumerate(free_points_list):
            self.data[point_id] = positions_2d[i].copy()

    def copy(self) -> "Positions":
        """Create a deep copy of the positions."""
        return Positions({pid: pos.copy() for pid, pos in self.data.items()})

    def items(self) -> Iterator[Tuple[PointID, np.ndarray]]:
        """Iterate over (point_id, position) pairs."""
        return iter(self.data.items())

    def keys(self) -> Iterator[PointID]:
        """Iterate over point IDs."""
        return iter(self.data.keys())

    def values(self) -> Iterator[np.ndarray]:
        """Iterate over positions."""
        return iter(self.data.values())

    def __getitem__(self, point_id: PointID) -> np.ndarray:
        """Allow dict-like access: positions[point_id]."""
        return self.data[point_id]

    def __setitem__(self, point_id: PointID, position: np.ndarray) -> None:
        """Allow dict-like assignment: positions[point_id] = pos."""
        self.data[point_id] = position.copy()

    def __contains__(self, point_id: PointID) -> bool:
        """Check if point exists: point_id in positions."""
        return point_id in self.data

    def __len__(self) -> int:
        """Get number of points."""
        return len(self.data)


# --- Core Data Models ---
@dataclass(frozen=True)
class KinematicsState:
    """Represents the state of a kinematic system at a specific configuration."""

    positions: Positions
    free_points: Set["PointID"]

    @property
    def fixed_points(self) -> Set["PointID"]:
        """Points that are fixed (not free to move)."""
        return set(self.positions.keys()) - self.free_points


@dataclass(frozen=True)
class GeometryDefinition:
    """Defines the geometry of a suspension system."""

    hard_points: Positions
    free_points: Set["PointID"]
