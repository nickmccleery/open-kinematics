"""
Core primitive types and data structures for the kinematics module.

This module contains fundamental type definitions, coordinate systems, and basic
data structures that form the foundation of the kinematics system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Set, TypeAlias

import numpy as np

# --- Core Data Types ---
Position: TypeAlias = np.ndarray
Positions: TypeAlias = Dict["PointID", Position]


# --- Coordinate System ---
class Direction:
    """Cardinal direction vectors in 3D space."""

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])


class CoordinateAxis(IntEnum):
    """Enumeration of coordinate axes."""

    X = 0
    Y = 1
    Z = 2


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
