from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from kinematics.geometry.points.ids import PointID

# Import the type aliases from the new location
from kinematics.core.types import Positions


@dataclass(frozen=True)
class KinematicsState:
    positions: Positions
    free_points: Set[PointID]

    @property
    def fixed_points(self) -> Set[PointID]:
        return set(self.positions.keys()) - self.free_points


@dataclass(frozen=True)
class GeometryDefinition:
    hard_points: Positions
    free_points: Set[PointID]