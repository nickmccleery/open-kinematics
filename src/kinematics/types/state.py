from dataclasses import dataclass
from typing import Dict, Set, TypeAlias

import numpy as np

from kinematics.geometry.points.ids import PointID

# A 3D point position is represented as a numpy array.
Position: TypeAlias = np.ndarray

# Maps point IDs to their positions.
Positions: TypeAlias = Dict[PointID, Position]


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
