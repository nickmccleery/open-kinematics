from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Set, Type, TypeAlias

import numpy as np

from kinematics.geometry.points.ids import PointID

if TYPE_CHECKING:
    from kinematics.suspensions.base import SuspensionProvider

# --- Core Data Types ---
Position: TypeAlias = np.ndarray
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

# --- Registry and Geometry Union Types ---
# These will be populated at runtime to avoid circular imports
PROVIDER_REGISTRY: Dict[Type, Type[SuspensionProvider]] = {}

# For now, use Any for GeometryType to avoid circular imports
# This can be refined later once the circular imports are resolved
from typing import Any

GeometryType = Any

GEOMETRY_TYPES: dict[str, Type[GeometryType]] = {}


def _populate_registries() -> None:
    """Populate registries at runtime to avoid circular import issues."""
    # Import here to avoid circular imports
    from kinematics.suspensions.double_wishbone.geometry import \
        DoubleWishboneGeometry
    from kinematics.suspensions.double_wishbone.provider import \
        DoubleWishboneProvider
    from kinematics.suspensions.macpherson.geometry import MacPhersonGeometry

    # Populate the registries
    PROVIDER_REGISTRY[DoubleWishboneGeometry] = DoubleWishboneProvider
    
    GEOMETRY_TYPES.update({
        "DOUBLE_WISHBONE": DoubleWishboneGeometry,
        "MACPHERSON_STRUT": MacPhersonGeometry,
    })    })