"""
Suspension geometry definitions and concrete implementations.

Contains the base suspension geometry class and all concrete suspension
type implementations like Double Wishbone and MacPherson strut.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Set

from kinematics.configs import SuspensionConfig, Units
from kinematics.constraints import (
    PointOnLine,
    make_point_point_distance,
    make_vector_angle,
)
from kinematics.points.collections import (
    LowerWishbonePoints,
    StrutPoints,
    TrackRodPoints,
    UpperWishbonePoints,
    WheelAxlePoints,
)
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.ids import PointID
from kinematics.points.utils import get_all_points
from kinematics.primitives import Direction


class SuspensionProvider(ABC):
    """
    Abstract base class defining the interface for a suspension type.

    Each implementation provides the specific rules (constraints, free points,
    derived point definitions, etc.) for its corresponding geometry.
    """

    def __init__(self, geometry: "SuspensionGeometry"):
        self.geometry = geometry

    @abstractmethod
    def get_initial_positions(self) -> Dict:
        """Constructs the initial dictionary of 3D point positions from the geometry."""
        ...

    @abstractmethod
    def get_free_points(self) -> Set:
        """Returns the set of PointIDs that the solver is allowed to move."""
        ...

    @abstractmethod
    def get_constraints(self, initial_positions) -> List:
        """Returns the full list of constraints that define the suspension's movement."""
        ...

    @abstractmethod
    def get_derived_point_definitions(self) -> Dict:
        """
        Returns a dictionary mapping derived PointIDs to their update function
        and dependencies.
        """
        ...


@dataclass
class SuspensionGeometry:
    """
    Base class for all suspension geometry types.
    """

    name: str
    version: str
    units: Units
    configuration: SuspensionConfig

    def validate(self) -> bool:
        raise NotImplementedError("Subclasses must implement validate()")


@dataclass
class DoubleWishboneHardPoints:
    """Hard point collection for double wishbone suspension."""

    lower_wishbone: LowerWishbonePoints
    upper_wishbone: UpperWishbonePoints
    track_rod: TrackRodPoints
    wheel_axle: WheelAxlePoints


@dataclass
class DoubleWishboneGeometry(SuspensionGeometry):
    """Double wishbone suspension geometry definition."""

    hard_points: DoubleWishboneHardPoints

    def validate(self) -> bool:
        return True


@dataclass
class MacPhersonHardPoints:
    """Hard point collection for MacPherson strut suspension."""

    lower_wishbone: LowerWishbonePoints
    strut: StrutPoints
    wheel_axle: WheelAxlePoints


@dataclass
class MacPhersonGeometry(SuspensionGeometry):
    """MacPherson strut suspension geometry definition."""

    hard_points: MacPhersonHardPoints

    def validate(self) -> bool:
        return True


class DoubleWishboneProvider(SuspensionProvider):
    """The concrete implementation of the SuspensionProvider for Double Wishbone geometry."""

    def __init__(self, geometry: DoubleWishboneGeometry):
        super().__init__(geometry)
        # Add type hint for more specific geometry access
        self.geometry: DoubleWishboneGeometry = geometry

    def get_initial_positions(self):
        """Extracts all hard points from the geometry file into the positions dictionary."""
        points = get_all_points(self.geometry.hard_points)
        return {p.id: p.as_array() for p in points}

    def get_free_points(self) -> Set:
        """Defines which points the solver is allowed to move."""
        return {
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            PointID.TRACKROD_OUTBOARD,
            PointID.TRACKROD_INBOARD,
        }

    def get_derived_point_definitions(self) -> Dict:
        """
        Defines all derived points, their calculation functions, and their dependencies.
        """
        wheel_cfg = self.geometry.configuration.wheel
        return {
            PointID.AXLE_MIDPOINT: (
                get_axle_midpoint,
                {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            ),
            PointID.WHEEL_CENTER: (
                partial(get_wheel_center, wheel_offset=wheel_cfg.offset),
                {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            ),
            PointID.WHEEL_INBOARD: (
                partial(get_wheel_inboard, wheel_width=wheel_cfg.width),
                {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            ),
            PointID.WHEEL_OUTBOARD: (
                partial(get_wheel_outboard, wheel_width=wheel_cfg.width),
                {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            ),
        }

    def get_constraints(self, initial_positions) -> List:
        """Builds the complete list of constraints that define the suspension's mechanics."""
        constraints: List = []

        # 1. Fixed distance constraints between points
        length_pairs = [
            (PointID.UPPER_WISHBONE_INBOARD_FRONT, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.UPPER_WISHBONE_INBOARD_REAR, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.UPPER_WISHBONE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.UPPER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.LOWER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.TRACKROD_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.TRACKROD_OUTBOARD),
        ]
        for p1, p2 in length_pairs:
            constraints.append(make_point_point_distance(initial_positions, p1, p2))

        # 2. Fixed angle constraints between vectors
        constraints.append(
            make_vector_angle(
                initial_positions,
                v1_start=PointID.UPPER_WISHBONE_OUTBOARD,
                v1_end=PointID.LOWER_WISHBONE_OUTBOARD,
                v2_start=PointID.AXLE_INBOARD,
                v2_end=PointID.AXLE_OUTBOARD,
            )
        )

        # 3. Point-on-line constraints (for steering rack)
        constraints.append(
            PointOnLine(
                point_id=PointID.TRACKROD_INBOARD,
                line_point=initial_positions[PointID.TRACKROD_INBOARD],
                line_direction=Direction.y,
            )
        )

        return constraints
