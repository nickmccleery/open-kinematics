"""
Complete MacPherson strut suspension implementation.

Contains geometry models, provider logic, and derived point calculations
all in one place for the MacPherson strut suspension type.
"""

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import numpy as np

from kinematics.constraints import Constraint, DistanceConstraint
from kinematics.core import PointID, SuspensionState
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.suspensions.base.geometry import SuspensionGeometry
from kinematics.suspensions.base.provider import SuspensionProvider
from kinematics.suspensions.common.collections import (
    LowerWishbonePoints,
    WheelAxlePoints,
)
from kinematics.vector_utils.geometric import compute_point_point_distance


@dataclass
class StrutPoints:
    """Points defining the strut geometry."""

    inboard: dict[str, float]
    outboard: dict[str, float]


@dataclass
class MacPhersonHardPoints:
    """Hard point collection for MacPherson strut suspension."""

    lower_wishbone: LowerWishbonePoints
    strut: StrutPoints
    wheel_axle: WheelAxlePoints


# Geometry model
@dataclass
class MacPhersonGeometry(SuspensionGeometry):
    """MacPherson strut suspension geometry definition."""

    hard_points: MacPhersonHardPoints

    def validate(self) -> bool:
        return True


# Provider implementation
class MacPhersonProvider(SuspensionProvider):
    """The concrete implementation for MacPherson strut geometry."""

    def __init__(self, geometry: MacPhersonGeometry):
        self.geometry = geometry

    def initial_state(self) -> SuspensionState:
        """Create initial suspension state from geometry."""
        positions = {}
        hard_points = self.geometry.hard_points

        # Lower wishbone
        lwb = hard_points.lower_wishbone
        positions[PointID.LOWER_WISHBONE_INBOARD_FRONT] = np.array(
            [lwb.inboard_front["x"], lwb.inboard_front["y"], lwb.inboard_front["z"]]
        )
        positions[PointID.LOWER_WISHBONE_INBOARD_REAR] = np.array(
            [lwb.inboard_rear["x"], lwb.inboard_rear["y"], lwb.inboard_rear["z"]]
        )
        positions[PointID.LOWER_WISHBONE_OUTBOARD] = np.array(
            [lwb.outboard["x"], lwb.outboard["y"], lwb.outboard["z"]]
        )

        # Strut
        strut = hard_points.strut
        positions[PointID.STRUT_INBOARD] = np.array(
            [strut.inboard["x"], strut.inboard["y"], strut.inboard["z"]]
        )
        positions[PointID.STRUT_OUTBOARD] = np.array(
            [strut.outboard["x"], strut.outboard["y"], strut.outboard["z"]]
        )

        # Wheel axle
        wa = hard_points.wheel_axle
        positions[PointID.AXLE_INBOARD] = np.array(
            [wa.inner["x"], wa.inner["y"], wa.inner["z"]]
        )
        positions[PointID.AXLE_OUTBOARD] = np.array(
            [wa.outer["x"], wa.outer["y"], wa.outer["z"]]
        )

        # Calculate derived points to create a complete initial state
        derived_spec = self.derived_spec()
        derived_resolver = DerivedPointsManager(derived_spec)
        all_positions = derived_resolver.update(positions)

        return SuspensionState(
            positions=all_positions, free_points=set(self.free_points())
        )

    def free_points(self) -> Sequence[PointID]:
        """Defines which points the solver is allowed to move."""
        return [
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.STRUT_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ]

    def derived_spec(self) -> DerivedPointsSpec:
        """Returns derived point specifications."""
        wheel_cfg = self.geometry.configuration.wheel

        functions = {
            PointID.AXLE_MIDPOINT: get_axle_midpoint,
            PointID.WHEEL_CENTER: partial(
                get_wheel_center, wheel_offset=wheel_cfg.offset
            ),
            PointID.WHEEL_INBOARD: partial(
                get_wheel_inboard, wheel_width=wheel_cfg.width
            ),
            PointID.WHEEL_OUTBOARD: partial(
                get_wheel_outboard, wheel_width=wheel_cfg.width
            ),
        }

        dependencies = {
            PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        }

        return DerivedPointsSpec(functions=functions, dependencies=dependencies)

    def constraints(self) -> list[Constraint]:
        """Builds the complete list of constraints."""
        constraints: list[Constraint] = []
        initial_state = self.initial_state()

        # Distance constraints
        length_pairs = [
            (PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.STRUT_INBOARD, PointID.STRUT_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.STRUT_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.STRUT_OUTBOARD),
        ]
        for p1, p2 in length_pairs:
            target_distance = compute_point_point_distance(
                initial_state.positions[p1], initial_state.positions[p2]
            )
            constraints.append(DistanceConstraint(p1, p2, target_distance))

        return constraints
