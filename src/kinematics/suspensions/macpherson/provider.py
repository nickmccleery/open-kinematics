"""
MacPherson strut suspension provider implementation.

Contains orchestration logic for constraint building, free points definition,
and derived point calculations specific to MacPherson strut suspensions.
"""

from functools import partial
from typing import Sequence

import numpy as np

from kinematics.constraints import Constraint, DistanceConstraint
from kinematics.core import SuspensionState
from kinematics.math import compute_point_point_distance
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.spec import DerivedSpec
from kinematics.points.ids import PointID
from kinematics.suspensions.base.provider import SuspensionProvider
from kinematics.suspensions.macpherson.model import MacPhersonGeometry


class MacPhersonProvider(SuspensionProvider):
    """The concrete implementation of the BaseProvider for MacPherson strut geometry."""

    def __init__(self, geometry: MacPhersonGeometry):
        self.geometry: MacPhersonGeometry = geometry

    def initial_state(self) -> SuspensionState:
        """Create initial suspension state from geometry."""
        positions = {}

        # Convert point collections to positions dict
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

        return SuspensionState(positions=positions, free_points=set(self.free_points()))

    def free_points(self) -> Sequence[PointID]:
        """Defines which points the solver is allowed to move."""
        return [
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.STRUT_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ]

    def derived_spec(self) -> DerivedSpec:
        """
        Returns a DerivedSpec with all derived points, their calculation functions, and dependencies.
        """
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

        return DerivedSpec(functions=functions, dependencies=dependencies)

    def constraints(self) -> list[Constraint]:
        """Builds the complete list of constraints that define the suspension's mechanics."""
        constraints: list[Constraint] = []
        initial_state = self.initial_state()

        # 1. Fixed distance constraints between points
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
