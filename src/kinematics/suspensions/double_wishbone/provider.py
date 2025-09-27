"""
Double wishbone suspension provider implementation.

Contains orchestration logic for constraint building, free points definition,
and derived point calculations specific to double wishbone suspensions.
"""

from functools import partial
from typing import Sequence

import numpy as np

from kinematics.constraints import (
    Constraint,
    PointOnLine,
    make_point_point_distance,
    make_vector_angle,
)
from kinematics.core import Direction, SuspensionState
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.spec import DerivedSpec
from kinematics.points.ids import PointID
from kinematics.suspensions.base.provider import SuspensionProvider
from kinematics.suspensions.double_wishbone.model import DoubleWishboneGeometry


class DoubleWishboneProvider(SuspensionProvider):
    """The concrete implementation of the BaseProvider for Double Wishbone geometry."""

    def __init__(self, geometry: DoubleWishboneGeometry):
        self.geometry: DoubleWishboneGeometry = geometry

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

        # Upper wishbone
        uwb = hard_points.upper_wishbone
        positions[PointID.UPPER_WISHBONE_INBOARD_FRONT] = np.array(
            [uwb.inboard_front["x"], uwb.inboard_front["y"], uwb.inboard_front["z"]]
        )
        positions[PointID.UPPER_WISHBONE_INBOARD_REAR] = np.array(
            [uwb.inboard_rear["x"], uwb.inboard_rear["y"], uwb.inboard_rear["z"]]
        )
        positions[PointID.UPPER_WISHBONE_OUTBOARD] = np.array(
            [uwb.outboard["x"], uwb.outboard["y"], uwb.outboard["z"]]
        )

        # Track rod
        tr = hard_points.track_rod
        positions[PointID.TRACKROD_INBOARD] = np.array(
            [tr.inner["x"], tr.inner["y"], tr.inner["z"]]
        )
        positions[PointID.TRACKROD_OUTBOARD] = np.array(
            [tr.outer["x"], tr.outer["y"], tr.outer["z"]]
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
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            PointID.TRACKROD_OUTBOARD,
            PointID.TRACKROD_INBOARD,
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
            constraints.append(
                make_point_point_distance(initial_state.positions, p1, p2)
            )

        # 2. Fixed angle constraints between vectors
        constraints.append(
            make_vector_angle(
                initial_state.positions,
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
                line_point=initial_state.positions[PointID.TRACKROD_INBOARD],
                line_direction=Direction.y,
            )
        )

        return constraints
