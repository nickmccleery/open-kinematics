"""
Double wishbone suspension provider implementation.

Contains orchestration logic for constraint building, free points definition,
and derived point calculations specific to double wishbone suspensions.
"""

from functools import partial
from typing import Sequence

from kinematics.constraints import (
    Constraint,
    PointOnLine,
    make_point_point_distance,
    make_vector_angle,
)
from kinematics.core import Direction, Positions
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.spec import DerivedSpec
from kinematics.points.ids import PointID
from kinematics.points.utils import get_all_points
from kinematics.suspensions.base.provider import SuspensionProvider
from kinematics.suspensions.double_wishbone.model import DoubleWishboneGeometry


class DoubleWishboneProvider(SuspensionProvider):
    """The concrete implementation of the BaseProvider for Double Wishbone geometry."""

    def __init__(self, geometry: DoubleWishboneGeometry):
        self.geometry: DoubleWishboneGeometry = geometry

    def initial_positions(self) -> Positions:
        """Extracts all hard points from the geometry file into a Positions object."""
        points = get_all_points(self.geometry.hard_points)
        data = {p.id: p.as_array() for p in points}
        return Positions(data)

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
        initial_positions = self.initial_positions()

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
