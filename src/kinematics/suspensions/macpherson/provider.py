"""
MacPherson strut suspension provider implementation.

Contains orchestration logic for constraint building, free points definition,
and derived point calculations specific to MacPherson strut suspensions.
"""

from functools import partial
from typing import Sequence

from kinematics.constraints import Constraint, make_point_point_distance
from kinematics.core import Positions
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.spec import DerivedSpec
from kinematics.points.ids import PointID
from kinematics.points.utils import get_all_points
from kinematics.suspensions.macpherson.model import MacPhersonModel
from kinematics.suspensions.provider import BaseProvider


class MacPhersonProvider(BaseProvider):
    """The concrete implementation of the BaseProvider for MacPherson strut geometry."""

    def __init__(self, geometry: MacPhersonModel):
        self.geometry: MacPhersonModel = geometry

    def initial_positions(self) -> Positions:
        """Extracts all hard points from the geometry file into a Positions object."""
        points = get_all_points(self.geometry.hard_points)
        data = {p.id: p.as_array() for p in points}
        return Positions(data)

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
        initial_positions = self.initial_positions()

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
            constraints.append(make_point_point_distance(initial_positions, p1, p2))

        return constraints
