"""
Double wishbone suspension implementation.
"""

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import numpy as np

from kinematics.constraints import (
    AngleConstraint,
    Constraint,
    DistanceConstraint,
    PointOnLineConstraint,
)
from kinematics.enums import PointID
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.state import SuspensionState
from kinematics.suspensions.base.geometry import SuspensionGeometry
from kinematics.suspensions.base.provider import SuspensionProvider
from kinematics.suspensions.common.collections import (
    LowerWishbonePoints,
    TrackRodPoints,
    UpperWishbonePoints,
    WheelAxlePoints,
)
from kinematics.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
)


@dataclass
class DoubleWishboneHardPoints:
    """
    Hard point collection for double wishbone suspension.

    Attributes:
        lower_wishbone: Points defining the lower wishbone geometry.
        upper_wishbone: Points defining the upper wishbone geometry.
        track_rod: Points defining the track rod geometry.
        wheel_axle: Points defining the wheel axle geometry.
    """

    lower_wishbone: LowerWishbonePoints
    upper_wishbone: UpperWishbonePoints
    track_rod: TrackRodPoints
    wheel_axle: WheelAxlePoints


# Geometry model
@dataclass
class DoubleWishboneGeometry(SuspensionGeometry):
    """
    Double wishbone suspension geometry definition.

    Extends the base SuspensionGeometry with double wishbone specific hard points.

    Attributes:
        hard_points: Collection of all hard point coordinates for the suspension.
    """

    hard_points: DoubleWishboneHardPoints

    def validate(self) -> bool:
        """
        Validate the double wishbone geometry configuration.

        Returns:
            True if geometry is valid.
        """
        # ! TODO: Actually validate this geometry.
        return True


# Provider implementation
class DoubleWishboneProvider(SuspensionProvider):
    """
    Concrete implementation of SuspensionProvider for double wishbone geometry.
    """

    def __init__(self, geometry: DoubleWishboneGeometry):
        """
        Initialize the double wishbone provider.

        Args:
            geometry: Double wishbone geometry configuration.
        """
        self.geometry = geometry

    def initial_state(self) -> SuspensionState:
        """
        Create initial suspension state from geometry hard points.

        Converts the hard point coordinates from the geometry into a SuspensionState
        with both explicitly defined and derived points.

        Returns:
            Initial suspension state with all point positions.
        """
        positions = {}
        hard_points = self.geometry.hard_points

        # Lower wishbone.
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

        # Upper wishbone.
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

        # Track rod.
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

        # Calculate derived points to create a complete initial state.
        derived_spec = self.derived_spec()
        derived_resolver = DerivedPointsManager(derived_spec)
        all_positions = derived_resolver.update(positions)

        return SuspensionState(
            positions=all_positions, free_points=set(self.free_points())
        )

    def free_points(self) -> Sequence[PointID]:
        """
        Define which points the solver can move during optimization.

        Returns:
            Sequence of point IDs that are free to move (outboard and axle points).
        """
        return [
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
            PointID.TRACKROD_OUTBOARD,
            PointID.TRACKROD_INBOARD,
        ]

    def derived_spec(self) -> DerivedPointsSpec:
        """
        Define specifications for computing derived points from free points.

        Returns:
            Specification containing functions and dependencies for derived points.
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

        return DerivedPointsSpec(functions=functions, dependencies=dependencies)

    def constraints(self) -> list[Constraint]:
        """
        Build the complete set of geometric constraints for double wishbone suspension.

        Returns:
            List of constraints that must be satisfied during kinematic solving.
        """
        constraints: list[Constraint] = []
        initial_state = self.initial_state()

        # Distance constraints.
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
            target_distance = compute_point_point_distance(
                initial_state.positions[p1], initial_state.positions[p2]
            )
            constraints.append(DistanceConstraint(p1, p2, target_distance))

        # Angle constraints.
        v1 = (
            initial_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]
            - initial_state.positions[PointID.UPPER_WISHBONE_OUTBOARD]
        )
        v2 = (
            initial_state.positions[PointID.AXLE_OUTBOARD]
            - initial_state.positions[PointID.AXLE_INBOARD]
        )
        target_angle = compute_vector_vector_angle(v1, v2)

        constraints.append(
            AngleConstraint(
                v1_start=PointID.UPPER_WISHBONE_OUTBOARD,
                v1_end=PointID.LOWER_WISHBONE_OUTBOARD,
                v2_start=PointID.AXLE_INBOARD,
                v2_end=PointID.AXLE_OUTBOARD,
                target_angle=target_angle,
            )
        )

        # Point-on-line constraints.
        constraints.append(
            PointOnLineConstraint(
                point_id=PointID.TRACKROD_INBOARD,
                line_point=initial_state.positions[PointID.TRACKROD_INBOARD],
                line_direction=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )
        )

        return constraints
