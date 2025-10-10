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
from kinematics.enums import Axis, PointID
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.state import SuspensionState
from kinematics.suspensions.core.collections import (
    LowerWishbonePoints,
    TrackRodPoints,
    UpperWishbonePoints,
    WheelAxlePoints,
)
from kinematics.suspensions.core.geometry import SuspensionGeometry
from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.generic import compute_2d_vector_vector_intersection
from kinematics.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
)
from kinematics.visualization.main import LinkVisualization


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
        derived_resolver.update_in_place(positions)

        return SuspensionState(positions=positions, free_points=set(self.free_points()))

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
        v1 = make_vec3(
            initial_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]
            - initial_state.positions[PointID.UPPER_WISHBONE_OUTBOARD]
        )
        v2 = make_vec3(
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

    def compute_side_view_instant_center(self, state: SuspensionState) -> Vec3:
        """
        Compute the side view instant center (SVIC) for a double wishbone suspension.

        This method follows the standard kinematic approach by projecting the suspension
        geometry onto the side-view (X-Z) plane and finding the intersection of the
        lines representing the effective swing arms.

        The process is as follows:
        1. For each wishbone, the 3D inboard pivot axis is defined by the vector
           connecting the two inboard mounting points.
        2. This axis vector is projected onto the 2D side-view plane to get a direction.
        3. A 2D line is constructed that passes through the projected outboard point
           and is parallel to the projected inboard axis direction.
        4. The 2D intersection of these two lines (one for the upper wishbone, one for
           the lower) is the side view instant center.

        Args:
            state: The current suspension state containing point positions.

        Returns:
            The (x, y, z) coordinates of the SVIC, with the y-coordinate set to 0.
            Returns [inf, 0.0, inf] if the projected lines are parallel.
        """
        # Get 3D positions of all relevant points from the current state.
        upper_front_3d = state.positions[PointID.UPPER_WISHBONE_INBOARD_FRONT]
        upper_rear_3d = state.positions[PointID.UPPER_WISHBONE_INBOARD_REAR]
        upper_outboard_3d = state.positions[PointID.UPPER_WISHBONE_OUTBOARD]

        lower_front_3d = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        lower_rear_3d = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        lower_outboard_3d = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

        # Upper wishbone side view projection.
        upper_axis_3d = upper_rear_3d - upper_front_3d

        # Project the axis and the outboard point to the 2D side view plane (X-Z).
        upper_axis_2d = np.array([upper_axis_3d[Axis.X], upper_axis_3d[Axis.Z]])
        upper_outboard_2d = np.array(
            [upper_outboard_3d[Axis.X], upper_outboard_3d[Axis.Z]]
        )

        # Lower wishbone side view projection.
        lower_axis_3d = lower_rear_3d - lower_front_3d

        # Project the axis and the outboard point to the 2D side-view plane (X-Z).
        lower_axis_2d = np.array([lower_axis_3d[Axis.X], lower_axis_3d[Axis.Z]])
        lower_outboard_2d = np.array(
            [lower_outboard_3d[Axis.X], lower_outboard_3d[Axis.Z]]
        )

        # Find the intersection of the two projected lines. Each line is defined by
        # its start point (the outboard point) and end point (outboard + inboard axis
        # vector). Basically, we use start point and direction, and we're working
        # with everything in the projected space.
        intersection = compute_2d_vector_vector_intersection(
            line1_start=lower_outboard_2d,
            line1_end=(lower_outboard_2d + lower_axis_2d).astype(np.float64),
            line2_start=upper_outboard_2d,
            line2_end=(upper_outboard_2d + upper_axis_2d).astype(np.float64),
            segments_only=False,
        )

        if intersection is None:
            # The projected lines are parallel, so the SVIC is at infinity.
            return make_vec3([np.inf, 0.0, np.inf])

        # Convert the 2D intersection point back to 3D space, treat y=0.
        svic_2d = intersection.point
        return make_vec3([svic_2d[0], 0.0, svic_2d[1]])

    def get_visualization_links(self) -> list[LinkVisualization]:
        """
        Get the visualization links for rendering the double wishbone suspension.

        Returns:
            List of link definitions specifying how to visualize the suspension
            structure, including wishbones, upright, track rod, and axle.
        """
        return [
            LinkVisualization(
                points=[
                    PointID.UPPER_WISHBONE_INBOARD_FRONT,
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.UPPER_WISHBONE_INBOARD_REAR,
                ],
                color="dodgerblue",
                label="Upper Wishbone",
            ),
            LinkVisualization(
                points=[
                    PointID.LOWER_WISHBONE_INBOARD_FRONT,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_INBOARD_REAR,
                ],
                color="dodgerblue",
                label="Lower Wishbone",
            ),
            LinkVisualization(
                points=[
                    PointID.TRACKROD_OUTBOARD,
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.TRACKROD_OUTBOARD,
                ],
                color="slategrey",
                label="Upright",
            ),
            LinkVisualization(
                points=[PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD],
                color="darkorange",
                label="Track Rod",
            ),
            LinkVisualization(
                points=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD],
                color="forestgreen",
                label="Axle",
            ),
        ]
