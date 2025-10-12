"""
Double wishbone suspension implementation.
"""

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import numpy as np

from kinematics.constants import EPSILON
from kinematics.constraints import (
    AngleConstraint,
    Constraint,
    DistanceConstraint,
    PointOnLineConstraint,
)
from kinematics.enums import Axis, PointID
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_contact_patch_center,
    get_wheel_center,
    get_wheel_center_on_ground,
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
from kinematics.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
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
        # Use the nominal tire radius from the configuration.
        tire_radius = self.geometry.configuration.wheel.tire.nominal_radius

        functions = {
            PointID.AXLE_MIDPOINT: get_axle_midpoint,
            PointID.WHEEL_CENTER: partial(
                get_wheel_center, wheel_offset=wheel_cfg.offset
            ),
            PointID.WHEEL_INBOARD: partial(
                get_wheel_inboard, wheel_width=wheel_cfg.tire.section_width
            ),
            PointID.WHEEL_OUTBOARD: partial(
                get_wheel_outboard, wheel_width=wheel_cfg.tire.section_width
            ),
            PointID.WHEEL_CENTER_ON_GROUND: partial(get_wheel_center_on_ground),
            PointID.CONTACT_PATCH_CENTER: partial(
                get_contact_patch_center, tire_radius=tire_radius
            ),
        }

        dependencies = {
            PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            PointID.WHEEL_CENTER_ON_GROUND: {
                PointID.WHEEL_CENTER,
                PointID.AXLE_INBOARD,
                PointID.AXLE_OUTBOARD,
            },
            PointID.CONTACT_PATCH_CENTER: {
                PointID.WHEEL_CENTER,
                PointID.AXLE_INBOARD,
                PointID.AXLE_OUTBOARD,
            },
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
            LinkVisualization(
                points=[PointID.WHEEL_CENTER_ON_GROUND],
                color="black",
                label="Wheel Center on Ground",
                linewidth=0.0,  # No line for single point
                marker="o",
                markersize=15.0,  # Large marker for visibility
            ),
        ]

    def compute_instant_axis(self, state: SuspensionState) -> tuple[Vec3, Vec3] | None:
        """
        Compute the 3D instantaneous axis of rotation for the upright.

        This axis is the line formed by the intersection of the two 3D planes
        defined by the upper and lower wishbones.

        Args:
            state: The current suspension state.

        Returns:
            A tuple containing a point on the axis and the axis direction vector.
            Returns None only if the wishbone planes are parallel (a valid state
            resulting in an instant center at infinity).

        Raises:
            ValueError: If either wishbone's points are collinear, as this
                        represents a degenerate geometry and not a valid plane.
        """
        # Gather the points that define the upper and lower wishbone planes.
        upper_front = state.positions[PointID.UPPER_WISHBONE_INBOARD_FRONT]
        upper_rear = state.positions[PointID.UPPER_WISHBONE_INBOARD_REAR]
        upper_outboard = state.positions[PointID.UPPER_WISHBONE_OUTBOARD]

        lower_front = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        lower_rear = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        lower_outboard = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

        # Construct the plane for each wishbone.
        upper_plane = plane_from_three_points(upper_front, upper_rear, upper_outboard)
        lower_plane = plane_from_three_points(lower_front, lower_rear, lower_outboard)

        if upper_plane is None or lower_plane is None:
            # A degenerate wishbone's points are collinear and cannot form a
            # plane. This is a modeling error, not a kinematic state.
            raise ValueError(
                "Degenerate wishbone geometry. Cannot compute instant axis."
            )

        # Find the 3D line of intersection between the two planes.
        # This will return None if the planes are parallel.
        return intersect_two_planes(
            n1=upper_plane[0],
            d1=upper_plane[1],
            n2=lower_plane[0],
            d2=lower_plane[1],
        )

    def compute_side_view_instant_center(self, state: SuspensionState) -> Vec3:
        """
        Compute the side view instant center (SVIC) in the plane of the wheel.

        This method follows the Milliken & Milliken approach by finding the
        intersection of the 3D instant axis with the vertical side-view plane
        that passes through the wheel center's Y-coordinate.

        Args:
            state: The current suspension state containing point positions.

        Returns:
            The (x, y, z) coordinates of the SVIC. Returns a vector with np.inf
            components if the instant axis is parallel to the viewing plane.
        """
        try:
            instant_axis = self.compute_instant_axis(state)
        except ValueError:
            # Re-raise modeling errors immediately.
            raise

        wheel_center_y = float(state.positions[PointID.WHEEL_CENTER][Axis.Y])
        wheel_center_z = float(state.positions[PointID.WHEEL_CENTER][Axis.Z])

        if instant_axis is None:
            # Wishbone planes are parallel, so the instant axis is undefined and
            # the instant center is considered to be at infinity.
            return make_vec3([np.inf, wheel_center_y, wheel_center_z])

        axis_point, axis_direction = instant_axis

        # Intersect the instant axis with the defined side-view plane.
        svic = intersect_line_with_vertical_plane(
            axis_point, axis_direction, wheel_center_y
        )

        if svic is None:
            # The instant axis is parallel to the side-view plane, meaning the
            # SVIC is at infinity in the X-Z plane.
            return make_vec3([np.inf, wheel_center_y, wheel_center_z])

        return svic

    def compute_front_view_instant_center(self, state: SuspensionState) -> Vec3:
        """
        Compute the front view instant center (FVIC) in the plane of the wheel.

        This method follows the Milliken & Milliken approach by finding the
        intersection of the 3D instant axis with the vertical front-view
        plane that passes through the wheel center's X-coordinate.

        Args:
            state: The current suspension state containing point positions.

        Returns:
            The (x, y, z) coordinates of the FVIC. Returns a vector with np.inf
            components if the instant axis is parallel to the viewing plane.
        """
        try:
            instant_axis = self.compute_instant_axis(state)
        except ValueError:
            # Re-raise modeling errors immediately.
            raise

        wheel_center_x = float(state.positions[PointID.WHEEL_CENTER][Axis.X])
        wheel_center_z = float(state.positions[PointID.WHEEL_CENTER][Axis.Z])

        if instant_axis is None:
            # Wishbone planes are parallel, so the instant center is at infinity.
            return make_vec3([wheel_center_x, np.inf, wheel_center_z])

        axis_point, axis_direction = instant_axis
        direction_x = float(axis_direction[Axis.X])

        # If the axis has no X-component, it is parallel to the Y-Z (front) plane.
        if abs(direction_x) < EPSILON:
            # The FVIC is at infinity in the Y-Z plane.
            return make_vec3([wheel_center_x, np.inf, wheel_center_z])

        # Solve for the parameter t where the line's X-coordinate equals the plane's.
        t = (wheel_center_x - float(axis_point[Axis.X])) / direction_x
        fvic = make_vec3(axis_point + t * axis_direction)

        return fvic
