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
from kinematics.suspensions.core.collections import ComponentsConfig
from kinematics.suspensions.core.geometry import SuspensionGeometry
from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.suspensions.core.shims import (
    compute_shim_offset,
    compute_upright_rotation_from_shim,
    rotate_point_about_axis,
)
from kinematics.types import Vec3, WorldAxisSystem, make_vec3
from kinematics.upright import Upright
from kinematics.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
)
from kinematics.visualization.main import LinkVisualization


@dataclass
class SuspensionHardpoints:
    """
    Hardpoint coordinates for double wishbone suspension.

    Keys match PointID enum names directly (snake_case).
    Hardpoints are the kinematic pivot points that define the suspension geometry.
    All coordinates are {x, y, z} dicts.

    Attributes:
        lower_wishbone_inboard_front: Front inboard lower wishbone pivot {x, y, z}
        lower_wishbone_inboard_rear: Rear inboard lower wishbone pivot {x, y, z}
        lower_wishbone_outboard: Lower ball joint {x, y, z}
        upper_wishbone_inboard_front: Front inboard upper wishbone pivot {x, y, z}
        upper_wishbone_inboard_rear: Rear inboard upper wishbone pivot {x, y, z}
        upper_wishbone_outboard: Upper ball joint {x, y, z}
        trackrod_inboard: Inner track rod pivot (rack end) {x, y, z}
        trackrod_outboard: Outer track rod pivot (steering arm) {x, y, z}
    """

    lower_wishbone_inboard_front: dict[str, float]
    lower_wishbone_inboard_rear: dict[str, float]
    lower_wishbone_outboard: dict[str, float]
    upper_wishbone_inboard_front: dict[str, float]
    upper_wishbone_inboard_rear: dict[str, float]
    upper_wishbone_outboard: dict[str, float]
    trackrod_inboard: dict[str, float]
    trackrod_outboard: dict[str, float]


# Geometry model.
@dataclass
class DoubleWishboneGeometry(SuspensionGeometry):
    """
    Double wishbone suspension geometry definition.

    The geometry is defined by:
    - hardpoints: Kinematic pivot points with flat PointID-matching keys
    - components: Rigid body configurations (upright with mounts and attachments)

    Attributes:
        hardpoints: Suspension hardpoint coordinates with PointID-matching keys
        components: Rigid body component configurations
    """

    hardpoints: SuspensionHardpoints
    components: ComponentsConfig

    def validate(self) -> bool:
        """
        Validate the double wishbone geometry configuration.

        Returns:
            True if geometry is valid.
        """
        # ! TODO: Actually validate this geometry.
        return True

    @staticmethod
    def _to_array(coord: dict[str, float]) -> np.ndarray:
        """
        Convert {x, y, z} dict to numpy array.
        """
        return np.array([coord["x"], coord["y"], coord["z"]])

    def get_hardpoints_dict(self) -> dict[PointID, np.ndarray]:
        """
        Build a hardpoints dictionary mapping PointID to coordinates.

        Returns:
            Dictionary mapping PointID to numpy array coordinates.
        """
        hp = self.hardpoints
        to_arr = self._to_array
        return {
            PointID.LOWER_WISHBONE_INBOARD_FRONT: to_arr(
                hp.lower_wishbone_inboard_front
            ),
            PointID.LOWER_WISHBONE_INBOARD_REAR: to_arr(hp.lower_wishbone_inboard_rear),
            PointID.LOWER_WISHBONE_OUTBOARD: to_arr(hp.lower_wishbone_outboard),
            PointID.UPPER_WISHBONE_INBOARD_FRONT: to_arr(
                hp.upper_wishbone_inboard_front
            ),
            PointID.UPPER_WISHBONE_INBOARD_REAR: to_arr(hp.upper_wishbone_inboard_rear),
            PointID.UPPER_WISHBONE_OUTBOARD: to_arr(hp.upper_wishbone_outboard),
            PointID.TRACKROD_INBOARD: to_arr(hp.trackrod_inboard),
            PointID.TRACKROD_OUTBOARD: to_arr(hp.trackrod_outboard),
        }

    def get_axle_attachments(self) -> dict[str, np.ndarray]:
        """
        Get axle attachment coordinates from the upright component.

        Returns:
            Dictionary with 'axle_inboard' and 'axle_outboard' coordinates.
        """
        attachments = self.components.upright.attachments
        return {
            "axle_inboard": self._to_array(attachments.axle_inboard),
            "axle_outboard": self._to_array(attachments.axle_outboard),
        }

    def get_upright_mount_ids(self) -> dict[str, PointID]:
        """
        Get the PointID references for upright mounts.

        Returns:
            Dictionary mapping mount roles to PointIDs.
        """
        mounts = self.components.upright.mounts
        return {
            "upper_ball_joint": PointID[mounts.upper_ball_joint.upper()],
            "lower_ball_joint": PointID[mounts.lower_ball_joint.upper()],
            "steering_pickup": PointID[mounts.steering_pickup.upper()],
        }


# Provider implementation.
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

    def build_upright_from_geometry(self) -> Upright:
        """
        Build an Upright rigid body from the geometry definition.

        Uses the geometry's helper methods to extract hardpoint and attachment data.

        Returns:
            Upright rigid body constructed from geometry, with LCS and local
            offsets computed.
        """
        # Get hardpoints and attachments using geometry helper methods.
        hardpoints = self.geometry.get_hardpoints_dict()
        attachments = self.geometry.get_axle_attachments()
        mount_ids = self.geometry.get_upright_mount_ids()

        # Convert attachments to Vec3.
        attachments_vec3 = {
            "axle_inboard": make_vec3(attachments["axle_inboard"]),
            "axle_outboard": make_vec3(attachments["axle_outboard"]),
        }

        # Convert hardpoints to Vec3.
        hardpoints_vec3 = {pid: make_vec3(pos) for pid, pos in hardpoints.items()}

        # Create upright from hardpoints and attachments.
        return Upright.from_hardpoints_and_attachments(
            mount_ids, hardpoints_vec3, attachments_vec3
        )

    def initial_state(self) -> SuspensionState:
        """
        Create initial suspension state from geometry.

        Converts the geometry into a SuspensionState with both hardpoints
        and derived points. If a camber shim is configured, applies the geometric
        transformation to rotate the upright attachments.

        Returns:
            Initial suspension state with all point positions.
        """
        # Build hardpoints from geometry.
        positions = self.geometry.get_hardpoints_dict()

        # Add axle attachments.
        axle_attachments = self.geometry.get_axle_attachments()
        positions[PointID.AXLE_INBOARD] = axle_attachments["axle_inboard"]
        positions[PointID.AXLE_OUTBOARD] = axle_attachments["axle_outboard"]

        # Apply camber shim transformation if configured.
        if self.geometry.configuration.camber_shim is not None:
            self.apply_camber_shim(positions)

        # Calculate derived points to create a complete initial state.
        derived_spec = self.derived_spec()
        derived_resolver = DerivedPointsManager(derived_spec)
        derived_resolver.update_in_place(positions)

        return SuspensionState(positions=positions, free_points=set(self.free_points()))

    def apply_camber_shim(self, positions: dict[PointID, np.ndarray]) -> None:
        """
        Apply camber shim transformation using the Upright rigid body.

        CRITICAL: The shim rotates ONLY the attachments (axle, brakes), NOT the
        hardpoints (ball joints). The shim sits between the structural upright
        (hardpoints) and the hub/bearing assembly (attachments).

        This method:
        1. Builds an Upright rigid body from current positions
        2. Computes shim rotation axis and angle
        3. Applies rotation to attachments via upright.apply_camber_shim()
        4. Updates positions dict with new attachment positions

        Args:
            positions: Dictionary of point positions to modify in-place
        """
        shim_config = self.geometry.configuration.camber_shim
        if shim_config is None:
            return

        # Build the upright rigid body from current positions.
        mount_ids = self.geometry.get_upright_mount_ids()
        hardpoints = {pid: make_vec3(positions[pid]) for pid in mount_ids.values()}
        attachments = {
            "axle_inboard": make_vec3(positions[PointID.AXLE_INBOARD]),
            "axle_outboard": make_vec3(positions[PointID.AXLE_OUTBOARD]),
        }
        upright = Upright.from_hardpoints_and_attachments(
            mount_ids, hardpoints, attachments
        )

        # Compute the shim offset vector.
        shim_offset = compute_shim_offset(shim_config)

        # Get the shim face center at design condition.
        sfc = shim_config.shim_face_center
        shim_face_center_design = make_vec3([sfc["x"], sfc["y"], sfc["z"]])

        # The lower ball joint is the pivot point (rotation center).
        # CRITICAL: Ball joints do NOT move - they're on the fixed part of the upright.
        lower_ball_joint = make_vec3(positions[PointID.LOWER_WISHBONE_OUTBOARD])

        # Compute the rotation axis and angle.
        rotation_axis, rotation_angle = compute_upright_rotation_from_shim(
            lower_ball_joint,
            shim_face_center_design,
            shim_offset,
        )

        # Apply shim to upright (rotates attachments only, not hardpoints)
        upright.apply_camber_shim(lower_ball_joint, rotation_axis, rotation_angle)

        # Update positions with new attachment positions.
        # The axle points have been rotated by the shim.
        positions[PointID.AXLE_INBOARD] = np.array(
            upright.get_world_position("axle_inboard")
        )
        positions[PointID.AXLE_OUTBOARD] = np.array(
            upright.get_world_position("axle_outboard")
        )

        # If track rod is mounted to the upright (common configuration),
        # it also needs to be rotated. Check the upright_mounted_points config.
        if self.geometry.configuration.upright_mounted_points:
            if (
                "trackrod_outboard"
                in self.geometry.configuration.upright_mounted_points
            ):
                # Track rod outer is mounted to upright - we need to rotate it.
                # This requires adding it as an attachment to the upright.
                # For now, use the old method for track rod to maintain compatibility.
                trackrod_pos = make_vec3(positions[PointID.TRACKROD_OUTBOARD])
                trackrod_rotated = rotate_point_about_axis(
                    trackrod_pos, lower_ball_joint, rotation_axis, rotation_angle
                )
                positions[PointID.TRACKROD_OUTBOARD] = np.array(trackrod_rotated)

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
                line_direction=WorldAxisSystem.Y,
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
                linewidth=0.0,  # No line for single point.
                marker="o",
                markersize=15.0,  # Large marker for visibility.
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
            # plane. This is a modelling error, not a kinematic state.
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
