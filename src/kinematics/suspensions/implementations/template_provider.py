"""
Template-based suspension provider.

This provider works with any suspension template, using the template's specification to
determine component structure, ownership, and constraints.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Sequence

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
    get_contact_patch_center,
    get_wheel_center,
    get_wheel_center_on_ground,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.state import SuspensionState
from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.suspensions.core.shims import (
    compute_shim_offset,
    compute_upright_rotation_from_shim,
    rotate_point_about_axis,
)
from kinematics.suspensions.templates.base import SuspensionTemplate
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

if TYPE_CHECKING:
    from kinematics.suspensions.core.template_geometry import TemplateGeometry


class TemplateSuspensionProvider(SuspensionProvider):
    """
    Generic suspension provider that works with any template.

    This provider uses the template's specification to determine:
    - Which points are free to move
    - How to construct rigid bodies
    - What constraints to apply

    Currently implements double wishbone topology. Additional topologies
    can be added by extending the constraint/visualization logic based
    on the template key.
    """

    def __init__(
        self,
        geometry: "TemplateGeometry",
        template: SuspensionTemplate | None = None,
    ) -> None:
        """
        Initialize the template provider.

        Args:
            geometry: The validated template geometry with hardpoints.
            template: The suspension template defining topology. If not provided,
                     will be looked up from the geometry's template_key.
        """
        # Look up template if not provided.
        if template is None:
            from kinematics.suspensions.templates.library import get_template

            template = get_template(geometry.template_key)
            if template is None:
                raise ValueError(f"No template found for key '{geometry.template_key}'")

        self.template = template
        self.geometry = geometry
        self._hardpoints = geometry.get_hardpoints_dict(template)
        self._upright: Upright | None = None

    def build_upright(self) -> Upright:
        """
        Build an upright rigid body from the hardpoints.

        Uses the template's upright component spec to determine mount
        mappings and attachment points.

        Returns:
            Constructed Upright with LCS established.
        """
        upright_spec = self.template.get_upright_component()
        if upright_spec is None:
            raise ValueError(
                f"Template '{self.template.key}' does not define an upright component"
            )

        mount_ids = upright_spec.mount_roles

        # Get axle attachments from hardpoints.
        attachments = {
            "axle_inboard": make_vec3(self._hardpoints[PointID.AXLE_INBOARD]),
            "axle_outboard": make_vec3(self._hardpoints[PointID.AXLE_OUTBOARD]),
        }

        # Convert hardpoints to Vec3 for construction.
        hardpoints_vec3 = {pid: make_vec3(pos) for pid, pos in self._hardpoints.items()}

        return Upright.from_hardpoints_and_attachments(
            mount_ids=mount_ids,
            hardpoints=hardpoints_vec3,
            attachments=attachments,
        )

    def initial_state(self) -> SuspensionState:
        """
        Create the initial suspension state from geometry.

        Builds hardpoint positions, applies camber shim if configured,
        and computes derived points.

        Returns:
            Initial SuspensionState with all point positions.
        """
        # Start with hardpoint positions.
        positions: dict[PointID, np.ndarray] = {
            pid: np.array(pos) for pid, pos in self._hardpoints.items()
        }

        # Apply camber shim transformation if configured.
        if self.geometry.configuration.camber_shim is not None:
            self._apply_camber_shim(positions)

        # Calculate derived points.
        derived_spec = self.derived_spec()
        derived_manager = DerivedPointsManager(derived_spec)
        derived_manager.update_in_place(positions)

        return SuspensionState(
            positions=positions,
            free_points=set(self.free_points()),
        )

    def _apply_camber_shim(self, positions: dict[PointID, np.ndarray]) -> None:
        """
        Apply camber shim transformation to attachment positions.

        CRITICAL: The shim rotates ONLY attachments (axle points), NOT
        the hardpoints (ball joints). The shim sits between the structural
        upright and the hub/bearing assembly.

        Args:
            positions: Dictionary of point positions to modify in-place
        """
        shim_config = self.geometry.configuration.camber_shim
        if shim_config is None:
            return

        # Build upright from current positions.
        upright_spec = self.template.get_upright_component()
        if upright_spec is None:
            return

        mount_ids = upright_spec.mount_roles
        hardpoints = {pid: make_vec3(positions[pid]) for pid in mount_ids.values()}
        attachments = {
            "axle_inboard": make_vec3(positions[PointID.AXLE_INBOARD]),
            "axle_outboard": make_vec3(positions[PointID.AXLE_OUTBOARD]),
        }
        upright = Upright.from_hardpoints_and_attachments(
            mount_ids, hardpoints, attachments
        )

        # Compute shim offset.
        shim_offset = compute_shim_offset(shim_config)

        # Get shim face center at design.
        sfc = shim_config.shim_face_center
        shim_face_center_design = make_vec3([sfc["x"], sfc["y"], sfc["z"]])

        # Lower ball joint is the pivot.
        lower_ball_joint = make_vec3(positions[PointID.LOWER_WISHBONE_OUTBOARD])

        # Compute rotation.
        rotation_axis, rotation_angle = compute_upright_rotation_from_shim(
            lower_ball_joint,
            shim_face_center_design,
            shim_offset,
        )

        # Apply shim to upright.
        upright.apply_camber_shim(lower_ball_joint, rotation_axis, rotation_angle)

        # Update positions with rotated attachments.
        positions[PointID.AXLE_INBOARD] = np.array(
            upright.get_world_position("axle_inboard")
        )
        positions[PointID.AXLE_OUTBOARD] = np.array(
            upright.get_world_position("axle_outboard")
        )

        # Rotate trackrod if it's upright-mounted.
        upright_mounted = self.geometry.configuration.upright_mounted_points
        if upright_mounted and "trackrod_outboard" in upright_mounted:
            trackrod_pos = make_vec3(positions[PointID.TRACKROD_OUTBOARD])
            trackrod_rotated = rotate_point_about_axis(
                trackrod_pos, lower_ball_joint, rotation_axis, rotation_angle
            )
            positions[PointID.TRACKROD_OUTBOARD] = np.array(trackrod_rotated)

    def free_points(self) -> Sequence[PointID]:
        """
        Get the points that can move during solving.

        For double wishbone, these are the outboard points and axle.

        Returns:
            Sequence of free point IDs.
        """
        # For double wishbone topology.
        if self.template.key in (
            "double_wishbone",
            "double_wishbone_front",
            "double_wishbone_rear",
        ):
            return [
                PointID.UPPER_WISHBONE_OUTBOARD,
                PointID.LOWER_WISHBONE_OUTBOARD,
                PointID.AXLE_INBOARD,
                PointID.AXLE_OUTBOARD,
                PointID.TRACKROD_OUTBOARD,
                PointID.TRACKROD_INBOARD,
            ]

        # Default: outboard and attachment points.
        free = []
        for component in self.template.components:
            for point_id in component.attachment_point_ids:
                if point_id not in free:
                    free.append(point_id)
            for role, point_id in component.mount_roles.items():
                if "outboard" in role.lower() and point_id not in free:
                    free.append(point_id)
        return free

    def derived_spec(self) -> DerivedPointsSpec:
        """
        Get the specification for computing derived points.

        Returns:
            Specification for wheel center, contact patch, etc.
        """
        wheel_cfg = self.geometry.configuration.wheel
        tire_radius = wheel_cfg.tire.nominal_radius

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
            PointID.WHEEL_CENTER_ON_GROUND: get_wheel_center_on_ground,
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
        Build geometric constraints for the suspension.

        Returns:
            List of constraints that must be satisfied during solving.
        """
        constraints: list[Constraint] = []
        initial_state = self.initial_state()

        # Distance constraints for double wishbone.
        if self.template.key in (
            "double_wishbone",
            "double_wishbone_front",
            "double_wishbone_rear",
        ):
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

            # Angle constraint for upright.
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

            # Point-on-line constraint for rack travel.
            constraints.append(
                PointOnLineConstraint(
                    point_id=PointID.TRACKROD_INBOARD,
                    line_point=initial_state.positions[PointID.TRACKROD_INBOARD],
                    line_direction=WorldAxisSystem.Y,
                )
            )

        return constraints

    def compute_side_view_instant_center(self, state: SuspensionState) -> Vec3 | None:
        """
        Compute the side view instant center.

        Args:
            state: Current suspension state

        Returns:
            SVIC coordinates, or None if not applicable.
        """
        if self.template.key not in (
            "double_wishbone",
            "double_wishbone_front",
            "double_wishbone_rear",
        ):
            return None

        try:
            instant_axis = self._compute_instant_axis(state)
        except ValueError:
            raise

        wheel_center_y = float(state.positions[PointID.WHEEL_CENTER][1])
        wheel_center_z = float(state.positions[PointID.WHEEL_CENTER][2])

        if instant_axis is None:
            return make_vec3([np.inf, wheel_center_y, wheel_center_z])

        axis_point, axis_direction = instant_axis
        svic = intersect_line_with_vertical_plane(
            axis_point, axis_direction, wheel_center_y
        )

        if svic is None:
            return make_vec3([np.inf, wheel_center_y, wheel_center_z])

        return svic

    def _compute_instant_axis(self, state: SuspensionState) -> tuple[Vec3, Vec3] | None:
        """
        Compute the 3D instant axis from wishbone planes.

        Args:
            state: Current suspension state

        Returns:
            Tuple of (point_on_axis, axis_direction), or None if parallel.
        """
        upper_front = state.positions[PointID.UPPER_WISHBONE_INBOARD_FRONT]
        upper_rear = state.positions[PointID.UPPER_WISHBONE_INBOARD_REAR]
        upper_outboard = state.positions[PointID.UPPER_WISHBONE_OUTBOARD]

        lower_front = state.positions[PointID.LOWER_WISHBONE_INBOARD_FRONT]
        lower_rear = state.positions[PointID.LOWER_WISHBONE_INBOARD_REAR]
        lower_outboard = state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

        upper_plane = plane_from_three_points(upper_front, upper_rear, upper_outboard)
        lower_plane = plane_from_three_points(lower_front, lower_rear, lower_outboard)

        if upper_plane is None or lower_plane is None:
            raise ValueError(
                "Degenerate wishbone geometry. Cannot compute instant axis."
            )

        return intersect_two_planes(
            n1=upper_plane[0],
            d1=upper_plane[1],
            n2=lower_plane[0],
            d2=lower_plane[1],
        )

    def get_visualization_links(self) -> list[LinkVisualization]:
        """
        Get visualization links for rendering the suspension.

        Returns:
            List of link definitions for visualisation.
        """
        if self.template.key in (
            "double_wishbone",
            "double_wishbone_front",
            "double_wishbone_rear",
        ):
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
                    linewidth=0.0,
                    marker="o",
                    markersize=15.0,
                ),
            ]

        # Generic fallback.
        return []
