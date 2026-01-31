"""
Double wishbone suspension implementation.

This module defines the DoubleWishboneSuspension class which combines topology
definition, geometry storage, and kinematic behavior in a single unified class.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Sequence

import numpy as np

from kinematics.components.upright import Upright
from kinematics.constraints import (
    AngleConstraint,
    Constraint,
    DistanceConstraint,
    PointOnLineConstraint,
)
from kinematics.core.enums import PointID, ShimType
from kinematics.core.types import Vec3, WorldAxisSystem, make_vec3
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
    intersect_line_with_vertical_plane,
    intersect_two_planes,
    plane_from_three_points,
)
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
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.config.shims import (
    compute_shim_offset,
    compute_upright_rotation_from_shim,
    rotate_point_about_axis,
)
from kinematics.visualization.main import LinkVisualization


@dataclass
class DoubleWishboneSuspension(Suspension):
    """
    Double wishbone suspension with all topology and behavior in one class.

    This class:
    - Defines valid points as class attributes (replacing SuspensionTemplate)
    - Stores hardpoints and config as instance data (replacing TemplateGeometry)
    - Implements constraints, visualization, and solver interface (replacing provider)
    """

    TYPE_KEY: ClassVar[str] = "double_wishbone"
    ALIASES: ClassVar[frozenset[str]] = frozenset(
        {"double_wishbone_front", "double_wishbone_rear"}
    )

    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.UPPER_WISHBONE_INBOARD_FRONT,
            PointID.UPPER_WISHBONE_INBOARD_REAR,
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.TRACKROD_INBOARD,
            PointID.TRACKROD_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        }
    )

    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = frozenset(
        {
            PointID.PUSHROD_INBOARD,
            PointID.PUSHROD_OUTBOARD,
        }
    )

    SUPPORTED_SHIMS: ClassVar[frozenset[ShimType]] = frozenset(
        {ShimType.OUTBOARD_CAMBER}
    )

    # Upright mount roles (for shim application).
    UPRIGHT_MOUNT_ROLES: ClassVar[dict[str, PointID]] = {
        "upper_ball_joint": PointID.UPPER_WISHBONE_OUTBOARD,
        "lower_ball_joint": PointID.LOWER_WISHBONE_OUTBOARD,
        "trackrod_outboard": PointID.TRACKROD_OUTBOARD,
    }

    # Free points that move during solving.
    FREE_POINTS: ClassVar[tuple[PointID, ...]] = (
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.TRACKROD_OUTBOARD,
        PointID.TRACKROD_INBOARD,
    )

    def free_points(self) -> Sequence[PointID]:
        """Points that move during solving."""
        return self.FREE_POINTS

    def initial_state(self) -> SuspensionState:
        """Build initial state from hardpoints, applying shims if configured."""
        if self._initial_state is not None:
            return self._initial_state

        positions = self.get_hardpoints_as_arrays()

        # Apply camber shim if configured.
        if self.config is not None and self.config.camber_shim is not None:
            self.apply_camber_shim(positions)

        # Compute derived points.
        derived_spec = self.derived_spec()
        derived_manager = DerivedPointsManager(derived_spec)
        derived_manager.update_in_place(positions)

        self._initial_state = SuspensionState(
            positions=positions,
            free_points=set(self.free_points()),
        )
        return self._initial_state

    def constraints(self) -> list[Constraint]:
        """Build geometric constraints for double wishbone."""
        initial_state = self.initial_state()
        constraints: list[Constraint] = []

        # Distance constraints (link lengths).
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

        # Angle constraint for upright rigidity.
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

    def derived_spec(self) -> DerivedPointsSpec:
        """Specification for derived points (wheel center, contact patch, etc.)."""
        if self.config is None:
            raise ValueError("Cannot compute derived spec without config")

        wheel_cfg = self.config.wheel
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

    def compute_side_view_instant_center(self, state: SuspensionState) -> Vec3 | None:
        """Compute side view instant center from wishbone planes."""
        try:
            instant_axis = self.compute_instant_axis(state)
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

    def compute_instant_axis(self, state: SuspensionState) -> tuple[Vec3, Vec3] | None:
        """Compute 3D instant axis from wishbone planes intersection."""
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
        """Visualization links for 3D rendering."""
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

    def apply_camber_shim(self, positions: dict[PointID, np.ndarray]) -> None:
        """
        Apply camber shim transformation to attachment positions.

        The shim rotates only attachments (axle points), not the hardpoints
        (ball joints). The shim sits between the structural upright and the hub/bearing
        assembly.
        """
        if self.config is None or self.config.camber_shim is None:
            return

        shim_config = self.config.camber_shim

        # Build upright from current positions.
        mount_ids = self.UPRIGHT_MOUNT_ROLES
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
        upright_mounted = self.config.upright_mounted_points
        if upright_mounted and "trackrod_outboard" in upright_mounted:
            trackrod_pos = make_vec3(positions[PointID.TRACKROD_OUTBOARD])
            trackrod_rotated = rotate_point_about_axis(
                trackrod_pos, lower_ball_joint, rotation_axis, rotation_angle
            )
            positions[PointID.TRACKROD_OUTBOARD] = np.array(trackrod_rotated)
