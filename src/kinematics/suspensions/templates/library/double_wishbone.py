"""Double wishbone suspension template."""

from __future__ import annotations

from kinematics.enums import PointID, ShimType
from kinematics.suspensions.templates.base import ComponentSpec, SuspensionTemplate

DOUBLE_WISHBONE_TEMPLATE = SuspensionTemplate(
    key="double_wishbone",
    required_point_ids=frozenset(
        {
            # Lower wishbone.
            PointID.LOWER_WISHBONE_INBOARD_FRONT,
            PointID.LOWER_WISHBONE_INBOARD_REAR,
            PointID.LOWER_WISHBONE_OUTBOARD,
            # Upper wishbone.
            PointID.UPPER_WISHBONE_INBOARD_FRONT,
            PointID.UPPER_WISHBONE_INBOARD_REAR,
            PointID.UPPER_WISHBONE_OUTBOARD,
            # Steering.
            PointID.TRACKROD_INBOARD,
            PointID.TRACKROD_OUTBOARD,
            # Axle/spindle.
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        }
    ),
    optional_point_ids=frozenset(
        {
            # Pushrod (for pushrod suspension variants).
            PointID.PUSHROD_INBOARD,
            PointID.PUSHROD_OUTBOARD,
        }
    ),
    components=(
        ComponentSpec(
            name="upright",
            mount_roles={
                "upper_ball_joint": PointID.UPPER_WISHBONE_OUTBOARD,
                "lower_ball_joint": PointID.LOWER_WISHBONE_OUTBOARD,
                "steering_pickup": PointID.TRACKROD_OUTBOARD,
            },
            # Axle points are rigidly attached to upright.
            attachment_point_ids=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD],
        ),
        ComponentSpec(
            name="upper_wishbone",
            mount_roles={
                "inboard_front": PointID.UPPER_WISHBONE_INBOARD_FRONT,
                "inboard_rear": PointID.UPPER_WISHBONE_INBOARD_REAR,
                "outboard": PointID.UPPER_WISHBONE_OUTBOARD,
            },
            attachment_point_ids=[],
        ),
        ComponentSpec(
            name="lower_wishbone",
            mount_roles={
                "inboard_front": PointID.LOWER_WISHBONE_INBOARD_FRONT,
                "inboard_rear": PointID.LOWER_WISHBONE_INBOARD_REAR,
                "outboard": PointID.LOWER_WISHBONE_OUTBOARD,
            },
            attachment_point_ids=[],
        ),
        ComponentSpec(
            name="trackrod",
            mount_roles={
                "inboard": PointID.TRACKROD_INBOARD,
                "outboard": PointID.TRACKROD_OUTBOARD,
            },
            attachment_point_ids=[],
        ),
    ),
    ownership={
        # Ball joints are owned by the upright (despite naming *_WISHBONE_OUTBOARD).
        # This reflects reality: the ball joint housing is part of the upright.
        PointID.UPPER_WISHBONE_OUTBOARD: "upright",
        PointID.LOWER_WISHBONE_OUTBOARD: "upright",
        PointID.TRACKROD_OUTBOARD: "upright",
        PointID.AXLE_INBOARD: "upright",
        PointID.AXLE_OUTBOARD: "upright",
        # Wishbone chassis mounts are owned by chassis (fixed).
        PointID.UPPER_WISHBONE_INBOARD_FRONT: "chassis",
        PointID.UPPER_WISHBONE_INBOARD_REAR: "chassis",
        PointID.LOWER_WISHBONE_INBOARD_FRONT: "chassis",
        PointID.LOWER_WISHBONE_INBOARD_REAR: "chassis",
        # Trackrod inboard (rack end) is chassis-mounted.
        PointID.TRACKROD_INBOARD: "chassis",
    },
    supported_shims=frozenset({ShimType.OUTBOARD_CAMBER}),
    aliases=frozenset({"double_wishbone_front", "double_wishbone_rear"}),
)
