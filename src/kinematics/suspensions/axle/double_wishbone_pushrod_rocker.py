"""
Double-wishbone pushrod-rocker axle with an anti-roll bar.

This concrete topology composes two ARB-ready pushrod-rocker corners and the
shared anti-roll bar axis. Unlike the basic axle, every ARB point is required.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Sequence

from kinematics.constraints import Constraint, DistanceConstraint
from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey, PointRef, Side
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
)
from kinematics.suspensions.axle.double_wishbone import DoubleWishboneAxleSuspension
from kinematics.suspensions.corner.double_wishbone_pushrod_rocker import (
    DoubleWishbonePushrodRockerArbSuspension,
)

if TYPE_CHECKING:
    from kinematics.state import SuspensionState
    from kinematics.visualization.main import LinkVisualization


@dataclass
class DoubleWishbonePushrodRockerAxleSuspension(DoubleWishboneAxleSuspension):
    """Two pushrod-rocker corners coupled by steering rack and anti-roll bar."""

    TYPE_KEY: ClassVar[str] = "double_wishbone_pushrod_rocker_axle"

    center_points: dict[PointID, Point3] = field(default_factory=dict)
    droplink_arb_points: dict[Side, Point3] = field(default_factory=dict)

    @property
    def has_arb(self) -> bool:
        """This topology always includes a complete anti-roll bar."""
        return True

    @property
    def has_rocker(self) -> bool:
        """Both corners in this axle topology have rockers."""
        return True

    @property
    def has_torsion_bar(self) -> bool:
        """Whether the axle corners use torsion bars."""
        return all(corner.has_torsion_bar for corner in self.corners.values())

    @property
    def has_strut(self) -> bool:
        """Whether the axle corners use inboard coilovers."""
        return all(corner.has_strut for corner in self.corners.values())

    def validate_hardpoints(self) -> None:
        """Validate both rocker corners and the required axle-level ARB."""
        if set(self.corners) != {Side.LEFT, Side.RIGHT}:
            raise ValueError("Axle requires exactly LEFT and RIGHT corner models.")
        for side, corner in self.corners.items():
            if not isinstance(corner, DoubleWishbonePushrodRockerArbSuspension):
                raise ValueError(
                    f"{side.name} corner must be an ARB-ready pushrod-rocker model."
                )
            corner.validate_hardpoints()

        expected_axis = {PointID.ARB_AXIS_A, PointID.ARB_AXIS_B}
        if set(self.center_points) != expected_axis:
            raise ValueError("ARB axle requires center ARB_AXIS_A and ARB_AXIS_B.")
        if set(self.droplink_arb_points) != {Side.LEFT, Side.RIGHT}:
            raise ValueError("ARB axle requires DROPLINK_ARB on both sides.")

        axis_a = self.center_points[PointID.ARB_AXIS_A]
        axis_b = self.center_points[PointID.ARB_AXIS_B]
        if compute_point_point_distance(axis_a, axis_b) <= EPS_GEOMETRIC:
            raise ValueError("ARB_AXIS_A and ARB_AXIS_B must be distinct points.")
        axis_direction = (axis_b - axis_a).normalize()
        for side, droplink in self.droplink_arb_points.items():
            radius = compute_point_to_line_distance(droplink, axis_a, axis_direction)
            if radius <= EPS_GEOMETRIC:
                raise ValueError(
                    f"{side.name} DROPLINK_ARB lies on the ARB axis (zero radius); "
                    "it must be off-axis to trace an ARB arm arc."
                )

    def initial_state(self) -> "SuspensionState":
        """Add the shared ARB axis and moving arm pickups to the axle state."""
        if self._initial_state is not None:
            return self._initial_state
        state = super().initial_state()
        for point, position in self.center_points.items():
            state.positions[PointRef(Side.CENTER, point)] = position.copy()
        for side, position in self.droplink_arb_points.items():
            key = PointRef(side, PointID.DROPLINK_ARB)
            state.positions[key] = position.copy()
            state.free_points.add(key)
        state.free_points_order = sorted(state.free_points)
        return state

    def free_points(self) -> Sequence[PointKey]:
        """Corner variables plus both moving ARB arm pickups."""
        return (
            *super().free_points(),
            PointRef(Side.LEFT, PointID.DROPLINK_ARB),
            PointRef(Side.RIGHT, PointID.DROPLINK_ARB),
        )

    def output_points(self) -> tuple[PointKey, ...]:
        """Per-side rocker corner outputs plus each ARB arm pickup."""
        result: list[PointKey] = []
        for side in (Side.LEFT, Side.RIGHT):
            result.extend(
                PointRef(side, point) for point in self.corners[side].output_points()
            )
            result.append(PointRef(side, PointID.DROPLINK_ARB))
        return tuple(result)

    def constraints(self) -> list[Constraint]:
        """Corner and rack constraints plus the anti-roll-bar linkage."""
        return [*super().constraints(), *self._arb_constraints()]

    def _arb_constraints(self) -> list[Constraint]:
        """Constrain each ARB arm to its axis and rocker droplink."""
        constraints: list[Constraint] = []
        axis_a = self.center_points[PointID.ARB_AXIS_A]
        axis_b = self.center_points[PointID.ARB_AXIS_B]
        axis_a_key = PointRef(Side.CENTER, PointID.ARB_AXIS_A)
        axis_b_key = PointRef(Side.CENTER, PointID.ARB_AXIS_B)

        for side in (Side.LEFT, Side.RIGHT):
            droplink = self.droplink_arb_points[side]
            arb_key = PointRef(side, PointID.DROPLINK_ARB)
            constraints.extend(
                (
                    DistanceConstraint(
                        arb_key,
                        axis_a_key,
                        compute_point_point_distance(droplink, axis_a),
                    ),
                    DistanceConstraint(
                        arb_key,
                        axis_b_key,
                        compute_point_point_distance(droplink, axis_b),
                    ),
                    DistanceConstraint(
                        PointRef(side, PointID.DROPLINK_ROCKER),
                        arb_key,
                        compute_point_point_distance(
                            self.corners[side]
                            .initial_state()
                            .positions[PointID.DROPLINK_ROCKER],
                            droplink,
                        ),
                    ),
                )
            )
        return constraints

    def get_visualization_links(self) -> list["LinkVisualization"]:
        """Base axle links plus the anti-roll bar and its two droplinks."""
        from kinematics.visualization.main import LinkVisualization

        links = super().get_visualization_links()
        design = self.initial_state()
        left_droplink = design.positions[PointRef(Side.LEFT, PointID.DROPLINK_ARB)]
        axis_a = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_A)]
        axis_b = design.positions[PointRef(Side.CENTER, PointID.ARB_AXIS_B)]
        if compute_point_point_distance(
            left_droplink, axis_a
        ) <= compute_point_point_distance(left_droplink, axis_b):
            left_end, right_end = PointID.ARB_AXIS_A, PointID.ARB_AXIS_B
        else:
            left_end, right_end = PointID.ARB_AXIS_B, PointID.ARB_AXIS_A

        links.append(
            LinkVisualization(
                points=[
                    PointRef(Side.LEFT, PointID.DROPLINK_ARB),
                    PointRef(Side.CENTER, left_end),
                    PointRef(Side.CENTER, right_end),
                    PointRef(Side.RIGHT, PointID.DROPLINK_ARB),
                ],
                color="teal",
                label="Anti-Roll Bar",
            )
        )
        for side in (Side.LEFT, Side.RIGHT):
            links.append(
                LinkVisualization(
                    points=[
                        PointRef(side, PointID.DROPLINK_ROCKER),
                        PointRef(side, PointID.DROPLINK_ARB),
                    ],
                    color="goldenrod",
                    label=f"{side.name.title()} Droplink",
                )
            )
        return links
