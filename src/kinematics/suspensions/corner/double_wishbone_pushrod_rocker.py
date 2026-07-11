"""
Double-wishbone corner suspension with pushrod and rocker actuation.

The spring medium is explicit: a torsion bar uses the rocker rotation directly,
while a coilover adds chassis and rocker pickups whose separation changes through
wheel travel. The ARB-ready corner subclass adds a required rocker droplink and
is used only by the full rocker/ARB axle model.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Sequence

from kinematics.constraints import (
    Constraint,
    DistanceConstraint,
    ScalarTripleProductConstraint,
)
from kinematics.core.constants import EPS_GEOMETRIC, MIN_CHIRALITY_VOLUME
from kinematics.core.enums import Axis, PointID
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_to_line_distance,
    compute_scalar_triple_product,
)
from kinematics.state import SuspensionState
from kinematics.suspensions.corner.attachments import rigid_point_constraints
from kinematics.suspensions.corner.double_wishbone import DoubleWishboneSuspension

RockerSpringType = Literal["torsion_bar", "coilover"]

ROCKER_POINTS = frozenset(
    {
        PointID.PUSHROD_OUTBOARD,
        PointID.PUSHROD_INBOARD,
        PointID.ROCKER_AXIS_FRONT,
        PointID.ROCKER_AXIS_REAR,
    }
)
COILOVER_POINTS = frozenset({PointID.STRUT_TOP, PointID.STRUT_BOTTOM})


@dataclass
class DoubleWishbonePushrodRockerSuspension(DoubleWishboneSuspension):
    """A double-wishbone corner actuated by a pushrod and rocker."""

    TYPE_KEY: ClassVar[str] = "double_wishbone_pushrod_rocker"
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = (
        DoubleWishboneSuspension.REQUIRED_POINTS | ROCKER_POINTS
    )
    OPTIONAL_POINTS: ClassVar[frozenset[PointID]] = COILOVER_POINTS

    spring_type: RockerSpringType = field(kw_only=True)

    @property
    def has_rocker(self) -> bool:
        """This topology always includes a pushrod and rocker."""
        return True

    @property
    def has_droplink(self) -> bool:
        """The basic corner has no ARB droplink pickup."""
        return False

    @property
    def has_strut(self) -> bool:
        """Whether this rocker is sprung by an inboard coilover."""
        return self.spring_type == "coilover"

    @property
    def has_torsion_bar(self) -> bool:
        """Whether this rocker is sprung by a coaxial torsion bar."""
        return self.spring_type == "torsion_bar"

    def validate_hardpoints(self) -> None:
        """Validate the explicit rocker topology and selected spring medium."""
        super().validate_hardpoints()
        present = set(self.hardpoints)
        coilover_present = COILOVER_POINTS & present
        if self.has_strut and coilover_present != COILOVER_POINTS:
            missing = sorted(point.name for point in COILOVER_POINTS - present)
            raise ValueError(
                "Pushrod-rocker spring type 'coilover' requires STRUT_TOP and "
                f"STRUT_BOTTOM; missing {missing}."
            )
        if self.has_torsion_bar and coilover_present:
            names = sorted(point.name for point in coilover_present)
            raise ValueError(
                "Pushrod-rocker spring type 'torsion_bar' does not accept "
                f"coilover points: {names}."
            )

        axis_front = self.hardpoints[PointID.ROCKER_AXIS_FRONT]
        axis_rear = self.hardpoints[PointID.ROCKER_AXIS_REAR]
        if compute_point_point_distance(axis_front, axis_rear) <= EPS_GEOMETRIC:
            raise ValueError(
                "ROCKER_AXIS_FRONT and ROCKER_AXIS_REAR must be distinct points."
            )
        y_extent = abs(float(axis_front[Axis.Y]) - float(axis_rear[Axis.Y]))
        if y_extent > EPS_GEOMETRIC:
            raise ValueError(
                "Rocker axis must be parallel to the XZ plane (zero Y-extent): "
                f"|Y(ROCKER_AXIS_FRONT) - Y(ROCKER_AXIS_REAR)| = {y_extent} "
                f"exceeds {EPS_GEOMETRIC}."
            )
        axis_direction = (axis_rear - axis_front).normalize()
        for point in self._rocker_pickups():
            radius = compute_point_to_line_distance(
                self.hardpoints[point], axis_front, axis_direction
            )
            if radius <= EPS_GEOMETRIC:
                raise ValueError(
                    f"{point.name} lies on the rocker axis (zero radius); it must "
                    "be off-axis to trace a rocker circle."
                )

    def _rocker_pickups(self) -> tuple[PointID, ...]:
        """Moving points rigidly attached to the rocker."""
        return (PointID.PUSHROD_INBOARD,)

    def free_points(self) -> Sequence[PointID]:
        """Base variables plus the pushrod, rocker, and optional coilover pickup."""
        points = [
            *super().free_points(),
            PointID.PUSHROD_OUTBOARD,
            *self._rocker_pickups(),
        ]
        if self.has_strut:
            points.append(PointID.STRUT_BOTTOM)
        return points

    def output_points(self) -> tuple[PointID, ...]:
        """Base outputs plus the explicit rocker topology."""
        points = [
            *super().output_points(),
            PointID.PUSHROD_OUTBOARD,
            PointID.PUSHROD_INBOARD,
        ]
        if self.has_droplink:
            points.append(PointID.DROPLINK_ROCKER)
        if self.has_strut:
            points.extend((PointID.STRUT_TOP, PointID.STRUT_BOTTOM))
        return tuple(points)

    def constraints(self) -> list[Constraint]:
        """Base corner constraints plus pushrod, rocker, and spring attachments."""
        constraints = super().constraints()
        constraints.extend(self._rocker_constraints(self.initial_state()))
        if self.has_strut:
            constraints.extend(
                rigid_point_constraints(
                    self.initial_state(),
                    PointID.STRUT_BOTTOM,
                    (
                        PointID.ROCKER_AXIS_FRONT,
                        PointID.ROCKER_AXIS_REAR,
                        PointID.PUSHROD_INBOARD,
                    ),
                )
            )
        return constraints

    def _rocker_constraints(self, initial_state: SuspensionState) -> list[Constraint]:
        """Build the fixed-length pushrod and rigid rocker constraints."""
        positions = initial_state.positions
        constraints: list[Constraint] = []

        def add_distance(p1: PointID, p2: PointID) -> None:
            constraints.append(
                DistanceConstraint(
                    p1,
                    p2,
                    compute_point_point_distance(positions[p1], positions[p2]),
                )
            )

        for anchor in (
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ):
            add_distance(PointID.PUSHROD_OUTBOARD, anchor)
        add_distance(PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD)
        add_distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_FRONT)
        add_distance(PointID.PUSHROD_INBOARD, PointID.ROCKER_AXIS_REAR)

        if self.has_droplink:
            add_distance(PointID.DROPLINK_ROCKER, PointID.ROCKER_AXIS_FRONT)
            add_distance(PointID.DROPLINK_ROCKER, PointID.ROCKER_AXIS_REAR)
            add_distance(PointID.PUSHROD_INBOARD, PointID.DROPLINK_ROCKER)
            design_triple = compute_scalar_triple_product(
                positions[PointID.ROCKER_AXIS_REAR]
                - positions[PointID.ROCKER_AXIS_FRONT],
                positions[PointID.PUSHROD_INBOARD]
                - positions[PointID.ROCKER_AXIS_FRONT],
                positions[PointID.DROPLINK_ROCKER]
                - positions[PointID.ROCKER_AXIS_FRONT],
            )
            if abs(design_triple) >= MIN_CHIRALITY_VOLUME:
                constraints.append(
                    ScalarTripleProductConstraint(
                        PointID.ROCKER_AXIS_FRONT,
                        PointID.ROCKER_AXIS_REAR,
                        PointID.PUSHROD_INBOARD,
                        PointID.DROPLINK_ROCKER,
                        target_volume=design_triple,
                        scale=max(abs(design_triple), 1.0),
                    )
                )
        return constraints

    def get_visualization_links(self):
        """Base links plus the pushrod, rocker, and selected spring."""
        from kinematics.visualization.main import LinkVisualization

        rocker_points = [PointID.ROCKER_AXIS_FRONT, PointID.PUSHROD_INBOARD]
        if self.has_droplink:
            rocker_points.append(PointID.DROPLINK_ROCKER)
        rocker_points.append(PointID.ROCKER_AXIS_REAR)
        links = [
            *super().get_visualization_links(),
            LinkVisualization(
                points=[PointID.PUSHROD_OUTBOARD, PointID.PUSHROD_INBOARD],
                color="crimson",
                label="Pushrod",
            ),
            LinkVisualization(
                points=rocker_points,
                color="mediumvioletred",
                label="Rocker",
            ),
        ]
        if self.has_strut:
            links.append(
                LinkVisualization(
                    points=[PointID.STRUT_TOP, PointID.STRUT_BOTTOM],
                    color="seagreen",
                    label="Spring/Damper",
                )
            )
        return links


@dataclass
class DoubleWishbonePushrodRockerArbSuspension(DoubleWishbonePushrodRockerSuspension):
    """Pushrod-rocker corner with an anti-roll-bar droplink pickup."""

    TYPE_KEY: ClassVar[str] = "double_wishbone_pushrod_rocker_arb"
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = (
        DoubleWishbonePushrodRockerSuspension.REQUIRED_POINTS
        | {PointID.DROPLINK_ROCKER}
    )

    @property
    def has_droplink(self) -> bool:
        """This corner always has an anti-roll-bar droplink pickup."""
        return True

    def _rocker_pickups(self) -> tuple[PointID, ...]:
        """Both moving pickups on the rocker body."""
        return (PointID.PUSHROD_INBOARD, PointID.DROPLINK_ROCKER)
