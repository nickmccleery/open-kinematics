"""
Double-wishbone axle suspension (two coupled corners).

This module defines :class:`DoubleWishboneAxleSuspension`, which composes two
:class:`~kinematics.suspensions.corner.DoubleWishboneSuspension` corner
instances (left and right) into a single constraint system solved together. The
corners are coupled through a rigid steering rack: the two inboard trackrod
points are held a fixed distance apart, so a single steering input drives both
wheels.

The axle keys its state, constraints, and derived points on
:class:`~kinematics.core.point_ref.PointRef` (``(side, point)``) rather than the
bare :class:`~kinematics.core.enums.PointID`. Everything below re-keys the
unchanged corner machinery into the two side namespaces; the corner classes
themselves are reused verbatim.

Coordinate system: ISO 8855 (X forward, Y left, Z up). The LEFT corner is the
+Y side; mirroring left <-> right is a reflection through the XZ plane
(``y -> -y`` for points, Y component negated for directions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Mapping, Sequence

from kinematics.constraints import Constraint, DistanceConstraint
from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey, PointRef, Side
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
)
from kinematics.points.derived.manager import (
    DerivedPointsSpec,
    PositionFn,
    PositionValue,
)
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.corner import DoubleWishboneSuspension

if TYPE_CHECKING:
    from kinematics.metrics.main import AxleMetricRows
    from kinematics.sensitivity import TangentField
    from kinematics.visualization.main import LinkVisualization, WheelAnchors


class _SideView(Mapping):
    """
    A value-type-agnostic view of one side of an axle positions dict.

    Corner-level code (constraints, derived-point functions) is written against
    plain :class:`~kinematics.core.enums.PointID` keys. The axle stores every
    position under a :class:`~kinematics.core.point_ref.PointRef`. This view
    forwards ``view[pid]`` to ``positions[PointRef(side, pid)]`` so the corner
    functions can run unchanged against the side-tagged dict.

    The view only forwards lookups, so it works identically for ``Point3`` and
    ``DualVec3`` values (the dual-number autodiff path in
    :meth:`~kinematics.solver.ResidualComputer.compute_jacobian` runs derived
    functions on dual-valued dicts).
    """

    __slots__ = ("_positions", "_side")

    def __init__(self, positions: Mapping[PointKey, Any], side: Side) -> None:
        self._positions = positions
        self._side = side

    def __getitem__(self, point: PointID) -> Any:
        return self._positions[PointRef(self._side, point)]

    def __setitem__(self, point: PointID, value: Any) -> None:
        # Derived-point functions are pure (they return values); __setitem__ is
        # provided for completeness so the view can stand in for a mutable dict.
        self._positions[PointRef(self._side, point)] = value  # type: ignore[index]

    def __contains__(self, point: object) -> bool:
        return PointRef(self._side, point) in self._positions  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[PointID]:
        for key in self._positions:
            if isinstance(key, PointRef) and key.side == self._side:
                yield key.point

    def __len__(self) -> int:
        return sum(
            1
            for key in self._positions
            if isinstance(key, PointRef) and key.side == self._side
        )


@dataclass
class DoubleWishboneAxleSuspension(Suspension):
    """
    A full double-wishbone axle: two corners solved in one coupled system.

    The two corners are independent double-wishbone models coupled by a rigid
    steering rack (a fixed distance between the two inboard trackrod points).
    Each corner also keeps its own ``PointOnLineConstraint`` holding its inboard
    trackrod on the world Y line, so the rack translates purely laterally.

    Degrees of freedom: two wheel-travel DOFs (one per corner) plus one shared
    rack DOF. A sweep MUST pin the rack DOF with a steering target -- e.g. a
    target on the LEFT ``TRACKROD_INBOARD`` along Y -- exactly as the
    single-corner model requires a steering target to be well posed. A typical
    axle sweep targets left wheel-centre Z, right wheel-centre Z, and left
    trackrod-inboard Y.
    """

    TYPE_KEY: ClassVar[str] = "double_wishbone_axle"

    # No axle-level required points: hardpoints live on the corners.
    REQUIRED_POINTS: ClassVar[frozenset[PointID]] = frozenset()

    # Left corner output points (corner order) followed by the right corner's.
    OUTPUT_POINTS: ClassVar[tuple[PointRef, ...]] = tuple(
        PointRef(Side.LEFT, p) for p in DoubleWishboneSuspension.OUTPUT_POINTS
    ) + tuple(PointRef(Side.RIGHT, p) for p in DoubleWishboneSuspension.OUTPUT_POINTS)

    corners: dict[Side, DoubleWishboneSuspension] = field(default_factory=dict)

    @property
    def has_arb(self) -> bool:
        """The basic axle topology does not include an anti-roll bar."""
        return False

    def validate_hardpoints(self) -> None:
        """Require exactly two valid double-wishbone corners."""
        if set(self.corners) != {Side.LEFT, Side.RIGHT}:
            raise ValueError("Axle requires exactly LEFT and RIGHT corner models.")
        for corner in self.corners.values():
            corner.validate_hardpoints()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def initial_state(self) -> SuspensionState:
        """Union of both corners' initial states, side-tagged by ``PointRef``."""
        if self._initial_state is not None:
            return self._initial_state

        positions: dict[PointKey, Point3] = {}
        free_points: set[PointKey] = set()
        for side, corner in self.corners.items():
            # Use the corner's own initial_state so any camber-shim adjustment is
            # applied per corner.
            corner_state = corner.initial_state()
            for pid, pos in corner_state.positions.items():
                positions[PointRef(side, pid)] = pos.copy()
            for pid in corner_state.free_points:
                free_points.add(PointRef(side, pid))

        self._initial_state = SuspensionState(
            positions=positions,
            free_points=free_points,
        )
        return self._initial_state

    def free_points(self) -> Sequence[PointKey]:
        """Each corner's free points, side-tagged."""
        result: list[PointKey] = []
        for side, corner in self.corners.items():
            result.extend(PointRef(side, pid) for pid in corner.free_points())
        return result

    def output_points(self) -> tuple[PointKey, ...]:
        """Per-side corner output points."""
        result: list[PointKey] = []
        for side in (Side.LEFT, Side.RIGHT):
            corner = self.corners[side]
            result.extend(PointRef(side, pid) for pid in corner.output_points())
        return tuple(result)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def constraints(self) -> list[Constraint]:
        """Both corners' constraints (re-keyed) plus the rigid rack coupling."""
        constraints: list[Constraint] = []
        for side, corner in self.corners.items():
            for constraint in corner.constraints():
                constraints.append(
                    constraint.remap(lambda pid, s=side: PointRef(s, pid))
                )

        # Rigid steering rack: hold the two inboard trackrod points a fixed
        # (design) distance apart so a single steering input drives both wheels.
        left_tri = (
            self.corners[Side.LEFT].initial_state().positions[PointID.TRACKROD_INBOARD]
        )
        right_tri = (
            self.corners[Side.RIGHT].initial_state().positions[PointID.TRACKROD_INBOARD]
        )
        rack_separation = compute_point_point_distance(left_tri, right_tri)
        constraints.append(
            DistanceConstraint(
                PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
                PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD),
                rack_separation,
            )
        )

        return constraints

    # ------------------------------------------------------------------
    # Derived points
    # ------------------------------------------------------------------

    def derived_spec(self) -> DerivedPointsSpec:
        """Each corner's derived spec, wrapped and re-keyed per side."""
        functions: dict[PointKey, PositionFn] = {}
        dependencies: dict[PointKey, set[PointKey]] = {}
        for side, corner in self.corners.items():
            spec = corner.derived_spec()
            for pid, fn in spec.functions.items():
                functions[PointRef(side, pid)] = self._wrap_derived_fn(fn, side)
            for pid, deps in spec.dependencies.items():
                dependencies[PointRef(side, pid)] = {
                    PointRef(side, dep) for dep in deps
                }
        return DerivedPointsSpec(functions=functions, dependencies=dependencies)

    @staticmethod
    def _wrap_derived_fn(fn: PositionFn, side: Side) -> PositionFn:
        """
        Adapt a corner derived function to the side-tagged positions dict.

        The returned function receives the full ``PointRef``-keyed positions
        dict and exposes it to the corner function as a plain ``PointID`` view
        for ``side``. This is value-type agnostic, so it works for both
        ``Point3`` and ``DualVec3`` dicts.
        """

        def wrapped(positions: dict[PointKey, PositionValue]) -> PositionValue:
            return fn(_SideView(positions, side))  # type: ignore[arg-type]

        return wrapped

    # ------------------------------------------------------------------
    # Per-side reuse
    # ------------------------------------------------------------------

    def corner_state(self, state: SuspensionState, side: Side) -> SuspensionState:
        """
        Strip a solved axle state down to one side's plain corner state.

        The returned state is keyed on plain ``PointID`` (side tag removed), so
        it can be fed to the unchanged corner suspension for metrics, instant
        centers, and geometry.

        Args:
            state: The solved axle state (``PointRef``-keyed).
            side: The side to extract.

        Returns:
            A ``PointID``-keyed :class:`SuspensionState` for that corner.
        """
        positions: dict[PointKey, Point3] = {}
        free_points: set[PointKey] = set()
        for key, pos in state.positions.items():
            if isinstance(key, PointRef) and key.side == side:
                positions[key.point] = pos
        for key in state.free_points:
            if isinstance(key, PointRef) and key.side == side:
                free_points.add(key.point)
        return SuspensionState(positions=positions, free_points=free_points)

    def compute_side_view_instant_center(self, state: SuspensionState) -> Point3 | None:
        """Not defined at axle level; instant centers are per side."""
        raise NotImplementedError("per-side; use corner_state()/corners[side]")

    def compute_front_view_instant_center(
        self, state: SuspensionState
    ) -> Point3 | None:
        """Not defined at axle level; instant centers are per side."""
        raise NotImplementedError("per-side; use corner_state()/corners[side]")

    # ------------------------------------------------------------------
    # Metrics dispatch
    # ------------------------------------------------------------------

    def compute_state_metrics(
        self,
        state: SuspensionState,
        tangents: "Sequence[TangentField] | None" = None,
    ) -> "AxleMetricRows":
        """Compute per-corner and axle-level metric rows for a solved axle state."""
        from kinematics.metrics.main import compute_metrics_for_axle_state

        if self.config is None:
            raise ValueError("Suspension has no configuration")
        return compute_metrics_for_axle_state(state, self, self.config, tangents)

    def resolve_target_key(self, point: PointID, side: Side | None) -> PointKey:
        """Axle sweep targets must name a side; returns a ``PointRef``."""
        if side is None:
            raise ValueError(
                f"Sweep target for '{point.name}' requires a 'side' "
                "(left/right) for an axle geometry."
            )
        if side == Side.CENTER:
            raise ValueError("Axle sweep target side must be 'left' or 'right'.")
        return PointRef(side, point)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_visualization_links(self) -> list["LinkVisualization"]:
        """Both corners' links (re-keyed) plus a rack link between the sides."""
        from kinematics.visualization.main import LinkVisualization

        links: list[LinkVisualization] = []
        for side, corner in self.corners.items():
            for link in corner.get_visualization_links():
                links.append(
                    LinkVisualization(
                        points=[PointRef(side, p) for p in link.points],
                        color=link.color,
                        label=f"{side.name.title()} {link.label}",
                        linewidth=link.linewidth,
                        linestyle=link.linestyle,
                        marker=link.marker,
                        markersize=link.markersize,
                    )
                )

        links.append(
            LinkVisualization(
                points=[
                    PointRef(Side.LEFT, PointID.TRACKROD_INBOARD),
                    PointRef(Side.RIGHT, PointID.TRACKROD_INBOARD),
                ],
                color="purple",
                label="Steering Rack",
            )
        )

        return links

    def wheel_visualization_anchors(self) -> "list[WheelAnchors]":
        """One wheel anchor set per side so both wheels are drawn."""
        from kinematics.visualization.main import WheelAnchors

        anchors: list[WheelAnchors] = []
        for side in self.corners:
            anchors.append(
                WheelAnchors(
                    center=PointRef(side, PointID.WHEEL_CENTER),
                    inboard=PointRef(side, PointID.WHEEL_INBOARD),
                    outboard=PointRef(side, PointID.WHEEL_OUTBOARD),
                    axle_inboard=PointRef(side, PointID.AXLE_INBOARD),
                    axle_outboard=PointRef(side, PointID.AXLE_OUTBOARD),
                )
            )
        return anchors
