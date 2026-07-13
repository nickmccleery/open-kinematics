"""Validated composition of suspension points and physical elements."""

from dataclasses import dataclass

from kinematics.core.elements import SuspensionElement
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.state import SuspensionState


@dataclass(frozen=True)
class PointCatalog:
    """Identifier-only classification of points in a suspension assembly."""

    fixed: frozenset[PointKey]
    free: frozenset[PointKey]
    derived: frozenset[PointKey]

    def __post_init__(self) -> None:
        """Require mutually exclusive point classifications."""
        overlaps = (
            (self.fixed & self.free)
            | (self.fixed & self.derived)
            | (self.free & self.derived)
        )
        if overlaps:
            raise ValueError(f"Point classifications overlap: {sorted(overlaps)!r}")

    @property
    def all(self) -> frozenset[PointKey]:
        """Return every point in the catalog."""
        return self.fixed | self.free | self.derived

    @classmethod
    def from_state(
        cls,
        state: SuspensionState,
        derived_spec: DerivedPointsSpec,
    ) -> "PointCatalog":
        """Classify point identifiers without copying solver positions."""
        state_points = frozenset[PointKey](state.positions)
        derived = frozenset[PointKey](derived_spec.functions)
        free = frozenset[PointKey](state.free_points)
        base = state_points - derived

        if not free <= base:
            invalid = sorted(free - base)
            raise ValueError(
                f"Free points must be non-derived state points: {invalid!r}"
            )

        catalog = cls(fixed=base - free, free=free, derived=derived)
        if state_points != catalog.all:
            raise ValueError("Initial-state points do not match the point catalog")

        missing_dependencies = {
            dependency
            for dependencies in derived_spec.dependencies.values()
            for dependency in dependencies
            if dependency not in catalog.all
        }
        if missing_dependencies:
            raise ValueError(
                "Derived-point dependencies are absent from the point catalog: "
                f"{sorted(missing_dependencies)!r}"
            )
        return catalog


@dataclass(frozen=True)
class SuspensionAssembly:
    """Complete physical composition of one suspension model."""

    points: PointCatalog
    elements: tuple[SuspensionElement, ...]
    output_points: tuple[PointKey, ...]

    def __post_init__(self) -> None:
        """Validate that every exported and element point exists."""
        element_points = {
            point for element in self.elements for point in element.point_keys
        }
        missing_element_points = element_points - self.points.all
        if missing_element_points:
            raise ValueError(
                "Assembly elements reference unknown points: "
                f"{sorted(missing_element_points)!r}"
            )

        missing_output_points = set(self.output_points) - self.points.all
        if missing_output_points:
            raise ValueError(
                "Assembly output references unknown points: "
                f"{sorted(missing_output_points)!r}"
            )

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return a stable, deduplicated view of every assembly point."""
        ordered: list[PointKey] = []
        seen: set[PointKey] = set()

        def append(point: PointKey) -> None:
            if point not in seen:
                ordered.append(point)
                seen.add(point)

        for point in self.output_points:
            append(point)
        for element in self.elements:
            for point in element.point_keys:
                append(point)
        for point in sorted(self.points.all - seen):
            append(point)
        return tuple(ordered)

    @classmethod
    def from_state(
        cls,
        state: SuspensionState,
        derived_spec: DerivedPointsSpec,
        elements: tuple[SuspensionElement, ...],
        output_points: tuple[PointKey, ...],
    ) -> "SuspensionAssembly":
        """Build and validate an assembly from existing solver declarations."""
        return cls(
            points=PointCatalog.from_state(state, derived_spec),
            elements=elements,
            output_points=output_points,
        )
