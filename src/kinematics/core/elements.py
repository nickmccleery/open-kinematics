"""Physical suspension element declarations."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from kinematics.core.primitives.enums import Axis
from kinematics.core.primitives.point_ref import PointKey


class RigidLinkType(StrEnum):
    """Fixed-length suspension link."""

    WISHBONE_LEG = "wishbone"
    TRACK_ROD = "track_rod"
    AXLE = "axle"
    PUSHROD = "pushrod"
    DROPLINK = "droplink"


class VariableLengthLinkType(StrEnum):
    """Link whose length changes with motion."""

    SPRING_DAMPER = "spring_damper"


class TorsionElementType(StrEnum):
    """Element that twists about an axis."""

    ANTI_ROLL_BAR = "anti_roll_bar"
    TORSION_BAR = "torsion_bar"


@dataclass(frozen=True)
class SuspensionElement(ABC):
    """Base type for one physical element in a suspension assembly."""

    label: str

    @property
    @abstractmethod
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return every point referenced by this element."""
        raise NotImplementedError


@dataclass(frozen=True)
class RigidLinkElement(SuspensionElement):
    """A two-point link that preserves its design length."""

    type: RigidLinkType
    point_a: PointKey
    point_b: PointKey

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return both link endpoints."""
        return self.point_a, self.point_b


@dataclass(frozen=True)
class VariableLengthLinkElement(SuspensionElement):
    """A two-point link whose length is free to change."""

    type: VariableLengthLinkType
    point_a: PointKey
    point_b: PointKey

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return both link endpoints."""
        return self.point_a, self.point_b


@dataclass(frozen=True)
class RackElement(SuspensionElement):
    """A steering rack connecting left and right inner trackrod joints."""

    left_inner: PointKey
    right_inner: PointKey
    translation_axis: Axis

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return the rack joint points in left-to-right order."""
        return self.left_inner, self.right_inner


@dataclass(frozen=True)
class UprightElement(SuspensionElement):
    """Rigid upright hardpoints, attachments, and visible body segments."""

    hardpoints: tuple[PointKey, ...]
    attachments: tuple[PointKey, ...]
    segments: tuple[tuple[PointKey, PointKey], ...]

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return the upright's hardpoints and rigid attachments."""
        return self.hardpoints + self.attachments


@dataclass(frozen=True)
class TorsionElement(SuspensionElement):
    """A torsion member with an arbitrary rotation axis and attachments."""

    type: TorsionElementType
    rotation_axis: tuple[PointKey, PointKey]
    attachments: tuple[PointKey, ...]
    path: tuple[PointKey, ...]

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return axis, attachment, and path points."""
        return self.rotation_axis + self.attachments + self.path


@dataclass(frozen=True)
class RockerElement(SuspensionElement):
    """A rigid rocker rotating about an arbitrarily oriented axis."""

    rotation_axis: tuple[PointKey, PointKey]
    pickups: tuple[PointKey, ...]

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return the rocker axis and pickup points."""
        return self.rotation_axis + self.pickups


@dataclass(frozen=True)
class WheelElement(SuspensionElement):
    """A wheel, hub axis, and contact patch."""

    center: PointKey
    inboard: PointKey
    outboard: PointKey
    axle_inboard: PointKey
    axle_outboard: PointKey
    contact_patch: PointKey

    @property
    def point_keys(self) -> tuple[PointKey, ...]:
        """Return all wheel and hub reference points."""
        return (
            self.center,
            self.inboard,
            self.outboard,
            self.axle_inboard,
            self.axle_outboard,
            self.contact_patch,
        )


def map_element_points(
    element: SuspensionElement,
    transform: Callable[[PointKey], PointKey],
    *,
    label: str | None = None,
) -> SuspensionElement:
    """Map every point reference while preserving the concrete element type."""
    mapped_label = element.label if label is None else label
    if isinstance(element, RigidLinkElement | VariableLengthLinkElement):
        return replace(
            element,
            label=mapped_label,
            point_a=transform(element.point_a),
            point_b=transform(element.point_b),
        )
    if isinstance(element, RackElement):
        return replace(
            element,
            label=mapped_label,
            left_inner=transform(element.left_inner),
            right_inner=transform(element.right_inner),
        )
    if isinstance(element, UprightElement):
        return replace(
            element,
            label=mapped_label,
            hardpoints=tuple(transform(point) for point in element.hardpoints),
            attachments=tuple(transform(point) for point in element.attachments),
            segments=tuple(
                (transform(start), transform(end)) for start, end in element.segments
            ),
        )
    if isinstance(element, TorsionElement):
        return replace(
            element,
            label=mapped_label,
            rotation_axis=tuple(transform(point) for point in element.rotation_axis),
            attachments=tuple(transform(point) for point in element.attachments),
            path=tuple(transform(point) for point in element.path),
        )
    if isinstance(element, RockerElement):
        return replace(
            element,
            label=mapped_label,
            rotation_axis=tuple(transform(point) for point in element.rotation_axis),
            pickups=tuple(transform(point) for point in element.pickups),
        )
    if isinstance(element, WheelElement):
        return replace(
            element,
            label=mapped_label,
            center=transform(element.center),
            inboard=transform(element.inboard),
            outboard=transform(element.outboard),
            axle_inboard=transform(element.axle_inboard),
            axle_outboard=transform(element.axle_outboard),
            contact_patch=transform(element.contact_patch),
        )
    raise TypeError(f"Unsupported suspension element: {type(element)!r}")
