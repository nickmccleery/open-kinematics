"""
Name-keyed analysis views derived from a suspension assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

import numpy as np

from kinematics.core.assembly import SuspensionAssembly
from kinematics.core.elements import (
    RackElement,
    RigidLinkElement,
    RockerElement,
    TorsionElement,
    UprightElement,
    VariableLengthLinkElement,
    WheelElement,
)
from kinematics.core.primitives.enums import PointID
from kinematics.core.primitives.geometry import extract_array
from kinematics.core.primitives.point_ref import PointKey, point_key_name
from kinematics.core.schema.config import SuspensionConfig

AXIS_FOOT_SUFFIX = "_axis_foot"


class DisplayElementType(StrEnum):
    """Kind of flattened element exposed to presentation consumers."""

    WISHBONE = "wishbone"
    UPRIGHT = "upright"
    TRACK_ROD = "track_rod"
    RACK = "rack"
    AXLE = "axle"
    CONTACT_PATCH = "contact_patch"
    PUSHROD = "pushrod"
    ROCKER = "rocker"
    SPRING_DAMPER = "spring_damper"
    ANTI_ROLL_BAR = "anti_roll_bar"
    TORSION_BAR = "torsion_bar"
    DROPLINK = "droplink"


@dataclass(frozen=True)
class DisplayElement:
    """One name-keyed point sequence in the presentation topology."""

    points: tuple[str, ...]
    type: DisplayElementType
    label: str


@dataclass(frozen=True)
class RockerDisplayGroup:
    """Display description for one rocker-equipped corner."""

    axis_start: str
    axis_end: str
    pickups: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class WheelDisplayDimensions:
    """Static tire dimensions in mm for drawing a wheel."""

    radius: float
    width: float
    rim_radius: float


@dataclass(frozen=True)
class WheelAnchorNames:
    """Name-keyed positions anchoring one displayed wheel."""

    center: str
    inboard: str
    outboard: str
    axle_inboard: str
    axle_outboard: str


def rocker_display_groups(assembly: SuspensionAssembly) -> list[RockerDisplayGroup]:
    """Convert rocker elements to public point names."""
    return [
        RockerDisplayGroup(
            axis_start=point_key_name(rocker.rotation_axis[0]),
            axis_end=point_key_name(rocker.rotation_axis[1]),
            pickups=tuple(point_key_name(point) for point in rocker.pickups),
            label=rocker.label,
        )
        for rocker in assembly.elements
        if isinstance(rocker, RockerElement)
    ]


def display_point_keys(assembly: SuspensionAssembly) -> tuple[PointKey, ...]:
    """Return all point keys required to resolve the presentation model."""
    points = list(assembly.output_points)
    seen = set(points)
    for element in assembly.elements:
        for key in element.point_keys:
            if key not in seen:
                points.append(key)
                seen.add(key)
    return tuple(points)


def display_positions(
    positions: Mapping[PointKey, object],
    point_keys: tuple[PointKey, ...],
    rocker_groups: list[RockerDisplayGroup],
) -> dict[str, tuple[float, float, float]]:
    """Flatten positions to name keys and append synthetic rocker-axis feet."""
    named: dict[str, tuple[float, float, float]] = {}
    for key in point_keys:
        position = positions.get(key)
        if position is None:
            continue
        raw = extract_array(position)
        named[point_key_name(key)] = (float(raw[0]), float(raw[1]), float(raw[2]))

    for group in rocker_groups:
        _append_axis_feet(named, group)
    return named


def _append_axis_feet(
    named_positions: dict[str, tuple[float, float, float]],
    group: RockerDisplayGroup,
) -> None:
    """Append the perpendicular projection of each pickup onto its rocker axis."""
    axis_a = named_positions.get(group.axis_start)
    axis_b = named_positions.get(group.axis_end)
    if axis_a is None or axis_b is None:
        return

    axis_origin = np.asarray(axis_a, dtype=np.float64)
    axis_direction = np.asarray(axis_b, dtype=np.float64) - axis_origin
    norm_sq = float(np.dot(axis_direction, axis_direction))
    if norm_sq <= 0.0:
        return

    for pickup in group.pickups:
        position = named_positions.get(pickup)
        if position is None:
            continue
        radius = np.asarray(position, dtype=np.float64) - axis_origin
        parameter = float(np.dot(radius, axis_direction)) / norm_sq
        foot = axis_origin + parameter * axis_direction
        named_positions[f"{pickup}{AXIS_FOOT_SUFFIX}"] = (
            float(foot[0]),
            float(foot[1]),
            float(foot[2]),
        )


def display_elements(assembly: SuspensionAssembly) -> list[DisplayElement]:
    """Flatten assembly elements to name-keyed presentation geometry."""
    elements: list[DisplayElement] = []
    for element in assembly.elements:
        if isinstance(element, RigidLinkElement | VariableLengthLinkElement):
            elements.append(
                DisplayElement(
                    points=(
                        point_key_name(element.point_a),
                        point_key_name(element.point_b),
                    ),
                    type=DisplayElementType(element.type.value),
                    label=element.label,
                )
            )
        elif isinstance(element, RackElement):
            elements.append(
                DisplayElement(
                    points=(
                        point_key_name(element.left_inner),
                        point_key_name(element.right_inner),
                    ),
                    type=DisplayElementType.RACK,
                    label=element.label,
                )
            )
        elif isinstance(element, UprightElement):
            elements.extend(
                DisplayElement(
                    points=(point_key_name(start), point_key_name(end)),
                    type=DisplayElementType.UPRIGHT,
                    label=element.label,
                )
                for start, end in element.segments
            )
        elif isinstance(element, WheelElement):
            elements.append(
                DisplayElement(
                    points=(point_key_name(element.contact_patch),),
                    type=DisplayElementType.CONTACT_PATCH,
                    label=f"{element.label} Contact Patch",
                )
            )
        elif isinstance(element, TorsionElement):
            elements.append(
                DisplayElement(
                    points=tuple(point_key_name(point) for point in element.path),
                    type=DisplayElementType(element.type.value),
                    label=element.label,
                )
            )
        elif isinstance(element, RockerElement):
            continue
        else:
            raise TypeError(f"Unsupported suspension element: {type(element)!r}")

    for group in rocker_display_groups(assembly):
        elements.append(
            DisplayElement(
                points=(group.axis_start, group.axis_end),
                type=DisplayElementType.ROCKER,
                label=f"{group.label} Axis",
            )
        )
        for pickup in group.pickups:
            arm = (
                "Droplink Arm"
                if pickup.endswith(PointID.DROPLINK_ROCKER.name.lower())
                else "Pushrod Arm"
            )
            elements.append(
                DisplayElement(
                    points=(pickup, f"{pickup}{AXIS_FOOT_SUFFIX}"),
                    type=DisplayElementType.ROCKER,
                    label=f"{group.label} {arm}",
                )
            )
    return elements


def wheel_display_dimensions(
    config: SuspensionConfig | None,
) -> WheelDisplayDimensions | None:
    """Return static tire dimensions, or None when no config is available."""
    if config is None:
        return None
    tire = config.wheel.tire
    return WheelDisplayDimensions(
        radius=float(tire.nominal_radius),
        width=float(tire.section_width),
        rim_radius=float(tire.rim_diameter_mm) / 2.0,
    )


def wheel_anchor_names(assembly: SuspensionAssembly) -> list[WheelAnchorNames]:
    """Return name-keyed drawing anchors for every wheel."""
    return [
        WheelAnchorNames(
            center=point_key_name(anchors.center),
            inboard=point_key_name(anchors.inboard),
            outboard=point_key_name(anchors.outboard),
            axle_inboard=point_key_name(anchors.axle_inboard),
            axle_outboard=point_key_name(anchors.axle_outboard),
        )
        for anchors in assembly.elements
        if isinstance(anchors, WheelElement)
    ]
