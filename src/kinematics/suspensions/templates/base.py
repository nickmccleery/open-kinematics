"""
Base classes for suspension templates.

Templates define the topology, ownership, and validation rules for suspension types.
They enable a unified YAML format while keeping type-specific logic in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from kinematics.enums import PointID


@dataclass(frozen=True)
class ComponentSpec:
    """
    Specification for a rigid body component in a suspension.

    Attributes:
        name: Component identifier (e.g., "upright", "upper_wishbone").
        mount_roles: Mapping from role names to PointIDs defining kinematic
            connections (e.g., {"upper_ball_joint": UPPER_WISHBONE_OUTBOARD}).
        attachment_point_ids: PointIDs rigidly attached to this component but
            not defining its kinematic frame (e.g., axle points on upright).
    """

    name: str
    mount_roles: dict[str, PointID]
    attachment_point_ids: Sequence[PointID] = field(default_factory=list)


@dataclass(frozen=True)
class SuspensionTemplate:
    """
    Template defining a suspension type's topology and validation rules.

    Templates specify:
    - What hardpoints are required/optional
    - What rigid body components exist and their structure
    - Which component "owns" each point (for ownership semantics)
    - Whether camber shim adjustment is supported

    The template key matches the YAML `type` field.

    Attributes:
        key: Template identifier matching YAML type field (e.g., "double_wishbone")
        required_point_ids: Set of PointIDs that must be provided in YAML
        optional_point_ids: Set of PointIDs that may optionally be provided
        components: List of ComponentSpecs defining rigid bodies
        ownership: Mapping from PointID to component name that owns it.
                  Note: Ball joint points like UPPER_WISHBONE_OUTBOARD are owned by the
                  upright even though their names suggest wishbone association.
        shim_support: Whether this template supports camber shim adjustment
        aliases: Alternative type names that map to this template
    """

    key: str
    required_point_ids: frozenset[PointID]
    optional_point_ids: frozenset[PointID] = field(default_factory=frozenset)
    components: tuple[ComponentSpec, ...] = field(default_factory=tuple)
    ownership: dict[PointID, str] = field(default_factory=dict)
    shim_support: bool = False
    aliases: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """
        Validate template consistency.
        """
        # Ensure no overlap between required and optional.
        overlap = self.required_point_ids & self.optional_point_ids
        if overlap:
            raise ValueError(f"Points cannot be both required and optional: {overlap}")

    @property
    def all_valid_point_ids(self) -> frozenset[PointID]:
        """
        All PointIDs that are valid for this template (required + optional).
        """
        return self.required_point_ids | self.optional_point_ids

    @property
    def all_valid_point_names(self) -> frozenset[str]:
        """
        All valid point names (uppercase PointID names).
        """
        return frozenset(p.name for p in self.all_valid_point_ids)

    def get_component_by_name(self, name: str) -> ComponentSpec | None:
        """
        Get a component specification by name.
        """
        for component in self.components:
            if component.name == name:
                return component
        return None

    def get_upright_component(self) -> ComponentSpec | None:
        """
        Get the upright component specification if it exists.
        """
        return self.get_component_by_name("upright")

    def point_id_from_name(self, name: str) -> PointID | None:
        """
        Convert a point name string to PointID.

        Accepts both SCREAMING_SNAKE_CASE (e.g., "UPPER_WISHBONE_OUTBOARD") and
        lower_snake_case (e.g., "upper_wishbone_outboard").
        """
        normalized = name.upper()
        for point_id in self.all_valid_point_ids:
            if point_id.name == normalized:
                return point_id
        return None
