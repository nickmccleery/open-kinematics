"""
Template-based geometry model for suspension systems.

This module provides a geometry model that works with the template system, allowing a
unified YAML format with flat hardpoint mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from kinematics.enums import PointID, Units
from kinematics.suspensions.core.geometry import SuspensionGeometry
from kinematics.suspensions.core.settings import SuspensionConfig
from kinematics.suspensions.templates.validation import (
    format_validation_errors,
    validate_hardpoints,
    validate_shim_config,
)
from kinematics.types import Vec3, make_vec3

if TYPE_CHECKING:
    from kinematics.suspensions.templates.base import SuspensionTemplate


@dataclass
class TemplateGeometry(SuspensionGeometry):
    """
    Geometry model for template-driven suspensions.

    This geometry model accepts a flat dictionary of hardpoints keyed by PointID names
    (e.g., "UPPER_WISHBONE_OUTBOARD") and converts them to the internal representation.

    Attributes:
        name: Human-readable name of the suspension geometry.
        version: Version string of the geometry implementation.
        units: Unit system used for all measurements.
        configuration: Detailed configuration parameters for the suspension.
        hardpoints: Raw hardpoint dictionary from YAML (name -> [x,y,z] or {x,y,z}).
        template_key: The template type key this geometry was loaded for.
    """

    hardpoints: dict[str, Any] = field(default_factory=dict)
    template_key: str = ""

    # Internal cache for converted hardpoints.
    _hardpoints_cache: dict[PointID, Vec3] | None = field(
        default=None, repr=False, compare=False
    )

    def get_hardpoints_dict(
        self, template: SuspensionTemplate | None = None
    ) -> dict[PointID, Vec3]:
        """
        Convert hardpoints to a dictionary mapping PointID to Vec3.

        Args:
            template: Optional template for validation. If not provided, does
                     best-effort conversion of all valid PointID names.

        Returns:
            Dictionary mapping PointID enum values to Vec3 coordinate arrays.

        Raises:
            ValueError: If a hardpoint key cannot be mapped to a PointID.
        """
        if self._hardpoints_cache is not None:
            return self._hardpoints_cache

        result: dict[PointID, Vec3] = {}

        for key, value in self.hardpoints.items():
            # Convert key to PointID.
            point_id = self._key_to_point_id(key, template)
            if point_id is None:
                raise ValueError(f"Unknown hardpoint key: {key}")

            # Convert value to Vec3.
            coords = self._value_to_vec3(value)
            result[point_id] = coords

        self._hardpoints_cache = result
        return result

    def _key_to_point_id(
        self, key: str, template: SuspensionTemplate | None = None
    ) -> PointID | None:
        """
        Convert a hardpoint key string to PointID.
        """
        normalized = key.upper()

        # If template provided, use its validation.
        if template is not None:
            return template.point_id_from_name(normalized)

        # Otherwise, try direct enum lookup.
        try:
            return PointID[normalized]
        except KeyError:
            return None

    def _value_to_vec3(self, value: Any) -> Vec3:
        """
        Convert a coordinate value to Vec3.
        """
        if isinstance(value, dict):
            # Dict format: {x: val, y: val, z: val}.
            return make_vec3([value["x"], value["y"], value["z"]])
        elif isinstance(value, (list, tuple)):
            # List format: [x, y, z].
            return make_vec3(value)
        elif isinstance(value, np.ndarray):
            return make_vec3(value)
        else:
            raise ValueError(f"Cannot convert {type(value)} to Vec3")

    def get_axle_attachments(self) -> tuple[Vec3, Vec3]:
        """
        Get axle inboard and outboard positions.

        Returns:
            Tuple of (axle_inboard, axle_outboard) Vec3 positions.
        """
        hardpoints = self.get_hardpoints_dict()
        return hardpoints[PointID.AXLE_INBOARD], hardpoints[PointID.AXLE_OUTBOARD]

    def get_upright_mount_ids(self) -> dict[str, PointID]:
        """
        Get the standard upright mount PointIDs.

        Returns:
            Dictionary mapping mount role names to PointIDs.
        """
        return {
            "upper_ball_joint": PointID.UPPER_WISHBONE_OUTBOARD,
            "lower_ball_joint": PointID.LOWER_WISHBONE_OUTBOARD,
            "steering_pickup": PointID.TRACKROD_OUTBOARD,
        }

    def validate(self, template: SuspensionTemplate | None = None) -> bool:
        """
        Validate the geometry configuration.

        Args:
            template: Optional template to validate against.

        Returns:
            True if geometry is valid.

        Raises:
            ValueError: If validation fails.
        """
        if template is None:
            # Without template, just verify we can convert all hardpoints.
            try:
                self.get_hardpoints_dict()
                return True
            except ValueError as e:
                raise ValueError(f"Geometry validation failed: {e}")

        # Validate hardpoints against template.
        errors = validate_hardpoints(self.hardpoints, template)

        # Validate shim config if present.
        shim_errors = validate_shim_config(self.configuration.camber_shim, template)
        errors.extend(shim_errors)

        if errors:
            raise ValueError(format_validation_errors(errors))

        return True

    def has_point(self, point_id: PointID) -> bool:
        """
        Check if a specific point is defined in this geometry.
        """
        hardpoints = self.get_hardpoints_dict()
        return point_id in hardpoints

    def get_point(self, point_id: PointID) -> Vec3 | None:
        """
        Get a specific point's coordinates, or None if not defined.
        """
        hardpoints = self.get_hardpoints_dict()
        return hardpoints.get(point_id)


def create_template_geometry(
    template: SuspensionTemplate,
    hardpoints: dict[str, Any],
    configuration: SuspensionConfig,
    name: str = "unnamed",
    version: str = "0.0.0",
    units: Units = Units.MILLIMETERS,
) -> TemplateGeometry:
    """
    Factory function to create a TemplateGeometry with validation.

    Args:
        template: The suspension template to use
        hardpoints: Raw hardpoint dictionary from YAML
        configuration: Suspension configuration
        name: Geometry name
        version: Geometry version
        units: Unit system

    Returns:
        Validated TemplateGeometry instance

    Raises:
        ValueError: If validation fails
    """
    geometry = TemplateGeometry(
        name=name,
        version=version,
        units=units,
        configuration=configuration,
        hardpoints=hardpoints,
        template_key=template.key,
    )

    # Validate against template.
    geometry.validate(template)

    return geometry
