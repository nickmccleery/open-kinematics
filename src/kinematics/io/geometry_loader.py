"""
YAML geometry loader with validation.

This module handles file I/O, schema parsing, and geometry validation for template-based
suspension types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import class_schema

from kinematics.enums import Units
from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.suspensions.core.settings import SuspensionConfig
from kinematics.suspensions.core.template_geometry import TemplateGeometry
from kinematics.suspensions.implementations.template_provider import (
    TemplateSuspensionProvider,
)
from kinematics.suspensions.templates.library import build_template_registry
from kinematics.suspensions.templates.validation import (
    format_validation_errors,
    validate_hardpoints,
    validate_shim_config,
)

if TYPE_CHECKING:
    from kinematics.suspensions.core.geometry import SuspensionGeometry
    from kinematics.suspensions.templates.base import SuspensionTemplate


@dataclass
class LoadedSuspension:
    """
    Result of loading a suspension geometry from a file.

    Attributes:
        geometry: The loaded and validated suspension geometry instance.
        provider: Instantiated provider bound to this geometry.
    """

    geometry: SuspensionGeometry
    provider: SuspensionProvider


def load_geometry(file_path: Path) -> LoadedSuspension:
    """
    Load suspension geometry from a YAML file.

    Uses the template system for validation and provider creation.
    The YAML format is:

        type: double_wishbone
        name: "My Suspension"
        version: "1.0.0"
        units: MILLIMETERS

        hardpoints:
          LOWER_WISHBONE_INBOARD_FRONT: [x, y, z]
          LOWER_WISHBONE_INBOARD_REAR: [x, y, z]
          ...

        config:
          steered: true
          wheel:
            offset: 0
            tire: {...}
          camber_shim:
            shim_face_center: {x: ..., y: ..., z: ...}
            shim_normal: {x: ..., y: ..., z: ...}
            design_thickness: 30.0
            setup_thickness: 30.0

    Args:
        file_path: Path to the YAML geometry file.

    Returns:
        LoadedSuspension dataclass containing the geometry and provider.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file contents are invalid or geometry type is unsupported.
        OSError: For general file handling errors.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            raise ValueError("Geometry file is empty.")

        if "type" not in yaml_data:
            raise ValueError("Geometry type not specified in file")

        geometry_type_key = yaml_data.pop("type")
        template_registry = build_template_registry()

        # Normalize key to lowercase for template lookup.
        template_key = geometry_type_key.lower()
        if template_key not in template_registry:
            available = sorted(template_registry.keys())
            raise ValueError(
                f"Unsupported geometry type: '{geometry_type_key}'. "
                f"Supported types: {', '.join(available)}"
            )

        template = template_registry[template_key]
        return _load_template_geometry(template_key, yaml_data, template)

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing geometry file: {e}") from e
    except MarshmallowValidationError as e:
        raise ValueError(f"Error validating geometry: {e}") from e
    except (IOError, OSError) as e:
        raise OSError(f"Error reading geometry file: {e}") from e


def _load_template_geometry(
    type_key: str,
    yaml_data: dict[str, Any],
    template: "SuspensionTemplate",
) -> LoadedSuspension:
    """
    Load geometry using the template system.

    Args:
        type_key: The geometry type key
        yaml_data: Parsed YAML data (with 'type' already removed)
        template: The template to use for validation

    Returns:
        LoadedSuspension with template geometry and provider

    Raises:
        ValueError: If validation fails
    """
    # Extract and validate hardpoints.
    hardpoints = yaml_data.get("hardpoints", {})
    errors = validate_hardpoints(hardpoints, template)
    if errors:
        raise ValueError(format_validation_errors(errors))

    # Load configuration using marshmallow.
    config_data = yaml_data.get("config", yaml_data.get("configuration", {}))
    ConfigSchema = class_schema(SuspensionConfig)
    try:
        config = ConfigSchema().load(config_data)
    except MarshmallowValidationError as e:
        raise ValueError(f"Configuration validation error: {e}") from e

    # Validate shim config if present.
    shim_errors = validate_shim_config(config.camber_shim, template)
    if shim_errors:
        raise ValueError(format_validation_errors(shim_errors))

    # Parse units.
    units_str = yaml_data.get("units", "MILLIMETERS")
    try:
        units = Units[units_str.upper()]
    except KeyError:
        raise ValueError(f"Unknown units: {units_str}")

    # Create geometry.
    geometry = TemplateGeometry(
        name=yaml_data.get("name", "unnamed"),
        version=yaml_data.get("version", "0.0.0"),
        units=units,
        configuration=config,
        hardpoints=hardpoints,
        template_key=type_key,
    )

    # Validate geometry against template.
    geometry.validate(template)

    # Create provider (geometry first for compatibility with re-instantiation).
    provider = TemplateSuspensionProvider(geometry, template)

    return LoadedSuspension(geometry=geometry, provider=provider)
