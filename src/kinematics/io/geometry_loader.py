"""
YAML geometry loader.

This module loads suspension geometry from YAML files and returns instantiated
Suspension subclass instances directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import class_schema

from kinematics.core.enums import PointID, ShimType, Units
from kinematics.core.types import make_vec3
from kinematics.suspensions.base import Suspension
from kinematics.suspensions.config.settings import SuspensionConfig
from kinematics.suspensions.registry import get_suspension_class, list_supported_types


def load_geometry(file_path: Path) -> Suspension:
    """
    Load suspension geometry from a YAML file.

    Returns the appropriate Suspension subclass instance directly.

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
        Instantiated Suspension subclass (e.g., DoubleWishboneSuspension).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If validation fails or type is unsupported.
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

        type_key = yaml_data.pop("type").lower()

        # Get the suspension class for this type.
        suspension_class = get_suspension_class(type_key)
        if suspension_class is None:
            available = list_supported_types()
            raise ValueError(
                f"Unsupported geometry type: '{type_key}'. "
                f"Supported types: {', '.join(available)}"
            )

        return _load_suspension(yaml_data, suspension_class)

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing geometry file: {e}") from e
    except MarshmallowValidationError as e:
        raise ValueError(f"Error validating geometry: {e}") from e
    except (IOError, OSError) as e:
        raise OSError(f"Error reading geometry file: {e}") from e


def _load_suspension(
    yaml_data: dict[str, Any],
    suspension_class: type[Suspension],
) -> Suspension:
    """
    Load and instantiate a suspension from parsed YAML data.

    Args:
        yaml_data: Parsed YAML data (with 'type' already removed).
        suspension_class: The Suspension subclass to instantiate.

    Returns:
        Instantiated suspension.

    Raises:
        ValueError: If validation fails.
    """
    # Extract and validate hardpoints.
    raw_hardpoints = yaml_data.get("hardpoints", {})
    errors = _validate_hardpoints(raw_hardpoints, suspension_class)
    if errors:
        raise ValueError("Validation failed:\n  - " + "\n  - ".join(errors))

    # Load configuration using marshmallow.
    config_data = yaml_data.get("config", yaml_data.get("configuration", {}))
    ConfigSchema = class_schema(SuspensionConfig)
    try:
        config = cast(SuspensionConfig, ConfigSchema().load(config_data))
    except MarshmallowValidationError as e:
        raise ValueError(f"Configuration validation error: {e}") from e

    # Validate shim config if present.
    shim_errors = _validate_shim_config(config.camber_shim, suspension_class)
    if shim_errors:
        raise ValueError("Validation failed:\n  - " + "\n  - ".join(shim_errors))

    # Parse units.
    units_str = yaml_data.get("units", "MILLIMETERS")
    try:
        units = Units[units_str.upper()]
    except KeyError:
        raise ValueError(f"Unknown units: {units_str}")

    # Convert hardpoints to dict[PointID, Vec3].
    hardpoints: dict[PointID, Any] = {}
    for key, value in raw_hardpoints.items():
        point_id = PointID[key.upper()]
        if isinstance(value, dict):
            coords = make_vec3([value["x"], value["y"], value["z"]])
        else:
            coords = make_vec3(value)
        hardpoints[point_id] = coords

    # Instantiate and return the suspension.
    return suspension_class(
        name=yaml_data.get("name", "unnamed"),
        version=yaml_data.get("version", "0.0.0"),
        units=units,
        hardpoints=hardpoints,
        config=config,
    )


def _validate_hardpoints(
    hardpoints: dict[str, Any],
    suspension_class: type[Suspension],
) -> list[str]:
    """
    Validate hardpoints against a suspension class definition.

    Args:
        hardpoints: Raw hardpoints dict from YAML.
        suspension_class: The suspension class to validate against.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    valid_points = suspension_class.all_valid_points()
    seen_point_ids: set[PointID] = set()

    for key, value in hardpoints.items():
        normalized_key = key.upper()

        # Check if key is valid.
        try:
            point_id = PointID[normalized_key]
            if point_id not in valid_points:
                raise KeyError()
        except KeyError:
            errors.append(f"Unknown hardpoint key '{key}'")
            continue

        seen_point_ids.add(point_id)
        errors.extend(_validate_vec3(key, value))

    # Check for missing required.
    missing = suspension_class.REQUIRED_POINTS - seen_point_ids
    if missing:
        missing_names = sorted(p.name for p in missing)
        errors.append(f"Missing required hardpoints: {', '.join(missing_names)}")

    return errors


def _validate_vec3(key: str, value: Any) -> list[str]:
    """Validate that a value is a valid [x, y, z] vec3."""
    errors: list[str] = []

    if isinstance(value, dict):
        # Dict format {x:, y:, z:}.
        required_keys = {"x", "y", "z"}
        provided_keys = set(value.keys())

        missing = required_keys - provided_keys
        if missing:
            errors.append(f"'{key}' missing keys: {', '.join(sorted(missing))}")
            return errors

        for coord in ["x", "y", "z"]:
            if coord in value and not isinstance(value[coord], (int, float)):
                errors.append(f"'{key}' {coord} must be numeric")
    elif isinstance(value, (list, tuple)):
        if len(value) != 3:
            errors.append(f"'{key}' needs 3 coordinates, got {len(value)}")
            return errors

        for i, v in enumerate(value):
            if not isinstance(v, (int, float)):
                coord_name = ["x", "y", "z"][i]
                errors.append(f"'{key}' {coord_name} must be numeric")
    else:
        errors.append(f"'{key}' must be [x, y, z], got {type(value).__name__}")

    return errors


def _validate_shim_config(
    shim_config: Any,
    suspension_class: type[Suspension],
) -> list[str]:
    """Validate shim config against suspension class supported shims."""
    errors: list[str] = []

    # If class doesn't support outboard camber shims, skip validation.
    if ShimType.OUTBOARD_CAMBER not in suspension_class.SUPPORTED_SHIMS:
        return errors

    # Shims are optional.
    if shim_config is None:
        return errors

    # Convert dataclass to dict if needed.
    if hasattr(shim_config, "__dataclass_fields__"):
        config = {
            "shim_face_center": shim_config.shim_face_center,
            "shim_normal": shim_config.shim_normal,
            "design_thickness": shim_config.design_thickness,
            "setup_thickness": shim_config.setup_thickness,
        }
    elif isinstance(shim_config, dict):
        config = shim_config
    else:
        errors.append("camber_shim must be a dict or CamberShimConfigOutboard")
        return errors

    # Validate required fields.
    required_fields = [
        "shim_face_center",
        "shim_normal",
        "design_thickness",
        "setup_thickness",
    ]
    for field in required_fields:
        if field not in config:
            errors.append(f"camber_shim missing required field: {field}")

    if errors:
        return errors

    errors.extend(
        _validate_vec3("camber_shim.shim_face_center", config["shim_face_center"])
    )
    errors.extend(_validate_vec3("camber_shim.shim_normal", config["shim_normal"]))

    if errors:
        return errors

    # Validate shim_normal magnitude.
    normal = config["shim_normal"]
    if isinstance(normal, dict):
        magnitude = np.sqrt(normal["x"] ** 2 + normal["y"] ** 2 + normal["z"] ** 2)
    elif isinstance(normal, (list, tuple)):
        magnitude = np.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    else:
        magnitude = 0.0

    if magnitude < 1e-6:
        errors.append("shim_normal vector is near-zero")

    # Validate thicknesses.
    for field in ["design_thickness", "setup_thickness"]:
        value = config.get(field)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"camber_shim.{field} must be numeric")

    return errors
