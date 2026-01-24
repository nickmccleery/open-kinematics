"""
Validation utilities for template-based suspension geometry.

Provides comprehensive validation with helpful error messages including Levenshtein-
based suggestions for typos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from kinematics.enums import PointID

if TYPE_CHECKING:
    from .base import SuspensionTemplate


@dataclass
class ValidationError:
    """
    A validation error with optional suggestion for correction.

    Attributes:
        message: The error message describing what's wrong.
        suggestion: Optional suggestion for how to fix the error.
    """

    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message} ({self.suggestion})"
        return self.message


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The minimum number of single-character edits (insertions, deletions,
        substitutions) required to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise.
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_closest_matches(
    key: str,
    valid_keys: set[str],
    max_results: int = 3,
    max_distance: int = 3,
) -> list[str]:
    """
    Find the closest matching keys using Levenshtein distance.

    Args:
        key: The key to find matches for.
        valid_keys: Set of valid keys to match against.
        max_results: Maximum number of suggestions to return.
        max_distance: Maximum edit distance to consider a match.

    Returns:
        List of closest matching keys, sorted by distance then alphabetically.
    """
    distances: list[tuple[int, str]] = []

    for valid_key in valid_keys:
        distance = levenshtein_distance(key.upper(), valid_key.upper())
        if distance <= max_distance:
            distances.append((distance, valid_key))

    # Sort by distance, then alphabetically.
    distances.sort(key=lambda x: (x[0], x[1]))

    return [key for _, key in distances[:max_results]]


def validate_hardpoints(
    hardpoints: dict[str, Any],
    template: SuspensionTemplate,
) -> list[ValidationError]:
    """
    Validate hardpoint keys and values against a template.

    Checks:
    - All required points are present
    - No unknown points (with Levenshtein suggestions for typos)
    - Each value is a valid triplet (list of 3 numeric values)

    Args:
        hardpoints: Dictionary of hardpoint name -> coordinate triplet.
        template: Template to validate against.

    Returns:
        List of ValidationError objects (empty if valid).
    """
    errors: list[ValidationError] = []
    valid_names = template.all_valid_point_names

    # Track which valid keys we've seen.
    seen_point_ids: set[PointID] = set()

    # Validate each provided hardpoint.
    for key, value in hardpoints.items():
        normalized_key = key.upper()

        # Check if key is valid.
        point_id = template.point_id_from_name(normalized_key)

        if point_id is None:
            # Unknown key - provide suggestions.
            suggestions = find_closest_matches(normalized_key, valid_names)
            suggestion_text = None
            if suggestions:
                suggestion_text = f"Did you mean: {', '.join(suggestions)}?"

            errors.append(
                ValidationError(
                    message=f"Unknown hardpoint key '{key}'",
                    suggestion=suggestion_text,
                )
            )
        else:
            seen_point_ids.add(point_id)

            # Validate value format.
            triplet_errors = _validate_triplet(key, value)
            errors.extend(triplet_errors)

    # Check for missing required points.
    missing = template.required_point_ids - seen_point_ids
    if missing:
        missing_names = sorted(p.name for p in missing)
        errors.append(
            ValidationError(
                message=f"Missing required hardpoints: {', '.join(missing_names)}",
                suggestion=None,
            )
        )

    return errors


def _validate_triplet(key: str, value: Any) -> list[ValidationError]:
    """
    Validate that a value is a valid [x, y, z] triplet.

    Args:
        key: The hardpoint key (for error messages).
        value: The value to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[ValidationError] = []

    # Check if it's a list/tuple.
    if not isinstance(value, (list, tuple)):
        if isinstance(value, dict):
            # Allow dict format {x:, y:, z:}.
            return _validate_dict_triplet(key, value)
        errors.append(
            ValidationError(
                message=f"'{key}' must be [x, y, z], got {type(value).__name__}",
                suggestion="Use format: [x, y, z] or {x: val, y: val, z: val}",
            )
        )
        return errors

    # Check length.
    if len(value) != 3:
        errors.append(
            ValidationError(
                message=f"'{key}' must have exactly 3 coordinates, got {len(value)}",
                suggestion="Provide [x, y, z] coordinates",
            )
        )
        return errors

    # Check each value is numeric.
    for i, v in enumerate(value):
        if not isinstance(v, (int, float)):
            coord_name = ["x", "y", "z"][i]
            errors.append(
                ValidationError(
                    message=f"'{key}' {coord_name} must be numeric",
                    suggestion=None,
                )
            )

    return errors


def _validate_dict_triplet(key: str, value: dict) -> list[ValidationError]:
    """
    Validate a dict-format coordinate {x:, y:, z:}.

    Args:
        key: The hardpoint key (for error messages).
        value: Dictionary with x, y, z keys.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[ValidationError] = []

    required_keys = {"x", "y", "z"}
    provided_keys = set(value.keys())

    # Check for missing keys.
    missing = required_keys - provided_keys
    if missing:
        errors.append(
            ValidationError(
                message=f"'{key}' dict is missing keys: {', '.join(sorted(missing))}",
                suggestion="Dict format requires {x: val, y: val, z: val}",
            )
        )
        return errors

    # Check for extra keys.
    extra = provided_keys - required_keys
    if extra:
        errors.append(
            ValidationError(
                message=f"'{key}' dict has unexpected keys: {', '.join(sorted(extra))}",
                suggestion="Dict format should only have x, y, z keys",
            )
        )

    # Check each value is numeric.
    for coord in ["x", "y", "z"]:
        if coord in value and not isinstance(value[coord], (int, float)):
            errors.append(
                ValidationError(
                    message=f"'{key}' {coord} must be numeric",
                    suggestion=None,
                )
            )

    return errors


def validate_shim_config(
    shim_config: Any,
    template: SuspensionTemplate,
) -> list[ValidationError]:
    """
    Validate camber shim configuration.

    Checks:
    - Required if template.shim_support is True (configurable)
    - shim_face_center has valid x, y, z
    - shim_normal has valid x, y, z
    - shim_normal is not near-zero
    - design_thickness and setup_thickness are present and numeric

    Args:
        shim_config: The camber_shim configuration object or dict.
        template: Template for context.

    Returns:
        List of ValidationError objects (empty if valid).
    """
    errors: list[ValidationError] = []

    # If template doesn't support shims and config is provided, just skip validation.
    if not template.shim_support:
        return errors

    # If no shim config provided, that's OK - shims are optional.
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
        errors.append(
            ValidationError(
                message="camber_shim must be a dict or CamberShimConfigOutboard",
                suggestion=None,
            )
        )
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
            errors.append(
                ValidationError(
                    message=f"camber_shim missing required field: {field}",
                    suggestion=None,
                )
            )

    if errors:
        return errors  # Can't continue validation without required fields.

    # Validate shim_face_center.
    center_errors = _validate_dict_triplet(
        "shim_face_center", config["shim_face_center"]
    )
    errors.extend(center_errors)

    # Validate shim_normal.
    normal_errors = _validate_dict_triplet("shim_normal", config["shim_normal"])
    errors.extend(normal_errors)

    # Check shim_normal magnitude.
    if not normal_errors:
        normal = config["shim_normal"]
        magnitude = np.sqrt(normal["x"] ** 2 + normal["y"] ** 2 + normal["z"] ** 2)
        if magnitude < 1e-6:
            errors.append(
                ValidationError(
                    message="shim_normal vector is near-zero",
                    suggestion="shim_normal must be non-zero (will be normalized)",
                )
            )
        elif abs(magnitude - 1.0) > 0.1:
            # Not an error, but worth noting - we'll normalize internally.
            pass

    # Validate thicknesses.
    for field in ["design_thickness", "setup_thickness"]:
        value = config.get(field)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(
                ValidationError(
                    message=f"camber_shim.{field} must be numeric",
                )
            )

    return errors


def format_validation_errors(errors: list[ValidationError]) -> str:
    """
    Format a list of validation errors into a human-readable string.

    Args:
        errors: List of ValidationError objects.

    Returns:
        Formatted error message string.
    """
    if not errors:
        return ""

    lines = ["Validation failed:"]
    for error in errors:
        lines.append(f"  - {error}")

    return "\n".join(lines)
