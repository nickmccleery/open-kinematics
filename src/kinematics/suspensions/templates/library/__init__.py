"""
Library of predefined suspension templates.

Each template defines the topology, required/optional points, and component structure
for a specific suspension type.
"""

from __future__ import annotations

from kinematics.suspensions.templates.base import SuspensionTemplate

from kinematics.suspensions.templates.library.double_wishbone import (
    DOUBLE_WISHBONE_TEMPLATE,
)


def build_template_registry() -> dict[str, SuspensionTemplate]:
    """
    Build the registry mapping type keys to templates.

    Returns:
        Dictionary mapping lowercase type keys to SuspensionTemplate instances.
        Includes both primary keys and aliases.
    """
    templates = [
        DOUBLE_WISHBONE_TEMPLATE,
        # Add more templates here as they are defined.
    ]

    registry: dict[str, SuspensionTemplate] = {}

    for template in templates:
        # Register by primary key.
        registry[template.key] = template

        # Register aliases.
        for alias in template.aliases:
            registry[alias] = template

    return registry


def get_template(type_key: str) -> SuspensionTemplate | None:
    """
    Get a template by its type key.

    Args:
        type_key: The type key (case-insensitive).

    Returns:
        The matching SuspensionTemplate, or None if not found.
    """
    registry = build_template_registry()
    return registry.get(type_key.lower())


__all__ = [
    "DOUBLE_WISHBONE_TEMPLATE",
    "build_template_registry",
    "get_template",
]
