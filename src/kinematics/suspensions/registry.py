"""
Suspension type registry.

Maps type keys to Suspension subclasses for loading from YAML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kinematics.suspensions.base import Suspension

SuspensionClass = type["Suspension"]

# Registry populated by importing suspension modules.
_SUSPENSION_REGISTRY: dict[str, SuspensionClass] = {}


def _ensure_registry_populated() -> None:
    """Import all suspension modules to populate registry."""
    if not _SUSPENSION_REGISTRY:
        # Import triggers registration.
        from kinematics.suspensions.double_wishbone import DoubleWishboneSuspension

        _register_class(DoubleWishboneSuspension)


def _register_class(cls: SuspensionClass) -> None:
    """Register a suspension class by its TYPE_KEY and ALIASES."""
    _SUSPENSION_REGISTRY[cls.TYPE_KEY] = cls
    for alias in cls.ALIASES:
        _SUSPENSION_REGISTRY[alias] = cls


def get_suspension_class(type_key: str) -> SuspensionClass | None:
    """
    Get a suspension class by type key.

    Args:
        type_key: The suspension type (e.g., "double_wishbone").

    Returns:
        The Suspension subclass, or None if not found.
    """
    _ensure_registry_populated()
    return _SUSPENSION_REGISTRY.get(type_key.lower())


def list_supported_types() -> list[str]:
    """
    List all supported suspension type keys.

    Returns:
        Sorted list of type keys (includes aliases).
    """
    _ensure_registry_populated()
    return sorted(_SUSPENSION_REGISTRY.keys())


__all__ = [
    "SuspensionClass",
    "get_suspension_class",
    "list_supported_types",
]
