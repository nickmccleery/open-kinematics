"""
Suspension type registry.

Contains the registry functionality for mapping suspension types to their
corresponding geometry and provider classes.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base.provider import SuspensionProvider

# Registry type definitions
ModelCls = Type[object]
ProviderCls = Type[SuspensionProvider]
Registry = Dict[str, Tuple[ModelCls, ProviderCls]]


def build_registry() -> Registry:
    """
    Return the plugin registry mapping: type_string -> (ModelClass, ProviderClass)
    """
    # Import here to avoid circular dependencies
    from .double_wishbone import DoubleWishboneGeometry, DoubleWishboneProvider
    from .macpherson import MacPhersonGeometry, MacPhersonProvider

    return {
        "DOUBLE_WISHBONE": (DoubleWishboneGeometry, DoubleWishboneProvider),
        "MACPHERSON_STRUT": (MacPhersonGeometry, MacPhersonProvider),
    }


__all__ = [
    "ModelCls",
    "ProviderCls",
    "Registry",
    "build_registry",
]
