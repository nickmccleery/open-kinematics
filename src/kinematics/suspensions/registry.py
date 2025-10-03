"""
Suspension type registry.
"""

from __future__ import annotations

from typing import Tuple, Type

from kinematics.suspensions.base.provider import SuspensionProvider
from kinematics.suspensions.double_wishbone import (
    DoubleWishboneGeometry,
    DoubleWishboneProvider,
)
from kinematics.suspensions.macpherson import MacPhersonGeometry, MacPhersonProvider

# Registry type definitions.
ModelCls = Type[object]
ProviderCls = Type[SuspensionProvider]
Registry = dict[str, Tuple[ModelCls, ProviderCls]]


def build_registry() -> Registry:
    """
    Return the plugin registry mapping suspension types to their classes.

    Returns:
        Dictionary mapping type strings to (ModelClass, ProviderClass) tuples.
    """
    return {
        "DOUBLE_WISHBONE": (DoubleWishboneGeometry, DoubleWishboneProvider),
        "MACPHERSON_STRUT": (MacPhersonGeometry, MacPhersonProvider),
    }
