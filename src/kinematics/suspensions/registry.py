"""
Suspension type registry.

Contains the registry functionality for mapping suspension types to their
corresponding geometry and provider classes.
"""

from __future__ import annotations

from typing import Tuple, Type

from kinematics.suspensions.base.provider import SuspensionProvider
from kinematics.suspensions.double_wishbone import (
    DoubleWishboneGeometry,
    DoubleWishboneProvider,
)
from kinematics.suspensions.macpherson import MacPhersonGeometry, MacPhersonProvider

# Registry type definitions
ModelCls = Type[object]
ProviderCls = Type[SuspensionProvider]
Registry = dict[str, Tuple[ModelCls, ProviderCls]]


def build_registry() -> Registry:
    """
    Return the plugin registry mapping: type_string -> (ModelClass, ProviderClass)
    """
    return {
        "DOUBLE_WISHBONE": (DoubleWishboneGeometry, DoubleWishboneProvider),
        "MACPHERSON_STRUT": (MacPhersonGeometry, MacPhersonProvider),
    }
