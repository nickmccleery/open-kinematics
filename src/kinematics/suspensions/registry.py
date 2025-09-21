"""
Single plugin registry for suspension geometry types.

Eliminates multiple registries by providing one source of truth for
geometry type lookup and instantiation.
"""

from typing import Type

from kinematics.suspensions.base import SuspensionProvider
from kinematics.suspensions.models import DoubleWishboneGeometry, SuspensionGeometry
from kinematics.suspensions.providers.double_wishbone import DoubleWishboneProvider

# Registry maps type string to (Model class, Provider class) tuple
REGISTRY: dict[str, tuple[Type[SuspensionGeometry], Type[SuspensionProvider]]] = {
    "DOUBLE_WISHBONE": (DoubleWishboneGeometry, DoubleWishboneProvider),
    # TODO: Add MacPherson when provider is implemented
    # "MACPHERSON_STRUT": (MacPhersonGeometry, MacPhersonProvider),
}
