from __future__ import annotations

from typing import Dict, Tuple, Type

from kinematics.suspensions.base.provider import SuspensionProvider

# Type alias for clarity; concrete Model classes are architecture-specific dataclasses
ModelCls = Type[object]
ProviderCls = Type[SuspensionProvider]
Registry = Dict[str, Tuple[ModelCls, ProviderCls]]


def build_registry() -> Registry:
    """
    Return the plugin registry mapping:
        type_string -> (ModelClass, ProviderClass)
    No imports occur at module import time; all imports are local to avoid cycles and side-effects.
    """
    # Local imports keep this file import-time cheap and avoid accidental cycles
    from kinematics.suspensions.double_wishbone.model import DoubleWishboneGeometry
    from kinematics.suspensions.double_wishbone.provider import DoubleWishboneProvider

    # Add more architectures here:
    from kinematics.suspensions.macpherson.model import MacPhersonGeometry
    from kinematics.suspensions.macpherson.provider import MacPhersonProvider

    return {
        "DOUBLE_WISHBONE": (DoubleWishboneGeometry, DoubleWishboneProvider),
        "MACPHERSON_STRUT": (MacPhersonGeometry, MacPhersonProvider),
    }
