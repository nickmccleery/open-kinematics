"""
Suspension system implementations and registry.

Contains base classes and the registry for all suspension types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Sequence, Tuple, Type

from kinematics.constraints import Constraint
from kinematics.core import PointID, SuspensionState
from kinematics.points.derived.manager import DerivedSpec


# Configuration classes
class Units(Enum):
    """Units of measurement for geometric parameters."""

    MILLIMETERS = "millimeters"


@dataclass
class WheelConfig:
    """Configuration parameters for a wheel."""

    diameter: float
    width: float
    offset: float


@dataclass
class StaticSetupConfig:
    """Static alignment configuration for a suspension."""

    static_camber: float
    static_toe: float
    static_caster: float


@dataclass
class SuspensionConfig:
    """Complete configuration for a suspension system."""

    steered: bool
    wheel: WheelConfig
    static_setup: StaticSetupConfig


# Base classes
@dataclass
class SuspensionGeometry(ABC):
    """Base class for all suspension geometry types."""

    name: str
    version: str
    units: Units
    configuration: SuspensionConfig

    @abstractmethod
    def validate(self) -> bool:
        """Subclasses must implement this validation method."""
        raise NotImplementedError


class SuspensionProvider(ABC):
    """
    Binds a concrete geometry model to initial positions, free points, derived points,
    and constraints.
    """

    @abstractmethod
    def initial_state(self) -> SuspensionState: ...

    @abstractmethod
    def free_points(self) -> Sequence[PointID]: ...

    @abstractmethod
    def derived_spec(self) -> DerivedSpec: ...

    @abstractmethod
    def constraints(self) -> list[Constraint]: ...


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


# Export the key classes
__all__ = [
    "SuspensionGeometry",
    "SuspensionProvider",
    "Units",
    "WheelConfig",
    "StaticSetupConfig",
    "SuspensionConfig",
    "build_registry",
]
