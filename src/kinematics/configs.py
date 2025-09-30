"""
Configuration structures for suspension systems.

Contains classes and enums for configuring suspension geometry,
wheel parameters, and static setup characteristics.
"""

from dataclasses import dataclass
from enum import Enum


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
