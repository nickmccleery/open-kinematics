"""
Suspension configuration classes and enums.

Contains all configuration-related classes for suspension systems.
"""

from __future__ import annotations

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
