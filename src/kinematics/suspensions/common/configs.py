"""
Suspension configuration classes and enums.

This module defines configuration structures for suspension systems, including units,
wheel parameters, and static alignment settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Units(Enum):
    """
    Units of measurement for geometric parameters.
    """

    MILLIMETERS = "millimeters"


@dataclass
class WheelConfig:
    """
    Configuration parameters for a wheel.

    Attributes:
        diameter: Wheel diameter in specified units.
        width: Wheel width in specified units.
        offset: Wheel offset from mounting surface.
    """

    diameter: float
    width: float
    offset: float


@dataclass
class StaticSetupConfig:
    """
    Static alignment configuration for a suspension.

    Attributes:
        static_camber: Static camber angle in degrees.
        static_toe: Static toe angle in degrees.
        static_caster: Static caster angle in degrees.
    """

    static_camber: float
    static_toe: float
    static_caster: float


@dataclass
class SuspensionConfig:
    """
    Complete configuration for a suspension system.

    Attributes:
        steered: Whether this suspension corner is steered.
        wheel: Wheel configuration parameters.
        static_setup: Static alignment settings.
    """

    steered: bool
    wheel: WheelConfig
    static_setup: StaticSetupConfig
