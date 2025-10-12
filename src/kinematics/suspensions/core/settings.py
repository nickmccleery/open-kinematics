"""
Suspension configuration classes and enums.

This module defines configuration structures for suspension systems, including units,
wheel parameters, and static alignment settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from kinematics.constants import MM_PER_INCH


@dataclass
class TireConfig:
    """
    Configuration parameters for a tire.

    Attributes:
        aspect_ratio: Aspect ratio, expressed as a fraction in the range [0, 1], e.g., 0.55 for 55%.
        section_width: Section width in mm.
        rim_diameter: Rim diameter in inches.
    """

    aspect_ratio: float
    section_width: float
    rim_diameter: float

    @property
    def sidewall_height(self) -> float:
        """
        Calculate sidewall height in mm.
        """
        return self.aspect_ratio * self.section_width

    @property
    def rim_diameter_mm(self) -> float:
        """
        Convert rim diameter from inches to mm.
        """
        return self.rim_diameter * MM_PER_INCH

    @property
    def nominal_radius(self) -> float:
        """
        Calculate nominal tire radius in mm.

        This makes no consideration of vertical load or speed growth effects.
        """
        return (self.rim_diameter_mm + 2 * self.sidewall_height) / 2


@dataclass
class WheelConfig:
    """
    Configuration parameters for a wheel and tire assembly.

    Attributes:
        offset: Wheel offset from mounting surface in mm.
        tire: Tire configuration parameters.
    """

    offset: float
    tire: TireConfig


@dataclass
class SuspensionConfig:
    """
    Complete configuration for a suspension system.

    Attributes:
        steered: Whether this suspension corner is steered.
        wheel: Wheel configuration parameters.
        cg_position: Center of gravity position coordinates (x, y, z) in mm (required for anti-dive/squat).
        wheelbase: Wheelbase distance in mm.
    """

    steered: bool
    wheel: WheelConfig
    cg_position: dict[str, float]
    wheelbase: float
