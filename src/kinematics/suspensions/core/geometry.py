"""
Base classes for suspension geometry.

This module defines the abstract base classes that all suspension geometries must
implement. These classes provide the common interface and structure for representing
different types of suspension systems in the kinematics framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kinematics.suspensions.core.settings import SuspensionConfig, Units


@dataclass
class SuspensionGeometry(ABC):
    """
    Base class for all suspension geometry types.

    Attributes:
        name: Human-readable name of the suspension geometry.
        version: Version string of the geometry implementation.
        units: Unit system used for all measurements and calculations.
        configuration: Detailed configuration parameters for the suspension.
    """

    name: str
    version: str
    units: Units
    configuration: SuspensionConfig

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the suspension geometry configuration. This method should perform
        checks to ensure the geometry is properly configured.

        Returns:
            True if the geometry is valid, False otherwise.
        """
        raise NotImplementedError
