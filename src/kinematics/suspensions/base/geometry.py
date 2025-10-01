"""
Base classes for suspension geometry.

Contains abstract base classes that all suspension geometries must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kinematics.suspensions.common.configs import SuspensionConfig, Units


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
