"""
Contains the abstract base classes for all suspension geometry models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kinematics.configs import SuspensionConfig, Units


@dataclass
class SuspensionGeometry(ABC):
    """
    Base class for all suspension geometry types.
    """

    name: str
    version: str
    units: Units
    configuration: SuspensionConfig

    @abstractmethod
    def validate(self) -> bool:
        """Subclasses must implement this validation method."""
        raise NotImplementedError
