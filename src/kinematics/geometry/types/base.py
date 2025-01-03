from dataclasses import dataclass
from enum import IntEnum

from kinematics.geometry.config.setup import SuspensionConfig
from kinematics.geometry.config.units import Units


class CoordinateAxis(IntEnum):
    X = 0
    Y = 1
    Z = 2


@dataclass
class SuspensionGeometry:
    """
    Base class for all suspension geometry types.
    """

    name: str
    version: str
    units: Units
    configuration: SuspensionConfig

    def validate(self) -> bool:
        raise NotImplementedError("Subclasses must implement validate()")
