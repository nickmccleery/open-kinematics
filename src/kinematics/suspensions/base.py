"""
Suspension provider interface and base classes.

Defines the abstract interface for suspension providers that handle
orchestration logic, constraints, and derived point calculations.
"""

from abc import ABC, abstractmethod
from typing import List, Set

from kinematics.core.positions import Positions
from kinematics.points.derived.spec import DerivedSpec
from kinematics.suspensions.models import SuspensionGeometry


class SuspensionProvider(ABC):
    """
    Abstract base class defining the interface for a suspension type.

    Each implementation provides the specific rules (constraints, free points,
    derived point definitions, etc.) for its corresponding geometry.
    """

    def __init__(self, geometry: SuspensionGeometry):
        self.geometry = geometry

    @abstractmethod
    def get_initial_positions(self) -> Positions:
        """Constructs the initial Positions object from the geometry."""
        ...

    @abstractmethod
    def get_free_points(self) -> Set:
        """Returns the set of PointIDs that the solver is allowed to move."""
        ...

    @abstractmethod
    def get_constraints(self, initial_positions: Positions) -> List:
        """Returns the full list of constraints that define the suspension's movement."""
        ...

    @abstractmethod
    def get_derived_point_definitions(self) -> DerivedSpec:
        """
        Returns a DerivedSpec containing derived point definitions and dependencies.
        """
        ...
