"""
Base classes for suspension providers.

This module defines the abstract interface for suspension providers that bind geometry
models to kinematic states, constraints, and derived point calculations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from kinematics.constraints import Constraint
from kinematics.enums import PointID
from kinematics.points.derived.manager import DerivedPointsSpec
from kinematics.state import SuspensionState


class SuspensionProvider(ABC):
    """
    Abstract base class for suspension providers.

    Suspension providers bind concrete geometry models to the kinematic framework by
    providing initial states, free points, derived point specifications, and geometric
    constraints.
    """

    @abstractmethod
    def initial_state(self) -> SuspensionState:
        """
        Get the initial suspension state.

        Returns:
            The starting configuration of all points in the suspension.
        """
        ...

    @abstractmethod
    def free_points(self) -> Sequence[PointID]:
        """
        Get the points that can move during solving.

        Returns:
            Sequence of point IDs that are free to move.
        """
        ...

    @abstractmethod
    def derived_spec(self) -> DerivedPointsSpec:
        """
        Get the specification for computing derived points.

        Returns:
            Specification defining how derived points are calculated from free points.
        """
        ...

    @abstractmethod
    def constraints(self) -> list[Constraint]:
        """
        Get the geometric constraints for the suspension.

        Returns:
            List of constraints that must be satisfied during solving.
        """
        ...
