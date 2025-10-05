"""
Base classes for suspension providers.

This module defines the abstract interface for suspension providers that bind geometry
models to kinematic states, constraints, and derived point calculations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from kinematics.constraints import Constraint
from kinematics.enums import PointID
from kinematics.points.derived.manager import DerivedPointsSpec

if TYPE_CHECKING:
    from kinematics.suspensions.core.geometry import SuspensionGeometry

from kinematics.state import SuspensionState


class SuspensionProvider(ABC):
    """
    Abstract base class for suspension providers.

    Suspension providers bind concrete geometry models to the kinematic framework by
    providing initial states, free points, derived point specifications, and geometric
    constraints.
    """

    def __init__(self, geometry: "SuspensionGeometry") -> None:  # type: ignore[name-defined]
        """
        Base initializer establishes the expected constructor signature for providers.

        Subclasses may override and are not required to call super().__init__. This
        exists primarily to satisfy static type checkers for calls like
        `provider_cls(geometry)`.
        """
        # Intentionally no-op; subclasses typically store geometry themselves.
        ...

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
