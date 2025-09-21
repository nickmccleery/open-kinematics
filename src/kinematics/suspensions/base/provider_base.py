from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from kinematics.constraints import Constraint
from kinematics.core import Positions
from kinematics.points.derived.spec import DerivedSpec
from kinematics.points.ids import PointID


class BaseProvider(ABC):
    """Binds a concrete geometry model to initial positions, free points, derived points, and constraints."""

    @abstractmethod
    def initial_positions(self) -> Positions: ...

    @abstractmethod
    def free_points(self) -> Sequence[PointID]:  # stable order
        ...

    @abstractmethod
    def derived_spec(self) -> DerivedSpec: ...

    @abstractmethod
    def constraints(self) -> list[Constraint]: ...
