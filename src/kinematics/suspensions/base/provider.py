from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from kinematics.constraints import Constraint
from kinematics.core import PointID, SuspensionState
from kinematics.points.derived.manager import DerivedSpec


class SuspensionProvider(ABC):
    """
    Binds a concrete geometry model to initial positions, free points, derived points,
    and constraints.
    """

    @abstractmethod
    def initial_state(self) -> SuspensionState: ...

    @abstractmethod
    def free_points(self) -> Sequence[PointID]: ...

    @abstractmethod
    def derived_spec(self) -> DerivedSpec: ...

    @abstractmethod
    def constraints(self) -> list[Constraint]: ...
