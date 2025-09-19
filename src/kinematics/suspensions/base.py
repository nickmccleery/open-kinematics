from abc import ABC, abstractmethod
from typing import Dict, List, Set

from kinematics.constraints.types import Constraint
from kinematics.core.types import Positions
from kinematics.geometry.base import SuspensionGeometry
from kinematics.geometry.points.ids import PointID
from kinematics.solver.manager import DerivedPointDefinition


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
        """Constructs the initial dictionary of 3D point positions from the geometry."""
        ...

    @abstractmethod
    def get_free_points(self) -> Set[PointID]:
        """Returns the set of PointIDs that the solver is allowed to move."""
        ...

    @abstractmethod
    def get_constraints(self, initial_positions: Positions) -> List[Constraint]:
        """Returns the full list of constraints that define the suspension's movement."""
        ...

    @abstractmethod
    def get_derived_point_definitions(self) -> Dict[PointID, DerivedPointDefinition]:
        """
        Returns a dictionary mapping derived PointIDs to their update function
        and dependencies.
        """
        ...
