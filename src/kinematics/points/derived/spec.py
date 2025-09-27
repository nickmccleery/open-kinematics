"""
Derived point specifications for explicit derived-point definitions.

Provides DerivedSpec class to make derived-point definitions explicit
and self-describing, replacing loose dicts/tuples.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Set

import numpy as np

from kinematics.points.ids import PointID

# Function signature for computing a derived point position
PositionFn = Callable[[Dict[PointID, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class DerivedSpec:
    """
    Specification for derived point calculations.

    Contains the functions to compute derived points and their dependencies
    in a self-describing format that can be validated and sorted.
    """

    functions: Dict[PointID, PositionFn]
    dependencies: Dict[PointID, Set[PointID]]

    def all_points(self) -> Set[PointID]:
        """Get all derived point IDs defined in this spec."""
        return set(self.functions.keys())

    def validate(self) -> None:
        """
        Validate the spec for consistency.

        Raises:
            ValueError: If spec is inconsistent
        """
        # Check that all points in functions have dependencies defined
        function_points = set(self.functions.keys())
        dependency_points = set(self.dependencies.keys())

        if function_points != dependency_points:
            missing_deps = function_points - dependency_points
            extra_deps = dependency_points - function_points

            msg_parts = []
            if missing_deps:
                msg_parts.append(f"Missing dependencies for: {missing_deps}")
            if extra_deps:
                msg_parts.append(f"Extra dependencies for: {extra_deps}")

            raise ValueError("; ".join(msg_parts))

    def __post_init__(self):
        """Validate the spec after initialization."""
        self.validate()
