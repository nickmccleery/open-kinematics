"""
Derived point specifications and management.
"""

from dataclasses import dataclass
from typing import Callable, Set

import numpy as np

from kinematics.enums import PointID
from kinematics.types import Vec3

# Function signature for computing a derived point position
PositionFn = Callable[[dict[PointID, Vec3]], np.ndarray]


@dataclass(frozen=True)
class DerivedPointsSpec:
    """
    Specification for derived point calculations.

    Contains the functions to compute derived points and their dependencies in a self-
    describing format that can be validated and sorted.
    """

    functions: dict[PointID, PositionFn]
    dependencies: dict[PointID, Set[PointID]]

    def all_points(self) -> Set[PointID]:
        """
        Get all derived point IDs defined in this spec.
        """
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
        """
        Validate the spec after initialization.
        """
        self.validate()


class DerivedPointsManager:
    """
    Manages the calculation of derived points by building and resolving a dependency
    graph to ensure the correct update order.
    """

    def __init__(self, spec: DerivedPointsSpec):
        self.spec = spec
        self.dependency_graph = spec.dependencies

        # This will raise an error if cycles are detected.
        self.update_order = self.get_topological_sort()

    def detect_cycles_util(
        self, node: PointID, visited: set, recursion_stack: set
    ) -> bool:
        visited.add(node)
        recursion_stack.add(node)

        for neighbor in self.dependency_graph.get(node, set()):
            if neighbor not in visited:
                if self.detect_cycles_util(neighbor, visited, recursion_stack):
                    return True
            elif neighbor in recursion_stack:
                return True  # Cycle detected

        recursion_stack.remove(node)
        return False

    def get_topological_sort(self) -> list[PointID]:
        """
        Performs a topological sort of the derived points to determine the correct
        calculation order.

        Raises:
            ValueError: If a circular dependency is detected in the graph.
        """
        visited = set()
        recursion_stack = set()
        nodes = set(self.dependency_graph.keys())

        for node in nodes:
            if node not in visited:
                if self.detect_cycles_util(node, visited, recursion_stack):
                    raise ValueError(
                        "Circular dependency detected in derived point definitions."
                    )

        visited.clear()
        order = []

        def dfs(node: PointID):
            if node in visited:
                return
            visited.add(node)

            # Recurse on dependencies that are also derived points
            for dep in self.dependency_graph.get(node, set()):
                if dep in self.spec.functions:
                    dfs(dep)

            order.append(node)

        for point_id in self.spec.functions:
            if point_id not in visited:
                dfs(point_id)

        return order

    def update_in_place(self, positions: dict[PointID, Vec3]) -> None:
        """
        Compute derived points and add them to positions dict in-place.

        Args:
            positions: Dictionary to mutate in-place by adding derived points.
        """
        for point_id in self.update_order:
            update_func = self.spec.functions[point_id]
            positions[point_id] = update_func(positions)
