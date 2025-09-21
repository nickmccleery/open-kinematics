"""
Derived point manager for handling point calculations with dependency resolution.

Moved from solver/manager.py as it's more logically part of the points subsystem.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

from kinematics.points.main import PointID
from kinematics.primitives import Position, Positions

# A definition consists of the function to call and a set of its dependencies.
DerivedPointDefinition = Tuple[Callable[[Positions], Position], Set[PointID]]


class DerivedPointManager:
    """
    Manages the calculation of derived points by building and resolving a
    dependency graph to ensure the correct update order.
    """

    def __init__(self, definitions: Dict[PointID, DerivedPointDefinition]):
        self.definitions = definitions
        self.dependency_graph = self.build_dependency_graph()

        # This will raise an error if cycles are detected.
        self.update_order = self.get_topological_sort()

    def build_dependency_graph(self) -> Dict[PointID, Set[PointID]]:
        graph = defaultdict(set)
        for point_id, (_, dependencies) in self.definitions.items():
            graph[point_id].update(dependencies)
        return graph

    def _detect_cycles_util(
        self, node: PointID, visited: set, recursion_stack: set
    ) -> bool:
        visited.add(node)
        recursion_stack.add(node)

        for neighbor in self.dependency_graph.get(node, set()):
            if neighbor not in visited:
                if self._detect_cycles_util(neighbor, visited, recursion_stack):
                    return True
            elif neighbor in recursion_stack:
                return True  # Cycle detected

        recursion_stack.remove(node)
        return False

    def get_topological_sort(self) -> List[PointID]:
        """
        Performs a topological sort of the derived points to determine
        the correct calculation order.

        Raises:
            ValueError: If a circular dependency is detected in the graph.
        """
        visited = set()
        recursion_stack = set()
        nodes = set(self.dependency_graph.keys())

        for node in nodes:
            if node not in visited:
                if self._detect_cycles_util(node, visited, recursion_stack):
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
                if dep in self.definitions:
                    dfs(dep)

            order.append(node)

        for point_id in self.definitions:
            if point_id not in visited:
                dfs(point_id)

        return order

    def update(self, positions: Positions) -> Positions:
        """
        Calculates all derived points based on the provided hard points and
        other derived points, respecting the dependency order.

        Args:
            positions: A dictionary containing at least all hard point positions.

        Returns:
            A new dictionary containing both original and all calculated derived points.
        """
        updated_positions = positions.copy()
        for point_id in self.update_order:
            update_func, _ = self.definitions[point_id]
            # The update function is called with the progressively updated dictionary
            updated_positions[point_id] = update_func(updated_positions)
        return updated_positions
