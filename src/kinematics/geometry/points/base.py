from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from kinematics.geometry.points.ids import PointID


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    fixed: bool = False
    id: PointID = PointID.NOT_ASSIGNED

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class DerivedPoint3D(Point3D):
    def get_dependencies(self) -> set[PointID]:
        raise NotImplementedError

    def update(self, points: dict[PointID, Point3D]) -> None:
        raise NotImplementedError


class PointSet:
    """
    Represents a set of points in 3D space.

    Attributes:
        points (dict[PointID, Point3D]): A dictionary mapping point identifiers to their
                                         3D coordinates.
    """

    def __init__(self, points: dict[PointID, Point3D]):
        self.points = points

    def __iter__(self) -> Iterator[Point3D]:
        return iter(self.points.values())

    def __getitem__(self, point_id: PointID) -> Point3D:
        return self.points[point_id]

    def as_array(self) -> np.ndarray:
        """
        Convert point positions to a flat array.
        """
        return np.concatenate([p.as_array() for p in self.points.values()])

    def update_from_array(self, arr: np.ndarray) -> None:
        """
        Update point positions from a flat array.
        """
        i = 0
        for point in self.points.values():
            point.x = float(arr[i])
            point.y = float(arr[i + 1])
            point.z = float(arr[i + 2])
            i += 3


class DerivedPointSet:
    """
    Collection of derived points with automatic dependency resolution.
    """

    def __init__(self, hard_points: dict[PointID, Point3D]):
        self.points: dict[PointID, DerivedPoint3D] = {}
        self.dependency_graph: dict[PointID, set[PointID]] = defaultdict(set)
        self.hard_points = hard_points

    def __getitem__(self, key: PointID) -> DerivedPoint3D:
        return self.points[key]

    def __iter__(self) -> Iterator[DerivedPoint3D]:
        return iter(self.points.values())

    def __len__(self) -> int:
        return len(self.points)

    def __contains__(self, key: PointID) -> bool:
        return key in self.points

    def items(self):
        return self.points.items()

    def values(self):
        return self.points.values()

    def add(self, point: DerivedPoint3D) -> None:
        """
        Add a derived point to the collection and compute its initial position.
        """
        self.points[point.id] = point
        self.dependency_graph[point.id].update(point.get_dependencies())
        self.update_point(point.id, self.hard_points)

    def update(self, hard_points: dict[PointID, Point3D]) -> None:
        """
        Update all derived points based on their dependencies.
        """
        target_order = self.get_update_order()
        for point_id in target_order:
            self.update_point(point_id, hard_points)

    def update_point(
        self,
        point_id: PointID,
        hard_points: dict[PointID, Point3D],
    ) -> None:
        all_points = {**hard_points, **self.points}
        self.points[point_id].update(all_points)

    def detect_cycles(self) -> list[PointID]:
        """
        Find any circular dependencies in the point definitions.
        """
        visited = set()
        path = set()

        # Get all nodes including both derived points and their dependencies.
        nodes = set(self.dependency_graph.keys())

        def dfs(node: PointID) -> bool:
            # If we see a node again in the current path, we have a cycle.
            if node in path:
                return True

            # If we've fully explored this node before, we know it's safe.
            if node in visited:
                return False

            visited.add(node)
            path.add(node)

            # Check all dependencies for cycles, including paths through hard points.
            for neighbor in self.dependency_graph[node]:
                if dfs(neighbor):
                    return True

            # Remove from current path as we backtrack.
            path.remove(node)

            return False

        for node in nodes:
            if dfs(node):
                return list(path)
        return []

    def get_update_order(self) -> list[PointID]:
        # Ensure no circular dependencies before attempting to order.
        if cycle := self.detect_cycles():
            raise ValueError(f"Circular dependency detected: {cycle}")

        visited_nodes = set()
        update_order = []

        # Start with all derived points - we'll follow their dependencies.
        nodes = set(self.points.keys())

        def dfs(node: PointID) -> None:
            # Skip or mark as visited.
            if node in visited_nodes:
                return
            visited_nodes.add(node)

            # Process dependencies first to ensure correct order.
            for dep in self.dependency_graph[node]:
                if dep in self.dependency_graph:
                    dfs(dep)

            # Only add derived points to the update list; hard points managed elsewhere.
            if node not in self.hard_points:
                update_order.append(node)

        for point_id in nodes:
            dfs(point_id)

        return update_order
