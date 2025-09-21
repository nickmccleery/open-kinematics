from abc import ABC, abstractmethod
from typing import Set, Union

import numpy as np
from numpy.typing import NDArray

from kinematics.math import compute_vector_angle, normalize_vector, point_distance
from kinematics.points.main import PointID
from kinematics.primitives import CoordinateAxis, Position, Positions


class BaseConstraint(ABC):
    """
    Abstract base class for all kinematic constraints.

    Each constraint must be able to calculate its residual error and report
    which points its calculations depend on.
    """

    @property
    @abstractmethod
    def involved_points(self) -> Set[PointID]:
        """Returns a set of all PointIDs that this constraint operates on."""
        ...

    @abstractmethod
    def get_residual(self, positions: Positions) -> np.ndarray:
        """
        Calculates the residual error for the constraint.

        The solver's goal is to drive this value to zero.
        Returns a 1D numpy array, as some constraints may have multiple residuals.
        """
        ...


class PointPointDistance(BaseConstraint):
    """Constrains the Euclidean distance between two points."""

    def __init__(self, p1: PointID, p2: PointID, distance: float):
        self.p1 = p1
        self.p2 = p2
        self.distance = distance

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2}

    def get_residual(self, positions: Positions) -> np.ndarray:
        current_distance = point_distance(positions[self.p1], positions[self.p2])
        return np.array([current_distance - self.distance])


class VectorAngle(BaseConstraint):
    """Constrains the angle between two vectors."""

    def __init__(
        self,
        v1_start: PointID,
        v1_end: PointID,
        v2_start: PointID,
        v2_end: PointID,
        angle: float,
    ):
        self.v1_start = v1_start
        self.v1_end = v1_end
        self.v2_start = v2_start
        self.v2_end = v2_end
        self.angle = angle

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.v1_start, self.v1_end, self.v2_start, self.v2_end}

    def get_residual(self, positions: Positions) -> np.ndarray:
        v1 = positions[self.v1_end] - positions[self.v1_start]
        v2 = positions[self.v2_end] - positions[self.v2_start]
        current_angle = compute_vector_angle(v1, v2)
        return np.array([current_angle - self.angle])


class PointFixedAxis(BaseConstraint):
    """Constrains a point's coordinate on a cardinal axis."""

    def __init__(self, point_id: PointID, axis: CoordinateAxis, value: float):
        self.point_id = point_id
        self.axis = axis
        self.value = value

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def get_residual(self, positions: Positions) -> np.ndarray:
        point_coord = positions[self.point_id][self.axis]
        return np.array([point_coord - self.value])


class PointOnLine(BaseConstraint):
    """Constrains a point to lie on an arbitrary vector."""

    def __init__(
        self, point_id: PointID, line_point: Position, line_direction: NDArray
    ):
        self.point_id = point_id
        self.line_point = line_point
        # Store the direction vector as-is (matching the original implementation)
        self.line_direction = line_direction

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def get_residual(self, positions: Positions) -> np.ndarray:
        current_point = positions[self.point_id]
        line_point = self.line_point
        direction = self.line_direction  # Don't normalize

        # Vector from line point to current point
        point_to_line = current_point - line_point

        # Use unnormalized cross product to maintain physical scale
        cross_product = np.cross(point_to_line, direction)
        direction_length = np.linalg.norm(direction)

        # Return actual physical distance
        distance = np.linalg.norm(cross_product) / direction_length
        return np.array([distance])


# Union type for all constraints
Constraint = Union[PointPointDistance, VectorAngle, PointFixedAxis, PointOnLine]


# Factory functions
def make_point_point_distance(
    positions: Positions, p1: PointID, p2: PointID
) -> PointPointDistance:
    """Factory function to create a PointPointDistance constraint from the initial state."""
    distance = float(np.linalg.norm(positions[p1] - positions[p2]))
    return PointPointDistance(p1=p1, p2=p2, distance=distance)


def make_vector_angle(
    positions: Positions,
    v1_start: PointID,
    v1_end: PointID,
    v2_start: PointID,
    v2_end: PointID,
) -> VectorAngle:
    """Factory function to create a VectorAngle constraint from the initial state."""
    v1 = normalize_vector(positions[v1_end] - positions[v1_start])
    v2 = normalize_vector(positions[v2_end] - positions[v2_start])
    angle = float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return VectorAngle(
        v1_start=v1_start, v1_end=v1_end, v2_start=v2_start, v2_end=v2_end, angle=angle
    )
