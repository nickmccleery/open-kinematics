from abc import ABC, abstractmethod
from typing import Set, Union

import numpy as np

from kinematics.core import PointID
from kinematics.types import Axis
from kinematics.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
)


class Constraint(ABC):
    """Base class for all kinematic constraints."""

    @property
    @abstractmethod
    def involved_points(self) -> Set[PointID]:
        """Returns a set of all PointIDs that this constraint operates on."""
        pass

    @abstractmethod
    def residual(
        self, positions: dict[PointID, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate constraint residual(s).

        Returns either:
        - float: For single-value constraints
        - np.ndarray: For multi-value constraints (rare cases)

        The solver's goal is to drive these values to zero.
        """
        pass


class DistanceConstraint(Constraint):
    """Constrains the Euclidean distance between two points."""

    def __init__(self, p1: PointID, p2: PointID, target_distance: float):
        self.p1 = p1
        self.p2 = p2
        self.target_distance = target_distance

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2}

    def residual(self, positions: dict[PointID, np.ndarray]) -> float:
        current_distance = compute_point_point_distance(
            positions[self.p1], positions[self.p2]
        )
        return float(current_distance - self.target_distance)


class AngleConstraint(Constraint):
    """Constrains the angle between two vectors."""

    def __init__(
        self,
        v1_start: PointID,
        v1_end: PointID,
        v2_start: PointID,
        v2_end: PointID,
        target_angle: float,
    ):
        self.v1_start = v1_start
        self.v1_end = v1_end
        self.v2_start = v2_start
        self.v2_end = v2_end
        self.target_angle = target_angle

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.v1_start, self.v1_end, self.v2_start, self.v2_end}

    def residual(self, positions: dict[PointID, np.ndarray]) -> float:
        v1 = positions[self.v1_end] - positions[self.v1_start]
        v2 = positions[self.v2_end] - positions[self.v2_start]

        current_angle = compute_vector_vector_angle(v1, v2)

        return float(current_angle - self.target_angle)


class FixedAxisConstraint(Constraint):
    """Constrains a point's coordinate on a cardinal axis."""

    def __init__(self, point_id: PointID, axis: Axis, value: float):
        self.point_id = point_id
        self.axis = axis
        self.value = value

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def residual(self, positions: dict[PointID, np.ndarray]) -> float:
        point_coord = positions[self.point_id][self.axis]
        return float(point_coord - self.value)


class PointOnLineConstraint(Constraint):
    """Constrains a point to lie on an arbitrary line."""

    def __init__(
        self, point_id: PointID, line_point: np.ndarray, line_direction: np.ndarray
    ):
        self.point_id = point_id
        self.line_point = line_point.copy()
        self.line_direction = line_direction.copy()

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def residual(self, positions: dict[PointID, np.ndarray]) -> float:
        current_point = positions[self.point_id]

        # Vector from line point to current point
        point_to_line = current_point - self.line_point

        # Distance from point to line using cross product
        cross_product = np.cross(point_to_line, self.line_direction)
        direction_length = np.linalg.norm(self.line_direction)

        # Return actual physical distance
        distance = np.linalg.norm(cross_product) / direction_length
        return float(distance)
