"""
Geometric constraints for kinematic systems.

This module defines constraint classes that enforce geometric relationships between
points in suspension kinematics, such as distances, angles, and positional constraints.
Each constraint computes a residual value that the solver attempts to drive to zero.
"""

from abc import ABC, abstractmethod
from typing import Set

import numpy as np

from kinematics.enums import Axis, PointID
from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.geometric import (
    compute_point_point_distance,
    compute_vector_vector_angle,
)


class Constraint(ABC):
    """
    Base class for all kinematics constraints.

    Constraints define geometric relationships that must be satisfied in the kinematic
    system. Each constraint computes a residual value representing the deviation from
    the desired condition. The solver minimizes these residuals to find valid
    configurations.
    """

    @property
    @abstractmethod
    def involved_points(self) -> Set[PointID]:
        """
        Returns a set of all PointIDs that this constraint operates on.
        """
        pass

    @abstractmethod
    def residual(self, positions: dict[PointID, Vec3]) -> float:
        """
        Calculate constraint residual.

        The solver's goal is to drive this value to zero. Returns a scalar float
        representing the constraint violation.
        """
        pass


class DistanceConstraint(Constraint):
    """
    Constrains the Euclidean distance between two points.

    This constraint enforces that the distance between two specified points remains
    constant at a target value, useful for rigid links or fixed separations in the
    suspension geometry.
    """

    def __init__(self, p1: PointID, p2: PointID, target_distance: float):
        """
        Initialize the distance constraint.

        Args:
            p1: First point identifier.
            p2: Second point identifier.
            target_distance: The required distance between the points.
        """
        self.p1 = p1
        self.p2 = p2
        self.target_distance = target_distance

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.p1, self.p2}

    def residual(self, positions: dict[PointID, Vec3]) -> float:
        """
        Compute the distance residual.

        Returns the difference between the current distance and target distance.
        Positive values indicate the points are too far apart.
        """
        current_distance = compute_point_point_distance(
            positions[self.p1], positions[self.p2]
        )
        return float(current_distance - self.target_distance)


class AngleConstraint(Constraint):
    """
    Constrains the angle between two vectors.

    This constraint enforces that the angle formed by two vectors (defined by point
    pairs) remains at a specified target angle. Useful for maintaining joint angles or
    geometric relationships in suspension linkages.
    """

    def __init__(
        self,
        v1_start: PointID,
        v1_end: PointID,
        v2_start: PointID,
        v2_end: PointID,
        target_angle: float,
    ):
        """
        Initialize the angle constraint.

        Args:
            v1_start: Starting point of the first vector.
            v1_end: Ending point of the first vector.
            v2_start: Starting point of the second vector.
            v2_end: Ending point of the second vector.
            target_angle: The required angle between the vectors in radians.
        """
        self.v1_start = v1_start
        self.v1_end = v1_end
        self.v2_start = v2_start
        self.v2_end = v2_end
        self.target_angle = target_angle

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.v1_start, self.v1_end, self.v2_start, self.v2_end}

    def residual(self, positions: dict[PointID, Vec3]) -> float:
        """
        Compute the angle residual.

        Returns the difference between the current angle and target angle in radians.
        Positive values indicate the angle is larger than the target.
        """
        v1 = positions[self.v1_end] - positions[self.v1_start]
        v2 = positions[self.v2_end] - positions[self.v2_start]

        current_angle = compute_vector_vector_angle(make_vec3(v1), make_vec3(v2))

        return float(current_angle - self.target_angle)


class FixedAxisConstraint(Constraint):
    """
    Constrains a point's coordinate on a cardinal axis.

    This constraint fixes a specific coordinate (X, Y, or Z) of a point to a constant
    value, useful for ground-fixed points or symmetry constraints in the suspension
    system.
    """

    def __init__(self, point_id: PointID, axis: Axis, value: float):
        """
        Initialize the fixed axis constraint.

        Args:
            point_id: The point whose coordinate is constrained.
            axis: The axis (X, Y, or Z) to constrain.
            value: The fixed coordinate value on the specified axis.
        """
        self.point_id = point_id
        self.axis = axis
        self.value = value

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def residual(self, positions: dict[PointID, Vec3]) -> float:
        """
        Compute the axis coordinate residual.

        Returns the difference between the current coordinate and the fixed value.
        Positive values indicate the coordinate is above the target.
        """
        point_coord = positions[self.point_id][self.axis]
        return float(point_coord - self.value)


class PointOnLineConstraint(Constraint):
    """
    Constrains a point to lie on an arbitrary line.

    This constraint enforces that a point remains on a specified infinite line defined
    by a point and direction vector. Useful for guiding points along linear paths or
    maintaining alignment in suspension mechanisms.
    """

    def __init__(
        self, point_id: PointID, line_point: np.ndarray, line_direction: np.ndarray
    ):
        """
        Initialize the point-on-line constraint.

        Args:
            point_id: The point that must lie on the line.
            line_point: A point on the line (3D numpy array).
            line_direction: The direction vector of the line (3D numpy array).
        """
        self.point_id = point_id
        self.line_point = line_point.copy()
        self.line_direction = line_direction.copy()

    @property
    def involved_points(self) -> Set[PointID]:
        return {self.point_id}

    def residual(self, positions: dict[PointID, Vec3]) -> float:
        """
        Compute the point-to-line distance residual.

        Returns the perpendicular distance from the point to the line. Zero indicates
        the point lies exactly on the line.
        """
        current_point = positions[self.point_id]

        # Vector from line point to current point.
        point_to_line = current_point - self.line_point

        # Distance from point to line using cross product.
        cross_product = np.cross(point_to_line, self.line_direction)
        direction_length = np.linalg.norm(self.line_direction)

        # Return actual physical distance.
        distance = np.linalg.norm(cross_product) / direction_length
        return float(distance)
