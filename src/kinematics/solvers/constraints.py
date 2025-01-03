from typing import Tuple

import numpy as np

from kinematics.geometry.constants import CoordinateAxis
from kinematics.geometry.points.base import Point3D
from kinematics.geometry.points.ids import PointID


class BaseConstraint:
    """
    A base class for defining constraints in the system to be solved.

    Methods:
        compute_residual(points: dict[PointID, Point3D]) -> float:
            Computes the residual for the constraint given a dictionary of points.
            This method must be implemented by subclasses.
    """

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        raise NotImplementedError()


class PointPointDistanceConstraint(BaseConstraint):
    """
    A constraint that ensures the distance between two points remains constant.

    Attributes:
        p1 (PointID): The identifier for the first point.
        p2 (PointID): The identifier for the second point.
        distance (float): The desired distance between the two points.

    Methods:
        compute_residual(points: dict[PointID, Point3D]) -> float:
            Computes the residual of the constraint, which is the difference
            between the current distance of the points and the desired distance.

            Args:
                points (dict[PointID, Point3D]): A dictionary mapping point identifiers
                                                 to their 3D coordinates.

            Returns:
                float: The residual value of the constraint.
    """

    def __init__(self, p1: PointID, p2: PointID, distance: float):
        self.p1 = p1
        self.p2 = p2
        self.distance = distance

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        p1 = points[self.p1].as_array()
        p2 = points[self.p2].as_array()
        current_length = float(np.linalg.norm(p1 - p2))
        return current_length - self.distance


class VectorVectorAngleConstraint(BaseConstraint):
    """
    A constraint that ensures the orientation between two vectors remains constant.

    Attributes:
        v1 (Tuple[PointID, PointID]): The identifiers for the first vector.
        v2 (Tuple[PointID, PointID]): The identifiers for the second vector.
        angle (float): The desired angle between the two vectors.

    Methods:
        compute_residual(points: dict[PointID, Point3D]) -> float:
            Computes the residual of the constraint, which is the difference
            between the current angle of the vectors and the desired angle.

            Args:
                points (dict[PointID, Point3D]): A dictionary mapping point identifiers
                                                 to their 3D coordinates.

            Returns:
                float: The residual value of the constraint.
    """

    def __init__(
        self,
        v1: Tuple[PointID, PointID],
        v2: Tuple[PointID, PointID],
        angle: float,
    ):
        self.v1 = v1
        self.v2 = v2
        self.angle = angle

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        v1_start = points[self.v1[0]].as_array()
        v1_end = points[self.v1[1]].as_array()
        v2_start = points[self.v2[0]].as_array()
        v2_end = points[self.v2[1]].as_array()

        v1 = v1_end - v1_start
        v2 = v2_end - v2_start

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        current_angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return current_angle - self.angle


class PointFixedAxisConstraint(BaseConstraint):
    """
    A constraint that fixes a point on a specified axis.

    Attributes:
        point_id (PointID): The identifier for the point.
        axis (Literal["x", "y", "z"]): The axis on which the point is fixed.
        value (float): The fixed value on the specified axis.

    Methods:
        compute_residual(points: dict[PointID, Point3D]) -> float:
            Computes the residual of the constraint, which is the difference
            between the current coordinate of the point on the specified axis
            and the fixed value.

            Args:
                points (dict[PointID, Point3D]): A dictionary mapping point identifiers
                                                 to their 3D coordinates.

            Returns:
                float: The residual value of the constraint.
    """

    def __init__(
        self,
        point_id: PointID,
        axis: CoordinateAxis,
        value: float,
    ):
        self.point_id = point_id
        self.axis = axis
        self.value = value

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        point = points[self.point_id]
        point_coord = point.as_array()[self.axis]
        return point_coord - self.value


class PointOnLineConstraint(BaseConstraint):
    """
    A constraint that ensures a point moves along a specified line.

    Attributes:
        point_id (PointID): The identifier for the point.
        line_point (PointID): The identifier for a point on the line.
        line_direction (np.ndarray): The direction vector of the line.

    Methods:
        compute_residual(points: dict[PointID, Point3D]) -> float:
            Computes the residual of the constraint, which is the distance
            from the point to the line.

            Args:
                points (dict[PointID, Point3D]): A dictionary mapping point identifiers
                                                 to their 3D coordinates.

            Returns:
                float: The residual value of the constraint.
    """

    def __init__(
        self, point_id: PointID, line_point: PointID, line_direction: np.ndarray
    ):
        self.point_id = point_id
        self.line_point = line_point
        self.line_direction = line_direction / np.linalg.norm(line_direction)

    def compute_residual(self, points: dict[PointID, Point3D]) -> float:
        point = points[self.point_id].as_array()
        line_point = points[self.line_point].as_array()
        line_direction = self.line_direction

        # Vector from line_point to point.
        point_vector = point - line_point

        # Project point_vector onto line_direction.
        projection_length = np.dot(point_vector, line_direction)
        projection_vector = projection_length * line_direction

        # Residual is the distance from the point to the line.
        residual_vector = point_vector - projection_vector
        residual = float(np.linalg.norm(residual_vector))

        return residual


# Utility methods to create constraints.
def lock_point_point_distance(p1: Point3D, p2: Point3D):
    length = float(np.linalg.norm(p1.as_array() - p2.as_array()))
    return PointPointDistanceConstraint(p1.id, p2.id, length)


def lock_vector_angles(v1: tuple[Point3D, Point3D], v2: tuple[Point3D, Point3D]):
    v1_vec = v1[1].as_array() - v1[0].as_array()
    v2_vec = v2[1].as_array() - v2[0].as_array()

    v1_vec = v1_vec / np.linalg.norm(v1_vec)
    v2_vec = v2_vec / np.linalg.norm(v2_vec)

    theta = np.arccos(np.clip(np.dot(v1_vec, v2_vec), -1.0, 1.0))

    return VectorVectorAngleConstraint(
        v1=(v1[0].id, v1[1].id), v2=(v2[0].id, v2[1].id), angle=theta
    )
