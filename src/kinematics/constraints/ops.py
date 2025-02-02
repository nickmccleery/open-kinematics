import numpy as np

from kinematics.constraints.types import (
    PointFixedAxis,
    PointOnLine,
    PointPointDistance,
    VectorAngle,
)
from kinematics.points.ops import normalize_vector
from kinematics.types.state import Positions


def point_point_distance_residual(
    positions: Positions, constraint: PointPointDistance
) -> float:
    p1 = positions[constraint.p1]
    p2 = positions[constraint.p2]
    current_length = float(np.linalg.norm(p1 - p2))
    return current_length - constraint.distance


def vector_angle_residual(positions: Positions, constraint: VectorAngle) -> float:
    v1_start = positions[constraint.v1_start]
    v1_end = positions[constraint.v1_end]
    v2_start = positions[constraint.v2_start]
    v2_end = positions[constraint.v2_end]

    v1 = normalize_vector(v1_end - v1_start)
    v2 = normalize_vector(v2_end - v2_start)

    current_angle = float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return current_angle - constraint.angle


def point_fixed_axis_residual(
    positions: Positions, constraint: PointFixedAxis
) -> float:
    point = positions[constraint.point_id]
    point_coord = point[constraint.axis]
    return point_coord - constraint.value


def point_on_line_residual(positions: Positions, constraint: PointOnLine) -> float:
    # Note that the constraint requires both point and line_point because
    # the line is defined by a point and a direction vector.
    point = positions[constraint.point_id]
    line_point = positions[constraint.line_point]
    line_direction = constraint.line_direction

    # Vector from line_point to point
    point_vector = point - line_point

    # Check if point and line_point are the same
    if np.allclose(point, line_point):
        return 0.0

    # Project point_vector onto line_direction
    projection_length = np.dot(point_vector, line_direction)
    projection_vector = projection_length * line_direction

    # Residual is the distance from the point to the line
    residual_vector = point_vector - projection_vector
    return float(np.linalg.norm(residual_vector))
