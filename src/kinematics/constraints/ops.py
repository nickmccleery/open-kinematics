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
    current_point = positions[constraint.point_id]
    line_point = constraint.line_point
    direction = constraint.line_direction  # Don't normalize

    # Vector from line point to current point
    point_to_line = current_point - line_point

    # Use unnormalized cross product to maintain physical scale
    cross_product = np.cross(point_to_line, direction)
    direction_length = np.linalg.norm(direction)

    # Return actual physical distance
    return float(np.linalg.norm(cross_product) / direction_length)
