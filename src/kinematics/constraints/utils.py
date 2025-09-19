import numpy as np

from kinematics.constraints.types import PointPointDistance, VectorAngle
from kinematics.geometry.points.ids import PointID
from kinematics.math import normalize_vector
from kinematics.core.types import Positions


def make_point_point_distance(
    positions: Positions, p1: PointID, p2: PointID
) -> PointPointDistance:
    distance = float(np.linalg.norm(positions[p1] - positions[p2]))
    return PointPointDistance(p1=p1, p2=p2, distance=distance)


def make_vector_angle(
    positions: Positions,
    v1_start: PointID,
    v1_end: PointID,
    v2_start: PointID,
    v2_end: PointID,
) -> VectorAngle:
    v1 = normalize_vector(positions[v1_end] - positions[v1_start])
    v2 = normalize_vector(positions[v2_end] - positions[v2_start])

    angle = float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    return VectorAngle(
        v1_start=v1_start, v1_end=v1_end, v2_start=v2_start, v2_end=v2_end, angle=angle
    )
