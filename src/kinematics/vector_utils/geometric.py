import numpy as np

from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.generic import normalize_vector


def compute_point_point_distance(p1: Vec3, p2: Vec3) -> float:
    return float(np.linalg.norm(p1 - p2))


def compute_point_point_midpoint(p1: Vec3, p2: Vec3) -> Vec3:
    return make_vec3((p1 + p2) / 2)


def compute_vector_vector_angle(v1: Vec3, v2: Vec3) -> float:
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)))
