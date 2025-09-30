import numpy as np

from kinematics.core import Position


def compute_point_point_distance(p1: Position, p2: Position) -> float:
    return float(np.linalg.norm(p1 - p2))


def compute_point_point_midpoint(p1: Position, p2: Position) -> Position:
    return (p1 + p2) / 2


def compute_vector_vector_angle(v1: Position, v2: Position) -> float:
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)))


def normalize_vector(v: Position) -> Position:
    return v / np.linalg.norm(v)
