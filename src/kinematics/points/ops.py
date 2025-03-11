from typing import Dict

import numpy as np

from kinematics.geometry.points.ids import PointID
from kinematics.types.state import Position, Positions


def point_distance(p1: Position, p2: Position) -> float:
    return float(np.linalg.norm(p1 - p2))


def compute_midpoint(p1: Position, p2: Position) -> Position:
    return (p1 + p2) / 2


def normalize_vector(v: Position) -> Position:
    return v / np.linalg.norm(v)


def compute_vector_angle(v1: Position, v2: Position) -> float:
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)))


def get_positions_as_array(
    positions: Dict[PointID, Position],
    point_ids: list[PointID],
) -> np.ndarray:
    """
    Args:
        positions: Dictionary mapping point IDs to their positions.
        point_ids: List of point IDs to extract.

    Returns:
        Array of shape (n, 3) containing the positions in order.
    """
    return np.array([positions[pid] for pid in point_ids])


def update_positions_from_array(
    positions: Positions,
    point_ids: list[PointID],
    array: np.ndarray,
) -> None:
    """
    Args:
        positions: Dictionary to update with new positions.
        point_ids: List of point IDs to update.
        array: Array of shape (n, 3) containing new positions.
    """
    for i, pid in enumerate(point_ids):
        positions[pid] = array[i].copy()
