import numpy as np

from kinematics.core import Positions
from kinematics.math import (
    compute_midpoint,
    compute_vector_angle,
    normalize_vector,
    point_distance,
)
from kinematics.points.ids import PointID


def test_point_distance():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    assert point_distance(p1, p2) == 1.0

    p3 = np.array([1.0, 1.0, 1.0])
    assert np.isclose(point_distance(p1, p3), np.sqrt(3.0))


def test_compute_midpoint():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 4.0, 6.0])
    mid = compute_midpoint(p1, p2)
    np.testing.assert_array_equal(mid, np.array([1.0, 2.0, 3.0]))


def test_normalize_vector():
    v = np.array([3.0, 0.0, 4.0])
    norm = normalize_vector(v)
    assert np.isclose(np.linalg.norm(norm), 1.0)
    np.testing.assert_array_almost_equal(norm, np.array([0.6, 0.0, 0.8]))


def test_compute_vector_angle():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    angle = compute_vector_angle(v1, v2)
    assert np.isclose(angle, np.pi / 2)


def test_positions_array_conversion():
    positions_dict = {
        PointID.LOWER_WISHBONE_OUTBOARD: np.array([1.0, 2.0, 3.0]),
        PointID.UPPER_WISHBONE_OUTBOARD: np.array([4.0, 5.0, 6.0]),
    }
    positions = Positions(positions_dict)
    point_ids = [PointID.LOWER_WISHBONE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD]

    # Test extraction
    arr = positions.array(point_ids)
    expected_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(arr, expected_flat)

    # Test update
    new_flat = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    positions.update_from_array(point_ids, new_flat)
    np.testing.assert_array_equal(
        positions[PointID.LOWER_WISHBONE_OUTBOARD], np.array([7.0, 8.0, 9.0])
    )
