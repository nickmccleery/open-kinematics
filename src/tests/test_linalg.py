import numpy as np

from kinematics.linalg import (
    compute_point_point_distance,
    compute_point_point_midpoint,
    compute_vector_vector_angle,
    normalize_vector,
)


def test_point_distance():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    assert compute_point_point_distance(p1, p2) == 1.0

    p3 = np.array([1.0, 1.0, 1.0])
    assert np.isclose(compute_point_point_distance(p1, p3), np.sqrt(3.0))


def test_compute_midpoint():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 4.0, 6.0])
    mid = compute_point_point_midpoint(p1, p2)
    np.testing.assert_array_equal(mid, np.array([1.0, 2.0, 3.0]))


def test_normalize_vector():
    v = np.array([3.0, 0.0, 4.0])
    norm = normalize_vector(v)
    assert np.isclose(np.linalg.norm(norm), 1.0)
    np.testing.assert_array_almost_equal(norm, np.array([0.6, 0.0, 0.8]))


def test_compute_vector_angle():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    angle = compute_vector_vector_angle(v1, v2)
    assert np.isclose(angle, np.pi / 2)
