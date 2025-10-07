import math

import numpy as np
import pytest

from kinematics.constants import EPSILON
from kinematics.vector_utils.geometric import (
    compute_point_point_distance,
    compute_point_point_midpoint,
    compute_point_to_line_distance,
    compute_point_to_plane_distance,
    compute_scalar_triple_product,
    compute_vector_vector_angle,
    compute_vectors_cross_product_magnitude,
    compute_vectors_dot_product,
)


def simple_positions():
    """
    Returns a dictionary of simple coordinate positions for testing.
    """
    return {
        "origin": np.array([0.0, 0.0, 0.0]),
        "x_unit": np.array([1.0, 0.0, 0.0]),
        "y_unit": np.array([0.0, 1.0, 0.0]),
        "z_unit": np.array([0.0, 0.0, 1.0]),
        "diagonal_xy": np.array([1.0, 1.0, 0.0]),
        "diagonal_xyz": np.array([1.0, 1.0, 1.0]),
    }


def test_point_distance_unit_vectors():
    """
    Test distances between points on coordinate axes.
    """
    pos = simple_positions()

    # Distance from origin to unit vector offset positions should be 1.
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["x_unit"]), 1.0, abs_tol=EPSILON
    )
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["y_unit"]), 1.0, abs_tol=EPSILON
    )
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["z_unit"]), 1.0, abs_tol=EPSILON
    )

    # Distance between orthogonal unit vectors should be sqrt(2).
    assert math.isclose(
        compute_point_point_distance(pos["x_unit"], pos["y_unit"]),
        math.sqrt(2),
        abs_tol=EPSILON,
    )

    # Distance from origin to (1,1,1) should be sqrt(3).
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["diagonal_xyz"]),
        math.sqrt(3),
        abs_tol=EPSILON,
    )


def test_point_distance_zero():
    """
    Test distance from a point to itself is zero.
    """
    pos = simple_positions()
    assert math.isclose(
        compute_point_point_distance(pos["origin"], pos["origin"]),
        0.0,
        abs_tol=EPSILON,
    )


def test_compute_midpoint_coordinate_axes():
    """
    Test midpoints using simple coordinate positions.
    """
    pos = simple_positions()

    # Midpoint between origin and x_unit should be (0.5, 0, 0).
    mid = compute_point_point_midpoint(pos["origin"], pos["x_unit"])
    expected = np.array([0.5, 0.0, 0.0])
    np.testing.assert_allclose(mid, expected, atol=EPSILON)

    # Midpoint between x_unit and y_unit should be (0.5, 0.5, 0).
    mid = compute_point_point_midpoint(pos["x_unit"], pos["y_unit"])
    expected = np.array([0.5, 0.5, 0.0])
    np.testing.assert_allclose(mid, expected, atol=EPSILON)


def test_compute_midpoint_diagonal():
    """
    Test midpoint calculation with diagonal vectors.
    """
    pos = simple_positions()

    # Midpoint between origin and (1,1,1) should be (0.5, 0.5, 0.5).
    mid = compute_point_point_midpoint(pos["origin"], pos["diagonal_xyz"])
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(mid, expected, atol=EPSILON)


def test_compute_vector_angle_perpendicular():
    """
    Test angle between perpendicular vectors is pi/2.
    """
    pos = simple_positions()

    # X and Y axes are perpendicular.
    angle = compute_vector_vector_angle(pos["x_unit"], pos["y_unit"])
    assert math.isclose(angle, math.pi / 2, abs_tol=EPSILON)

    # Y and Z axes are perpendicular.
    angle = compute_vector_vector_angle(pos["y_unit"], pos["z_unit"])
    assert math.isclose(angle, math.pi / 2, abs_tol=EPSILON)


def test_compute_vector_angle_parallel():
    """
    Test angle between parallel vectors is 0.
    """
    pos = simple_positions()

    # Vector with itself.
    angle = compute_vector_vector_angle(pos["x_unit"], pos["x_unit"])
    assert math.isclose(angle, 0.0, abs_tol=EPSILON)

    # Parallel vectors (same direction).
    v1 = np.array([2.0, 0.0, 0.0])  # 2x along X.
    angle = compute_vector_vector_angle(pos["x_unit"], v1)
    assert math.isclose(angle, 0.0, abs_tol=EPSILON)


def test_compute_vector_angle_antiparallel():
    """
    Test angle between anti-parallel vectors is pi.
    """
    pos = simple_positions()

    # Opposite directions along X axis.
    v_neg_x = np.array([-1.0, 0.0, 0.0])
    angle = compute_vector_vector_angle(pos["x_unit"], v_neg_x)
    assert math.isclose(angle, math.pi, abs_tol=EPSILON)


def test_compute_vector_angle_45_degrees():
    """
    Test angle for 45-degree case.
    """
    pos = simple_positions()

    # X unit vector and diagonal (1,1,0) should form 45 degrees.
    angle = compute_vector_vector_angle(pos["x_unit"], pos["diagonal_xy"])
    assert math.isclose(angle, math.pi / 4, abs_tol=EPSILON)


def test_compute_vector_angle_zero_vector():
    """
    Test that zero vectors raise ValueError.
    """
    pos = simple_positions()
    zero_vec = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        compute_vector_vector_angle(pos["x_unit"], zero_vec)

    with pytest.raises(ValueError):
        compute_vector_vector_angle(zero_vec, pos["y_unit"])


def test_cross_product_magnitude_perpendicular():
    """
    Test cross product magnitude for perpendicular vectors is 1.
    """
    pos = simple_positions()

    # Perpendicular unit vectors have cross product magnitude 1.
    cross_mag = compute_vectors_cross_product_magnitude(pos["x_unit"], pos["y_unit"])
    assert math.isclose(cross_mag, 1.0, abs_tol=EPSILON)


def test_cross_product_magnitude_parallel():
    """
    Test cross product magnitude for parallel vectors is 0.
    """
    pos = simple_positions()

    # Parallel vectors have cross product magnitude 0.
    v_parallel = np.array([2.0, 0.0, 0.0])  # Parallel to x_unit
    cross_mag = compute_vectors_cross_product_magnitude(pos["x_unit"], v_parallel)
    assert math.isclose(cross_mag, 0.0, abs_tol=EPSILON)


def test_cross_product_magnitude_45_degrees():
    """
    Test cross product magnitude for 45-degree angle.
    """
    pos = simple_positions()

    # 45 degree angle should give cross product magnitude sin(pi/4) = sqrt(2)/2
    cross_mag = compute_vectors_cross_product_magnitude(
        pos["x_unit"], pos["diagonal_xy"]
    )
    expected = math.sin(math.pi / 4)  # sqrt(2)/2 ~= 0.707
    assert math.isclose(cross_mag, expected, abs_tol=EPSILON)


def test_dot_product_perpendicular():
    """
    Test dot product for perpendicular vectors is 0.
    """
    pos = simple_positions()

    # Perpendicular unit vectors have dot product 0.
    dot = compute_vectors_dot_product(pos["x_unit"], pos["y_unit"])
    assert math.isclose(dot, 0.0, abs_tol=EPSILON)


def test_dot_product_parallel():
    """
    Test dot product for parallel vectors is 1.
    """
    pos = simple_positions()

    # Parallel unit vectors have dot product 1.
    v_parallel = np.array([2.0, 0.0, 0.0])  # Parallel to x_unit
    dot = compute_vectors_dot_product(pos["x_unit"], v_parallel)
    assert math.isclose(dot, 1.0, abs_tol=EPSILON)


def test_dot_product_antiparallel():
    """
    Test dot product for anti-parallel vectors is -1.
    """
    pos = simple_positions()

    # Anti-parallel unit vectors have dot product -1.
    v_antiparallel = np.array([-1.0, 0.0, 0.0])
    dot = compute_vectors_dot_product(pos["x_unit"], v_antiparallel)
    assert math.isclose(dot, -1.0, abs_tol=EPSILON)


def test_dot_product_45_degrees():
    """
    Test dot product for 45-degree angle.
    """
    pos = simple_positions()

    # 45 degree angle should give dot product cos(pi/4) = sqrt(2)/2
    dot = compute_vectors_dot_product(pos["x_unit"], pos["diagonal_xy"])
    expected = math.cos(math.pi / 4)  # sqrt(2)/2 ~= 0.707
    assert math.isclose(dot, expected, abs_tol=EPSILON)


def test_point_to_line_distance_on_line():
    """
    Test distance from point on line to line is 0.
    """
    pos = simple_positions()

    # Point on the line (Y axis) should have distance 0.
    line_point = pos["origin"]
    line_direction = pos["y_unit"]
    test_point = np.array([0.0, 2.0, 0.0])  # On Y axis

    distance = compute_point_to_line_distance(test_point, line_point, line_direction)
    assert math.isclose(distance, 0.0, abs_tol=EPSILON)


def test_point_to_line_distance_perpendicular():
    """
    Test distance from point perpendicular to line.
    """
    pos = simple_positions()

    # Point (1,0,0) to Y axis should have distance 1.
    line_point = pos["origin"]
    line_direction = pos["y_unit"]
    test_point = pos["x_unit"]

    distance = compute_point_to_line_distance(test_point, line_point, line_direction)
    assert math.isclose(distance, 1.0, abs_tol=EPSILON)


def test_point_to_line_distance_diagonal():
    """
    Test distance calculation with diagonal geometry.
    """
    pos = simple_positions()

    # Point (1,1,0) to X axis should have distance 1 (Y component).
    line_point = pos["origin"]
    line_direction = pos["x_unit"]
    test_point = pos["diagonal_xy"]

    distance = compute_point_to_line_distance(test_point, line_point, line_direction)
    assert math.isclose(distance, 1.0, abs_tol=EPSILON)


def test_point_to_plane_distance_on_plane():
    """
    Test distance from point on plane to plane is 0.
    """
    pos = simple_positions()

    # Point on Z=0 plane should have distance 0.
    plane_point = pos["origin"]
    plane_normal = pos["z_unit"]
    test_point = pos["diagonal_xy"]  # (1,1,0) - on Z=0 plane

    distance = compute_point_to_plane_distance(test_point, plane_point, plane_normal)
    assert math.isclose(distance, 0.0, abs_tol=EPSILON)


def test_point_to_plane_distance_positive():
    """
    Test positive signed distance to plane.
    """
    pos = simple_positions()

    # Point above Z=0 plane should have positive distance.
    plane_point = pos["origin"]
    plane_normal = pos["z_unit"]
    test_point = np.array([0.0, 0.0, 2.0])  # 2 units above Z=0.

    distance = compute_point_to_plane_distance(test_point, plane_point, plane_normal)
    assert math.isclose(distance, 2.0, abs_tol=EPSILON)


def test_point_to_plane_distance_negative():
    """
    Test negative signed distance to plane.
    """
    pos = simple_positions()

    # Point below Z=0 plane should have negative distance.
    plane_point = pos["origin"]
    plane_normal = pos["z_unit"]
    test_point = np.array([0.0, 0.0, -1.5])  # 1.5 units below Z=0

    distance = compute_point_to_plane_distance(test_point, plane_point, plane_normal)
    assert math.isclose(distance, -1.5, abs_tol=EPSILON)


def test_scalar_triple_product_right_handed_basis():
    """
    Test scalar triple product for right-handed unit basis is 1.
    """
    pos = simple_positions()

    # X dot (Y cross Z) = 1 for right-handed coordinate system.
    triple_product = compute_scalar_triple_product(
        pos["x_unit"],
        pos["y_unit"],
        pos["z_unit"],
    )
    assert math.isclose(triple_product, 1.0, abs_tol=EPSILON)


def test_scalar_triple_product_left_handed_basis():
    """
    Test scalar triple product for left-handed basis is -1.
    """
    pos = simple_positions()

    # X dot (Z cross Y) = -1 (reversed order).
    triple_product = compute_scalar_triple_product(
        pos["x_unit"],
        pos["z_unit"],
        pos["y_unit"],
    )
    assert math.isclose(triple_product, -1.0, abs_tol=EPSILON)


def test_scalar_triple_product_coplanar():
    """
    Test scalar triple product for coplanar vectors is 0.
    """
    pos = simple_positions()

    # Three vectors in XY plane should have triple product 0.
    v1 = pos["x_unit"]
    v2 = pos["y_unit"]
    v3 = pos["diagonal_xy"]  # Also in XY plane

    triple_product = compute_scalar_triple_product(v1, v2, v3)
    assert math.isclose(triple_product, 0.0, abs_tol=EPSILON)


def test_zero_vector_errors():
    """
    Test that functions properly raise errors for zero-length vectors.
    """
    zero_vec = np.array([0.0, 0.0, 0.0])
    unit_vec = np.array([1.0, 0.0, 0.0])

    # These functions should raise ValueError for zero vectors.
    with pytest.raises(ValueError):
        compute_vectors_cross_product_magnitude(zero_vec, unit_vec)

    with pytest.raises(ValueError):
        compute_vectors_dot_product(unit_vec, zero_vec)

    with pytest.raises(ValueError):
        compute_point_to_line_distance(unit_vec, unit_vec, zero_vec)

    with pytest.raises(ValueError):
        compute_point_to_plane_distance(unit_vec, unit_vec, zero_vec)
