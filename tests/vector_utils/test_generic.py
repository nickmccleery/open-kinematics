"""
Tests for the generic vector utility functions.
"""

import numpy as np
import pytest

from kinematics.constants import EPSILON, TEST_TOLERANCE
from kinematics.vector_utils.generic import normalize_vector, project_coordinate


class TestNormalizeVector:
    """
    Tests for the normalize_vector function.
    """

    def test_normalize_unit_vector_x(self):
        """
        Test normalizing a unit vector in x direction.
        """
        v = np.array([1.0, 0.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_unit_vector_y(self):
        """
        Test normalizing a unit vector in y direction.
        """
        v = np.array([0.0, 1.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_unit_vector_z(self):
        """
        Test normalizing a unit vector in z direction.
        """
        v = np.array([0.0, 0.0, 1.0])
        result = normalize_vector(v)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_scaled_vector(self):
        """
        Test normalizing a scaled vector.
        """
        v = np.array([3.0, 4.0, 0.0])  # Magnitude = 5.
        result = normalize_vector(v)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_3d_vector(self):
        """
        Test normalizing a 3D vector.
        """
        v = np.array([1.0, 2.0, 2.0])  # Magnitude = 3.
        result = normalize_vector(v)
        expected = np.array([1 / 3, 2 / 3, 2 / 3])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_negative_vector(self):
        """
        Test normalizing a vector with negative components.
        """
        v = np.array([-3.0, -4.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([-0.6, -0.8, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_mixed_sign_vector(self):
        """
        Test normalizing a vector with mixed positive/negative components.
        """
        v = np.array([3.0, -4.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([0.6, -0.8, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_2d_vector(self):
        """
        Test normalizing a 2D vector.
        """
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_higher_dimension_vector(self):
        """
        Test normalizing a higher dimensional vector.
        """
        v = np.array([1.0, 1.0, 1.0, 1.0])  # Magnitude = 2.
        result = normalize_vector(v)
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_very_small_vector(self):
        """
        Test normalizing a very small but non-zero vector.
        """
        v = np.array([TEST_TOLERANCE * 10, 0.0, 0.0])
        result = normalize_vector(v)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=TEST_TOLERANCE)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=TEST_TOLERANCE)

    def test_normalize_zero_vector_raises_error(self):
        """
        Test that normalizing a zero vector raises ValueError.
        """
        v = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot normalize zero-length vector"):
            normalize_vector(v)

    def test_normalize_near_zero_vector_raises_error(self):
        """
        Test that normalizing a near-zero vector raises ValueError.
        """
        v = np.array([EPSILON / 2, 0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot normalize zero-length vector"):
            normalize_vector(v)

    def test_normalize_returns_float64(self):
        """
        Test that the result is of type float64.
        """
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = normalize_vector(v)
        assert result.dtype == np.float64


class TestProjectCoordinate:
    """
    Tests for the project_coordinate function.
    """

    def test_project_along_x_axis(self):
        """
        Test projection along x-axis.
        """
        position = np.array([3.0, 2.0, 1.0])
        direction = np.array([1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 3.0, atol=TEST_TOLERANCE)

    def test_project_along_y_axis(self):
        """
        Test projection along y-axis.
        """
        position = np.array([3.0, 2.0, 1.0])
        direction = np.array([0.0, 1.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 2.0, atol=TEST_TOLERANCE)

    def test_project_along_z_axis(self):
        """
        Test projection along z-axis.
        """
        position = np.array([3.0, 2.0, 1.0])
        direction = np.array([0.0, 0.0, 1.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 1.0, atol=TEST_TOLERANCE)

    def test_project_along_diagonal(self):
        """
        Test projection along a diagonal direction.
        """
        position = np.array([1.0, 1.0, 0.0])
        direction = np.array(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]
        )  # Normalized diagonal.
        result = project_coordinate(position, direction)
        expected = np.sqrt(2)  # ||(1,1,0)|| projected onto (1/sqrt(2), 1/sqrt(2), 0).
        assert np.isclose(result, expected, atol=TEST_TOLERANCE)

    def test_project_negative_direction(self):
        """
        Test projection in negative direction.
        """
        position = np.array([3.0, 2.0, 1.0])
        direction = np.array([-1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, -3.0, atol=TEST_TOLERANCE)

    def test_project_orthogonal_vectors(self):
        """
        Test projection of orthogonal vectors gives zero.
        """
        position = np.array([1.0, 0.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 0.0, atol=TEST_TOLERANCE)

    def test_project_zero_position(self):
        """
        Test projection of zero position vector.
        """
        position = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert np.isclose(result, 0.0, atol=TEST_TOLERANCE)

    def test_project_arbitrary_unit_direction(self):
        """
        Test projection along an arbitrary unit direction.
        """
        position = np.array([2.0, 3.0, 6.0])
        # Create normalized direction vector.
        direction_unnormalized = np.array([1.0, 2.0, 2.0])
        direction = direction_unnormalized / np.linalg.norm(direction_unnormalized)
        result = project_coordinate(position, direction)

        # Manual calculation: dot product.
        expected = np.dot(position, direction)
        assert np.isclose(result, expected, atol=TEST_TOLERANCE)

    def test_project_returns_float(self):
        """
        Test that the result is a float.
        """
        position = np.array([1.0, 2.0, 3.0])
        direction = np.array([1.0, 0.0, 0.0])
        result = project_coordinate(position, direction)
        assert isinstance(result, float)

    def test_project_non_unit_direction_raises_error(self):
        """
        Test that non-unit direction vector raises ValueError.
        """
        position = np.array([1.0, 2.0, 3.0])
        direction = np.array([2.0, 0.0, 0.0])  # Magnitude = 2, not unit.
        with pytest.raises(ValueError, match="Direction vector not normalized"):
            project_coordinate(position, direction)

    def test_project_zero_direction_raises_error(self):
        """
        Test that zero direction vector raises ValueError.
        """
        position = np.array([1.0, 2.0, 3.0])
        direction = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Direction vector not normalized"):
            project_coordinate(position, direction)

    def test_project_near_unit_direction_passes(self):
        """
        Test that nearly unit direction vector passes within tolerance.
        """
        position = np.array([1.0, 2.0, 3.0])
        # Create a direction vector that's almost but not exactly unit length.
        direction = np.array([1.0 + EPSILON / 2, 0.0, 0.0])
        # This should pass since the difference is within TEST_TOLERANCE tolerance.
        result = project_coordinate(position, direction)
        # Should be close to projecting onto [1,0,0].
        assert np.isclose(result, 1.0, atol=1e-3)

    def test_project_slightly_off_unit_direction_raises_error(self):
        """
        Test that direction vector outside tolerance raises ValueError.
        """
        position = np.array([1.0, 2.0, 3.0])
        # Create a direction vector that's outside the tolerance.
        direction = np.array(
            [1.1, 0.0, 0.0]
        )  # Magnitude = 1.1, clearly outside tolerance.
        with pytest.raises(ValueError, match="Direction vector not normalized"):
            project_coordinate(position, direction)
