from functools import partial

import numpy as np
import pytest

from kinematics.enums import PointID
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_center_on_ground,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec


@pytest.fixture
def sample_positions():
    # Set up a simple configuration:
    # - Axle inboard at origin
    # - Axle outboard at x=2
    # - This makes the axle 2 units wide pointing along x-axis
    return {
        PointID.AXLE_INBOARD: np.array([0.0, 0.0, 0.0]),
        PointID.AXLE_OUTBOARD: np.array([2.0, 0.0, 0.0]),
    }


def test_axle_midpoint(sample_positions):
    result = get_axle_midpoint(sample_positions)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 0.0]))


def test_wheel_center(sample_positions):
    # With axle outboard at x=2.0 and offset of 0.5, wheel center should be at x=2.5
    result = get_wheel_center(sample_positions, wheel_offset=0.5)
    np.testing.assert_array_equal(result, np.array([2.5, 0.0, 0.0]))


def test_wheel_inboard_outboard(sample_positions):
    # First compute wheel center
    wheel_center = get_wheel_center(sample_positions, wheel_offset=0.5)
    positions_dict = sample_positions.copy()
    positions_dict[PointID.WHEEL_CENTER] = wheel_center

    # Test inboard point (should be 0.5 units inboard of wheel center)
    inboard = get_wheel_inboard(positions_dict, wheel_width=1.0)
    np.testing.assert_array_equal(inboard, np.array([2.0, 0.0, 0.0]))

    # Test outboard point (should be 0.5 units outboard of wheel center)
    outboard = get_wheel_outboard(positions_dict, wheel_width=1.0)
    np.testing.assert_array_equal(outboard, np.array([3.0, 0.0, 0.0]))


def test_dependency_manager_basic(sample_positions):
    """
    Test that the DerivedPointsManager can calculate axle midpoint.
    """
    functions = {
        PointID.AXLE_MIDPOINT: get_axle_midpoint,
    }
    dependencies = {
        PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
    }
    spec = DerivedPointsSpec(functions=functions, dependencies=dependencies)
    manager = DerivedPointsManager(spec)

    manager.update_in_place(sample_positions)

    # Should have original positions plus the derived point
    assert len(sample_positions) == 3
    assert PointID.AXLE_MIDPOINT in sample_positions
    np.testing.assert_array_equal(
        sample_positions[PointID.AXLE_MIDPOINT], np.array([1.0, 0.0, 0.0])
    )


def test_dependency_manager_complex(sample_positions):
    """
    Test that the DerivedPointsManager can handle dependencies between derived points.
    """
    functions = {
        PointID.AXLE_MIDPOINT: get_axle_midpoint,
        PointID.WHEEL_CENTER: partial(get_wheel_center, wheel_offset=0.5),
        PointID.WHEEL_INBOARD: partial(get_wheel_inboard, wheel_width=1.0),
        PointID.WHEEL_OUTBOARD: partial(get_wheel_outboard, wheel_width=1.0),
    }
    dependencies = {
        PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
    }
    spec = DerivedPointsSpec(functions=functions, dependencies=dependencies)
    manager = DerivedPointsManager(spec)

    manager.update_in_place(sample_positions)

    # Should have original positions plus all derived points
    assert len(sample_positions) == 6
    assert PointID.AXLE_MIDPOINT in sample_positions
    assert PointID.WHEEL_CENTER in sample_positions
    assert PointID.WHEEL_INBOARD in sample_positions
    assert PointID.WHEEL_OUTBOARD in sample_positions

    # Check values
    np.testing.assert_array_equal(
        sample_positions[PointID.AXLE_MIDPOINT], np.array([1.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        sample_positions[PointID.WHEEL_CENTER], np.array([2.5, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        sample_positions[PointID.WHEEL_INBOARD], np.array([2.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        sample_positions[PointID.WHEEL_OUTBOARD], np.array([3.0, 0.0, 0.0])
    )


def test_circular_dependency_detection():
    """
    Test that circular dependencies are detected.
    """

    def dummy_func(positions):
        return positions[PointID.AXLE_INBOARD]

    functions = {
        PointID.WHEEL_CENTER: dummy_func,
        PointID.WHEEL_OUTBOARD: dummy_func,
    }
    dependencies = {
        PointID.WHEEL_CENTER: {PointID.WHEEL_OUTBOARD},
        PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER},
    }

    spec = DerivedPointsSpec(functions=functions, dependencies=dependencies)

    with pytest.raises(ValueError, match="Circular dependency detected"):
        DerivedPointsManager(spec)


def test_wheel_center_on_ground_zero_camber():
    """
    Test ground projection with zero camber (vertical wheel).
    """
    positions = {
        PointID.WHEEL_CENTER: np.array([0.0, 900.0, 350.0]),
        PointID.AXLE_INBOARD: np.array([0.0, 850.0, 350.0]),
        PointID.AXLE_OUTBOARD: np.array([0.0, 950.0, 350.0]),  # Horizontal axle
    }

    result = get_wheel_center_on_ground(positions, ground_plane_z=0.0)

    # With zero camber, projection should be directly below wheel center
    np.testing.assert_allclose(result[0], 0.0, atol=0.1)  # Same X
    np.testing.assert_allclose(result[1], 900.0, atol=0.1)  # Same Y
    np.testing.assert_allclose(result[2], 0.0, atol=0.1)  # On ground


def test_wheel_center_on_ground_with_camber():
    """
    Test ground projection with negative camber.
    """
    positions = {
        PointID.WHEEL_CENTER: np.array([0.0, 900.0, 350.0]),
        PointID.AXLE_INBOARD: np.array([0.0, 850.0, 350.0]),
        PointID.AXLE_OUTBOARD: np.array([0.0, 950.0, 344.8]),  # ~3° camber
    }

    result = get_wheel_center_on_ground(positions, ground_plane_z=0.0)

    # With camber, Y coordinate should differ from wheel center
    assert abs(result[1] - 900.0) > 5.0  # Y offset due to camber

    # Should still be on ground plane
    np.testing.assert_allclose(result[2], 0.0, atol=0.1)

    # X coordinate should remain close to wheel center (pure camber, no toe)
    np.testing.assert_allclose(result[0], 0.0, atol=0.5)


def test_wheel_center_on_ground_with_toe():
    """
    Test ground projection with toe angle.
    """
    positions = {
        PointID.WHEEL_CENTER: np.array([0.0, 900.0, 350.0]),
        PointID.AXLE_INBOARD: np.array([0.0, 850.0, 350.0]),
        PointID.AXLE_OUTBOARD: np.array([5.2, 950.0, 350.0]),  # ~3° toe
    }

    result = get_wheel_center_on_ground(positions, ground_plane_z=0.0)

    # With pure toe (no camber), should project straight down
    # Y coordinate should equal wheel center Y
    np.testing.assert_allclose(result[1], 900.0, atol=0.1)

    # X coordinate should equal wheel center X
    np.testing.assert_allclose(result[0], 0.0, atol=0.1)

    # Should be on ground
    np.testing.assert_allclose(result[2], 0.0, atol=0.1)


def test_wheel_center_on_ground_with_camber_and_toe():
    """
    Test ground projection with both camber and toe.
    """
    positions = {
        PointID.WHEEL_CENTER: np.array([0.0, 900.0, 350.0]),
        PointID.AXLE_INBOARD: np.array([0.0, 850.0, 350.0]),
        PointID.AXLE_OUTBOARD: np.array([5.2, 950.0, 344.8]),  # Both camber and toe
    }

    result = get_wheel_center_on_ground(positions, ground_plane_z=0.0)

    # Should be on ground plane
    np.testing.assert_allclose(result[2], 0.0, atol=0.1)

    # Both X and Y should differ from wheel center due to combined effects
    assert abs(result[1] - 900.0) > 5.0  # Y changed by camber
    # X may have small offset due to interaction of camber and toe


def test_wheel_center_on_ground_custom_ground_plane():
    """
    Test ground projection with non-zero ground plane.
    """
    positions = {
        PointID.WHEEL_CENTER: np.array([0.0, 900.0, 450.0]),
        PointID.AXLE_INBOARD: np.array([0.0, 850.0, 450.0]),
        PointID.AXLE_OUTBOARD: np.array([0.0, 950.0, 450.0]),
    }

    # Project onto ground plane at Z = 100mm
    result = get_wheel_center_on_ground(positions, ground_plane_z=100.0)

    # Should be on the specified ground plane
    np.testing.assert_allclose(result[2], 100.0, atol=0.1)

    # X and Y should match wheel center (zero camber/toe)
    np.testing.assert_allclose(result[0], 0.0, atol=0.1)
    np.testing.assert_allclose(result[1], 900.0, atol=0.1)


def test_wheel_center_on_ground_in_provider():
    """
    Test that WHEEL_CENTER_ON_GROUND is computed through provider.
    """
    from pathlib import Path

    from kinematics.io.geometry_loader import load_geometry

    # Load test geometry
    test_data_dir = Path(__file__).parent.parent / "data"
    geometry_file = test_data_dir / "geometry.yaml"

    if geometry_file.exists():
        loaded = load_geometry(geometry_file)
        state = loaded.provider.initial_state()

        # Verify the point exists
        assert PointID.WHEEL_CENTER_ON_GROUND in state.positions

        # Verify it's on the ground
        ground_point = state.positions[PointID.WHEEL_CENTER_ON_GROUND]
        assert ground_point[2] < 10.0  # Should be near Z=0
