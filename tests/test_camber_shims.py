"""
Tests for camber shim functionality.

These tests verify that camber shims correctly rotate the upright about
the lower ball joint, modifying the as-built geometry without changing
the design hard points.
"""

import numpy as np

from kinematics.enums import Axis, PointID
from kinematics.io.geometry_loader import load_geometry
from kinematics.suspensions.core.settings import CamberShimConfigOutboard
from kinematics.suspensions.core.shims import (
    compute_shim_offset,
    compute_upright_rotation_from_shim,
    rotate_point_about_axis,
)
from kinematics.types import make_vec3


def test_compute_shim_offset_positive():
    """Test that increasing shim thickness creates outboard offset."""
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},  # Unit vector pointing outboard
        design_thickness=0.0,
        setup_thickness=10.0,  # 10mm of shim added
    )

    offset = compute_shim_offset(shim_config)

    # Should move 10mm outboard (positive Y direction)
    assert np.isclose(offset[Axis.X], 0.0)
    assert np.isclose(offset[Axis.Y], 10.0)
    assert np.isclose(offset[Axis.Z], 0.0)


def test_compute_shim_offset_negative():
    """Test that removing shims creates inboard offset."""
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
        design_thickness=10.0,
        setup_thickness=5.0,  # Removed 5mm of shim
    )

    offset = compute_shim_offset(shim_config)

    # Should move 5mm inboard (negative Y direction)
    assert np.isclose(offset[Axis.X], 0.0)
    assert np.isclose(offset[Axis.Y], -5.0)
    assert np.isclose(offset[Axis.Z], 0.0)


def test_compute_shim_offset_zero():
    """Test that no shim change creates no offset."""
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
        design_thickness=5.0,
        setup_thickness=5.0,  # No change
    )

    offset = compute_shim_offset(shim_config)

    # Should have no offset
    assert np.isclose(offset[Axis.X], 0.0)
    assert np.isclose(offset[Axis.Y], 0.0)
    assert np.isclose(offset[Axis.Z], 0.0)


def test_rotate_point_about_axis_90_degrees():
    """Test rotation of a point 90 degrees about Z axis."""
    point = make_vec3(np.array([1.0, 0.0, 0.0]))
    pivot = make_vec3(np.array([0.0, 0.0, 0.0]))
    axis = make_vec3(np.array([0.0, 0.0, 1.0]))
    angle = np.pi / 2  # 90 degrees

    rotated = rotate_point_about_axis(point, pivot, axis, angle)

    # Should rotate to (0, 1, 0)
    assert np.isclose(rotated[Axis.X], 0.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Y], 1.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Z], 0.0, atol=1e-10)


def test_rotate_point_about_axis_with_offset_pivot():
    """Test rotation about an axis that doesn't pass through origin."""
    point = make_vec3(np.array([2.0, 0.0, 0.0]))
    pivot = make_vec3(np.array([1.0, 0.0, 0.0]))
    axis = make_vec3(np.array([0.0, 0.0, 1.0]))
    angle = np.pi  # 180 degrees

    rotated = rotate_point_about_axis(point, pivot, axis, angle)

    # Should rotate to (0, 0, 0)
    assert np.isclose(rotated[Axis.X], 0.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Y], 0.0, atol=1e-10)
    assert np.isclose(rotated[Axis.Z], 0.0, atol=1e-10)


def test_compute_upright_rotation_simple_case():
    """Test upright rotation calculation for a simple geometry."""
    lower_ball_joint = make_vec3(np.array([0.0, 900.0, 200.0]))
    shim_face_center_design = make_vec3(np.array([0.0, 350.0, 500.0]))

    # 10mm outboard offset
    shim_offset = make_vec3(np.array([0.0, 10.0, 0.0]))

    axis, angle = compute_upright_rotation_from_shim(
        lower_ball_joint,
        shim_face_center_design,
        shim_offset,
    )

    # Axis should be primarily in X direction (perpendicular to Y-Z plane)
    assert abs(axis[Axis.X]) > 0.9  # Mostly X direction

    # Angle should be small but non-zero
    assert angle > 0.0
    assert angle < np.deg2rad(5.0)  # Less than 5 degrees for 10mm shim


def test_shim_application_changes_camber(double_wishbone_geometry_file):
    """
    Test that applying a camber shim rotates the upright and changes camber angle.
    """
    # Load base geometry
    loaded = load_geometry(double_wishbone_geometry_file)

    # Get initial camber (no shim)
    initial_state = loaded.provider.initial_state()
    initial_axle_in = initial_state.positions[PointID.AXLE_INBOARD]
    initial_axle_out = initial_state.positions[PointID.AXLE_OUTBOARD]

    # Calculate initial camber angle
    initial_axle_vector = initial_axle_out - initial_axle_in
    initial_camber_rad = np.arctan2(
        initial_axle_vector[Axis.Y], initial_axle_vector[Axis.Z]
    )

    # Apply a camber shim
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
        design_thickness=0.0,
        setup_thickness=10.0,
    )
    loaded.geometry.configuration.camber_shim = shim_config

    # Re-initialize provider
    provider_class = type(loaded.provider)
    shimmed_provider = provider_class(loaded.geometry)
    shimmed_state = shimmed_provider.initial_state()

    # Get shimmed camber
    shimmed_axle_in = shimmed_state.positions[PointID.AXLE_INBOARD]
    shimmed_axle_out = shimmed_state.positions[PointID.AXLE_OUTBOARD]

    shimmed_axle_vector = shimmed_axle_out - shimmed_axle_in
    shimmed_camber_rad = np.arctan2(
        shimmed_axle_vector[Axis.Y], shimmed_axle_vector[Axis.Z]
    )

    # Camber should have changed (become less negative / more positive)
    # Adding shim outboard should reduce negative camber
    assert shimmed_camber_rad > initial_camber_rad, (
        f"Expected camber to increase (less negative), "
        f"got initial={np.degrees(initial_camber_rad):.3f}°, "
        f"shimmed={np.degrees(shimmed_camber_rad):.3f}°"
    )


def test_shim_does_not_move_lower_ball_joint(double_wishbone_geometry_file):
    """
    Test that the lower ball joint (pivot point) doesn't move when shims are applied.
    """
    loaded = load_geometry(double_wishbone_geometry_file)

    # Get initial lower ball joint position
    initial_state = loaded.provider.initial_state()
    initial_lbj = initial_state.positions[PointID.LOWER_WISHBONE_OUTBOARD].copy()

    # Apply shim
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
        design_thickness=0.0,
        setup_thickness=10.0,
    )
    loaded.geometry.configuration.camber_shim = shim_config

    provider_class = type(loaded.provider)
    shimmed_provider = provider_class(loaded.geometry)
    shimmed_state = shimmed_provider.initial_state()

    shimmed_lbj = shimmed_state.positions[PointID.LOWER_WISHBONE_OUTBOARD]

    # Lower ball joint should not move
    assert np.allclose(initial_lbj, shimmed_lbj), (
        "Lower ball joint (pivot) should not move when shims are applied"
    )


def test_shim_does_not_move_inboard_points(double_wishbone_geometry_file):
    """
    Test that chassis-mounted inboard points don't move when shims are applied.
    """
    loaded = load_geometry(double_wishbone_geometry_file)

    initial_state = loaded.provider.initial_state()
    initial_points = {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: initial_state.positions[
            PointID.LOWER_WISHBONE_INBOARD_FRONT
        ].copy(),
        PointID.LOWER_WISHBONE_INBOARD_REAR: initial_state.positions[
            PointID.LOWER_WISHBONE_INBOARD_REAR
        ].copy(),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: initial_state.positions[
            PointID.UPPER_WISHBONE_INBOARD_FRONT
        ].copy(),
        PointID.UPPER_WISHBONE_INBOARD_REAR: initial_state.positions[
            PointID.UPPER_WISHBONE_INBOARD_REAR
        ].copy(),
        PointID.TRACKROD_INBOARD: initial_state.positions[
            PointID.TRACKROD_INBOARD
        ].copy(),
    }

    # Apply shim
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
        design_thickness=0.0,
        setup_thickness=10.0,
    )
    loaded.geometry.configuration.camber_shim = shim_config

    provider_class = type(loaded.provider)
    shimmed_provider = provider_class(loaded.geometry)
    shimmed_state = shimmed_provider.initial_state()

    # All inboard points should remain unchanged
    for point_id, initial_pos in initial_points.items():
        shimmed_pos = shimmed_state.positions[point_id]
        assert np.allclose(initial_pos, shimmed_pos), (
            f"{point_id.name} (chassis-mounted) should not move when shims are applied"
        )


def test_shim_does_not_move_upper_ball_joint(double_wishbone_geometry_file):
    """
    Test that the upper ball joint does NOT move when shims are applied.

    The shim is internal to the upright, so ball joints stay fixed.
    """
    loaded = load_geometry(double_wishbone_geometry_file)

    initial_state = loaded.provider.initial_state()
    initial_ubj = initial_state.positions[PointID.UPPER_WISHBONE_OUTBOARD].copy()

    # Apply shim
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
        design_thickness=0.0,
        setup_thickness=10.0,
    )
    loaded.geometry.configuration.camber_shim = shim_config

    provider_class = type(loaded.provider)
    shimmed_provider = provider_class(loaded.geometry)
    shimmed_state = shimmed_provider.initial_state()

    shimmed_ubj = shimmed_state.positions[PointID.UPPER_WISHBONE_OUTBOARD]

    # Upper ball joint should NOT move (it's on the fixed part of the upright)
    assert np.allclose(initial_ubj, shimmed_ubj), (
        "Upper ball joint should not move when shims are applied"
    )


def test_shim_moves_axle_points(double_wishbone_geometry_file):
    """
    Test that all upright-mounted points DO move when shims are applied.

    This includes axle points and trackrod outboard (test geometry doesn't have pushrod).
    """
    loaded = load_geometry(double_wishbone_geometry_file)

    initial_state = loaded.provider.initial_state()
    initial_axle_in = initial_state.positions[PointID.AXLE_INBOARD].copy()
    initial_axle_out = initial_state.positions[PointID.AXLE_OUTBOARD].copy()
    initial_trackrod_out = initial_state.positions[PointID.TRACKROD_OUTBOARD].copy()

    # Apply shim
    shim_config = CamberShimConfigOutboard(
        shim_face_center={"x": 0.0, "y": 350.0, "z": 500.0},
        shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
        design_thickness=0.0,
        setup_thickness=10.0,
    )
    loaded.geometry.configuration.camber_shim = shim_config

    provider_class = type(loaded.provider)
    shimmed_provider = provider_class(loaded.geometry)
    shimmed_state = shimmed_provider.initial_state()

    shimmed_axle_in = shimmed_state.positions[PointID.AXLE_INBOARD]
    shimmed_axle_out = shimmed_state.positions[PointID.AXLE_OUTBOARD]
    shimmed_trackrod_out = shimmed_state.positions[PointID.TRACKROD_OUTBOARD]

    # All upright-mounted points should move
    distance_moved_axle_in = np.linalg.norm(shimmed_axle_in - initial_axle_in)
    distance_moved_axle_out = np.linalg.norm(shimmed_axle_out - initial_axle_out)
    distance_moved_trackrod = np.linalg.norm(
        shimmed_trackrod_out - initial_trackrod_out
    )

    assert distance_moved_axle_in > 0.1, (
        f"Axle inboard should move (moved {distance_moved_axle_in:.3f}mm)"
    )
    assert distance_moved_axle_out > 0.1, (
        f"Axle outboard should move (moved {distance_moved_axle_out:.3f}mm)"
    )
    assert distance_moved_trackrod > 0.1, (
        f"Trackrod outboard should move (moved {distance_moved_trackrod:.3f}mm)"
    )


def test_backward_compatibility_no_shim(double_wishbone_geometry_file):
    """
    Test that when design_thickness == setup_thickness, there's no effect.
    """
    loaded = load_geometry(double_wishbone_geometry_file)

    # Geometry has shim config, but design == as_built (no delta)
    assert loaded.geometry.configuration.camber_shim is not None
    shim = loaded.geometry.configuration.camber_shim
    assert shim.design_thickness == shim.setup_thickness

    # Should initialize without error
    state = loaded.provider.initial_state()

    # Should have all expected points
    assert PointID.UPPER_WISHBONE_OUTBOARD in state.positions
    assert PointID.LOWER_WISHBONE_OUTBOARD in state.positions
    assert PointID.AXLE_INBOARD in state.positions
    assert PointID.AXLE_OUTBOARD in state.positions
