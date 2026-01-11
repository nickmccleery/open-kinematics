"""
Camber shim geometry calculations.

This module handles the geometric transformations required when camber shims
are added or removed from a suspension system.
"""

from __future__ import annotations

import numpy as np

from kinematics.constants import EPSILON
from kinematics.suspensions.core.settings import CamberShimConfigOutboard
from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.generic import normalize_vector
from kinematics.vector_utils.geometric import compute_vector_vector_angle


def compute_shim_offset(shim_config: CamberShimConfigOutboard) -> Vec3:
    """
    Compute the offset vector caused by shim thickness change.

    Args:
        shim_config: Camber shim configuration.

    Returns:
        Offset vector in mm (points outboard from chassis).
    """
    # Calculate thickness delta.
    delta_thickness = shim_config.setup_thickness - shim_config.design_thickness

    # Get and normalise the shim normal vector.
    normal = np.array(
        [
            shim_config.shim_normal["x"],
            shim_config.shim_normal["y"],
            shim_config.shim_normal["z"],
        ],
        dtype=np.float64,
    )
    normal_unit = normalize_vector(normal)

    # Offset is along the normal direction.
    offset = normal_unit * delta_thickness

    return make_vec3(offset)


def rotate_point_about_axis(
    point: Vec3, pivot: Vec3, axis: Vec3, angle_rad: float
) -> Vec3:
    """
    Rotate a point about an arbitrary axis using Rodrigues' rotation formula.

    Args:
        point: Point to rotate.
        pivot: Point on the rotation axis.
        axis: Unit vector defining the rotation axis direction.
        angle_rad: Rotation angle in radians.

    Returns:
        Rotated point coordinates.
    """
    # Translate point to origin (pivot at origin).
    p = point - pivot

    # Rodrigues' rotation formula
    # v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1 - cos(θ)).
    k = axis
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Cross product k × p.
    k_cross_p = np.cross(k, p)

    # Dot product k · p.
    k_dot_p = np.dot(k, p)

    # Apply Rodrigues formula.
    p_rot = p * cos_angle + k_cross_p * sin_angle + k * k_dot_p * (1 - cos_angle)

    # Translate back.
    return make_vec3(p_rot + pivot)


def compute_upright_rotation_from_shim(
    lower_ball_joint: Vec3,
    shim_face_center_design: Vec3,
    shim_offset: Vec3,
) -> tuple[Vec3, float]:
    """
    Compute the rotation axis and angle for the upright given a shim offset.

    The upright rotates about an axis through the lower ball joint. The rotation
    is determined by the constraint that the shim face center must move by the
    shim offset vector.

    The rotation axis is perpendicular to both:
    - The radial vector from lower ball joint to shim face center
    - The shim offset direction (parallel to shim normal)

    This means the axis is also perpendicular to the shim normal, which makes
    physical sense: pushing the shim face outboard causes rotation about an axis
    perpendicular to that push direction.

    Args:
        lower_ball_joint: Position of the lower ball joint (rotation pivot).
        shim_face_center_design: Design position of the shim face center.
        shim_offset: Offset vector from shim thickness change (parallel to shim normal).

    Returns:
        Tuple of (rotation_axis, rotation_angle_rad).
        rotation_axis is a unit vector through the lower ball joint.
        rotation_angle_rad is the angle to rotate the upright.
    """
    # The shim face must move to: shim_face_center_design + shim_offset.
    shim_face_target = make_vec3(shim_face_center_design + shim_offset)

    # Vector from lower ball joint to design shim face centre.
    r_design = make_vec3(shim_face_center_design - lower_ball_joint)

    # Vector from lower ball joint to target shim face centre.
    r_target = make_vec3(shim_face_target - lower_ball_joint)

    # The rotation axis is perpendicular to both r_design and r_target.
    # Note: r_target = r_design + shim_offset, so by cross product properties:
    #   r_design × r_target = r_design × (r_design + shim_offset)
    #                       = r_design × shim_offset
    # Therefore the axis is perpendicular to both the radial arm and the shim normal.
    cross = np.cross(r_design, r_target)
    cross_magnitude = np.linalg.norm(cross)

    if cross_magnitude < EPSILON:
        # Vectors are parallel - no rotation needed.
        return make_vec3(np.array([0.0, 0.0, 1.0])), 0.0

    rotation_axis = make_vec3(cross / cross_magnitude)

    # Compute the rotation angle between the two radial vectors.
    rotation_angle = compute_vector_vector_angle(r_design, r_target)

    return rotation_axis, rotation_angle
