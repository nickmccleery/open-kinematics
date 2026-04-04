"""
Common derived point calculation functions.

These functions calculate positions of derived points based on the positions of other
points in the suspension system. They are shared across different suspension types to
avoid code duplication.
"""

import numpy as np

from kinematics.core.constants import EPS_GEOMETRIC
from kinematics.core.enums import Axis, PointID
from kinematics.core.types import Vec3, WorldAxisSystem, make_vec3
from kinematics.core.vector_utils.generic import normalize_vector


def get_wheel_plane_down_vector(positions: dict[PointID, Vec3]) -> Vec3:
    """
    Calculates the 'down' direction vector in the wheel's plane of rotation.

    This vector is always perpendicular to the axle's direction and is calculated
    by finding the component of the global down vector that is orthogonal to
    the axle vector (using Gram-Schmidt orthogonalization).

    Args:
        positions: Dictionary of point coordinates. Must contain AXLE_INBOARD
                   and AXLE_OUTBOARD.

    Returns:
        A normalized 3D vector representing the 'down' direction in the
        wheel's plane.

    Raises:
        ValueError: If the axle has zero length or the resulting projected
                    down vector is a zero vector (i.e., axle is vertical).
    """
    axle_inboard = positions[PointID.AXLE_INBOARD]
    axle_outboard = positions[PointID.AXLE_OUTBOARD]

    # Compute the normalized axle direction (wheel's spin axis).
    axle_vector = axle_outboard - axle_inboard
    axle_direction = normalize_vector(axle_vector)

    # Find the 'down' direction within the wheel plane (perpendicular to the axle).
    global_down = -1 * WorldAxisSystem.Z

    # Project global down onto the plane perpendicular to the axle. This removes
    # the component of 'down' that is parallel to the axle.
    down_parallel_to_axle = np.dot(global_down, axle_direction) * axle_direction
    wheel_down = global_down - down_parallel_to_axle

    # Normalize to get the final unit vector. This will raise a ValueError if
    # the axle is vertical, which is the correct fail-fast behavior.
    return normalize_vector(wheel_down)


def get_axle_midpoint(positions: dict[PointID, Vec3]) -> Vec3:
    """
    Computes the center point between the inboard and outboard axle positions.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                  Must contain AXLE_INBOARD and AXLE_OUTBOARD entries.

    Returns:
        A numpy array representing the 3D coordinates of the axle midpoint.
    """
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.AXLE_OUTBOARD]
    midpoint = make_vec3((p1 + p2) / 2)
    return midpoint


def get_wheel_center(positions: dict[PointID, Vec3], wheel_offset: float) -> Vec3:
    """
    Determine wheel center from hub face using ISO/SAE wheel-offset convention.

    Starting at `AXLE_OUTBOARD` (hub mounting face), this moves along the axle
    axis by `wheel_offset` toward axle inboard for positive values.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                Must contain AXLE_INBOARD and AXLE_OUTBOARD entries.
        wheel_offset: Wheel offset (ET) from hub mounting face to wheel center
                  plane in mm. Positive values place the wheel centerline
                  inboard of the hub face; negative values place it outboard.

    Returns:
        A numpy array representing the 3D coordinates of the wheel center.
    """
    p1 = positions[PointID.AXLE_OUTBOARD]  # Hub face.
    p2 = positions[PointID.AXLE_INBOARD]  # Axle inboard point.
    v = p1 - p2  # Points outboard; from inboard to axle outboard (hub face).
    v = normalize_vector(v)

    # ISO/SAE wheel offset convention: positive offset places centerline inboard.
    wheel_center = make_vec3(p1 - v * wheel_offset)
    return wheel_center


def get_wheel_inboard(positions: dict[PointID, Vec3], wheel_width: float) -> Vec3:
    """
    Determines the inboard edge position of the wheel by moving inward from the wheel
    center by half the wheel width along the axle axis.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                Must contain AXLE_INBOARD and WHEEL_CENTER entries.
        wheel_width: Total width of the wheel across its axial dimension.

    Returns:
        A numpy array representing the 3D coordinates of the wheel's inboard lip/edge.
    """
    p1 = positions[PointID.AXLE_INBOARD]
    p2 = positions[PointID.WHEEL_CENTER]
    v = p2 - p1  # Points outboard; from inboard to wheel center.
    v = normalize_vector(v)
    wheel_inboard = make_vec3(p2 - v * (wheel_width / 2))
    return wheel_inboard


def get_wheel_outboard(positions: dict[PointID, Vec3], wheel_width: float) -> Vec3:
    """
    Determines the outboard edge position of the wheel by moving outward from the wheel
    center by half the wheel width along the axle axis.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                Must contain WHEEL_CENTER and AXLE_INBOARD entries.
        wheel_width: Total width of the wheel across its axial dimension.

    Returns:
        A numpy array representing the 3D coordinates of the wheel's outboard lip/edge.
    """
    p1 = positions[PointID.WHEEL_CENTER]
    p2 = positions[PointID.AXLE_INBOARD]
    v = p1 - p2  # Points outboard; from axle inboard to wheel center.
    v = normalize_vector(v)
    wheel_outboard = make_vec3(p1 + v * (wheel_width / 2))
    return wheel_outboard


def get_wheel_center_on_ground(
    positions: dict[PointID, Vec3],
    ground_plane_z: float = 0.0,
) -> Vec3:
    """
    Computes the intersection of the ground plane with the wheel-center line in the
    wheel plane.

    Starting at `WHEEL_CENTER`, this traces the wheel-plane 'down' direction
    (the component of global down that is perpendicular to the axle) until it
    reaches `z = ground_plane_z`. The result is the ground intercept of the
    wheel centerline in the wheel's central plane, so it moves with camber.

    This is a geometric projection and does not use tire radius; for a
    radius-based tire contact point, use `get_contact_patch_center`.

    Args:
        positions: Dictionary of point coordinates.
        ground_plane_z: Z-coordinate of the ground plane in mm (default: 0.0).

    Returns:
        The 3D coordinates where the wheel center projects onto the ground plane.
    """
    wheel_center = positions[PointID.WHEEL_CENTER]
    wheel_down_normalized = get_wheel_plane_down_vector(positions)

    # Find where a ray from the wheel center intersects the horizontal ground plane.
    wheel_down_z = wheel_down_normalized[Axis.Z]

    if abs(wheel_down_z) < EPS_GEOMETRIC:
        # The projection direction is horizontal (e.g., at 90° camber).
        raise ValueError(
            "Wheel plane normal is parallel to ground plane; cannot project."
        )

    # Solve for parameter t where the ray intersects the ground plane.
    t = (ground_plane_z - wheel_center[Axis.Z]) / wheel_down_z
    ground_projection = wheel_center + t * wheel_down_normalized

    return make_vec3(ground_projection)


def get_contact_patch_center(
    positions: dict[PointID, Vec3], tire_radius: float
) -> Vec3:
    """
    Computes the position of the geometric contact patch center.

    This is the lowest point on an ideal tire circle in the wheel's center
    plane. It is found by moving from the wheel center in the wheel-plane
    'down' direction by a distance equal to the tire radius. Its Z-coordinate
    is not fixed and will move with the suspension.

    Args:
        positions: Dictionary of point coordinates.
        tire_radius: The radius of the tire in mm.

    Returns:
        The 3D coordinates of the geometric contact point.
    """
    wheel_center = positions[PointID.WHEEL_CENTER]
    wheel_down_normalized = get_wheel_plane_down_vector(positions)

    # Calculate the contact point by moving from the wheel center by the radius.
    contact_point = wheel_center + wheel_down_normalized * tire_radius

    return make_vec3(contact_point)
