"""
Common derived point calculation functions.

These functions calculate positions of derived points based on the positions of other
points in the suspension system. They are shared across different suspension types to
avoid code duplication.
"""

import numpy as np

from kinematics.constants import EPSILON
from kinematics.enums import Axis, PointID
from kinematics.types import Vec3, make_vec3
from kinematics.vector_utils.generic import normalize_vector


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
    Determines the wheel center by projecting inboard from the axle outboard position
    along the axle axis by the specified wheel offset distance.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                Must contain AXLE_INBOARD and AXLE_OUTBOARD entries.
        wheel_offset: Distance from the hub face to the wheel center plane.
                    Positive offset means the wheel centerline is inboard of the hub face.

    Returns:
        A numpy array representing the 3D coordinates of the wheel center.
    """
    p1 = positions[PointID.AXLE_OUTBOARD]  # Hub face.
    p2 = positions[PointID.AXLE_INBOARD]  # Axle inboard point.
    v = p1 - p2  # Points outboard; from inboard to axle outboard (hub face).
    v = normalize_vector(v)
    wheel_center = make_vec3(p1 + v * wheel_offset)
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
    Project the wheel center onto the ground plane along the wheel plane normal.

    This function computes where the wheel center projects onto the ground when
    following the wheel's plane normal direction (perpendicular to the axle).
    This accounts for both camber and toe.

    The projection direction is found by:
    1. Taking the global down vector (0, 0, -1).
    2. Removing its component parallel to the axle direction.
    3. Normalizing the result to get the wheel plane normal.
    4. Following this direction from wheel center to the ground plane.

    Args:
        positions: Dictionary mapping point IDs to their 3D coordinates.
                  Must contain WHEEL_CENTER, AXLE_INBOARD, and AXLE_OUTBOARD.
        ground_plane_z: Z-coordinate of the ground plane in mm (default: 0.0).

    Returns:
        A numpy array representing the 3D coordinates where the wheel center
        projects onto the ground plane.

    Note:
        This is a purely geometric calculation. In a full suspension analysis,
        this point can be used as a reference for calculations that need a
        ground-referenced position accounting for wheel orientation.
    """

    wheel_center = positions[PointID.WHEEL_CENTER]
    axle_inboard = positions[PointID.AXLE_INBOARD]
    axle_outboard = positions[PointID.AXLE_OUTBOARD]

    # Compute axle direction (wheel spin axis).
    axle_vector = axle_outboard - axle_inboard
    axle_direction = normalize_vector(axle_vector)

    # Find the wheel plane normal (points 'down' in wheel's reference frame).
    # Start with global down direction.
    global_down = np.array([0.0, 0.0, -1.0])

    # Project global down onto plane perpendicular to axle (Gram-Schmidt orthogonalization).
    # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    # This removes the component of 'down' that's parallel to the axle.
    down_parallel_to_axle = np.dot(global_down, axle_direction) * axle_direction
    wheel_down = global_down - down_parallel_to_axle

    # Normalize to get unit vector.
    wheel_down_normalized = normalize_vector(wheel_down)

    # Find where ray from wheel center intersects ground plane
    # Ray equation: P(t) = wheel_center + t * wheel_down_normalized
    # Ground plane equation: Z = ground_plane_z
    # Solve for t: wheel_center[2] + t * wheel_down_normalized[2] = ground_plane_z
    wheel_down_z = wheel_down_normalized[Axis.Z]

    if abs(wheel_down_z) < EPSILON:
        # Wheel down vector is horizontal; parallel to ground plane. Something
        # is not happy here.
        raise ValueError(
            "Wheel plane normal is parallel to ground plane; cannot project."
        )
    else:
        # Solve for parameter t where ray intersects ground plane.
        t = (ground_plane_z - wheel_center[Axis.Z]) / wheel_down_z

        # Compute intersection point.
        ground_projection = wheel_center + t * wheel_down_normalized

    return make_vec3(ground_projection)
