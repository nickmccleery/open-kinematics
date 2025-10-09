"""
Common derived point calculation functions.

These functions calculate positions of derived points based on the positions of other
points in the suspension system. They are shared across different suspension types to
avoid code duplication.
"""

from kinematics.enums import PointID
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
