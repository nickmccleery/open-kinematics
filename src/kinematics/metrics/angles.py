"""
This module contains functions for calculating fundamental wheel alignment angles.

All functions operate on a solved SuspensionState and return angles in degrees.

Coordinate System Assumption: ISO 8855 (X-Forward, Y-Left, Z-Up).
"""

import numpy as np

from kinematics.core.enums import Axis, PointID
from kinematics.core.types import WorldAxisSystem
from kinematics.core.vector_utils.generic import normalize_vector
from kinematics.state import SuspensionState


def _detect_side(state: SuspensionState) -> float:
    """
    Detects the side of the vehicle based on the Y-coordinate of the wheel center.

    Returns:
        1.0 for Left side (Y > 0), -1.0 for Right side (Y < 0).
    """
    # Use Axle Outboard to detect side
    y_pos = state.get(PointID.AXLE_OUTBOARD)[Axis.Y]
    return -1.0 if y_pos < 0 else 1.0


def calculate_camber(state: SuspensionState) -> float:
    """
    Calculates the camber angle in degrees.

    Camber is the angle of the wheel's vertical centerline with respect to the
    vehicle's vertical axis (Z-axis), viewed from the front (YZ plane).
    Negative camber means the top of the wheel is tilted inwards (towards
    vehicle centerline).

    Args:
        state: The solved SuspensionState to analyze.

    Returns:
        The camber angle in degrees.
    """
    side = _detect_side(state)

    axle_vector = normalize_vector(
        state.get(PointID.AXLE_OUTBOARD) - state.get(PointID.AXLE_INBOARD)
    )
    # Wheel's "up" vector is perpendicular to both the axle and the vehicle's
    # longitudinal axis (X-axis).
    # We multiply by -side to ensure the vector points roughly +Z (Up) for both sides.
    wheel_up_vector = np.cross(axle_vector, WorldAxisSystem.X) * -side

    # Project the wheel's up vector onto the front view plane (YZ plane)
    wheel_up_proj_y = wheel_up_vector[Axis.Y]
    wheel_up_proj_z = wheel_up_vector[Axis.Z]

    # Calculate angle with the global Z-axis.
    angle = np.arctan2(wheel_up_proj_y, wheel_up_proj_z)

    # For right side, inward tilt is +Y, which gives positive angle. Invert to match convention.
    camber_rad = angle if side > 0 else -angle

    return np.rad2deg(camber_rad)


def calculate_caster(state: SuspensionState) -> float:
    """
    Calculates the caster angle in degrees.

    Caster is the angle of the steering axis with respect to the vehicle's
    vertical axis (Z-axis), viewed from the side (XZ plane). Positive caster
    means the top of the steering axis is tilted rearward.

    The steering axis is defined by the upper and lower outboard pivot points.
    For a double wishbone, these are the outboard wishbone points.

    Args:
        state: The solved SuspensionState to analyze.

    Returns:
        The caster angle in degrees.
    """
    # Define steering axis (upper pivot to lower pivot).
    upper_pivot = state.get(PointID.UPPER_WISHBONE_OUTBOARD)
    lower_pivot = state.get(PointID.LOWER_WISHBONE_OUTBOARD)
    steering_axis_vector = upper_pivot - lower_pivot

    # Project the steering axis vector onto the side view plane (XZ plane).
    steering_axis_proj_x = steering_axis_vector[Axis.X]
    steering_axis_proj_z = steering_axis_vector[Axis.Z]

    # Positive caster means top tilted rearward (negative X direction relative to bottom).
    # We negate x so that rearward tilt results in a positive angle.
    caster_rad = np.arctan2(-steering_axis_proj_x, steering_axis_proj_z)
    return np.rad2deg(caster_rad)


def calculate_toe(state: SuspensionState) -> float:
    """
    Calculates the toe angle in degrees.

    Toe is the angle of the wheel's longitudinal axis with respect to the
    vehicle's longitudinal axis (X-axis), viewed from the top (XY plane).
    Positive toe (toe-in) means the front of the wheel points inwards.

    Args:
        state: The solved SuspensionState to analyze.

    Returns:
        The toe angle in degrees.
    """
    side = _detect_side(state)

    axle_vector = state.get(PointID.AXLE_OUTBOARD) - state.get(PointID.AXLE_INBOARD)

    # Project the axle vector onto the top view plane (XY plane).
    axle_proj_x = axle_vector[Axis.X]
    axle_proj_y = axle_vector[Axis.Y]

    # Toe-in results in the axle vector pointing slightly forward (+X).
    if side > 0:  # Left side
        toe_rad = np.arctan2(axle_proj_x, axle_proj_y)
    else:  # Right side
        # Measure relative to -Y axis
        toe_rad = np.arctan2(axle_proj_x, -axle_proj_y)

    return np.rad2deg(toe_rad)
