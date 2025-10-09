"""
This module contains functions for calculating fundamental wheel alignment angles.

All functions operate on a solved SuspensionState and return angles in degrees.
"""

import numpy as np

from kinematics.enums import PointID
from kinematics.state import SuspensionState
from kinematics.types import WorldAxisSystem
from kinematics.vector_utils.generic import normalize_vector


def calculate_camber(state: SuspensionState) -> float:
    """
    Calculates the camber angle in degrees.

    Camber is the angle of the wheel's vertical centerline with respect to the
    vehicle's vertical axis (Z-axis), viewed from the front (YZ plane).
    Negative camber means the top of the wheel is tilted inwards.

    Args:
        state: The solved SuspensionState to analyze.

    Returns:
        The camber angle in degrees.
    """
    axle_vector = normalize_vector(
        state.get(PointID.AXLE_OUTBOARD) - state.get(PointID.AXLE_INBOARD)
    )
    # Wheel's "up" vector is perpendicular to both the axle and the vehicle's
    # longitudinal axis (X-axis).
    wheel_up_vector = np.cross(axle_vector, WorldAxisSystem.X)

    # Project the wheel's up vector onto the front view plane (YZ plane)
    wheel_up_proj_y = wheel_up_vector[1]
    wheel_up_proj_z = wheel_up_vector[2]

    # Calculate angle with the global Z-axis
    camber_rad = np.arctan2(wheel_up_proj_y, wheel_up_proj_z)
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
    steering_axis_proj_x = steering_axis_vector[0]
    steering_axis_proj_z = steering_axis_vector[2]

    # Angle with the global Z-axis. Positive angle means rearward tilt (positive caster).
    caster_rad = np.arctan2(steering_axis_proj_x, steering_axis_proj_z)
    return np.rad2deg(caster_rad)


def calculate_toe(state: SuspensionState) -> float:
    """
    Calculates the toe angle in degrees.

    Toe is the angle of the wheel's longitudinal axis with respect to the
    vehicle's longitudinal axis (Y-axis), viewed from the top (XY plane).
    Positive toe (toe-in) means the front of the wheel points inwards.

    Args:
        state: The solved SuspensionState to analyze.

    Returns:
        The toe angle in degrees.
    """
    axle_vector = state.get(PointID.AXLE_OUTBOARD) - state.get(PointID.AXLE_INBOARD)

    # Project the axle vector onto the top view plane (XY plane).
    axle_proj_x = axle_vector[0]
    axle_proj_y = axle_vector[1]

    # Angle with the global Y-axis (vehicle transverse axis is X).
    # We use arctan2(y, x) to get the angle from the X-axis, so we swap here.
    toe_rad = np.arctan2(axle_proj_x, axle_proj_y)
    return np.rad2deg(toe_rad)
