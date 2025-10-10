"""
Enumeration types for suspension kinematics.
"""

from enum import Enum, IntEnum


class PointID(IntEnum):
    """
    Enumeration of all point identifiers used in the suspension system.
    """

    NOT_ASSIGNED = 0

    LOWER_WISHBONE_INBOARD_FRONT = 1
    LOWER_WISHBONE_INBOARD_REAR = 2
    LOWER_WISHBONE_OUTBOARD = 3

    UPPER_WISHBONE_INBOARD_FRONT = 4
    UPPER_WISHBONE_INBOARD_REAR = 5
    UPPER_WISHBONE_OUTBOARD = 6

    PUSHROD_INBOARD = 7
    PUSHROD_OUTBOARD = 8

    TRACKROD_INBOARD = 9
    TRACKROD_OUTBOARD = 10

    AXLE_INBOARD = 11
    AXLE_OUTBOARD = 12
    AXLE_MIDPOINT = 13

    STRUT_TOP = 14
    STRUT_BOTTOM = 15

    WHEEL_CENTER = 16
    WHEEL_INBOARD = 17
    WHEEL_OUTBOARD = 18

    # Contact patch center is effectively the wheel and tyre centerline's Z minimum,
    # while wheel center on ground is the projection of the wheel centre line onto
    # the ground plane, i.e., world axis system Z=0.
    CONTACT_PATCH_CENTER = 19
    WHEEL_CENTER_ON_GROUND = 20


class Axis(IntEnum):
    """
    Enumeration of the three principal axes in 3D space.
    """

    X = 0
    Y = 1
    Z = 2


class TargetPositionMode(Enum):
    """
    Specifies how a target value should be interpreted.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class Units(Enum):
    """
    Units of measurement for geometric parameters.
    """

    MILLIMETERS = "millimeters"
