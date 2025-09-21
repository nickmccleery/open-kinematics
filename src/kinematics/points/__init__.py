"""
Point-related functionality for the kinematics module.

This package consolidates all point-related functionality including:
- Point IDs and identifiers
- 3D point data structures
- Point collections for suspension components
- Derived point calculations
- Derived point manager
"""

from .collections import (
    LowerWishbonePoints,
    StrutPoints,
    TrackRodPoints,
    UpperWishbonePoints,
    WheelAxlePoints,
)
from .derived import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)

# Re-export all point-related functionality for clean imports
from .main import Point3D, PointID, get_all_points
from .manager import DerivedPointDefinition, DerivedPointManager
