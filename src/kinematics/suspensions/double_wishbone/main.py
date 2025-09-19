# In src/kinematics/suspensions/double_wishbone/main.py
from typing import List

from kinematics.api import solve_kinematics
from kinematics.solvers.core import PointTargetSet
from kinematics.suspensions.double_wishbone.geometry import DoubleWishboneGeometry
from kinematics.types.state import Positions


def solve_suspension(
    geometry: DoubleWishboneGeometry, point_targets: List[PointTargetSet]
) -> List[Positions]:
    """
    High-level function to solve a double wishbone suspension sweep.

    This is a convenience wrapper around the generic `solve_kinematics` API.
    """
    return solve_kinematics(geometry=geometry, point_targets=point_targets)
    return solve_kinematics(geometry=geometry, point_targets=point_targets)
