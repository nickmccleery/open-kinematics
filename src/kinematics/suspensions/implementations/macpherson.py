"""
MacPherson strut suspension implementation.
"""

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import numpy as np

from kinematics.constraints import Constraint, DistanceConstraint
from kinematics.enums import PointID
from kinematics.points.derived.definitions import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.points.derived.manager import DerivedPointsManager, DerivedPointsSpec
from kinematics.state import SuspensionState
from kinematics.suspensions.core.collections import LowerWishbonePoints, WheelAxlePoints
from kinematics.suspensions.core.geometry import SuspensionGeometry
from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.types import Vec3
from kinematics.vector_utils.geometric import compute_point_point_distance
from kinematics.visualization.main import LinkVisualization


@dataclass
class StrutPoints:
    """
    Points defining the strut geometry.

    Attributes:
        inboard: Inboard strut mounting point coordinates.
        outboard: Outboard strut mounting point coordinates.
    """

    inboard: dict[str, float]
    outboard: dict[str, float]


@dataclass
class MacPhersonHardPoints:
    """
    Hard point collection for MacPherson strut suspension.

    Attributes:
        lower_wishbone: Points defining the lower wishbone geometry.
        strut: Points defining the strut geometry.
        wheel_axle: Points defining the wheel axle geometry.
    """

    lower_wishbone: LowerWishbonePoints
    strut: StrutPoints
    wheel_axle: WheelAxlePoints


# Geometry model
@dataclass
class MacPhersonGeometry(SuspensionGeometry):
    """
    MacPherson strut suspension geometry definition.

    Extends the base SuspensionGeometry with MacPherson specific hard points.

    Attributes:
        hard_points: Collection of all hard point coordinates for the suspension.
    """

    hard_points: MacPhersonHardPoints

    def validate(self) -> bool:
        """
        Validate the MacPherson geometry configuration.

        Returns:
            True if geometry is valid.
        """
        # ! TODO: Actually validate this geometry.
        return True


# Provider implementation
class MacPhersonProvider(SuspensionProvider):
    """
    Concrete implementation of SuspensionProvider for MacPherson strut geometry.
    """

    def __init__(self, geometry: MacPhersonGeometry):
        """
        Initialize the MacPherson provider.

        Args:
            geometry: MacPherson geometry configuration.
        """
        self.geometry = geometry

    def initial_state(self) -> SuspensionState:
        """
        Create initial suspension state from geometry hard points.

        Converts the hard point coordinates from the geometry into a SuspensionState
        with both explicitly defined and derived points.

        Returns:
            Initial suspension state with all point positions.
        """
        positions = {}
        hard_points = self.geometry.hard_points

        # Lower wishbone.
        lwb = hard_points.lower_wishbone
        positions[PointID.LOWER_WISHBONE_INBOARD_FRONT] = np.array(
            [lwb.inboard_front["x"], lwb.inboard_front["y"], lwb.inboard_front["z"]]
        )
        positions[PointID.LOWER_WISHBONE_INBOARD_REAR] = np.array(
            [lwb.inboard_rear["x"], lwb.inboard_rear["y"], lwb.inboard_rear["z"]]
        )
        positions[PointID.LOWER_WISHBONE_OUTBOARD] = np.array(
            [lwb.outboard["x"], lwb.outboard["y"], lwb.outboard["z"]]
        )

        # Strut.
        strut = hard_points.strut
        positions[PointID.STRUT_INBOARD] = np.array(
            [strut.inboard["x"], strut.inboard["y"], strut.inboard["z"]]
        )
        positions[PointID.STRUT_OUTBOARD] = np.array(
            [strut.outboard["x"], strut.outboard["y"], strut.outboard["z"]]
        )

        # Wheel axle.
        wa = hard_points.wheel_axle
        positions[PointID.AXLE_INBOARD] = np.array(
            [wa.inner["x"], wa.inner["y"], wa.inner["z"]]
        )
        positions[PointID.AXLE_OUTBOARD] = np.array(
            [wa.outer["x"], wa.outer["y"], wa.outer["z"]]
        )

        # Calculate derived points to create a complete initial state.
        derived_spec = self.derived_spec()
        derived_resolver = DerivedPointsManager(derived_spec)
        derived_resolver.update_in_place(positions)

        return SuspensionState(positions=positions, free_points=set(self.free_points()))

    def free_points(self) -> Sequence[PointID]:
        """
        Define which points the solver can move during optimization.

        Returns:
            Sequence of point IDs that are free to move.
        """
        return [
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.STRUT_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        ]

    def derived_spec(self) -> DerivedPointsSpec:
        """
        Define specifications for computing derived points from free points.

        Returns:
            Specification containing functions and dependencies for derived points.
        """
        wheel_cfg = self.geometry.configuration.wheel

        functions = {
            PointID.AXLE_MIDPOINT: get_axle_midpoint,
            PointID.WHEEL_CENTER: partial(
                get_wheel_center, wheel_offset=wheel_cfg.offset
            ),
            PointID.WHEEL_INBOARD: partial(
                get_wheel_inboard, wheel_width=wheel_cfg.width
            ),
            PointID.WHEEL_OUTBOARD: partial(
                get_wheel_outboard, wheel_width=wheel_cfg.width
            ),
        }

        dependencies = {
            PointID.AXLE_MIDPOINT: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_CENTER: {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
            PointID.WHEEL_INBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
            PointID.WHEEL_OUTBOARD: {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        }

        return DerivedPointsSpec(functions=functions, dependencies=dependencies)

    def constraints(self) -> list[Constraint]:
        """
        Build the complete set of geometric constraints for MacPherson strut suspension.

        Returns:
            List of constraints that must be satisfied during kinematic solving.
        """
        constraints: list[Constraint] = []
        initial_state = self.initial_state()

        # Distance constraints.
        length_pairs = [
            (PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.STRUT_INBOARD, PointID.STRUT_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD),
            (PointID.AXLE_INBOARD, PointID.STRUT_OUTBOARD),
            (PointID.AXLE_OUTBOARD, PointID.STRUT_OUTBOARD),
        ]
        for p1, p2 in length_pairs:
            target_distance = compute_point_point_distance(
                initial_state.positions[p1], initial_state.positions[p2]
            )
            constraints.append(DistanceConstraint(p1, p2, target_distance))

        return constraints

    def compute_side_view_instant_center(self, state: SuspensionState) -> Vec3:
        raise NotImplementedError("SVIC calcs not defined for MacPherson strut.")

    def get_visualization_links(self) -> list[LinkVisualization]:
        """
        Get the visualization links for rendering the MacPherson strut suspension.

        Returns:
            List of link definitions specifying how to visualize the suspension
            structure, including lower control arm, strut, and upright connections.
        """
        return [
            LinkVisualization(
                points=[
                    PointID.LOWER_WISHBONE_INBOARD_FRONT,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_INBOARD_REAR,
                ],
                color="dodgerblue",
                label="Lower Control Arm",
            ),
            LinkVisualization(
                points=[PointID.STRUT_INBOARD, PointID.STRUT_OUTBOARD],
                color="darkorange",
                label="Strut",
            ),
            LinkVisualization(
                points=[PointID.LOWER_WISHBONE_OUTBOARD, PointID.STRUT_OUTBOARD],
                color="slategrey",
                label="Upright",
            ),
            LinkVisualization(
                points=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD],
                color="forestgreen",
                label="Axle",
            ),
        ]
