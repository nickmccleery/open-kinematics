from copy import deepcopy

import numpy as np

from kinematics.geometry.points.base import Point3D
from kinematics.geometry.points.collections import AxleMidPoint, WheelCenterPoint
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.geometry.utils import get_all_points
from kinematics.solvers.common import BaseSolver
from kinematics.solvers.constraints import (
    BaseConstraint,
    FixedAxisConstraint,
    PointPointDistanceConstraint,
    VectorOrientationConstraint,
)
from kinematics.solvers.targets import AxisDisplacementTarget


class DoubleWishboneSolver(BaseSolver):
    """Solver for double wishbone suspension geometry."""

    def __init__(self, geometry: DoubleWishboneGeometry):
        # Get point map.
        point_map = {p.id: deepcopy(p) for p in get_all_points(geometry.hard_points)}

        # Create derived points.
        derived_points = {
            PointID.AXLE_MIDPOINT: AxleMidPoint(
                deps=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD]
            ),
            PointID.WHEEL_CENTER: WheelCenterPoint(
                deps=[PointID.AXLE_OUTBOARD, PointID.AXLE_INBOARD],
                wheel_offset=geometry.configuration.wheel.offset,
            ),
        }

        # Initial update of derived points.
        for point in derived_points.values():
            point.update(point_map)

        # Create target
        target = AxisDisplacementTarget(
            point_id=PointID.AXLE_MIDPOINT,
            axis=2,  # Z-axis
            reference_position=derived_points[PointID.AXLE_MIDPOINT].as_array(),
        )

        super().__init__(
            geometry=geometry, derived_points=derived_points, motion_target=target
        )

    def initialize_constraints(self) -> list[BaseConstraint]:
        """Initialize all constraints specific to double wishbone geometry."""
        constraints = []
        constraints.extend(self.create_length_constraints())
        constraints.extend(self.create_orientation_constraints())
        constraints.extend(self.create_linear_constraints())
        return constraints

    def create_length_constraints(self) -> list[PointPointDistanceConstraint]:
        """Creates fixed-length constraints for double wishbone geometry."""
        hp = self.geometry.hard_points
        constraints = []

        def make_constraint(p1: Point3D, p2: Point3D):
            """Creates a constraint between two points."""
            length = float(np.linalg.norm(p1.as_array() - p2.as_array()))
            constraints.append(PointPointDistanceConstraint(p1.id, p2.id, length))

        # Wishbone inboard to outboard constraints.
        make_constraint(hp.upper_wishbone.inboard_front, hp.upper_wishbone.outboard)
        make_constraint(hp.upper_wishbone.inboard_rear, hp.upper_wishbone.outboard)
        make_constraint(hp.lower_wishbone.inboard_front, hp.lower_wishbone.outboard)
        make_constraint(hp.lower_wishbone.inboard_rear, hp.lower_wishbone.outboard)

        # Upright length constraint.
        make_constraint(hp.upper_wishbone.outboard, hp.lower_wishbone.outboard)

        # Axle length constraint.
        make_constraint(hp.wheel_axle.inner, hp.wheel_axle.outer)

        # Axle to ball joint constraints.
        make_constraint(hp.wheel_axle.inner, hp.upper_wishbone.outboard)
        make_constraint(hp.wheel_axle.inner, hp.lower_wishbone.outboard)
        make_constraint(hp.wheel_axle.outer, hp.upper_wishbone.outboard)
        make_constraint(hp.wheel_axle.outer, hp.lower_wishbone.outboard)

        # Trackrod length constraint.
        make_constraint(hp.track_rod.inner, hp.track_rod.outer)

        # Trackrod constraints.
        make_constraint(hp.upper_wishbone.outboard, hp.track_rod.outer)
        make_constraint(hp.lower_wishbone.outboard, hp.track_rod.outer)

        # Axle to TRE constraints.
        make_constraint(hp.wheel_axle.inner, hp.track_rod.outer)
        make_constraint(hp.wheel_axle.outer, hp.track_rod.outer)

        return constraints

    def create_orientation_constraints(self) -> list[VectorOrientationConstraint]:
        """Creates orientation constraints for double wishbone geometry."""
        hp = self.geometry.hard_points
        constraints = []

        # Upright to axle orientation constraint.
        initial_upright = (
            hp.upper_wishbone.outboard.as_array()
            - hp.lower_wishbone.outboard.as_array()
        )
        initial_axle = hp.wheel_axle.outer.as_array() - hp.wheel_axle.inner.as_array()

        initial_upright = initial_upright / np.linalg.norm(initial_upright)
        initial_axle = initial_axle / np.linalg.norm(initial_axle)

        initial_angle = np.arccos(
            np.clip(np.dot(initial_upright, initial_axle), -1.0, 1.0)
        )

        axle_to_upright = VectorOrientationConstraint(
            v1=(hp.upper_wishbone.outboard.id, hp.lower_wishbone.outboard.id),
            v2=(hp.wheel_axle.inner.id, hp.wheel_axle.outer.id),
            angle=initial_angle,
        )
        constraints.append(axle_to_upright)

        return constraints

    def create_linear_constraints(self) -> list[FixedAxisConstraint]:
        """Creates linear motion constraints for double wishbone geometry."""
        hp = self.geometry.hard_points
        constraints = []

        # Track rod inner point should only move in Y direction.
        # Constrain X and Z position.
        constraints.append(
            FixedAxisConstraint(
                point_id=hp.track_rod.inner.id, axis="x", value=hp.track_rod.inner.x
            )
        )

        constraints.append(
            FixedAxisConstraint(
                point_id=hp.track_rod.inner.id, axis="z", value=hp.track_rod.inner.z
            )
        )

        return constraints
