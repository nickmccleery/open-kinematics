from copy import deepcopy

import numpy as np

from kinematics.geometry.constants import CoordinateAxis, Direction
from kinematics.geometry.points.base import DerivedPoint3D, Point3D
from kinematics.geometry.points.collections import AxleMidPoint, WheelCenterPoint
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.solvers.common import BaseSolver
from kinematics.solvers.constraints import (
    BaseConstraint,
    PointFixedAxisConstraint,
    PointOnLineConstraint,
    PointPointDistanceConstraint,
    VectorVectorAngleConstraint,
)
from kinematics.solvers.targets import AxisDisplacementTarget, MotionTarget


class DoubleWishboneSolver(BaseSolver):
    def __init__(self, geometry: DoubleWishboneGeometry):
        super().__init__(geometry)

    def create_derived_points(self) -> dict[PointID, DerivedPoint3D]:
        derived_points = {
            PointID.AXLE_MIDPOINT: AxleMidPoint(
                deps=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD]
            ),
            PointID.WHEEL_CENTER: WheelCenterPoint(
                deps=[PointID.AXLE_OUTBOARD, PointID.AXLE_INBOARD],
                wheel_offset=self.geometry.configuration.wheel.offset,
            ),
        }
        return derived_points

    def create_motion_target(
        self, derived_points: dict[PointID, DerivedPoint3D]
    ) -> MotionTarget:
        return AxisDisplacementTarget(
            point_id=PointID.AXLE_MIDPOINT,
            axis=CoordinateAxis.Z,
            reference_point=deepcopy(derived_points[PointID.AXLE_MIDPOINT]),
        )

    def initialize_constraints(self) -> list[BaseConstraint]:
        """Initialize all constraints specific to double wishbone geometry."""
        constraints = []
        constraints.extend(self.create_length_constraints())
        constraints.extend(self.create_angle_constraints())
        constraints.extend(self.create_linear_constraints())
        return constraints

    def create_length_constraints(self) -> list[PointPointDistanceConstraint]:
        """Creates fixed-length constraints for double wishbone geometry."""
        hp = self.geometry.hard_points
        constraints = []

        def make_constraint(p1: Point3D, p2: Point3D):
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

    def create_angle_constraints(self) -> list[VectorVectorAngleConstraint]:
        """Creates orientation constraints for double wishbone geometry."""
        hp = self.geometry.hard_points
        constraints = []

        def make_constraint(v1: tuple[Point3D, Point3D], v2: tuple[Point3D, Point3D]):
            v1_vec = v1[1].as_array() - v1[0].as_array()
            v2_vec = v2[1].as_array() - v2[0].as_array()

            v1_vec = v1_vec / np.linalg.norm(v1_vec)
            v2_vec = v2_vec / np.linalg.norm(v2_vec)

            theta = np.arccos(np.clip(np.dot(v1_vec, v2_vec), -1.0, 1.0))

            constraints.append(
                VectorVectorAngleConstraint(
                    v1=(v1[0].id, v1[1].id), v2=(v2[0].id, v2[1].id), angle=theta
                )
            )

        # Kingpin axis to axle orientation constraint.
        make_constraint(
            (hp.upper_wishbone.outboard, hp.lower_wishbone.outboard),
            (hp.wheel_axle.inner, hp.wheel_axle.outer),
        )

        return constraints

    def create_linear_constraints(self) -> list[PointFixedAxisConstraint]:
        """Creates linear motion constraints for double wishbone geometry."""
        hp = self.geometry.hard_points
        constraints = []

        # Track rod inner point should only move in Y direction. Note that this
        # could also be achieved with two PointFixedAxisConstraints, but this is
        # more concise.
        constraints.append(
            PointOnLineConstraint(
                point_id=hp.track_rod.inner.id,
                line_point=hp.track_rod.inner.id,
                line_direction=Direction.y,
            ),
        )

        return constraints
