from copy import deepcopy

from kinematics.geometry.constants import CoordinateAxis, Direction
from kinematics.geometry.points.base import DerivedPoint3D, Point3D
from kinematics.geometry.points.collections import (
    AxleMidPoint,
    WheelCenterPoint,
    WheelOutboardPoint,
)
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.solvers.common import BaseSolver, KinematicState
from kinematics.solvers.constraints import (
    PointFixedAxisConstraint,
    PointOnLineConstraint,
    PointPointDistanceConstraint,
    VectorVectorAngleConstraint,
    lock_point_point_distance,
    lock_vector_angles,
)
from kinematics.solvers.targets import AxisDisplacementTarget, MotionTarget


class DoubleWishboneSolver(BaseSolver):
    def __init__(self, geometry: DoubleWishboneGeometry):
        super().__init__(geometry)

    def create_derived_points(self) -> dict[PointID, DerivedPoint3D]:
        derived_points = {}
        derived_points[PointID.AXLE_MIDPOINT] = AxleMidPoint()
        derived_points[PointID.WHEEL_CENTER] = WheelCenterPoint(
            wheel_offset=self.geometry.configuration.wheel.offset
        )
        derived_points[PointID.WHEEL_OUTBOARD] = WheelOutboardPoint(
            wheel_width=self.geometry.configuration.wheel.width
        )

        return derived_points

    def create_motion_target(self, state: KinematicState) -> MotionTarget:
        """
        Create motion target using the initialized state to get correct derived point
        positions.
        """
        return AxisDisplacementTarget(
            point_id=PointID.AXLE_MIDPOINT,
            axis=CoordinateAxis.Z,
            reference_point=deepcopy(state.derived_points[PointID.AXLE_MIDPOINT]),
        )

    def initialize_constraints(self) -> None:
        """Initialize all constraints specific to double wishbone geometry."""
        self.constraints.extend(self.create_length_constraints())
        self.constraints.extend(self.create_angle_constraints())
        self.constraints.extend(self.create_linear_constraints())
        return

    def create_length_constraints(self) -> list[PointPointDistanceConstraint]:
        """Creates fixed-length constraints for double wishbone geometry."""
        hp = self.geometry.hard_points
        constraints = []

        def make_constraint(p1: Point3D, p2: Point3D):
            constraints.append(lock_point_point_distance(p1, p2))

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
        hp = self.geometry.hard_points
        constraints = []

        def make_constraint(v1: tuple[Point3D, Point3D], v2: tuple[Point3D, Point3D]):
            constraints.append(lock_vector_angles(v1, v2))

        # Kingpin axis to axle orientation constraint.
        make_constraint(
            (hp.upper_wishbone.outboard, hp.lower_wishbone.outboard),
            (hp.wheel_axle.inner, hp.wheel_axle.outer),
        )

        return constraints

    def create_linear_constraints(self) -> list[PointFixedAxisConstraint]:
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
