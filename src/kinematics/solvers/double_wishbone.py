import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.schemas import DoubleWishboneGeometry, Point3D, PointID
from kinematics.geometry.utils import get_all_points
from kinematics.solvers.constraints import (
    PointPointDistanceConstraint,
    VectorOrientationConstraint,
)

FTOL = 1e-8
XTOL = 1e-8


class SuspensionState:
    """Contains all point positions for a given suspension state."""

    def __init__(self, points: dict[PointID, Point3D]):
        """Initialize with mapping of point IDs to Point3D objects."""
        self.points = points
        self.free_points = {id: p for id, p in points.items() if not p.fixed}

    @property
    def free_array(self) -> np.ndarray:
        """Returns coordinates of free points as flat array."""
        return np.concatenate([p.as_array() for p in self.free_points.values()])

    def update_from_array(self, arr: np.ndarray) -> None:
        """Updates free point positions from array."""
        i = 0
        for point in self.free_points.values():
            point.x = arr[i]
            point.y = arr[i + 1]
            point.z = arr[i + 2]
            i += 3

    @classmethod
    def from_geometry(cls, points: list[Point3D]) -> "SuspensionState":
        """Creates initial state from list of geometry points."""
        return cls({p.id: p for p in points})


class DoubleWishboneSolver:
    """
    Solves for suspension point positions throughout the range of motion.
    """

    def __init__(
        self, geometry: DoubleWishboneGeometry, point_lookup: dict[PointID, Point3D]
    ):
        """
        Initializes the solver with a specific suspension geometry.
        """
        self.geometry = geometry
        self.point_lookup = point_lookup
        self.length_constraints = self.compute_distance_constraints()
        self.orientation_constraints = self.compute_orientation_constraints()
        self.initial_state = self.compute_initial_state()
        self.target_z_displacement = 0.0

    def compute_orientation_constraints(self) -> list[VectorOrientationConstraint]:
        """Computes orientation constraints from the initial geometry."""
        constraints = []
        hp = self.geometry.hard_points

        # Upright to axle orientation constraint
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

        # Upper balljoint-TRE to kingpin axis orientation constraint
        p1 = hp.upper_wishbone.outboard.as_array()
        p2 = hp.track_rod.outer.as_array()
        v1 = p2 - p1

        p1 = hp.upper_wishbone.outboard.as_array()
        p2 = hp.lower_wishbone.outboard.as_array()
        v2 = p2 - p1

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        theta = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        upper_tre_to_axle = VectorOrientationConstraint(
            v1=(hp.upper_wishbone.outboard.id, hp.track_rod.outer.id),
            v2=(hp.wheel_axle.outer.id, hp.wheel_axle.inner.id),
            angle=theta,
        )
        constraints.append(upper_tre_to_axle)

        return constraints

    def compute_distance_constraints(self) -> list[PointPointDistanceConstraint]:
        """Computes all fixed-length constraints from the suspension geometry."""
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

        # Upright length constraint (distance between upper and lower ball joints).
        make_constraint(hp.upper_wishbone.outboard, hp.lower_wishbone.outboard)

        # Axle length constraint.
        make_constraint(hp.wheel_axle.inner, hp.wheel_axle.outer)

        # Axle end to end constraints.
        make_constraint(hp.wheel_axle.inner, hp.wheel_axle.outer)

        # Axle to ball joint constraints.
        make_constraint(hp.wheel_axle.inner, hp.upper_wishbone.outboard)
        make_constraint(hp.wheel_axle.inner, hp.lower_wishbone.outboard)
        make_constraint(hp.wheel_axle.outer, hp.upper_wishbone.outboard)
        make_constraint(hp.wheel_axle.outer, hp.lower_wishbone.outboard)

        # Trackrod constraints.
        make_constraint(hp.upper_wishbone.outboard, hp.track_rod.outer)
        make_constraint(hp.lower_wishbone.outboard, hp.track_rod.outer)

        # Axle to track rod constraints.
        make_constraint(hp.wheel_axle.inner, hp.track_rod.outer)
        make_constraint(hp.wheel_axle.outer, hp.track_rod.outer)

        # Add axle midpoint constraints.
        axle_midpoint_xyz = (
            hp.wheel_axle.inner.as_array() + hp.wheel_axle.outer.as_array()
        ) / 2
        axle_midpoint = Point3D(
            x=axle_midpoint_xyz[0],
            y=axle_midpoint_xyz[1],
            z=axle_midpoint_xyz[2],
            id=PointID.AXLE_MIDPOINT,
        )
        self.point_lookup[axle_midpoint.id] = axle_midpoint

        make_constraint(axle_midpoint, hp.upper_wishbone.outboard)
        make_constraint(axle_midpoint, hp.lower_wishbone.outboard)

        return constraints

    def compute_initial_state(self) -> SuspensionState:
        """Computes the initial suspension state from the geometry."""
        points = get_all_points(self.geometry.hard_points)
        return SuspensionState.from_geometry(points)

    def solve_positions(self, z_displacement: float) -> SuspensionState:
        """Solves for the suspension state at a given vertical displacement."""
        self.target_z_displacement = z_displacement

        initial_guess = self.initial_state.free_array
        initial_guess[2::3] += z_displacement  # Apply z displacement to all points.

        result = least_squares(
            self.compute_residuals, initial_guess, method="lm", ftol=FTOL, xtol=XTOL
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve suspension position for displacement {z_displacement}m."
            )

        new_state = SuspensionState(self.initial_state.points.copy())
        new_state.update_from_array(result.x)
        return new_state

    def compute_residuals(self, state_array: np.ndarray) -> np.ndarray:
        """Computes constraint residuals for the current suspension state."""
        state = SuspensionState(self.initial_state.points.copy())
        state.update_from_array(state_array)

        residuals = []

        # Length constraints
        for constraint in self.length_constraints:
            p1 = state.points[constraint.p1].as_array()
            p2 = state.points[constraint.p2].as_array()
            current_length = np.linalg.norm(p1 - p2)
            residuals.append(current_length - constraint.distance)

        # Orientation constraints
        for constraint in self.orientation_constraints:
            v1_start = state.points[constraint.v1[0]].as_array()
            v1_end = state.points[constraint.v1[1]].as_array()
            v2_start = state.points[constraint.v2[0]].as_array()
            v2_end = state.points[constraint.v2[1]].as_array()

            v1 = v1_end - v1_start
            v2 = v2_end - v2_start

            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)

            current_angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            residuals.append(current_angle - constraint.angle)

        # Target position constraint
        axle_inner_id = state.points[PointID.AXLE_INBOARD].as_array()
        axle_outer_id = state.points[PointID.AXLE_OUTBOARD].as_array()
        initial_midpoint = (
            self.initial_state.points[PointID.AXLE_INBOARD].as_array()
            + self.initial_state.points[PointID.AXLE_OUTBOARD].as_array()
        ) / 2
        current_midpoint = (axle_inner_id + axle_outer_id) / 2
        target_z = initial_midpoint[2] + self.target_z_displacement
        residuals.append(current_midpoint[2] - target_z)

        return np.array(residuals)
