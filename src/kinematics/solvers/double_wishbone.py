from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.schemas import DoubleWishboneGeometry, Point3D, PointID
from kinematics.geometry.utils import get_all_points
from kinematics.solvers.constraints import (
    LinearMotionConstraint,
    PointPointDistanceConstraint,
    VectorOrientationConstraint,
)

FTOL = 1e-8
XTOL = 1e-8


class SuspensionState:
    """Contains all point positions for a given suspension state."""

    def __init__(self, points: dict[PointID, Point3D]):
        """Initialize with mapping of point IDs to Point3D objects."""
        # Create a deep copy of points to ensure independence
        self.points = deepcopy(points)
        self.free_points = {id: p for id, p in self.points.items() if not p.fixed}

    @property
    def free_array(self) -> np.ndarray:
        """Returns coordinates of free points as flat array."""
        return np.concatenate([p.as_array() for p in self.free_points.values()])

    def update_from_array(self, arr: np.ndarray) -> None:
        """Updates free point positions from array."""
        i = 0
        for point in self.free_points.values():
            point.x = float(arr[i])
            point.y = float(arr[i + 1])
            point.z = float(arr[i + 2])
            i += 3

    @classmethod
    def from_geometry(cls, points: list[Point3D]) -> "SuspensionState":
        """Creates initial state from list of geometry points."""
        return cls({p.id: deepcopy(p) for p in points})


class DoubleWishboneSolver:
    """Solves for suspension point positions throughout the range of motion."""

    def __init__(self, geometry: DoubleWishboneGeometry):
        """Initializes the solver with a specific suspension geometry."""
        self.geometry = geometry
        self.points = get_all_points(self.geometry.hard_points)
        self.initial_state = SuspensionState.from_geometry(self.points)

        # Calculate and store initial axle midpoint
        initial_axle_inboard = self.initial_state.points[
            PointID.AXLE_INBOARD
        ].as_array()
        initial_axle_outboard = self.initial_state.points[
            PointID.AXLE_OUTBOARD
        ].as_array()
        self.initial_axle_midpoint = (initial_axle_inboard + initial_axle_outboard) / 2

        # Create a fresh copy for current state
        self.current_state = deepcopy(self.initial_state)

        # Compute constraints
        self.length_constraints = self.compute_distance_constraints()
        self.orientation_constraints = self.compute_orientation_constraints()
        self.linear_constraints = self.compute_linear_constraints()

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

        # Wishbone inboard to outboard constraints
        make_constraint(hp.upper_wishbone.inboard_front, hp.upper_wishbone.outboard)
        make_constraint(hp.upper_wishbone.inboard_rear, hp.upper_wishbone.outboard)
        make_constraint(hp.lower_wishbone.inboard_front, hp.lower_wishbone.outboard)
        make_constraint(hp.lower_wishbone.inboard_rear, hp.lower_wishbone.outboard)

        # Upright length constraint
        make_constraint(hp.upper_wishbone.outboard, hp.lower_wishbone.outboard)

        # Axle length constraint
        make_constraint(hp.wheel_axle.inner, hp.wheel_axle.outer)

        # Axle to ball joint constraints
        make_constraint(hp.wheel_axle.inner, hp.upper_wishbone.outboard)
        make_constraint(hp.wheel_axle.inner, hp.lower_wishbone.outboard)
        make_constraint(hp.wheel_axle.outer, hp.upper_wishbone.outboard)
        make_constraint(hp.wheel_axle.outer, hp.lower_wishbone.outboard)

        # Trackrod length constraint
        make_constraint(hp.track_rod.inner, hp.track_rod.outer)

        # Trackrod constraints
        make_constraint(hp.upper_wishbone.outboard, hp.track_rod.outer)
        make_constraint(hp.lower_wishbone.outboard, hp.track_rod.outer)

        return constraints

    def compute_initial_state(self) -> SuspensionState:
        """Computes the initial suspension state from the geometry."""
        points = get_all_points(self.geometry.hard_points)
        return SuspensionState.from_geometry(points)

    def solve_sweep(self, displacements: list[float]) -> list[SuspensionState]:
        """Solves suspension positions through a sweep of displacements."""
        states = []

        # Reset current state to initial state at the start of sweep
        self.current_state = deepcopy(self.initial_state)

        for z_displacement in displacements:
            # Create a fresh copy of the current state for this iteration
            iteration_state = deepcopy(self.current_state)

            # Prepare initial guess
            initial_guess = iteration_state.free_array.copy()
            initial_guess[2::3] += z_displacement

            result = least_squares(
                self.compute_residuals,
                initial_guess,
                method="lm",
                ftol=FTOL,
                xtol=XTOL,
                args=(z_displacement,),
            )

            if not result.success:
                raise RuntimeError(
                    f"Failed to solve for displacement {z_displacement}m"
                )

            # Update the iteration state with results
            iteration_state.update_from_array(result.x)

            # Store a deep copy of the solved state
            states.append(deepcopy(iteration_state))

            # Update current state for next iteration
            self.current_state = iteration_state

        return states

    def compute_linear_constraints(self) -> list[LinearMotionConstraint]:
        """Computes linear motion constraints for track rod inner point."""
        hp = self.geometry.hard_points
        constraints = []

        # Track rod inner point should only move in Y direction
        # Constrain X and Z position
        constraints.append(
            LinearMotionConstraint(
                point_id=hp.track_rod.inner.id, axis="x", value=hp.track_rod.inner.x
            )
        )

        constraints.append(
            LinearMotionConstraint(
                point_id=hp.track_rod.inner.id, axis="z", value=hp.track_rod.inner.z
            )
        )

        return constraints

    def compute_residuals(
        self, state_array: np.ndarray, target_dz: float
    ) -> np.ndarray:
        """Computes constraint residuals for current suspension state."""
        # Create a fresh state object for residual computation
        state = SuspensionState(self.current_state.points)
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

        # Linear motion constraints
        for constraint in self.linear_constraints:
            point = state.points[constraint.point_id]
            point_coord = getattr(point, constraint.axis)
            residuals.append(point_coord - constraint.value)

        # Target position constraint - using pre-calculated initial midpoint
        current_axle_inboard = state.points[PointID.AXLE_INBOARD].as_array()
        current_axle_outboard = state.points[PointID.AXLE_OUTBOARD].as_array()
        current_axle_midpoint = (current_axle_inboard + current_axle_outboard) / 2

        target_z = self.initial_axle_midpoint[2] + target_dz
        residuals.append(current_axle_midpoint[2] - target_z)

        return np.array(residuals)
