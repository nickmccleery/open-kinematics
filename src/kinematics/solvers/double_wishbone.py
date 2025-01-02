from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.schemas import Point3D, PointID
from kinematics.geometry.utils import get_all_points
from kinematics.solvers.constraints import (
    BaseConstraint,
    FixedAxisConstraint,
    PointPointDistanceConstraint,
    VectorOrientationConstraint,
)

FTOL = 1e-8  # Convergence tolerance for function value.
XTOL = 1e-8  # Convergence tolerance for independent variables.


class SuspensionState:
    """Represents the state of a suspension system, tracking point positions."""

    def __init__(self, points: dict[PointID, Point3D]):
        # Create a deep copy of points to ensure independence.
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


class BaseSolver(ABC):
    """Base class for suspension solvers."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.points = get_all_points(self.geometry.hard_points)
        self.initial_state = SuspensionState.from_geometry(self.points)

        # Calculate and store initial axle midpoint.
        initial_axle_inboard = self.initial_state.points[
            PointID.AXLE_INBOARD
        ].as_array()
        initial_axle_outboard = self.initial_state.points[
            PointID.AXLE_OUTBOARD
        ].as_array()
        self.initial_axle_midpoint = (initial_axle_inboard + initial_axle_outboard) / 2

        # Create a fresh copy for current state.
        self.current_state = deepcopy(self.initial_state)

        # Initialize all constraints.
        self.constraints = self.initialize_constraints()

    @abstractmethod
    def initialize_constraints(self) -> list[BaseConstraint]:
        """Initialize all constraints for the suspension geometry."""
        pass

    def solve_sweep(self, displacements: list[float]) -> list[SuspensionState]:
        """Solves suspension positions through a sweep of displacements."""
        states = []

        # Reset current state to initial state at the start of sweep.
        self.current_state = deepcopy(self.initial_state)

        for z_displacement in displacements:
            # Create a fresh copy of the current state for this iteration.
            iteration_state = deepcopy(self.current_state)

            # Prepare initial guess.
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
                    f"Failed to solve for displacement {z_displacement}m."
                )

            # Update the iteration state with results.
            iteration_state.update_from_array(result.x)

            # Store a deep copy of the solved state.
            states.append(deepcopy(iteration_state))

            # Update current state for next iteration.
            self.current_state = iteration_state

        return states

    def compute_residuals(
        self,
        state_array: np.ndarray,
        target_dz: float,
    ) -> np.ndarray:
        """Computes constraint residuals for current suspension state."""
        # Create a fresh state object for residual computation.
        state = SuspensionState(self.current_state.points)
        state.update_from_array(state_array)

        # Compute residuals for all constraints.
        residuals = [
            constraint.compute_residual(state.points) for constraint in self.constraints
        ]

        # Add target position constraint.
        current_axle_inboard = state.points[PointID.AXLE_INBOARD].as_array()
        current_axle_outboard = state.points[PointID.AXLE_OUTBOARD].as_array()
        current_axle_midpoint = (current_axle_inboard + current_axle_outboard) / 2

        target_z = self.initial_axle_midpoint[2] + target_dz
        residuals.append(current_axle_midpoint[2] - target_z)

        return np.array(residuals)


class DoubleWishboneSolver(BaseSolver):
    """Solver for double wishbone suspension geometry."""

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
