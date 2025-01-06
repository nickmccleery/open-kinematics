from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.points.base import DerivedPointSet, Point3D, PointSet
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.utils import build_point_map, get_all_points
from kinematics.solvers.targets import MotionTarget

FTOL = 1e-8  # Convergence tolerance for function value.
XTOL = 1e-8  # Convergence tolerance for independent variables.


class KinematicState:
    """
    Represents the kinematic state of a system.

    Attributes:
        points (dict[PointID, Point3D]): A dictionary of point IDs to 3D points.
        free_points (PointSet): A set of points that are not fixed.
        fixed_points (PointSet): A set of points that are fixed.
        derived_points (dict[PointID, DerivedPoint3D]): A dictionary of derived point IDs to derived 3D points.
        motion_target (MotionTarget): The target motion state.
    """

    def __init__(
        self,
        hard_points: dict[PointID, Point3D],
        derived_points: DerivedPointSet,
        motion_target: MotionTarget,
    ):
        # Take a deep copy of the points at initialization.
        self.hard_points = deepcopy(hard_points)

        # Get point sets; these will be used in numpy array format, so we use the
        # PointSet class to handle that.
        self.free_points = PointSet(
            {id: p for id, p in self.hard_points.items() if not p.fixed}
        )
        self.fixed_points = PointSet(
            {id: p for id, p in self.hard_points.items() if p.fixed}
        )

        self.derived_points = derived_points or {}

        self.motion_target = motion_target

        return

    def update_derived_points(self) -> None:
        self.derived_points.update()

    def update_free_points(self, arr: np.ndarray) -> None:
        self.free_points.update_from_array(arr)
        self.update_derived_points()

    def get_point_position(self, point_id: PointID) -> np.ndarray:
        if point_id in self.derived_points:
            return self.derived_points[point_id].as_array()
        return self.hard_points[point_id].as_array()

    def compute_target_residual(self, displacement: float) -> float:
        if not self.motion_target:
            raise ValueError("No motion target set.")

        current_pos = self.get_point_position(self.motion_target.point_id)
        target = self.motion_target.get_target_value(
            reference_point=self.motion_target.reference_point,
            displacement=displacement,
        )
        error = (
            current_pos[self.motion_target.axis]
            - target.as_array()[self.motion_target.axis]
        )
        return error


class BaseSolver:
    def __init__(self, geometry):
        # Store geometry and compute points.
        self.geometry = geometry
        self.hard_points = build_point_map(get_all_points(self.geometry.hard_points))
        self.derived_points = self.create_derived_points()
        self.motion_target = self.create_motion_target(
            hard_points=self.hard_points,
            derived_points=self.derived_points,
        )

        # Create initial state with points.
        initial_state = KinematicState(
            hard_points=self.hard_points,
            derived_points=self.derived_points,
            motion_target=self.motion_target,
        )

        # Store initial and current state; use deepcopy so they're independent.
        self.initial_state = deepcopy(initial_state)
        self.current_state = deepcopy(initial_state)

        # Initialize constraints.
        self.constraints = []
        self.initialize_constraints()

    def create_derived_points(self) -> DerivedPointSet:
        raise NotImplementedError

    def create_motion_target(
        self,
        hard_points: dict[PointID, Point3D],
        derived_points: DerivedPointSet,
    ) -> MotionTarget:
        raise NotImplementedError

    def initialize_constraints(self) -> None:
        raise NotImplementedError(
            "Derived solvers must implement initialize_constraints."
        )

    def solve_sweep(self, displacements: list[float]) -> list[KinematicState]:
        states = []

        for displacement in displacements:
            iteration_state = deepcopy(self.current_state)
            initial_guess = iteration_state.free_points.as_array().copy()
            initial_guess[2::3] += displacement

            result = least_squares(
                self.compute_residuals,
                initial_guess,
                method="lm",
                ftol=FTOL,
                xtol=XTOL,
                args=(displacement,),
            )

            if not result.success:
                raise RuntimeError(f"Failed to solve for displacement {displacement}m.")

            iteration_state.free_points.update_from_array(result.x)
            states.append(deepcopy(iteration_state))
            self.current_state = iteration_state

        return states

    def compute_residuals(
        self,
        state_array: np.ndarray,
        target_dz: float,
    ) -> np.ndarray:
        state = deepcopy(self.current_state)
        state.free_points.update_from_array(state_array)
        state.update_derived_points()

        residuals = [
            constraint.compute_residual(state.hard_points)
            for constraint in self.constraints
        ]
        target_residual = state.compute_target_residual(target_dz)

        residuals.append(target_residual)
        return np.array(residuals)
