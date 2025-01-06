from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.points.base import (
    DerivedPoint3D,
    DerivedPointSet,
    Point3D,
    PointSet,
)
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.utils import build_point_map, get_all_points
from kinematics.solvers.targets import MotionTarget

FTOL = 1e-8  # Convergence tolerance for function value.
XTOL = 1e-8  # Convergence tolerance for independent variables.


class KinematicState:
    """
    Represents the point positions in a kinematic system at a given instant.

    Attributes:
        all_points (dict[PointID, Point3D]): Complete dictionary of all point positions.
        free_points (PointSet): Set of points that can move.
        fixed_points (PointSet): Set of points with fixed positions.
        derived_points (DerivedPointSet): Set of points derived from other points.
    """

    def __init__(
        self,
        hard_points: dict[PointID, Point3D],
        derived_points: dict[PointID, DerivedPoint3D],
    ):
        # Store the complete set of points.
        self.all_points = deepcopy(hard_points)

        # Separate into free and fixed point sets.
        self.free_points = PointSet(
            {id: p for id, p in self.all_points.items() if not p.fixed}
        )
        self.fixed_points = PointSet(
            {id: p for id, p in self.all_points.items() if p.fixed}
        )

        # Initialize derived points—requires a hard point reference.
        self.derived_points = DerivedPointSet(
            hard_points={id: p for id, p in hard_points.items()}
        )
        for dp in derived_points.values():
            self.derived_points.add(dp)

    def update_free_points(self, arr: np.ndarray) -> None:
        self.free_points.update_from_array(arr)

        # Sync changes back to main points dictionary.
        for point_id, point in self.free_points.points.items():
            self.all_points[point_id] = point

        # Update derived points with new positions.
        self.update_derived_points()

    def update_derived_points(self) -> None:
        """
        Update all derived point positions based on current point positions.
        """
        self.derived_points.update(self.all_points)

    def get_point_position(self, point_id: PointID) -> np.ndarray:
        """
        Get the current position of any point (derived or hard) by ID.
        """
        if point_id in self.derived_points:
            return self.derived_points[point_id].as_array()
        return self.all_points[point_id].as_array()


class BaseSolver:
    def __init__(self, geometry):
        # Store geometry and compute initial points.
        self.geometry = geometry
        self.hard_points = build_point_map(get_all_points(self.geometry.hard_points))
        self.derived_points = self.create_derived_points()

        # Create and initialize states first.
        self.initial_state = KinematicState(
            hard_points=self.hard_points,
            derived_points=self.derived_points,
        )
        # Make sure derived points are initialized.
        self.initial_state.update_derived_points()

        # Create current state as a copy of the initial state.
        self.current_state = deepcopy(self.initial_state)

        # Create motion target using initialized state.
        self.motion_target = self.create_motion_target(state=self.current_state)

        # Initialize constraints.
        self.constraints = []
        self.initialize_constraints()

    def create_derived_points(self) -> dict[PointID, DerivedPoint3D]:
        raise NotImplementedError

    def create_motion_target(self, state: KinematicState) -> MotionTarget:
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
            constraint.compute_residual(state.all_points)
            for constraint in self.constraints
        ]
        target_residual = self.compute_target_residual(
            state=state, displacement=target_dz
        )

        residuals.append(target_residual)
        return np.array(residuals)

    def compute_target_residual(
        self, state: KinematicState, displacement: float
    ) -> float:
        if not self.motion_target:
            raise ValueError("No motion target set.")

        current_pos = state.get_point_position(self.motion_target.point_id)
        target = self.motion_target.get_target_value(
            reference_point=self.motion_target.reference_point,
            displacement=displacement,
        )
        error = (
            current_pos[self.motion_target.axis]
            - target.as_array()[self.motion_target.axis]
        )
        return error
