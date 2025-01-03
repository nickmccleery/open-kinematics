from copy import deepcopy
from typing import Dict, Iterator

import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.points.base import Point3D
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.utils import get_all_points
from kinematics.solvers.constraints import BaseConstraint

FTOL = 1e-8  # Convergence tolerance for function value.
XTOL = 1e-8  # Convergence tolerance for independent variables.


class PointSet:
    """
    Represents a set of points in 3D space.

    Attributes:
        points (dict[PointID, Point3D]): A dictionary mapping point identifiers to their
                                         3D coordinates.
    """

    def __init__(self, points: dict[PointID, Point3D]):
        self.points = points

    def __iter__(self) -> Iterator[Point3D]:
        return iter(self.points.values())

    def __getitem__(self, point_id: PointID) -> Point3D:
        return self.points[point_id]

    def as_array(self) -> np.ndarray:
        """
        Convert point positions to a flat array.
        """
        return np.concatenate([p.as_array() for p in self.points.values()])

    def update_from_array(self, arr: np.ndarray) -> None:
        """
        Update point positions from a flat array."
        """
        i = 0
        for point in self.points.values():
            point.x = float(arr[i])
            point.y = float(arr[i + 1])
            point.z = float(arr[i + 2])
            i += 3


class SuspensionState:
    """
    Represents the state of the suspension system.

    Attributes:
        points (dict[PointID, Point3D]): A dictionary mapping point identifiers to their
                                         3D coordinates.
        free_points (PointSet): A set of points that are not fixed.
        fixed_points (PointSet): A set of points that are fixed.
    """

    def __init__(self, points: Dict[PointID, Point3D]):
        self.points = deepcopy(points)
        self.free_points = PointSet(
            {id: p for id, p in self.points.items() if not p.fixed}
        )
        self.fixed_points = PointSet(
            {id: p for id, p in self.points.items() if p.fixed}
        )

    @classmethod
    def from_geometry(cls, points: list[Point3D]) -> "SuspensionState":
        """
        Create an initial state from a list of hardpoints.
        """
        return cls({p.id: deepcopy(p) for p in points})


class BaseSolver:
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

    def initialize_constraints(self) -> list[BaseConstraint]:
        raise NotImplementedError(
            "Derived solvers must implement initialize_constraints."
        )

    def solve_sweep(self, displacements: list[float]) -> list[SuspensionState]:
        states = []

        # Reset current state to initial state at the start of sweep.
        self.current_state = deepcopy(self.initial_state)

        for z_displacement in displacements:
            # Create a fresh copy of the current state for this iteration.
            iteration_state = deepcopy(self.current_state)

            # Prepare initial guess.
            initial_guess = iteration_state.free_points.as_array().copy()
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
            iteration_state.free_points.update_from_array(result.x)

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
        # Create a fresh state object for residual computation.
        state = SuspensionState(self.current_state.points)
        state.free_points.update_from_array(state_array)

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
