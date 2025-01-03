from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.points.base import DerivedPoint3D, Point3D
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


class MotionTarget(Protocol):
    point_id: PointID
    axis: int
    reference_position: np.ndarray

    def get_current_position(self, points: dict[PointID, Point3D]) -> np.ndarray: ...
    def get_target_value(
        self, reference_position: np.ndarray, displacement: float
    ) -> float: ...


@dataclass
class AxisDisplacementTarget(MotionTarget):
    """Target that moves along a specified axis."""

    point_id: PointID
    axis: int  # 0=x, 1=y, 2=z
    reference_position: np.ndarray

    def get_current_position(self, points: dict[PointID, Point3D]) -> np.ndarray:
        return points[self.point_id].as_array()

    def get_target_value(
        self, reference_position: np.ndarray, displacement: float
    ) -> float:
        return reference_position[self.axis] + displacement


class KinematicState:
    def __init__(
        self,
        points: dict[PointID, Point3D],
        derived_points: dict[PointID, DerivedPoint3D] | None = None,
        motion_target: MotionTarget | None = None,
    ):
        self.points = deepcopy(points)
        self.free_points = PointSet(
            {id: p for id, p in self.points.items() if not p.fixed}
        )
        self.fixed_points = PointSet(
            {id: p for id, p in self.points.items() if p.fixed}
        )
        self.derived_points = derived_points or {}
        self.motion_target = motion_target

        self.update_derived_points()

    def update_derived_points(self) -> None:
        """Update positions of all derived points."""
        for point in self.derived_points.values():
            point.update(self.points)

    def update_free_points(self, arr: np.ndarray) -> None:
        """Update free point positions and recompute derived points."""
        self.free_points.update_from_array(arr)
        self.update_derived_points()

    @classmethod
    def from_geometry(
        cls,
        points: list[Point3D],
        derived_points: dict[PointID, DerivedPoint3D] | None = None,
        motion_target: MotionTarget | None = None,
    ) -> "KinematicState":
        """Create an initial state from a list of hardpoints."""
        return cls(
            {p.id: deepcopy(p) for p in points},
            derived_points=derived_points,
            motion_target=motion_target,
        )

    def get_point_position(self, point_id: PointID) -> np.ndarray:
        if point_id in self.derived_points:
            return self.derived_points[point_id].as_array()
        return self.points[point_id].as_array()

    def compute_target_residual(self, displacement: float) -> float:
        if not self.motion_target:
            raise ValueError("No motion target set.")
        current_pos = self.get_point_position(self.motion_target.point_id)
        target = self.motion_target.get_target_value(
            reference_position=self.motion_target.reference_position,
            displacement=displacement,
        )
        return current_pos[self.motion_target.axis] - target


class BaseSolver:
    def __init__(
        self,
        geometry,
        derived_points: dict[PointID, DerivedPoint3D],
        motion_target: MotionTarget,
    ):
        self.geometry = geometry
        self.all_points = get_all_points(self.geometry.hard_points)
        self.initial_state = KinematicState.from_geometry(
            self.all_points,
            derived_points=derived_points,
            motion_target=motion_target,
        )
        self.current_state = deepcopy(self.initial_state)
        self.constraints = self.initialize_constraints()

    def initialize_constraints(self) -> list[BaseConstraint]:
        raise NotImplementedError(
            "Derived solvers must implement initialize_constraints."
        )

    def solve_sweep(self, displacements: list[float]) -> list[KinematicState]:
        states = []
        self.current_state = deepcopy(self.initial_state)

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
        state = KinematicState(
            self.current_state.points,
            derived_points=self.current_state.derived_points,
            motion_target=self.current_state.motion_target,
        )
        state.free_points.update_from_array(state_array)
        state.update_derived_points()

        residuals = [
            constraint.compute_residual(state.points) for constraint in self.constraints
        ]
        residuals.append(state.compute_target_residual(target_dz))

        return np.array(residuals)
