from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares

from kinematics.geometry.schemas import DoubleWishboneGeometry, Point3D

FTOL = 1e-8
XTOL = 1e-8


class LinkLengthConstraint(NamedTuple):
    """
    Records a fixed-length constraint between two points in the suspension system.
    """

    point1_name: str
    point2_name: str
    length: float


class VectorOrientationConstraint(NamedTuple):
    """
    Maintains the angle between two vectors defined by points.
    """

    base_vector_start_name: str  # e.g., "upper_outboard"
    base_vector_end_name: str  # e.g., "lower_outboard"
    target_vector_start_name: str  # e.g., "axle_inner"
    target_vector_end_name: str  # e.g., "axle_outer"
    target_angle: float  # Angle to maintain (in radians)


@dataclass
class SuspensionState:
    """Contains all moving point positions for a given suspension state."""

    upper_outboard: np.ndarray  # Upper ball joint.
    lower_outboard: np.ndarray  # Lower ball joint.
    axle_inner: np.ndarray  # Inner end of wheel axle.
    axle_outer: np.ndarray  # Outer end of wheel axle.
    track_rod_outer: np.ndarray  # Outer end of track rod.
    # track_rod_inner: np.ndarray  # Inner end of track rod.

    def as_array(self) -> np.ndarray:
        """Converts the suspension state to a flat array for optimization."""
        return np.concatenate(
            [
                self.upper_outboard,
                self.lower_outboard,
                self.axle_inner,
                self.axle_outer,
                self.track_rod_outer,
            ]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SuspensionState":
        """Creates a suspension state from a flat array of coordinates."""
        return cls(
            upper_outboard=arr[0:3],
            lower_outboard=arr[3:6],
            axle_inner=arr[6:9],
            axle_outer=arr[9:12],
            track_rod_outer=arr[12:15],
        )


class DoubleWishboneSolver:
    """
    Solves for suspension point positions throughout the range of motion.
    """

    def __init__(self, geometry: DoubleWishboneGeometry):
        """
        Initializes the solver with a specific suspension geometry.
        """
        self.geometry = geometry
        self.length_constraints = self.compute_distance_constraints()
        self.orientation_constraints = self.compute_orientation_constraints()
        self.initial_state = self.compute_initial_state()
        self.target_z_displacement = 0.0

    def compute_orientation_constraints(self) -> list[VectorOrientationConstraint]:
        """
        Computes orientation constraints from the initial geometry.
        """
        constraints = []

        hp = self.geometry.hard_points

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
            base_vector_start_name="upper_outboard",
            base_vector_end_name="lower_outboard",
            target_vector_start_name="axle_inner",
            target_vector_end_name="axle_outer",
            target_angle=initial_angle,
        )

        constraints.append(axle_to_upright)

        # Upper balljoint-TRE to kingpin axis orientation constraint.
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
            base_vector_start_name="upper_outboard",
            base_vector_end_name="track_rod_outer",
            target_vector_start_name="axle_outer",
            target_vector_end_name="axle_inner",
            target_angle=theta,
        )

        constraints.append(upper_tre_to_axle)

        return constraints

    def compute_distance_constraints(self) -> list[LinkLengthConstraint]:
        """Computes all fixed-length constraints from the suspension geometry."""
        hp = self.geometry.hard_points
        constraints = []

        def make_constraint(p1: Point3D, p2: Point3D, name1: str, name2: str):
            """Creates a constraint between two points."""
            length = float(np.linalg.norm(p1.as_array() - p2.as_array()))
            constraints.append(LinkLengthConstraint(name1, name2, length))

        # Wishbone inboard to outboard constraints.
        make_constraint(
            hp.upper_wishbone.inboard_front,
            hp.upper_wishbone.outboard,
            "upper_inboard_front",
            "upper_outboard",
        )
        make_constraint(
            hp.upper_wishbone.inboard_rear,
            hp.upper_wishbone.outboard,
            "upper_inboard_rear",
            "upper_outboard",
        )
        make_constraint(
            hp.lower_wishbone.inboard_front,
            hp.lower_wishbone.outboard,
            "lower_inboard_front",
            "lower_outboard",
        )
        make_constraint(
            hp.lower_wishbone.inboard_rear,
            hp.lower_wishbone.outboard,
            "lower_inboard_rear",
            "lower_outboard",
        )

        # Upright length constraint (distance between upper and lower ball joints).
        make_constraint(
            hp.upper_wishbone.outboard,
            hp.lower_wishbone.outboard,
            "upper_outboard",
            "lower_outboard",
        )

        # Axle length constraint.
        make_constraint(
            hp.wheel_axle.inner,
            hp.wheel_axle.outer,
            "axle_inner",
            "axle_outer",
        )

        # Axle end to end constraints.
        make_constraint(
            hp.wheel_axle.inner,
            hp.wheel_axle.outer,
            "axle_inner",
            "axle_outer",
        )

        # Axle to ball joint constraints.
        make_constraint(
            hp.wheel_axle.inner,
            hp.upper_wishbone.outboard,
            "axle_inner",
            "upper_outboard",
        )
        make_constraint(
            hp.wheel_axle.inner,
            hp.lower_wishbone.outboard,
            "axle_inner",
            "lower_outboard",
        )
        make_constraint(
            hp.wheel_axle.outer,
            hp.upper_wishbone.outboard,
            "axle_outer",
            "upper_outboard",
        )
        make_constraint(
            hp.wheel_axle.outer,
            hp.lower_wishbone.outboard,
            "axle_outer",
            "lower_outboard",
        )

        # Trackrod constraints.
        make_constraint(
            hp.upper_wishbone.outboard,
            hp.track_rod.outer,
            "upper_outboard",
            "track_rod_outer",
        )

        make_constraint(
            hp.lower_wishbone.outboard,
            hp.track_rod.outer,
            "lower_outboard",
            "track_rod_outer",
        )

        # Axle to track rod constraints.
        make_constraint(
            hp.wheel_axle.inner,
            hp.track_rod.outer,
            "axle_inner",
            "track_rod_outer",
        )

        make_constraint(
            hp.wheel_axle.outer,
            hp.track_rod.outer,
            "axle_outer",
            "track_rod_outer",
        )

        # Add axle midpoint constraints.
        axle_midpoint = (
            hp.wheel_axle.inner.as_array() + hp.wheel_axle.outer.as_array()
        ) / 2

        make_constraint(
            Point3D(x=axle_midpoint[0], y=axle_midpoint[1], z=axle_midpoint[2]),
            hp.upper_wishbone.outboard,
            "axle_midpoint",
            "upper_outboard",
        )

        make_constraint(
            Point3D(x=axle_midpoint[0], y=axle_midpoint[1], z=axle_midpoint[2]),
            hp.lower_wishbone.outboard,
            "axle_midpoint",
            "lower_outboard",
        )

        return constraints

    def compute_initial_state(self) -> SuspensionState:
        """
        Computes the initial suspension state from the geometry.
        """
        hp = self.geometry.hard_points
        return SuspensionState(
            upper_outboard=hp.upper_wishbone.outboard.as_array(),
            lower_outboard=hp.lower_wishbone.outboard.as_array(),
            axle_inner=hp.wheel_axle.inner.as_array(),
            axle_outer=hp.wheel_axle.outer.as_array(),
            track_rod_outer=hp.track_rod.outer.as_array(),
        )

    def solve_positions(self, z_displacement: float) -> SuspensionState:
        """
        Solves for the suspension state at a given vertical displacement.
        """
        self.target_z_displacement = z_displacement

        initial_guess = self.initial_state.as_array()
        initial_guess[2::3] += z_displacement  # Apply z displacement to all points.

        result = least_squares(
            self.compute_residuals, initial_guess, method="lm", ftol=FTOL, xtol=XTOL
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve suspension position for displacement {z_displacement}m."
            )

        return SuspensionState.from_array(result.x)

    def compute_residuals(self, state_array: np.ndarray) -> np.ndarray:
        """
        Computes constraint residuals for the current suspension state.
        """
        state = SuspensionState.from_array(state_array)
        hp = self.geometry.hard_points

        axle_midpoint = (state.axle_inner + state.axle_outer) / 2
        point_map = {
            "upper_inboard_front": hp.upper_wishbone.inboard_front.as_array(),
            "upper_inboard_rear": hp.upper_wishbone.inboard_rear.as_array(),
            "lower_inboard_front": hp.lower_wishbone.inboard_front.as_array(),
            "lower_inboard_rear": hp.lower_wishbone.inboard_rear.as_array(),
            "upper_outboard": state.upper_outboard,
            "lower_outboard": state.lower_outboard,
            "track_rod_outer": state.track_rod_outer,
            # "track_rod_inner": state.track_rod_inner,
            "axle_inner": state.axle_inner,
            "axle_outer": state.axle_outer,
            "axle_midpoint": axle_midpoint,
        }

        residuals = []

        # Length constraints
        for constraint in self.length_constraints:
            p1 = point_map[constraint.point1_name]
            p2 = point_map[constraint.point2_name]
            current_length = np.linalg.norm(p1 - p2)
            residuals.append(current_length - constraint.length)

        # Orientation constraints.
        for constraint in self.orientation_constraints:
            v1_start = point_map[constraint.base_vector_start_name]
            v1_end = point_map[constraint.base_vector_end_name]
            v2_start = point_map[constraint.target_vector_start_name]
            v2_end = point_map[constraint.target_vector_end_name]

            v1 = v1_end - v1_start
            v2 = v2_end - v2_start

            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)

            current_angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            residuals.append(current_angle - constraint.target_angle)

        # Target position constraint.
        initial_midpoint = (
            self.initial_state.axle_inner + self.initial_state.axle_outer
        ) / 2
        target_z = initial_midpoint[2] + self.target_z_displacement
        residuals.append(axle_midpoint[2] - target_z)

        return np.array(residuals)
