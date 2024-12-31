from dataclasses import dataclass
from typing import List, NamedTuple

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


@dataclass
class SuspensionState:
    """Contains all moving point positions for a given suspension state."""

    upper_outboard: np.ndarray  # Upper ball joint.
    lower_outboard: np.ndarray  # Lower ball joint.
    axle_inner: np.ndarray  # Inner end of wheel axle.
    axle_outer: np.ndarray  # Outer end of wheel axle.

    def as_array(self) -> np.ndarray:
        """Converts the suspension state to a flat array for optimization."""
        return np.concatenate(
            [self.upper_outboard, self.lower_outboard, self.axle_inner, self.axle_outer]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SuspensionState":
        """Creates a suspension state from a flat array of coordinates."""
        return cls(
            upper_outboard=arr[0:3],
            lower_outboard=arr[3:6],
            axle_inner=arr[6:9],
            axle_outer=arr[9:12],
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
        self.constraints = self.compute_constraints()
        self.initial_state = self.compute_initial_state()

    def compute_constraints(self) -> List[LinkLengthConstraint]:
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

        # Add axle midpoint constraints
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
            "axle_inner": state.axle_inner,
            "axle_outer": state.axle_outer,
            "axle_midpoint": axle_midpoint,
        }

        residuals = []
        for constraint in self.constraints:
            p1 = point_map[constraint.point1_name]
            p2 = point_map[constraint.point2_name]
            current_length = np.linalg.norm(p1 - p2)
            residuals.append(current_length - constraint.length)

        # Target position constraint.
        initial_midpoint = (
            self.initial_state.axle_inner + self.initial_state.axle_outer
        ) / 2
        target_z = initial_midpoint[2] + self.target_z_displacement
        residuals.append(axle_midpoint[2] - target_z)

        return np.array(residuals)
