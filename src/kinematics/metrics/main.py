from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from kinematics.core.types import Vec3, make_vec3
from kinematics.metrics.angles import calculate_camber, calculate_caster, calculate_toe
from kinematics.metrics.antis import (
    calculate_geometric_anti_dive,
    calculate_geometric_anti_squat,
)
from kinematics.state import SuspensionState
from kinematics.suspensions.config.settings import SuspensionConfig

if TYPE_CHECKING:
    from kinematics.suspensions.base import Suspension


@dataclass(frozen=True)
class SuspensionMetrics:
    """
    A container for all calculated suspension metrics for a single state.

    Attributes:
        camber: Camber angle in degrees.
        caster: Caster angle in degrees.
        toe: Toe angle in degrees.
        anti_dive: (Geometric) anti-dive as a percentage (%).
        anti_squat: (Geometric) anti-squat as a percentage (%).
    """

    camber: float
    caster: float
    toe: float
    anti_dive: float
    anti_squat: float


class MetricsCalculator:
    """
    Computes and caches a full set of kinematic metrics for a single SuspensionState.

    This class uses a suspension to access suspension-specific calculations (like SVIC)
    and implements caching to avoid redundant computations of expensive metrics.
    """

    def __init__(
        self,
        state: SuspensionState,
        suspension: "Suspension",
        geometry_config: "SuspensionConfig",
    ):
        """
        Initialize the metrics calculator.

        Args:
            state: The solved suspension state to analyze.
            suspension: The suspension for type-specific calculations.
            geometry_config: Suspension configuration with vehicle parameters.
        """
        self.state = state
        self.suspension = suspension
        self.geometry_config = geometry_config

    @cached_property
    def side_view_ic(self) -> Vec3 | None:
        """
        Compute and cache the side view instant center.

        This uses the suspension's type-specific SVIC calculation. Returns None
        if SVIC cannot be computed (e.g., parallel links).
        """
        return self.suspension.compute_side_view_instant_center(self.state)

    @property
    def camber(self) -> float:
        """
        Camber angle in degrees.
        """
        return calculate_camber(self.state)

    @property
    def caster(self) -> float:
        """
        Caster angle in degrees.
        """
        return calculate_caster(self.state)

    @property
    def toe(self) -> float:
        """
        Toe angle in degrees.
        """
        return calculate_toe(self.state)

    @property
    def anti_dive(self) -> float:
        """
        Anti-dive percentage.

        Returns 0.0 if SVIC cannot be computed.
        """
        if self.side_view_ic is None:
            return 0.0

        cg = self.geometry_config.cg_position
        return calculate_geometric_anti_dive(
            self.state,
            self.side_view_ic,
            make_vec3([cg["x"], cg["y"], cg["z"]]),
            self.geometry_config.wheelbase,
            self.geometry_config.wheel.tire.nominal_radius,
        )

    @property
    def anti_squat(self) -> float:
        """
        Anti-squat percentage.

        Returns 0.0 if SVIC cannot be computed.
        """
        if self.side_view_ic is None:
            return 0.0

        cg = self.geometry_config.cg_position
        return calculate_geometric_anti_squat(
            self.state,
            self.side_view_ic,
            make_vec3([cg["x"], cg["y"], cg["z"]]),
            self.geometry_config.wheelbase,
            self.geometry_config.wheel.tire.nominal_radius,
        )


def compute_all_metrics(
    state: SuspensionState,
    suspension: "Suspension",
    geometry_config: "SuspensionConfig",
) -> SuspensionMetrics:
    """
    Compute a comprehensive set of kinematic metrics for a given suspension state.

    This is the primary high-level function for metrics calculation. It efficiently
    computes all standard metrics by resolving shared intermediate geometries
    (like instant centers) only once through caching.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The suspension for type-specific calculations.
        geometry_config: The suspension geometry configuration.

    Returns:
        A SuspensionMetrics dataclass containing all calculated values.
    """
    calc = MetricsCalculator(state, suspension, geometry_config)

    return SuspensionMetrics(
        camber=calc.camber,
        caster=calc.caster,
        toe=calc.toe,
        anti_dive=calc.anti_dive,
        anti_squat=calc.anti_squat,
    )


def compute_all_metrics_from_suspension(
    state: SuspensionState,
    suspension: "Suspension",
) -> SuspensionMetrics:
    """
    Compute metrics using parameters from the suspension configuration.

    This convenience function extracts tire radius and CG height from the suspension
    configuration, then delegates to compute_all_metrics(). It's useful when all
    vehicle parameters are already defined in the geometry file.

    Args:
        state: The solved SuspensionState to analyze.
        suspension: The suspension containing wheel/tire configuration.

    Returns:
        A SuspensionMetrics dataclass containing all calculated values.
    """
    if suspension.config is None:
        raise ValueError("Suspension has no configuration")

    return compute_all_metrics(
        state=state,
        suspension=suspension,
        geometry_config=suspension.config,
    )
