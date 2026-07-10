"""
Main orchestration functions for suspension kinematics.

This module provides high-level functions to coordinate the solving of suspension
geometries.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from kinematics.core.types import SweepConfig
from kinematics.points.derived.manager import DerivedPointsManager
from kinematics.sensitivity import (
    TangentField,
    TangentSolveInfo,
    compute_state_tangents,
)
from kinematics.solver import (
    SolverInfo,
    convert_targets_to_absolute,
    solve_suspension_sweep,
)
from kinematics.state import SuspensionState
from kinematics.suspensions.base import Suspension

if TYPE_CHECKING:
    from kinematics.metrics.main import AxleMetricRows, MetricRow


def solve_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
) -> tuple[List[SuspensionState], List[SolverInfo]]:
    """
    Orchestrates the solving of suspension kinematics for a parametric sweep.

    This function coordinates the complete process of solving suspension kinematics
    by setting up derived point calculations and running the solver across target
    configurations.

    Args:
        suspension: The Suspension instance containing geometry and behavior.
        sweep_config: Configuration for the parametric sweep.

    Returns:
        Tuple containing the list of solved suspension states and corresponding
        solver information for each step in the sweep.
    """
    derived_spec = suspension.derived_spec()
    derived_manager = DerivedPointsManager(derived_spec)

    kinematic_states, solver_stats = solve_suspension_sweep(
        initial_state=suspension.initial_state(),
        constraints=suspension.constraints(),
        sweep_config=sweep_config,
        derived_manager=derived_manager,
    )

    return kinematic_states, solver_stats


@dataclass(frozen=True)
class SweepTangents:
    """
    Solution-manifold tangents for a full sweep, with solve health.

    Attributes:
        per_step: One list of TangentField per state, each ordered like the
            sweep's target dimensions.
        solve_infos: The numerical health of each state's tangent solve,
            aligned with ``per_step``. Rank-deficient entries mean that
            state's tangents (and any motion ratios derived from them) are
            minimum-norm picks, not unique solutions.
    """

    per_step: list[list[TangentField]]
    solve_infos: list[TangentSolveInfo]


def compute_sweep_tangents(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> SweepTangents:
    """
    Compute solution-manifold tangents for every solved state of a sweep.

    This is the post-solve analysis companion to :func:`solve_sweep`: for
    each solved state it evaluates the analytical residual Jacobian and
    extracts d(position)/d(target) for every sweep target via the implicit
    function theorem (see kinematics.sensitivity). The result feeds the
    derivative metrics (motion ratios, camber gain, bump steer, ...).

    Args:
        suspension: The suspension the states were solved for.
        sweep_config: The sweep configuration that produced the states.
        states: The solved states, one per sweep step.

    Returns:
        The per-state tangent fields plus per-state solve health.
    """
    derived_manager = DerivedPointsManager(suspension.derived_spec())
    constraints = suspension.constraints()
    initial_state = suspension.initial_state()

    tangents_per_step: list[list[TangentField]] = []
    solve_infos: list[TangentSolveInfo] = []
    for step_index, state in enumerate(states):
        step_targets = convert_targets_to_absolute(
            [sweep[step_index] for sweep in sweep_config.target_sweeps],
            initial_state,
        )
        fields, info = compute_state_tangents(
            state, constraints, derived_manager, step_targets
        )
        tangents_per_step.append(fields)
        solve_infos.append(info)
    return SweepTangents(per_step=tangents_per_step, solve_infos=solve_infos)


@dataclass(frozen=True)
class SweepMetricsResult:
    """
    The metric rows for a sweep, plus how the derivative columns fared.

    Attributes:
        rows: One entry per state: an ordered metric row for corner models,
            or structured per-corner plus axle-level rows (AxleMetricRows)
            for axle models.
        derivative_error: ``None`` when tangents computed cleanly. Otherwise
            a short description of the failure that made every derivative
            column (motion ratios, camber gain, bump steer, modal rates) be
            omitted from the rows.
        tangent_solve_infos: Per-state tangent solve health, aligned with
            ``rows``; ``None`` when tangents were not computed (no config,
            or ``derivative_error`` is set).
    """

    rows: list["MetricRow | AxleMetricRows"]
    derivative_error: str | None = None
    tangent_solve_infos: list[TangentSolveInfo] | None = None


def compute_sweep_metrics(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: List[SuspensionState],
) -> SweepMetricsResult:
    """
    Compute the full metric rows for every solved state of a sweep.

    This is the single high-level metrics entry point for sweep consumers
    (CLI, API adapters): it computes the solution-manifold tangents internally
    and feeds them to the per-state metric dispatch, so every consumer gets
    the complete column set -- per-state metrics plus the derivative metrics
    (motion ratios, camber gain, bump steer, modal roll/heave rates) -- without
    knowing anything about tangents or automatic differentiation.

    The tangent computation is best-effort: if it fails (e.g. a pathological
    configuration), the per-state metrics are still returned and only the
    derivative columns are omitted, consistently across all rows. The failure
    is never silent, though -- it is reported on the result so callers can
    tell "absent by design" from "failed to compute", and per-state solve
    health is reported so rank-deficient (non-unique, minimum-norm) tangents
    can be flagged downstream.

    Args:
        suspension: The suspension the states were solved for.
        sweep_config: The sweep configuration that produced the states.
        states: The solved states, one per sweep step.

    Returns:
        The rows plus derivative status. Rows are empty when the suspension
        has no configuration (metrics need vehicle parameters).
    """
    if suspension.config is None:
        return SweepMetricsResult(rows=[OrderedDict() for _ in states])

    tangents: SweepTangents | None
    derivative_error: str | None = None
    try:
        tangents = compute_sweep_tangents(suspension, sweep_config, states)
    except Exception as exc:  # noqa: BLE001 - derivative metrics degrade gracefully
        tangents = None
        derivative_error = f"{type(exc).__name__}: {exc}"

    rows: list["MetricRow | AxleMetricRows"] = [
        suspension.compute_state_metrics(
            state, tangents.per_step[index] if tangents is not None else None
        )
        for index, state in enumerate(states)
    ]
    return SweepMetricsResult(
        rows=rows,
        derivative_error=derivative_error,
        tangent_solve_infos=tangents.solve_infos if tangents is not None else None,
    )
