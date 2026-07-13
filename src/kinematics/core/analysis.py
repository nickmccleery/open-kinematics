"""
High-level, structured suspension sweep analysis for front-end consumers.

The API in this module keeps corner locations structural. Side suffixes are an
export concern and are not embedded in analysis metric keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from kinematics.core.diagnostics import DiagnosticIssue
from kinematics.core.metrics.main import AxleMetricRows, MetricRow
from kinematics.core.metrics.metadata import MetricDisplay, metric_display_for_keys
from kinematics.core.metrics.registry import MetricSpec, metric_specs_for_suspension
from kinematics.core.presentation import (
    DisplayElement,
    RockerDisplayGroup,
    WheelAnchorNames,
    WheelDisplayDimensions,
    display_elements,
    display_point_keys,
    display_positions,
    rocker_display_groups,
    wheel_anchor_names,
    wheel_display_dimensions,
)
from kinematics.core.primitives.enums import TargetPositionMode
from kinematics.core.primitives.point_ref import (
    PointKey,
    PointRef,
    Side,
    point_key_name,
)
from kinematics.core.solver import SolverInfo
from kinematics.core.state import SuspensionState
from kinematics.core.suspensions.base import Suspension
from kinematics.core.sweep import (
    EvaluatedSweep,
    evaluate_solved_sweep,
    solve_evaluated_sweep,
)
from kinematics.core.targeting import PointTarget, SweepConfig

Positions = dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class SuspensionInfo:
    """Identifying metadata for an analyzed suspension."""

    name: str
    type_key: str
    units: str


@dataclass(frozen=True)
class SweepParameter:
    """One principal-axis sweep dimension usable as a chart axis."""

    point: str
    axis: str
    side: str | None


@dataclass(frozen=True)
class AnalyzedFrame:
    """One solved and analyzed sweep step."""

    index: int
    positions: Positions
    metrics: MetricRow
    corner_metrics: dict[str, MetricRow]
    solver: SolverInfo


@dataclass(frozen=True)
class ReferenceCondition:
    """A solved reference pose for comparison with the sweep."""

    label: str
    positions: Positions
    metrics: MetricRow
    corner_metrics: dict[str, MetricRow]


@dataclass(frozen=True)
class StaticPose:
    """The as-assembled initial pose of a suspension geometry."""

    suspension: SuspensionInfo
    point_keys: list[str]
    positions: Positions
    wheel: WheelDisplayDimensions | None
    elements: list[DisplayElement]
    wheel_anchors: list[WheelAnchorNames]


@dataclass(frozen=True)
class SweepAnalysis:
    """Complete structured result of a suspension sweep."""

    suspension: SuspensionInfo
    point_keys: list[str]
    metric_keys: list[str]
    corner_metric_keys: list[str]
    locations: list[str]
    metric_display: list[MetricDisplay]
    sweep_parameters: list[SweepParameter]
    references: dict[str, ReferenceCondition]
    wheel: WheelDisplayDimensions | None
    elements: list[DisplayElement]
    wheel_anchors: list[WheelAnchorNames]
    diagnostics: list[DiagnosticIssue]
    frames: list[AnalyzedFrame] = field(default_factory=list)

    @property
    def steps(self) -> int:
        """Return the number of solved frames."""
        return len(self.frames)


def _suspension_info(suspension: Suspension) -> SuspensionInfo:
    return SuspensionInfo(
        name=suspension.name,
        type_key=suspension.TYPE_KEY,
        units=suspension.units.symbol,
    )


def sweep_parameters(sweep_config: SweepConfig) -> list[SweepParameter]:
    """Describe every principal-axis dimension in a sweep."""
    parameters: list[SweepParameter] = []
    for dimension in sweep_config.target_sweeps:
        if not dimension:
            continue
        target = dimension[0]
        axis = getattr(target.direction, "axis", None)
        if axis is None:
            continue
        key = target.point_id
        side = None
        if isinstance(key, PointRef) and key.side is not Side.CENTER:
            side = key.side.name.lower()
        parameters.append(
            SweepParameter(point=point_key_name(key), axis=axis.name.lower(), side=side)
        )
    return parameters


def _hold_sweep_config(sweep_config: SweepConfig) -> SweepConfig | None:
    hold_dimensions: list[list[PointTarget]] = []
    for dimension in sweep_config.target_sweeps:
        if not dimension:
            continue
        target = dimension[0]
        hold_dimensions.append(
            [
                PointTarget(
                    point_id=target.point_id,
                    direction=target.direction,
                    value=0.0,
                    mode=TargetPositionMode.RELATIVE,
                )
            ]
        )
    return SweepConfig(hold_dimensions) if hold_dimensions else None


def _split_metric_rows(
    rows: MetricRow | AxleMetricRows,
) -> tuple[MetricRow, dict[str, MetricRow]]:
    if isinstance(rows, AxleMetricRows):
        return rows.axle, rows.corners
    return rows, {}


def _setup_reference(
    suspension: Suspension,
    sweep_config: SweepConfig,
    point_keys: tuple[PointKey, ...],
    rocker_groups: list[RockerDisplayGroup],
) -> tuple[ReferenceCondition | None, DiagnosticIssue | None]:
    """Solve the nominal setup pose without making it a hard dependency."""
    hold_config = _hold_sweep_config(sweep_config)
    if hold_config is None:
        return None, None
    try:
        evaluated = solve_evaluated_sweep(suspension, hold_config)
        if not evaluated.states:
            return None, None
        row = evaluated.metrics.rows[0]
    except Exception as error:  # noqa: BLE001 - the reference is optional
        return None, DiagnosticIssue(
            step=None,
            category="reference",
            severity="warning",
            message=(
                "Setup reference unavailable: reference solve failed "
                f"({type(error).__name__}: {error})."
            ),
            value=None,
        )
    metrics, corner_metrics = _split_metric_rows(row)
    return (
        ReferenceCondition(
            label="Setup",
            positions=display_positions(
                evaluated.states[0].positions,
                point_keys,
                rocker_groups,
            ),
            metrics=metrics,
            corner_metrics=corner_metrics,
        ),
        None,
    )


def _metric_specs(suspension: Suspension) -> Mapping[str, MetricSpec]:
    """Collect canonical static and topology-specific derivative metadata."""
    return metric_specs_for_suspension(suspension)


def analyze_sweep(suspension: Suspension, sweep_config: SweepConfig) -> SweepAnalysis:
    """Solve a sweep and assemble a complete structured analysis."""
    return analyze_evaluated_sweep(
        suspension,
        sweep_config,
        solve_evaluated_sweep(suspension, sweep_config),
    )


def analyze_solved_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
    states: list[SuspensionState],
    solver_stats: list[SolverInfo],
) -> SweepAnalysis:
    """Assemble structured analysis from an already solved suspension sweep."""
    return analyze_evaluated_sweep(
        suspension,
        sweep_config,
        evaluate_solved_sweep(
            suspension,
            sweep_config,
            states,
            solver_stats,
        ),
    )


def analyze_evaluated_sweep(
    suspension: Suspension,
    sweep_config: SweepConfig,
    evaluated: EvaluatedSweep,
) -> SweepAnalysis:
    """Build the rich presentation model for an evaluated sweep."""
    assembly = suspension.assembly()
    point_keys = display_point_keys(assembly)
    rocker_groups = rocker_display_groups(assembly)

    frames: list[AnalyzedFrame] = []
    for index, (state, info, row) in enumerate(
        zip(evaluated.states, evaluated.solver_stats, evaluated.metrics.rows)
    ):
        metrics, corner_metrics = _split_metric_rows(row)
        frames.append(
            AnalyzedFrame(
                index=index,
                positions=display_positions(state.positions, point_keys, rocker_groups),
                metrics=metrics,
                corner_metrics=corner_metrics,
                solver=info,
            )
        )

    metric_keys: list[str] = []
    corner_metric_keys: list[str] = []
    locations: list[str] = []
    for frame in frames:
        if not frame.metrics and not frame.corner_metrics:
            continue
        metric_keys = list(frame.metrics)
        locations = list(frame.corner_metrics)
        if frame.corner_metrics:
            corner_metric_keys = list(next(iter(frame.corner_metrics.values())))
        break

    display_keys = corner_metric_keys + [
        key for key in metric_keys if key not in corner_metric_keys
    ]
    references: dict[str, ReferenceCondition] = {}
    setup, reference_issue = _setup_reference(
        suspension,
        sweep_config,
        point_keys,
        rocker_groups,
    )
    if setup is not None:
        references["setup"] = setup
    diagnostics = list(evaluated.diagnostics)
    if reference_issue is not None:
        diagnostics.append(reference_issue)

    return SweepAnalysis(
        suspension=_suspension_info(suspension),
        point_keys=[point_key_name(key) for key in point_keys],
        metric_keys=metric_keys,
        corner_metric_keys=corner_metric_keys,
        locations=locations,
        metric_display=metric_display_for_keys(display_keys, _metric_specs(suspension)),
        sweep_parameters=sweep_parameters(sweep_config),
        references=references,
        wheel=wheel_display_dimensions(suspension.config),
        elements=display_elements(assembly),
        wheel_anchors=wheel_anchor_names(assembly),
        diagnostics=diagnostics,
        frames=frames,
    )


def initial_pose(suspension: Suspension) -> StaticPose:
    """Return the as-assembled pose without running a sweep."""
    state = suspension.initial_state()
    assembly = suspension.assembly()
    point_keys = display_point_keys(assembly)
    rocker_groups = rocker_display_groups(assembly)
    return StaticPose(
        suspension=_suspension_info(suspension),
        point_keys=[point_key_name(key) for key in point_keys],
        positions=display_positions(state.positions, point_keys, rocker_groups),
        wheel=wheel_display_dimensions(suspension.config),
        elements=display_elements(assembly),
        wheel_anchors=wheel_anchor_names(assembly),
    )
