"""File-to-file sweep command service."""

from dataclasses import dataclass
from pathlib import Path

from kinematics.cli.io.results_writer import SolutionFrame, create_writer_for_path
from kinematics.cli.io.yaml import load_geometry, load_sweep
from kinematics.core.export import (
    flat_specs_for_suspension,
    flatten_metric_result,
    flatten_positions,
)
from kinematics.core.suspensions.base import Suspension
from kinematics.core.sweep import EvaluatedSweep, solve_evaluated_sweep


@dataclass(frozen=True)
class SweepRun:
    """Solved objects retained for terminal reporting and optional rendering."""

    suspension: Suspension
    evaluated: EvaluatedSweep


def run_sweep_files(
    geometry_path: Path,
    sweep_path: Path,
    output_path: Path,
) -> SweepRun:
    """Load, solve, analyze, and write one sweep without terminal behavior."""
    suspension = load_geometry(geometry_path)
    sweep_config = load_sweep(sweep_path, suspension)
    evaluated = solve_evaluated_sweep(suspension, sweep_config)

    writer = create_writer_for_path(
        output_path,
        geometry_path=str(geometry_path),
        sweep_path=str(sweep_path),
    )
    output_points = suspension.output_points()
    metric_specs = flat_specs_for_suspension(suspension)
    for index, (state, solver_info, metric_row) in enumerate(
        zip(
            evaluated.states,
            evaluated.solver_stats,
            evaluated.metrics.rows,
        )
    ):
        writer.add_frame(
            index,
            SolutionFrame(
                positions=flatten_positions(state.positions, output_points),
                solver_info=solver_info,
                metrics=flatten_metric_result(metric_row),
                metric_specs=metric_specs,
            ),
        )
    writer.write()

    return SweepRun(
        suspension=suspension,
        evaluated=evaluated,
    )
