from pathlib import Path

import typer

from kinematics.enums import Axis, PointID
from kinematics.io.geometry_loader import load_geometry
from kinematics.io.results_writer import ResultsWriter, SolutionFrame
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import solve_suspension_sweep
from kinematics.solver import PointTarget
from kinematics.types import PointTargetAxis, SweepConfig

app = typer.Typer(add_completion=False)


@app.command()
def solve(
    geometry: Path,
    steps: int = typer.Option(1, help="Number of sweep steps"),
):
    loaded = load_geometry(geometry)

    # We should actually define a sweep file format for this to be any use,
    # but for now just do a simple Z bump sweep on the wheel center.
    targets = [
        PointTarget(
            point_id=PointID.WHEEL_CENTER,
            direction=PointTargetAxis(Axis.Z),
            value=0.01 * i,
        )
        for i in range(steps)
    ]
    sweep_config = SweepConfig([targets])

    solution = solve_suspension_sweep(loaded.provider, sweep_config)
    typer.echo(f"converged=True steps={len(solution)}")


@app.command()
def sweep(
    geometry: Path = typer.Option(..., exists=True, help="Path to geometry YAML"),
    sweep: Path = typer.Option(..., exists=True, help="Path to sweep YAML"),
    out: Path = typer.Option(..., help="Output Parquet path"),
):
    """
    Run a sweep from file and write results to Parquet (wide schema).

    Example:
        kinematics sweep --geometry=tests/data/geometry.yaml --sweep=tests/data/sweep.yaml --out=results.parquet
    """
    loaded = load_geometry(geometry)
    sweep_config = parse_sweep_file(sweep)

    solution_states = solve_suspension_sweep(loaded.provider, sweep_config)

    # Write out in wide format.
    writer = ResultsWriter(str(out), geometry_path=str(geometry), sweep_path=str(sweep))
    for idx, st in enumerate(solution_states):
        positions = {
            (PointID(pid).name if isinstance(pid, PointID) else str(pid)): (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
            )
            for pid, pos in st.positions.items()
        }
        frame = SolutionFrame(positions=positions)
        writer.add_frame(idx, frame)
    writer.write()

    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
