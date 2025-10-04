from pathlib import Path

import typer

from kinematics.enums import Axis, PointID
from kinematics.loader import load_geometry
from kinematics.main import solve_suspension_sweep
from kinematics.parquet_writer import write_parquet
from kinematics.solver import PointTarget
from kinematics.sweep_io import parse_sweep_file
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

    solution = solve_suspension_sweep(
        loaded.geometry, loaded.provider_cls, sweep_config
    )
    typer.echo(f"converged=True steps={len(solution)}")


@app.command()
def sweep(
    geometry: Path = typer.Option(..., exists=True, help="Path to geometry YAML"),
    sweep: Path = typer.Option(..., exists=True, help="Path to sweep YAML/JSON"),
    out: Path = typer.Option(..., help="Output Parquet path"),
):
    """
    Run a sweep from file and write results to Parquet.
    """
    loaded = load_geometry(geometry)
    sweep_config = parse_sweep_file(sweep)
    states = solve_suspension_sweep(loaded.geometry, loaded.provider_cls, sweep_config)

    write_parquet(
        states,
        out,
        geometry_type=type(loaded.geometry).__name__,
        provider_class=loaded.provider_cls.__name__,
        sweep=sweep_config,
    )
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
