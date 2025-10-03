from __future__ import annotations

from pathlib import Path

import typer

from kinematics.core import PointID
from kinematics.loader import load_geometry
from kinematics.main import solve_suspension_sweep
from kinematics.solver import PointTarget
from kinematics.types import Axis, PointTargetAxis, SweepConfig

app = typer.Typer(add_completion=False)


@app.command()
def solve(
    geometry: Path,
    steps: int = typer.Option(1, help="Number of sweep steps"),
    tol: float = typer.Option(1e-8, help="Convergence tolerance"),
    max_iters: int = typer.Option(50, help="Max solver iterations"),
):
    geom, provider = load_geometry(geometry)

    # Create a dummy target for now, this will be expanded later
    targets = [
        PointTarget(
            point_id=PointID.WHEEL_CENTER,
            direction=PointTargetAxis(Axis.Z),
            value=0.01 * i,
        )
        for i in range(steps)
    ]
    sweep_config = SweepConfig([targets])

    solution = solve_suspension_sweep(geom, provider, sweep_config)
    typer.echo(f"converged=True steps={len(solution)}")


if __name__ == "__main__":
    app()
