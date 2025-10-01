from __future__ import annotations

from pathlib import Path

import typer

from . import load_geometry
from .core import PointID
from .main import solve_kinematics
from .solver import PointTarget, PointTargetSet
from .types import Axis, AxisTarget

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
            direction=AxisTarget(Axis.Z),
            value=0.01 * i,
        )
        for i in range(steps)
    ]
    target_set = PointTargetSet(values=targets)

    solution = solve_kinematics(geom, provider, [target_set])
    typer.echo(f"converged=True steps={len(solution)}")


if __name__ == "__main__":
    app()
