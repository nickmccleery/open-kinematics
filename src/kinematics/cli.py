from pathlib import Path

import typer

from kinematics.enums import PointID
from kinematics.io.geometry_loader import load_geometry
from kinematics.io.results_writer import SolutionFrame, create_writer_for_path
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import solve_sweep

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def sweep(
    geometry: Path = typer.Option(..., exists=True, help="Path to geometry YAML"),
    sweep: Path = typer.Option(..., exists=True, help="Path to sweep YAML"),
    out: Path = typer.Option(..., help="Output path (.parquet or .csv)"),
    animation_out: Path | None = typer.Option(
        None, help="Optional animation output path (.mp4, .gif, etc.)"
    ),
):
    """
    Run a sweep from file and write results to Parquet or CSV format.

    Example:
        kinematics --geometry=tests/data/geometry.yaml --sweep=tests/data/sweep.yaml --out=results.parquet
        kinematics --geometry=tests/data/geometry.yaml --sweep=tests/data/sweep.yaml --out=results.csv --animation-out=anim.mp4
    """
    loaded = load_geometry(geometry)
    sweep_config = parse_sweep_file(sweep)

    solution_states, solver_stats = solve_sweep(loaded.provider, sweep_config)

    # Write out in wide format.
    writer = create_writer_for_path(
        out, geometry_path=str(geometry), sweep_path=str(sweep)
    )
    for idx, (st, solver_info) in enumerate(zip(solution_states, solver_stats)):
        positions = {
            (PointID(pid).name if isinstance(pid, PointID) else str(pid)): (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
            )
            for pid, pos in st.positions.items()
        }

        frame = SolutionFrame(positions=positions, solver_info=solver_info)

        writer.add_frame(idx, frame)
    writer.write()

    typer.echo(f"wrote {out}")

    # Generate animation if requested.
    if animation_out:
        try:
            from kinematics.visualization.api import visualize_suspension_sweep

            # Get wheel parameters from geometry configuration.
            wheel_cfg = loaded.geometry.configuration.wheel

            # Create animation.
            visualize_suspension_sweep(
                provider=loaded.provider,
                solution_states=solution_states,
                output_path=animation_out,
                wheel_diameter=wheel_cfg.diameter,
                wheel_width=wheel_cfg.width,
                fps=20,
                show_live=False,
            )

            typer.echo(f"Wrote animation: {animation_out}")

        except ImportError as e:
            typer.echo(
                f"Error: Visualization dependencies not installed.\n"
                f'Install with: pip install "kinematics[viz]"\n'
                f"Details: {e}",
                err=True,
            )
            typer.Exit(1)


if __name__ == "__main__":
    app()
