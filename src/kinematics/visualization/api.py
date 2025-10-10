"""
Public API for visualization features with lazy imports for optional dependencies.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import typer

from kinematics.enums import Axis, PointID
from kinematics.suspensions.implementations.double_wishbone import (
    DoubleWishboneProvider,
)
from kinematics.visualization.plots import create_four_view_plot

if TYPE_CHECKING:
    from kinematics.state import SuspensionState
    from kinematics.suspensions.core.provider import SuspensionProvider


def visualize_suspension_sweep(
    provider: "SuspensionProvider",
    solution_states: list["SuspensionState"],
    output_path: Path,
    wheel_diameter: float,
    wheel_width: float,
    fps: int = 20,
    show_live: bool = False,
) -> None:
    """
    Create an animation of a suspension sweep.

    This function requires matplotlib and related visualization dependencies.
    Install with: pip install "kinematics[viz]"

    Args:
        provider: The suspension provider used to generate the solutions.
        solution_states: List of solved suspension states to animate.
        output_path: Path where the animation file will be saved.
        wheel_diameter: Wheel diameter in millimeters.
        wheel_width: Wheel width in millimeters.
        fps: Frames per second for the animation.
        show_live: Whether to show the animation during creation.

    Raises:
        ImportError: If visualization dependencies are not installed.
    """
    try:
        from kinematics.visualization.animation import create_animation
        from kinematics.visualization.main import (
            SuspensionVisualizer,
            WheelVisualization,
        )
    except ImportError as e:
        raise ImportError(
            'Visualization dependencies not found. Install with: pip install "kinematics[viz]"\n'
            f"Original error: {e}"
        ) from e

    # Configure wheel visualization
    wheel_config = WheelVisualization(
        diameter=wheel_diameter,
        width=wheel_width,
    )

    # Get visualization links from provider
    visualization_links = provider.get_visualization_links()

    # Create visualizer
    visualizer = SuspensionVisualizer(visualization_links, wheel_config)

    # Get initial positions for animation baseline
    initial_state = provider.initial_state()
    initial_positions = initial_state.positions.copy()

    # Extract position dictionaries from states
    position_states = [state.positions for state in solution_states]

    # Create the animation
    create_animation(
        position_states,
        initial_positions,
        visualizer,
        output_path,
        fps=fps,
        show_live=show_live,
    )


def visualize_geometry(
    provider: "SuspensionProvider",
    output_path: Path,
) -> None:
    """
    Creates a debug plot for a single suspension state and checks ground tangency.

    Args:
        provider: The suspension provider for the geometry.
        output_path: Path where the plot image will be saved.
        ground_geometry: If True, temporarily adjusts all points vertically so the
                         geometric contact point is at Z=0 for the visualization.
    """
    try:
        # Test for matplotlib availability; plotting is handled by plots module.
        import matplotlib.pyplot as plt

        plt.figure()  # Test that matplotlib works.
        plt.close()
    except ImportError as e:
        raise ImportError(
            'Visualization dependencies not found. Install with: pip install "kinematics[viz]"\n'
            f"Original error: {e}"
        ) from e

    if not isinstance(provider, DoubleWishboneProvider):
        raise NotImplementedError(
            "Geometry visualization is currently only supported for DoubleWishbone suspensions."
        )

    typer.secho(
        "Checking and visualizing suspension geometry...",
    )

    state = provider.initial_state()
    z_offset = state.get(PointID.CONTACT_PATCH_CENTER)[Axis.Z]

    # Report the final status.
    if np.isclose(z_offset, 0.0, atol=1e-2):
        typer.secho(
            f"Geometry Check: OK. Contact patch center is on the ground (Z = {z_offset:.3f} mm).",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            "Geometry Check: WARNING. Contact patch center is not on the ground.",
            fg=typer.colors.RED,
        )
        typer.echo("─" * 60)
        typer.secho(
            f"The contact patch center currently located at Z = {z_offset:.3f}mm."
        )
        typer.secho(
            "Hard points must be adjusted to place the contact patch center at Z=0.",
        )
        typer.echo("─" * 60)

    # Create visualization.
    dw_provider = cast(DoubleWishboneProvider, provider)
    wheel_cfg = dw_provider.geometry.configuration.wheel

    # Create the four-view plot
    create_four_view_plot(
        state=state,
        provider=dw_provider,
        output_path=output_path,
        wheel_diameter=wheel_cfg.tire.nominal_radius * 2,
        wheel_width=wheel_cfg.tire.section_width,
        title="Suspension Geometry Visualization",
        dpi=150,
    )

    typer.secho(f"Visualization saved to: {output_path}", fg=typer.colors.GREEN)
