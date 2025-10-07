"""
Public API for visualization features with lazy imports for optional dependencies.
"""

from pathlib import Path
from typing import TYPE_CHECKING

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
        from kinematics.visualization.debug import create_animation
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
