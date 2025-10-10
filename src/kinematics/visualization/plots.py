"""
Standard plotting functions for suspension visualization.

This module provides reusable plotting functionality for both single states
and animation sequences.
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from kinematics.enums import PointID
from kinematics.state import SuspensionState
from kinematics.suspensions.core.provider import SuspensionProvider
from kinematics.visualization.main import SuspensionVisualizer, WheelVisualization


def create_four_view_plot(
    state: SuspensionState,
    provider: SuspensionProvider,
    output_path: Path,
    wheel_diameter: float,
    wheel_width: float,
    title: str = "Suspension Geometry Visualization",
    dpi: int = 150,
) -> None:
    """
    Create a four-view plot (front, top, side, isometric) of a suspension state.

    Args:
        state: The suspension state to visualize.
        provider: The suspension provider for getting visualization links.
        output_path: Path where the plot image will be saved.
        wheel_diameter: Wheel diameter in millimeters.
        wheel_width: Wheel width in millimeters.
        title: Main title for the plot.
        dpi: DPI for the saved image.
    """
    # Configure wheel visualization
    wheel_config = WheelVisualization(
        diameter=wheel_diameter,
        width=wheel_width,
    )

    # Get visualization links from provider
    visualization_links = provider.get_visualization_links()

    # Create visualizer
    visualizer = SuspensionVisualizer(visualization_links, wheel_config)

    # Create figure with four subplots
    fig_scalar = 1.25
    fig = plt.figure(figsize=(16 * fig_scalar, 10 * fig_scalar))
    gs = fig.add_gridspec(2, 2)

    axes = {
        "front": fig.add_subplot(gs[0, 0], projection="3d"),
        "top": fig.add_subplot(gs[1, 0], projection="3d"),
        "side": fig.add_subplot(gs[0, 1], projection="3d"),
        "iso": fig.add_subplot(gs[1, 1], projection="3d"),
    }

    # Compute global bounds for consistent scaling
    all_points = np.array(list(state.positions.values()))
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    # Common axis limits and aspect
    x_mid = (max_bounds[0] + min_bounds[0]) * 0.5
    y_mid = (max_bounds[1] + min_bounds[1]) * 0.5
    z_mid = (max_bounds[2] + min_bounds[2]) * 0.5
    max_range = max(
        max_bounds[0] - min_bounds[0],
        max_bounds[1] - min_bounds[1],
        max_bounds[2] - min_bounds[2],
    )

    # Configure each view
    for view_name, ax in axes.items():
        ax3d = cast(Axes3D, ax)

        # Set view-specific properties
        if view_name == "top":
            ax3d.view_init(elev=90, azim=0)
            ax3d.set_title("Top View [X-Y]")
            ax3d.set_proj_type("ortho")
            ax3d.set_zticklabels([])  # type: ignore[attr-defined]
        elif view_name == "front":
            ax3d.view_init(elev=0, azim=0)
            ax3d.set_title("Front View [Y-Z]")
            ax3d.set_proj_type("ortho")
            ax3d.set_xticklabels([])
        elif view_name == "side":
            ax3d.view_init(elev=0, azim=90)
            ax3d.set_title("Side View [X-Z]")
            ax3d.set_proj_type("ortho")
            ax3d.set_yticklabels([])
        else:  # isometric
            ax3d.view_init(elev=20, azim=45)
            ax3d.set_title("Isometric View")
            ax3d.set_proj_type("ortho")

        # Set consistent axis limits and aspect ratio
        ax3d.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax3d.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax3d.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])
        ax3d.set_box_aspect([1, 1, 1])  # type: ignore[arg-type]
        ax3d.set_xlabel("X [mm]")
        ax3d.set_ylabel("Y [mm]")
        ax3d.set_zlabel("Z [mm]")

        # Plot suspension links
        for link in visualizer.links:
            if len(link.points) > 1:
                pts = np.array([state.positions[pid] for pid in link.points])
                ax3d.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=link.color,
                    linewidth=link.linewidth,
                    linestyle=link.linestyle,
                    marker=link.marker,
                    markersize=link.markersize,
                    label=link.label if view_name == "iso" else None,
                )
            else:
                # Single point
                pt = state.positions[link.points[0]]
                ax3d.scatter(
                    pt[0],
                    pt[1],
                    pt[2],
                    color=link.color,
                    s=int(link.markersize**2),
                    marker=link.marker,
                    label=link.label if view_name == "iso" else None,
                )

        # Draw the wheel
        visualizer.draw_wheel(ax3d, state.positions)

        # Add contact patch center if it exists
        if PointID.CONTACT_PATCH_CENTER in state.positions:
            contact_pt = state.positions[PointID.CONTACT_PATCH_CENTER]
            ax3d.scatter(
                contact_pt[0],
                contact_pt[1],
                contact_pt[2],
                color="red",
                s=100,
                marker="o",
                label="Contact Patch" if view_name == "iso" else None,
            )

    # Add legend only to isometric view
    axes["iso"].legend(loc="upper left")

    # Set main title and layout
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(
        left=0.0, right=1, bottom=0.025, top=0.95, wspace=0.01, hspace=0.01
    )

    # Save the plot
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def create_single_view_plot(
    state: SuspensionState,
    provider: SuspensionProvider,
    output_path: Path,
    wheel_diameter: float,
    wheel_width: float,
    view: str = "iso",
    title: str = "Suspension Geometry Visualization",
    dpi: int = 150,
) -> None:
    """
    Create a single view plot of a suspension state.

    Args:
        state: The suspension state to visualize.
        provider: The suspension provider for getting visualization links.
        output_path: Path where the plot image will be saved.
        wheel_diameter: Wheel diameter in millimeters.
        wheel_width: Wheel width in millimeters.
        view: View type ("front", "top", "side", "iso").
        title: Title for the plot.
        dpi: DPI for the saved image.
    """
    # Configure wheel visualization
    wheel_config = WheelVisualization(
        diameter=wheel_diameter,
        width=wheel_width,
    )

    # Get visualization links from provider
    visualization_links = provider.get_visualization_links()

    # Create visualizer
    visualizer = SuspensionVisualizer(visualization_links, wheel_config)

    # Create single plot
    fig = plt.figure(figsize=(12, 8))
    ax_raw = fig.add_subplot(111, projection="3d")
    ax = cast(Axes3D, ax_raw)

    # Set view-specific properties
    if view == "top":
        ax.view_init(elev=90, azim=0)
        ax.set_title("Top View [X-Y]")
    elif view == "front":
        ax.view_init(elev=0, azim=0)
        ax.set_title("Front View [Y-Z]")
    elif view == "side":
        ax.view_init(elev=0, azim=90)
        ax.set_title("Side View [X-Z]")
    else:  # isometric
        ax.view_init(elev=20, azim=45)
        ax.set_title(title)

    ax.set_proj_type("ortho")

    # Compute bounds and set axis limits
    all_points = np.array(list(state.positions.values()))
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    x_mid = (max_bounds[0] + min_bounds[0]) * 0.5
    y_mid = (max_bounds[1] + min_bounds[1]) * 0.5
    z_mid = (max_bounds[2] + min_bounds[2]) * 0.5
    max_range = max(
        max_bounds[0] - min_bounds[0],
        max_bounds[1] - min_bounds[1],
        max_bounds[2] - min_bounds[2],
    )

    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])
    ax.set_box_aspect([1, 1, 1])  # type: ignore[arg-type]
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # Plot suspension links
    for link in visualizer.links:
        if len(link.points) > 1:
            pts = np.array([state.positions[pid] for pid in link.points])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=link.color,
                linewidth=link.linewidth,
                linestyle=link.linestyle,
                marker=link.marker,
                markersize=link.markersize,
                label=link.label,
            )
        else:
            # Single point
            pt = state.positions[link.points[0]]
            ax.scatter(
                pt[0],
                pt[1],
                pt[2],
                color=link.color,
                s=link.markersize**2,  # type: ignore[arg-type]
                marker=link.marker,
                label=link.label,
            )

    # Draw the wheel
    visualizer.draw_wheel(ax, state.positions)

    # Add contact patch center if it exists
    if PointID.CONTACT_PATCH_CENTER in state.positions:
        contact_pt = state.positions[PointID.CONTACT_PATCH_CENTER]
        ax.scatter(
            contact_pt[0],
            contact_pt[1],
            contact_pt[2],
            color="red",
            s=100,
            marker="*",
            label="Contact Patch Center",
        )

    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
