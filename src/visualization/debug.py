from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from kinematics.geometry.points.ids import PointID
from kinematics.solvers.common import KinematicState
from visualization.main import SuspensionVisualizer


def create_animation(
    states: list[KinematicState],
    visualizer: SuspensionVisualizer,
    output_path: Path,
    fps: int = 20,
    interval: int = 200,
) -> None:
    """Create and save animation of suspension movement"""
    fig_scalar = 1.25
    fig = plt.figure(figsize=(16 * fig_scalar, 10 * fig_scalar))
    gs = fig.add_gridspec(2, 2)

    axes = {
        "front": fig.add_subplot(gs[0, 0], projection="3d"),
        "top": fig.add_subplot(gs[1, 0], projection="3d"),
        "side": fig.add_subplot(gs[0, 1], projection="3d"),
        "iso": fig.add_subplot(gs[1, 1], projection="3d"),
    }

    # Calculate bounds
    all_points = []
    for state in states:
        for point_id in PointID:
            if point_id in state.hard_points:
                all_points.append(state.hard_points[point_id].as_array())
            elif point_id in state.derived_points:
                all_points.append(state.derived_points[point_id].as_array())

    all_points = np.array(all_points)
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    def update(frame: int) -> None:
        state = states[frame]

        for view_name, ax in axes.items():
            ax.clear()

            # Set view angles and configure orthographic projections
            if view_name == "top":
                ax.view_init(elev=90, azim=0)
                ax.set_title("Top View [X-Y]")
                ax.set_proj_type("ortho")
                ax.set_zticklabels([])  # Hide Z-axis ticks
            elif view_name == "front":
                ax.view_init(elev=0, azim=0)
                ax.set_title("Front View [Y-Z]")
                ax.set_proj_type("ortho")
                ax.set_xticklabels([])  # Hide X-axis ticks
            elif view_name == "side":
                ax.view_init(elev=0, azim=90)
                ax.set_title("Side View [X-Z]")
                ax.set_proj_type("ortho")
                ax.set_yticklabels([])  # Hide Y-axis ticks
            else:  # isometric
                ax.view_init(elev=20, azim=45)
                ax.set_title("Isometric View")
                ax.set_proj_type("ortho")

            # Draw all links
            for link in visualizer.links:
                points = []
                for point_id in link.points:
                    if point_id in state.hard_points:
                        points.append(state.hard_points[point_id].as_array())
                points = np.array(points)

                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    color=link.color,
                    linewidth=link.linewidth,
                    linestyle=link.linestyle,
                    marker=link.marker,
                    markersize=link.markersize,
                    label=link.label if view_name == "iso" else None,
                )

            # Draw wheel
            wheel_center = state.derived_points[PointID.WHEEL_CENTER].as_array()
            wheel_inboard = state.derived_points[PointID.WHEEL_INBOARD].as_array()
            wheel_outboard = state.derived_points[PointID.WHEEL_OUTBOARD].as_array()
            axle_vector = (
                state.hard_points[PointID.AXLE_OUTBOARD].as_array()
                - state.hard_points[PointID.AXLE_INBOARD].as_array()
            )
            visualizer.draw_wheel(
                ax,
                wheel_center,
                wheel_inboard,
                wheel_outboard,
                axle_vector,
                state,
            )

            # Calculate cube bounds
            x_mid = (max_bounds[0] + min_bounds[0]) * 0.5
            y_mid = (max_bounds[1] + min_bounds[1]) * 0.5
            z_mid = (max_bounds[2] + min_bounds[2]) * 0.5

            max_range = max(
                max_bounds[0] - min_bounds[0],
                max_bounds[1] - min_bounds[1],
                max_bounds[2] - min_bounds[2],
            )

            # Set square limits
            ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
            ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
            ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])

            # Force equal aspect ratio
            ax.set_box_aspect([1, 1, 1])

            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_zlabel("Z [mm]")

            if view_name == "iso":
                ax.legend(loc="upper left")

        target_z = state.derived_points.derived_points[PointID.WHEEL_CENTER].z
        fig.suptitle(
            f"Wheel Center Z: {target_z:.1f} [mm]",
            fontsize=16,
        )

    # Create animation
    plt.subplots_adjust(
        left=0.0,  # Move left edge closer to figure edge
        right=1,  # Move right edge closer to figure edge
        bottom=0.025,  # Move bottom edge closer to figure edge
        top=0.95,  # Move top edge closer to figure edge
        wspace=0.01,  # Reduce horizontal space between subplots
        hspace=0.01,  # Reduce vertical space between subplots
    )
    anim = animation.FuncAnimation(
        fig, update, frames=len(states), interval=interval, blit=False
    )

    # Save animation and close figure
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close()
