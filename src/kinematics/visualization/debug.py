from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from kinematics.core import PointID
from kinematics.visualization.main import SuspensionVisualizer


def create_animation(
    position_states: list[dict[PointID, np.ndarray]],
    initial_positions: dict[PointID, np.ndarray],
    visualizer: SuspensionVisualizer,
    output_path: Path,
    fps: int = 20,
    interval: int = 200,
) -> None:
    fig_scalar = 1.25
    fig = plt.figure(figsize=(16 * fig_scalar, 10 * fig_scalar))
    gs = fig.add_gridspec(2, 2)

    axes = {
        "front": fig.add_subplot(gs[0, 0], projection="3d"),
        "top": fig.add_subplot(gs[1, 0], projection="3d"),
        "side": fig.add_subplot(gs[0, 1], projection="3d"),
        "iso": fig.add_subplot(gs[1, 1], projection="3d"),
    }

    all_points = []
    for positions in position_states:
        all_points.extend([pos for pos in positions.values()])

    all_points = np.array(all_points)
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    def update(frame: int) -> None:
        positions = position_states[frame]

        for view_name, ax in axes.items():
            ax.clear()

            if view_name == "top":
                ax.view_init(elev=90, azim=0)
                ax.set_title("Top View [X-Y]")
                ax.set_proj_type("ortho")
                ax.set_zticklabels([])  # type: ignore
            elif view_name == "front":
                ax.view_init(elev=0, azim=0)
                ax.set_title("Front View [Y-Z]")
                ax.set_proj_type("ortho")
                ax.set_xticklabels([])
            elif view_name == "side":
                ax.view_init(elev=0, azim=90)
                ax.set_title("Side View [X-Z]")
                ax.set_proj_type("ortho")
                ax.set_yticklabels([])
            else:
                ax.view_init(elev=20, azim=45)
                ax.set_title("Isometric View")
                ax.set_proj_type("ortho")

            for link in visualizer.links:
                points = np.array([positions[point_id] for point_id in link.points])
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

            visualizer.draw_wheel(ax, positions)

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

            ax.set_box_aspect([1, 1, 1])

            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_zlabel("Z [mm]")

            if view_name == "iso":
                ax.legend(loc="upper left")

        title_string = (
            f"Wheel Center Z: {positions[PointID.WHEEL_CENTER][2] - initial_positions[PointID.WHEEL_CENTER][2]:.1f} [mm]",
            f"Rack Displacement: {positions[PointID.TRACKROD_INBOARD][1] - initial_positions[PointID.TRACKROD_INBOARD][1]:.1f} [mm]",
        )
        fig.suptitle(
            "\n".join(title_string),
            fontsize=16,
        )

    plt.subplots_adjust(
        left=0.0,
        right=1,
        bottom=0.025,
        top=0.95,
        wspace=0.01,
        hspace=0.01,
    )
    anim = animation.FuncAnimation(
        fig,
        update,  # type: ignore
        frames=len(position_states),
        interval=interval,
        blit=False,
    )

    anim.save(output_path, writer="pillow", fps=fps)
    plt.close()
