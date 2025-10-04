from pathlib import Path
from typing import cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from kinematics.enums import PointID
from kinematics.types import Vec3
from kinematics.visualization.main import SuspensionVisualizer


def create_animation(
    position_states: list[dict[PointID, Vec3]],
    initial_positions: dict[PointID, Vec3],
    visualizer: SuspensionVisualizer,
    output_path: Path,
    fps: int = 20,
    writer: str | None = None,
    codec: str = "libx264",
    dpi: int = 200,
    show_live: bool = True,
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

    # Compute global bounds.
    all_points = np.array([p for state in position_states for p in state.values()])
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    # Common axis limits and aspect.
    x_mid = (max_bounds[0] + min_bounds[0]) * 0.5
    y_mid = (max_bounds[1] + min_bounds[1]) * 0.5
    z_mid = (max_bounds[2] + min_bounds[2]) * 0.5
    max_range = max(
        max_bounds[0] - min_bounds[0],
        max_bounds[1] - min_bounds[1],
        max_bounds[2] - min_bounds[2],
    )

    # Configure axes once.
    for view_name, ax in axes.items():
        ax3d = cast(Axes3D, ax)
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
        else:
            ax3d.view_init(elev=20, azim=45)
            ax3d.set_title("Isometric View")
            ax3d.set_proj_type("ortho")

        ax3d.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax3d.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax3d.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])
        ax3d.set_box_aspect([1, 1, 1])  # type: ignore[arg-type]
        ax3d.set_xlabel("X [mm]")
        ax3d.set_ylabel("Y [mm]")
        ax3d.set_zlabel("Z [mm]")

    # Pre-create link line artists per axis so we can just update data each frame.
    link_artists: dict[str, list] = {k: [] for k in axes.keys()}
    for view_name, ax in axes.items():
        for link in visualizer.links:
            pts = np.array([initial_positions[pid] for pid in link.points])
            (line,) = ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=link.color,
                linewidth=link.linewidth,
                linestyle=link.linestyle,
                marker=link.marker,
                markersize=link.markersize,
                label=link.label if view_name == "iso" else None,
                animated=False,
            )
            link_artists[view_name].append(line)

    # Add legend once on iso view.
    axes["iso"].legend(loc="upper left")

    # Create artists.
    wheel_cfg = visualizer.wheel_config
    theta = np.linspace(0, 2 * np.pi, wheel_cfg.num_points)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    radius = wheel_cfg.diameter / 2
    spoke_indices = np.linspace(0, wheel_cfg.num_points - 1, 12, dtype=int)

    wheel_artists: dict[str, dict[str, list]] = {
        k: {"rims": [], "spokes": []} for k in axes.keys()
    }

    # Helper to compute wheel geometry for a given positions dict.
    def compute_wheel_points(positions: dict[PointID, Vec3]):
        wheel_center = positions[PointID.WHEEL_CENTER]
        wheel_inboard = positions[PointID.WHEEL_INBOARD]
        wheel_outboard = positions[PointID.WHEEL_OUTBOARD]
        axle_vec = positions[PointID.AXLE_OUTBOARD] - positions[PointID.AXLE_INBOARD]
        e1 = axle_vec / np.linalg.norm(axle_vec)
        e2 = np.array([1.0, 0.0, 0.0])
        if float(abs(np.dot(e1, e2))) > 0.9:
            e2 = np.array([0.0, 1.0, 0.0])
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)

        ring = (cos_t[:, None] * e2) + (sin_t[:, None] * e3)
        rim_center = wheel_center + radius * ring
        rim_inboard = wheel_inboard + radius * ring
        rim_outboard = wheel_outboard + radius * ring
        return rim_center, rim_inboard, rim_outboard

    # Create initial wheel artists using the initial positions.
    rim_center, rim_inboard, rim_outboard = compute_wheel_points(initial_positions)
    for view_name, ax in axes.items():
        (rim_center_line,) = ax.plot(
            rim_center[:, 0],
            rim_center[:, 1],
            rim_center[:, 2],
            color=wheel_cfg.color,
            alpha=0.25,
            linestyle=wheel_cfg.linestyle,
            animated=False,
        )
        (rim_in_line,) = ax.plot(
            rim_inboard[:, 0],
            rim_inboard[:, 1],
            rim_inboard[:, 2],
            color=wheel_cfg.color,
            alpha=wheel_cfg.alpha,
            linestyle=wheel_cfg.linestyle,
            animated=False,
        )
        (rim_out_line,) = ax.plot(
            rim_outboard[:, 0],
            rim_outboard[:, 1],
            rim_outboard[:, 2],
            color=wheel_cfg.color,
            alpha=wheel_cfg.alpha,
            linestyle=wheel_cfg.linestyle,
            animated=False,
        )
        wheel_artists[view_name]["rims"].extend(
            [rim_center_line, rim_in_line, rim_out_line]
        )

        for idx in spoke_indices:
            (sp_line,) = ax.plot(
                [rim_inboard[idx, 0], rim_outboard[idx, 0]],
                [rim_inboard[idx, 1], rim_outboard[idx, 1]],
                [rim_inboard[idx, 2], rim_outboard[idx, 2]],
                color=wheel_cfg.color,
                alpha=wheel_cfg.alpha,
                linestyle=wheel_cfg.linestyle,
                animated=False,
            )
            wheel_artists[view_name]["spokes"].append(sp_line)

    # Layout.
    plt.subplots_adjust(
        left=0.0, right=1, bottom=0.025, top=0.95, wspace=0.01, hspace=0.01
    )

    # Persistent title updated each frame (cheaper than re-creating).
    title_artist = fig.suptitle("", fontsize=16)

    # Update function that only updates artist data (no clears/plots).
    def update(frame: int):
        positions = position_states[frame]

        # Update links.
        for view_name in axes.keys():
            for line, link in zip(link_artists[view_name], visualizer.links):
                pts = np.array([positions[pid] for pid in link.points])
                line.set_data(pts[:, 0], pts[:, 1])
                # For 3D lines, z is set via set_3d_properties
                line.set_3d_properties(pts[:, 2])  # type: ignore[attr-defined]

        # Update wheel geometry.
        rim_center_u, rim_inboard_u, rim_outboard_u = compute_wheel_points(positions)
        for view_name in axes.keys():
            rims = wheel_artists[view_name]["rims"]
            # center rim
            rims[0].set_data(rim_center_u[:, 0], rim_center_u[:, 1])
            rims[0].set_3d_properties(rim_center_u[:, 2])  # type: ignore[attr-defined]
            # inboard rim
            rims[1].set_data(rim_inboard_u[:, 0], rim_inboard_u[:, 1])
            rims[1].set_3d_properties(rim_inboard_u[:, 2])  # type: ignore[attr-defined]
            # outboard rim
            rims[2].set_data(rim_outboard_u[:, 0], rim_outboard_u[:, 1])
            rims[2].set_3d_properties(rim_outboard_u[:, 2])  # type: ignore[attr-defined]

            for sp_line, idx in zip(wheel_artists[view_name]["spokes"], spoke_indices):
                sp_line.set_data(
                    [rim_inboard_u[idx, 0], rim_outboard_u[idx, 0]],
                    [rim_inboard_u[idx, 1], rim_outboard_u[idx, 1]],
                )
                sp_line.set_3d_properties(
                    [rim_inboard_u[idx, 2], rim_outboard_u[idx, 2]]
                )  # type: ignore[attr-defined]

        # Update global title.
        title_string = (
            f"Wheel Center Z: {positions[PointID.WHEEL_CENTER][2] - initial_positions[PointID.WHEEL_CENTER][2]:.1f} [mm]",
            f"Rack Displacement: {positions[PointID.TRACKROD_INBOARD][1] - initial_positions[PointID.TRACKROD_INBOARD][1]:.1f} [mm]",
        )
        title_artist.set_text("\n".join(title_string))

        artists = []
        for view_name in axes.keys():
            artists.extend(link_artists[view_name])
            artists.extend(wheel_artists[view_name]["rims"])
            artists.extend(wheel_artists[view_name]["spokes"])
        return artists

    # Allow frame skipping for faster renders.
    frame_indices = range(0, len(position_states), 1)

    # Choose writer automatically if not provided.
    out_suffix = output_path.suffix.lower()
    chosen_writer: str
    if writer is not None:
        chosen_writer = writer
    elif out_suffix in {".mp4", ".m4v", ".mov"}:
        chosen_writer = "ffmpeg"
    else:
        chosen_writer = "pillow"

    try:
        if chosen_writer == "ffmpeg":
            Writer = animation.writers["ffmpeg"]
            writer_inst = Writer(fps=fps, codec=codec)
        else:
            Writer = animation.writers[chosen_writer]
            writer_inst = Writer(fps=fps)
    except Exception:
        # Fallback to pillow
        Writer = animation.writers["pillow"]
        writer_inst = Writer(fps=fps)

    if show_live:
        plt.ion()
        plt.show(block=False)

    try:
        with writer_inst.saving(fig, str(output_path), dpi):
            for frame in frame_indices:
                update(frame)
                if show_live:
                    # Draw immediately and let GUI process events.
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                writer_inst.grab_frame()
    finally:
        if show_live:
            plt.ioff()
        plt.close(fig)
