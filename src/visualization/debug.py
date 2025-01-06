from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.solvers.common import KinematicState

CHECK_TOLERANCE = 1e-2


def create_suspension_animation(
    geometry: DoubleWishboneGeometry, states: list[KinematicState], output_path: Path
) -> None:
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)

    # Create four subplots.
    ax_top = fig.add_subplot(gs[0, 0], projection="3d")
    ax_front = fig.add_subplot(gs[0, 1], projection="3d")
    ax_side = fig.add_subplot(gs[1, 0], projection="3d")
    ax_iso = fig.add_subplot(gs[1, 1], projection="3d")

    hp = geometry.hard_points
    inboard_points = np.array(
        [
            hp.upper_wishbone.inboard_front.as_array(),
            hp.upper_wishbone.inboard_rear.as_array(),
            hp.lower_wishbone.inboard_front.as_array(),
            hp.lower_wishbone.inboard_rear.as_array(),
        ]
    )

    def get_state_points(state: KinematicState) -> np.ndarray:
        return np.array(
            [
                state.hard_points[PointID.UPPER_WISHBONE_OUTBOARD].as_array(),
                state.hard_points[PointID.LOWER_WISHBONE_OUTBOARD].as_array(),
                state.hard_points[PointID.AXLE_INBOARD].as_array(),
                state.hard_points[PointID.AXLE_OUTBOARD].as_array(),
                state.hard_points[PointID.TRACKROD_INBOARD].as_array(),
                state.hard_points[PointID.TRACKROD_OUTBOARD].as_array(),
            ]
        )

    moving_points = np.vstack([get_state_points(state) for state in states])
    all_points = np.vstack([inboard_points, moving_points])
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    def plot_state(ax, state: KinematicState, view_name: str) -> None:
        """Plot suspension state on a given axis."""
        ax.clear()

        # Set view based on subplot type
        if view_name == "top":
            ax.view_init(elev=90, azim=0)
            ax.set_title("Top View (X-Y)")
        elif view_name == "front":
            ax.view_init(elev=0, azim=0)
            ax.set_title("Front View (Y-Z)")
        elif view_name == "side":
            ax.view_init(elev=0, azim=90)
            ax.set_title("Side View (X-Z)")
        else:  # isometric
            ax.view_init(elev=20, azim=45)
            ax.set_title("Isometric View")

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")

        # Plot fixed hardpoints
        ax.scatter(
            inboard_points[:, 0],
            inboard_points[:, 1],
            inboard_points[:, 2],
            color="red",
            marker="o",
            s=30,
            label="Inboard Points",
        )

        # Get current moving points from state
        moving_points = get_state_points(state)
        ax.scatter(
            moving_points[:, 0],
            moving_points[:, 1],
            moving_points[:, 2],
            color="blue",
            marker="o",
            s=30,
            label="Outboard Points",
        )

        # Draw upper wishbone legs
        upper_outboard = state.hard_points[PointID.UPPER_WISHBONE_OUTBOARD].as_array()
        ax.plot(
            [hp.upper_wishbone.inboard_front.x, upper_outboard[0]],
            [hp.upper_wishbone.inboard_front.y, upper_outboard[1]],
            [hp.upper_wishbone.inboard_front.z, upper_outboard[2]],
            "k-",
        )
        ax.plot(
            [hp.upper_wishbone.inboard_rear.x, upper_outboard[0]],
            [hp.upper_wishbone.inboard_rear.y, upper_outboard[1]],
            [hp.upper_wishbone.inboard_rear.z, upper_outboard[2]],
            "k-",
        )

        # Draw lower wishbone legs
        lower_outboard = state.hard_points[PointID.LOWER_WISHBONE_OUTBOARD].as_array()
        ax.plot(
            [hp.lower_wishbone.inboard_front.x, lower_outboard[0]],
            [hp.lower_wishbone.inboard_front.y, lower_outboard[1]],
            [hp.lower_wishbone.inboard_front.z, lower_outboard[2]],
            "k-",
        )
        ax.plot(
            [hp.lower_wishbone.inboard_rear.x, lower_outboard[0]],
            [hp.lower_wishbone.inboard_rear.y, lower_outboard[1]],
            [hp.lower_wishbone.inboard_rear.z, lower_outboard[2]],
            "k-",
        )

        # Draw upright (connecting upper and lower ball joints)
        ax.plot(
            [upper_outboard[0], lower_outboard[0]],
            [upper_outboard[1], lower_outboard[1]],
            [upper_outboard[2], lower_outboard[2]],
            "k-",
        )

        # Draw wheel axle
        axle_inner = state.hard_points[PointID.AXLE_INBOARD].as_array()
        axle_outer = state.hard_points[PointID.AXLE_OUTBOARD].as_array()
        ax.plot(
            [axle_inner[0], axle_outer[0]],
            [axle_inner[1], axle_outer[1]],
            [axle_inner[2], axle_outer[2]],
            "k-",
        )

        # Draw track rod
        track_rod_inner = state.hard_points[PointID.TRACKROD_INBOARD].as_array()
        track_rod_outer = state.hard_points[PointID.TRACKROD_OUTBOARD].as_array()
        ax.plot(
            [track_rod_inner[0], track_rod_outer[0]],
            [track_rod_inner[1], track_rod_outer[1]],
            [track_rod_inner[2], track_rod_outer[2]],
            "k-",
        )

        # Draw wheel center and outboard.
        wheel_center = state.derived_points[PointID.WHEEL_CENTER].as_array()
        wheel_outboard = state.derived_points[PointID.WHEEL_OUTBOARD].as_array()
        ax.plot(
            [wheel_center[0], wheel_outboard[0]],
            [wheel_center[1], wheel_outboard[1]],
            [wheel_center[2], wheel_outboard[2]],
            "r-",
        )

        # Set fixed axis limits
        ax.set_xlim(min_bounds[0], max_bounds[0])
        ax.set_ylim(min_bounds[1], max_bounds[1])
        ax.set_zlim(min_bounds[2], max_bounds[2])

        if view_name == "isometric":
            ax.legend()

    def print_state_points(state: KinematicState, frame: int, file_path: str) -> None:
        """Print a formatted table of point positions for the current frame."""
        points_of_interest = {
            "Upper Outboard": PointID.UPPER_WISHBONE_OUTBOARD,
            "Lower Outboard": PointID.LOWER_WISHBONE_OUTBOARD,
            "Axle Inboard": PointID.AXLE_INBOARD,
            "Axle Outboard": PointID.AXLE_OUTBOARD,
            "Trackrod Inboard": PointID.TRACKROD_INBOARD,
            "Trackrod Outboard": PointID.TRACKROD_OUTBOARD,
        }

        with open(file_path, "a") as file:
            if frame == 0:
                file.write("\nPoint Positions Table:\n")
                file.write(
                    f"{'Frame':<6} {'Point':<20} {'X':>10} {'Y':>10} {'Z':>10}\n"
                )
                file.write("-" * 60 + "\n")

            for name, point_id in points_of_interest.items():
                point = state.hard_points[point_id].as_array()
                file.write(
                    f"{frame:<6} {name:<20} {point[0]:>10.2f} {point[1]:>10.2f} {point[2]:>10.2f}\n"
                )

            # Add a blank line between frames
            file.write("\n")

            # At last frame, print fixed points for reference
            if frame == len(states) - 1:
                file.write("\nFixed Points Reference:\n")
                file.write("-" * 60 + "\n")
                hp = geometry.hard_points
                fixed_points = {
                    "UW Inboard Front": hp.upper_wishbone.inboard_front,
                    "UW Inboard Rear": hp.upper_wishbone.inboard_rear,
                    "LW Inboard Front": hp.lower_wishbone.inboard_front,
                    "LW Inboard Rear": hp.lower_wishbone.inboard_rear,
                }
                for name, point in fixed_points.items():
                    pos = point.as_array()
                    file.write(
                        f"{'--':<6} {name:<20} {pos[0]:>10.2f} {pos[1]:>10.2f} {pos[2]:>10.2f}\n"
                    )

    def update(frame):
        state = states[frame]
        plot_state(ax_top, state, "top")
        plot_state(ax_front, state, "front")
        plot_state(ax_side, state, "side")
        plot_state(ax_iso, state, "isometric")
        fig.suptitle(f"Frame {frame}", fontsize=16)
        print_state_points(state, frame, "point_positions.txt")

    # Create animation
    plt.subplots_adjust(
        left=0.05,  # Move left edge closer to figure edge
        right=0.95,  # Move right edge closer to figure edge
        bottom=0.05,  # Move bottom edge closer to figure edge
        top=0.95,  # Move top edge closer to figure edge
        wspace=0.1,  # Reduce horizontal space between subplots
        hspace=0.1,  # Reduce vertical space between subplots
    )
    anim = animation.FuncAnimation(
        fig, update, frames=len(states), interval=100, blit=False
    )

    # Save as GIF
    anim.save(output_path, writer="pillow", fps=20)
    plt.close()
