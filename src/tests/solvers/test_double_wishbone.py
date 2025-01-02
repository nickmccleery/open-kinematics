from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from kinematics.geometry.loader import load_geometry
from kinematics.geometry.schemas import DoubleWishboneGeometry, PointID
from kinematics.solvers.double_wishbone import DoubleWishboneSolver, SuspensionState

CHECK_TOLERANCE = 1e-2


def create_suspension_animation(
    geometry: DoubleWishboneGeometry, states: list[SuspensionState], output_path: Path
) -> None:
    """Creates an animated visualization of suspension movement."""
    fig = plt.figure(figsize=(20, 20))

    # Create four subplots
    ax_top = fig.add_subplot(221, projection="3d")
    ax_front = fig.add_subplot(222, projection="3d")
    ax_side = fig.add_subplot(223, projection="3d")
    ax_iso = fig.add_subplot(224, projection="3d")
    axes = [ax_top, ax_front, ax_side, ax_iso]

    # Calculate fixed limits once from initial geometry and states
    hp = geometry.hard_points
    inboard_points = np.array(
        [
            hp.upper_wishbone.inboard_front.as_array(),
            hp.upper_wishbone.inboard_rear.as_array(),
            hp.lower_wishbone.inboard_front.as_array(),
            hp.lower_wishbone.inboard_rear.as_array(),
        ]
    )

    # Modified to use points dictionary
    def get_state_points(state: SuspensionState) -> np.ndarray:
        return np.array(
            [
                state.points[PointID.UPPER_WISHBONE_OUTBOARD].as_array(),
                state.points[PointID.LOWER_WISHBONE_OUTBOARD].as_array(),
                state.points[PointID.AXLE_INBOARD].as_array(),
                state.points[PointID.AXLE_OUTBOARD].as_array(),
                state.points[PointID.TRACKROD_INBOARD].as_array(),
                state.points[PointID.TRACKROD_OUTBOARD].as_array(),
            ]
        )

    moving_points = np.vstack([get_state_points(state) for state in states])
    all_points = np.vstack([inboard_points, moving_points])
    min_bounds = all_points.min(axis=0) - 100
    max_bounds = all_points.max(axis=0) + 100

    def plot_state(ax, state: SuspensionState, view_name: str) -> None:
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
            s=50,
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
            s=50,
            label="Outboard Points",
        )

        # Draw upper wishbone legs
        upper_outboard = state.points[PointID.UPPER_WISHBONE_OUTBOARD].as_array()
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
        lower_outboard = state.points[PointID.LOWER_WISHBONE_OUTBOARD].as_array()
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
        axle_inner = state.points[PointID.AXLE_INBOARD].as_array()
        axle_outer = state.points[PointID.AXLE_OUTBOARD].as_array()
        ax.plot(
            [axle_inner[0], axle_outer[0]],
            [axle_inner[1], axle_outer[1]],
            [axle_inner[2], axle_outer[2]],
            "k-",
        )

        # Draw track rod
        track_rod_inner = state.points[PointID.TRACKROD_INBOARD].as_array()
        track_rod_outer = state.points[PointID.TRACKROD_OUTBOARD].as_array()
        ax.plot(
            [track_rod_inner[0], track_rod_outer[0]],
            [track_rod_inner[1], track_rod_outer[1]],
            [track_rod_inner[2], track_rod_outer[2]],
            "k-",
        )

        # Set fixed axis limits
        ax.set_xlim(min_bounds[0], max_bounds[0])
        ax.set_ylim(min_bounds[1], max_bounds[1])
        ax.set_zlim(min_bounds[2], max_bounds[2])

        if view_name == "isometric":
            ax.legend()

    def update(frame):
        state = states[frame]
        plot_state(ax_top, state, "top")
        plot_state(ax_front, state, "front")
        plot_state(ax_side, state, "side")
        plot_state(ax_iso, state, "isometric")
        fig.suptitle(f"Frame {frame}", fontsize=16)

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(states), interval=100, blit=False
    )

    # Save as GIF
    anim.save(output_path, writer="pillow", fps=20)
    plt.close()


def test_run_solver(double_wishbone_geometry_file: Path) -> None:
    """Tests the double wishbone solver through its range of motion."""
    geometry = load_geometry(double_wishbone_geometry_file)

    if not isinstance(geometry, DoubleWishboneGeometry):
        raise ValueError("Invalid geometry type")

    solver = DoubleWishboneSolver(geometry=geometry)

    # Create displacement sweep
    displacement_range = [-80, 80]
    n_steps = 21
    displacements = list(
        np.linspace(displacement_range[0], displacement_range[1], n_steps)
    )

    # Solve for all positions
    states = solver.solve_sweep(displacements)

    # Verify constraints are maintained
    for state, displacement in zip(states, displacements):
        # Verify length constraints
        for constraint in solver.length_constraints:
            p1 = state.points[constraint.p1].as_array()
            p2 = state.points[constraint.p2].as_array()
            current_length = np.linalg.norm(p1 - p2)

            assert np.abs(current_length - constraint.distance) < CHECK_TOLERANCE, (
                f"Constraint violation at displacement {displacement}: "
                f"{constraint.p1} to {constraint.p2}"
            )

        # Verify axle midpoint z position
        axle_inner = state.points[PointID.AXLE_INBOARD].as_array()
        axle_outer = state.points[PointID.AXLE_OUTBOARD].as_array()
        axle_midpoint = (axle_inner + axle_outer) / 2
        initial_midpoint = (
            solver.initial_state.points[PointID.AXLE_INBOARD].as_array()
            + solver.initial_state.points[PointID.AXLE_OUTBOARD].as_array()
        ) / 2
        target_z = initial_midpoint[2] + displacement

        assert (
            np.abs(axle_midpoint[2] - target_z) < CHECK_TOLERANCE
        ), f"Failed to maintain axle midpoint at displacement {displacement}"

    # Create animation
    states_animate = states + states[::-1]
    output_path = Path("suspension_motion.gif")
    create_suspension_animation(geometry, states_animate, output_path)
