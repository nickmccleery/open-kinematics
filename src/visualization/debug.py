from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.solvers.common import KinematicState


@dataclass
class LinkVisualization:
    points: list[PointID]
    color: str
    label: str
    linewidth: float = 3.0
    linestyle: str = "-"
    marker: str = "o"
    markersize: float = 10.0


@dataclass
class WheelVisualization:
    diameter: float
    width: float
    num_points: int = 50
    color: str = "black"
    alpha: float = 0.75
    linestyle: str = "-"


class SuspensionVisualizer:
    def __init__(
        self, geometry: DoubleWishboneGeometry, wheel_config: WheelVisualization
    ):
        self.geometry = geometry
        self.wheel_config = wheel_config
        self.links = self.define_links()

    def define_links(self) -> List[LinkVisualization]:
        return [
            LinkVisualization(
                points=[
                    PointID.UPPER_WISHBONE_INBOARD_FRONT,
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.UPPER_WISHBONE_INBOARD_REAR,
                ],
                color="dodgerblue",
                label="Upper Wishbone",
            ),
            LinkVisualization(
                points=[
                    PointID.LOWER_WISHBONE_INBOARD_FRONT,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_INBOARD_REAR,
                ],
                color="dodgerblue",
                label="Lower Wishbone",
            ),
            LinkVisualization(
                points=[
                    PointID.TRACKROD_OUTBOARD,
                    PointID.UPPER_WISHBONE_OUTBOARD,
                    PointID.LOWER_WISHBONE_OUTBOARD,
                    PointID.TRACKROD_OUTBOARD,
                ],
                color="slategrey",
                label="Upright",
            ),
            LinkVisualization(
                points=[PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD],
                color="darkorange",
                label="Track Rod",
            ),
            LinkVisualization(
                points=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD],
                color="forestgreen",
                label="Axle",
            ),
        ]

    def draw_wheel(
        self,
        ax,
        wheel_center,
        wheel_inboard,
        wheel_outboard,
        axle_vector,
        state: KinematicState,
    ) -> None:
        """Draw a 3D wheel representation"""
        # Normalize axle vector
        axle_vector = axle_vector / np.linalg.norm(axle_vector)

        # Create basis vectors for wheel plane
        # First basis vector is the axle direction
        e1 = axle_vector

        # Second basis vector can be any perpendicular vector
        e2 = np.array([1, 0, 0])
        if np.abs(np.dot(e1, e2)) > 0.9:
            e2 = np.array([0, 1, 0])
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)

        # Third basis vector completes right-handed system
        e3 = np.cross(e1, e2)

        # Generate points for wheel rim
        theta = np.linspace(0, 2 * np.pi, self.wheel_config.num_points)
        radius = self.wheel_config.diameter / 2

        # Create wheel points.
        rim_points_center = np.zeros((self.wheel_config.num_points, 3))
        rim_points_inboard = np.zeros((self.wheel_config.num_points, 3))
        rim_points_outboard = np.zeros((self.wheel_config.num_points, 3))

        for i, angle in enumerate(theta):
            # Combine basis vectors with sin/cos for circle
            rim_points_center[i] = wheel_center + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_inboard[i] = wheel_inboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_outboard[i] = wheel_outboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )

        # Plot center points
        ax.plot(
            rim_points_center[:, 0],
            rim_points_center[:, 1],
            rim_points_center[:, 2],
            color=self.wheel_config.color,
            alpha=0.5,
            linestyle=self.wheel_config.linestyle,
        )

        # Plot inboard points
        ax.plot(
            rim_points_inboard[:, 0],
            rim_points_inboard[:, 1],
            rim_points_inboard[:, 2],
            color=self.wheel_config.color,
            alpha=self.wheel_config.alpha,
            linestyle=self.wheel_config.linestyle,
        )

        # Plot outboard points
        ax.plot(
            rim_points_outboard[:, 0],
            rim_points_outboard[:, 1],
            rim_points_outboard[:, 2],
            color=self.wheel_config.color,
            alpha=self.wheel_config.alpha,
            linestyle=self.wheel_config.linestyle,
        )

    def create_animation(
        self,
        states: List[KinematicState],
        output_path: Path,
        fps: int = 20,
        interval: int = 100,
    ) -> None:
        """Create and save animation of suspension movement"""
        fig = plt.figure(figsize=(16, 16))  # Make figure square
        gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15)

        axes = {
            "top": fig.add_subplot(gs[0, 0], projection="3d"),
            "front": fig.add_subplot(gs[0, 1], projection="3d"),
            "side": fig.add_subplot(gs[1, 0], projection="3d"),
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

                # Set view angles
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

                # Draw all links
                for link in self.links:
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
                self.draw_wheel(
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
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            fig.suptitle(f"Frame {frame}", fontsize=16)

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
            fig, update, frames=len(states), interval=interval, blit=False
        )

        # Save animation and close figure
        anim.save(output_path, writer="pillow", fps=fps)
        plt.close()
