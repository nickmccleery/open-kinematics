from dataclasses import dataclass
from typing import List

import numpy as np

from kinematics.geometry.points.ids import PointID
from kinematics.suspensions.double_wishbone.geometry import DoubleWishboneGeometry
from kinematics.types.state import Positions


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
        positions: Positions,
    ) -> None:
        wheel_center = positions[PointID.WHEEL_CENTER]
        wheel_inboard = positions[PointID.WHEEL_INBOARD]
        wheel_outboard = positions[PointID.WHEEL_OUTBOARD]
        axle_vector = positions[PointID.AXLE_OUTBOARD] - positions[PointID.AXLE_INBOARD]

        axle_vector = axle_vector / np.linalg.norm(axle_vector)

        e1 = axle_vector
        e2 = np.array([1, 0, 0])
        if np.abs(np.dot(e1, e2)) > 0.9:
            e2 = np.array([0, 1, 0])
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)

        e3 = np.cross(e1, e2)

        theta = np.linspace(0, 2 * np.pi, self.wheel_config.num_points)
        radius = self.wheel_config.diameter / 2

        rim_points_center = np.zeros((self.wheel_config.num_points, 3))
        rim_points_inboard = np.zeros((self.wheel_config.num_points, 3))
        rim_points_outboard = np.zeros((self.wheel_config.num_points, 3))

        for i, angle in enumerate(theta):
            rim_points_center[i] = wheel_center + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_inboard[i] = wheel_inboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_outboard[i] = wheel_outboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )

        ax.plot(
            rim_points_center[:, 0],
            rim_points_center[:, 1],
            rim_points_center[:, 2],
            color=self.wheel_config.color,
            alpha=0.25,
            linestyle=self.wheel_config.linestyle,
        )

        ax.plot(
            rim_points_inboard[:, 0],
            rim_points_inboard[:, 1],
            rim_points_inboard[:, 2],
            color=self.wheel_config.color,
            alpha=self.wheel_config.alpha,
            linestyle=self.wheel_config.linestyle,
        )

        ax.plot(
            rim_points_outboard[:, 0],
            rim_points_outboard[:, 1],
            rim_points_outboard[:, 2],
            color=self.wheel_config.color,
            alpha=self.wheel_config.alpha,
            linestyle=self.wheel_config.linestyle,
        )

        spoke_indices = np.linspace(0, self.wheel_config.num_points - 1, 12, dtype=int)

        for idx in spoke_indices:
            ax.plot(
                [rim_points_inboard[idx, 0], rim_points_outboard[idx, 0]],
                [rim_points_inboard[idx, 1], rim_points_outboard[idx, 1]],
                [rim_points_inboard[idx, 2], rim_points_outboard[idx, 2]],
                color=self.wheel_config.color,
                alpha=self.wheel_config.alpha,
                linestyle=self.wheel_config.linestyle,
            )
