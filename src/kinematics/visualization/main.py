from dataclasses import dataclass

import numpy as np

from kinematics.enums import PointID
from kinematics.types import Vec3


@dataclass
class LinkVisualization:
    """Configuration for visualizing a suspension link."""

    points: list[PointID]
    color: str
    label: str
    linewidth: float = 3.0
    linestyle: str = "-"
    marker: str = "o"
    markersize: float = 10.0


@dataclass
class WheelVisualization:
    """Configuration for visualizing the wheel."""

    diameter: float
    width: float
    rim_diameter: float
    num_points: int = 50
    color: str = "black"
    alpha: float = 0.75
    linestyle: str = "-"
    # Number of radial hoop lines that wrap around the tire.
    num_hoops: int = 20
    # Position of shoulder circles as fraction of half-width from center.
    # 0 = at center, 1 = at edge. Lower values move shoulders closer to center.
    shoulder_position: float = 0.85


class SuspensionVisualizer:
    """Renders suspension geometry to matplotlib 3D axes."""

    def draw_links(
        self,
        ax,
        positions: dict[PointID, Vec3],
    ) -> list:
        """
        Draws all links and returns a list of matplotlib line artists.
        """
        link_artists = []
        for link in self.links:
            pts = np.array([positions[pid] for pid in link.points])
            (line,) = ax.plot(
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
            link_artists.append(line)
        return link_artists

    def update_links(
        self,
        artists: list,
        positions: dict[PointID, Vec3],
    ) -> None:
        """
        Update all link artists with new geometry for animation.
        """
        for line, link in zip(artists, self.links):
            pts = np.array([positions[pid] for pid in link.points])
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])

    @staticmethod
    def get_band_endpoints(
        wheel_inboard: np.ndarray,
        wheel_outboard: np.ndarray,
        e2: np.ndarray,
        e3: np.ndarray,
        num_bands: int,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the endpoints for cross-tyre bands at true radial angles.

        Returns two arrays of shape (num_bands, 3): inboard and outboard endpoints.
        """
        thetas = np.linspace(0, 2 * np.pi, num_bands, endpoint=False)
        band_inboard = np.array(
            [
                wheel_inboard + radius * (np.cos(theta) * e2 + np.sin(theta) * e3)
                for theta in thetas
            ]
        )
        band_outboard = np.array(
            [
                wheel_outboard + radius * (np.cos(theta) * e2 + np.sin(theta) * e3)
                for theta in thetas
            ]
        )
        return band_inboard, band_outboard

    def __init__(
        self, links: list[LinkVisualization], wheel_config: WheelVisualization
    ):
        self.links = links
        self.wheel_config = wheel_config

    def _compute_wheel_frame(
        self,
        positions: dict[PointID, Vec3],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute orthonormal frame for wheel visualization.

        Returns:
            Tuple of (e1, e2, e3) where e1 is axle direction.
        """
        axle_vector = positions[PointID.AXLE_OUTBOARD] - positions[PointID.AXLE_INBOARD]
        e1 = axle_vector / np.linalg.norm(axle_vector)

        e2 = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(e1, e2)) > 0.9:
            e2 = np.array([0.0, 1.0, 0.0])
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)

        e3 = np.cross(e1, e2)
        return e1, e2, e3

    def _compute_circle_points(
        self,
        center: np.ndarray,
        radius: float,
        e2: np.ndarray,
        e3: np.ndarray,
    ) -> np.ndarray:
        """Compute points for a circle in the plane defined by e2, e3."""
        theta = np.linspace(0, 2 * np.pi, self.wheel_config.num_points)
        points = np.zeros((self.wheel_config.num_points, 3))
        for i, angle in enumerate(theta):
            points[i] = center + radius * (np.cos(angle) * e2 + np.sin(angle) * e3)
        return points

    def _compute_hoop_points(
        self,
        wheel_inboard: np.ndarray,
        wheel_outboard: np.ndarray,
        wheel_center: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        e3: np.ndarray,
        theta: float,
        tire_radius: float,
        rim_radius: float,
    ) -> np.ndarray:
        """
        Compute points for a single hoop that wraps around the tire profile.

        The hoop passes through the 5 circumferential circle intersection points:
        rim inboard -> shoulder inboard -> crown center ->
        shoulder outboard -> rim outboard.

        Uses straight line segments between these points.
        """
        half_width = np.linalg.norm(wheel_outboard - wheel_inboard) / 2

        # Radial direction for this hoop angle.
        radial = np.cos(theta) * e2 + np.sin(theta) * e3

        # Compute shoulder positions using configurable parameter.
        # shoulder_position=0.7 means shoulders are 70% of half-width from center.
        shoulder_offset = self.wheel_config.shoulder_position * half_width
        shoulder_inboard = wheel_center - shoulder_offset * e1
        shoulder_outboard = wheel_center + shoulder_offset * e1

        points = np.array(
            [
                wheel_inboard + rim_radius * radial,  # Rim inboard edge
                shoulder_inboard + tire_radius * radial,  # Shoulder inboard
                wheel_center + tire_radius * radial,  # Crown center
                shoulder_outboard + tire_radius * radial,  # Shoulder outboard
                wheel_outboard + rim_radius * radial,  # Rim outboard edge
            ]
        )

        return points

    def draw_wheel(
        self,
        ax,
        positions: dict[PointID, Vec3],
    ) -> dict:
        """
        Draws a 3D wheel/tire representation and returns the matplotlib artists.

        The visualization includes:
        - 5 circumferential circles: 2 at rim radius (edges), 3 at tire radius (crown)
        - Radial hoop lines that wrap around the tire profile

        Returns dict with 'rims' (list of 5 circle lines) and 'bands' (list of hoops).
        """
        wheel_center = positions[PointID.WHEEL_CENTER]
        wheel_inboard = positions[PointID.WHEEL_INBOARD]
        wheel_outboard = positions[PointID.WHEEL_OUTBOARD]

        e1, e2, e3 = self._compute_wheel_frame(positions)

        tire_radius = self.wheel_config.diameter / 2
        rim_radius = self.wheel_config.rim_diameter / 2

        # Compute axial positions for the 5 circles.
        half_width = np.linalg.norm(wheel_outboard - wheel_inboard) / 2

        # Crown circles at tire radius: center and shoulder positions.
        crown_center = wheel_center
        shoulder_offset = self.wheel_config.shoulder_position * half_width
        shoulder_inboard = wheel_center - shoulder_offset * e1
        shoulder_outboard = wheel_center + shoulder_offset * e1

        # Compute circle points.
        rim_inboard_pts = self._compute_circle_points(wheel_inboard, rim_radius, e2, e3)
        rim_outboard_pts = self._compute_circle_points(
            wheel_outboard, rim_radius, e2, e3
        )
        crown_center_pts = self._compute_circle_points(
            crown_center, tire_radius, e2, e3
        )
        shoulder_inboard_pts = self._compute_circle_points(
            shoulder_inboard, tire_radius, e2, e3
        )
        shoulder_outboard_pts = self._compute_circle_points(
            shoulder_outboard, tire_radius, e2, e3
        )

        # Draw the 5 circumferential circles.
        rim_lines = []
        linewidth = 1.0

        # Rim edges (lower radius, full alpha).
        rim_lines.append(
            ax.plot(
                rim_inboard_pts[:, 0],
                rim_inboard_pts[:, 1],
                rim_inboard_pts[:, 2],
                color=self.wheel_config.color,
                alpha=self.wheel_config.alpha,
                linestyle=self.wheel_config.linestyle,
                linewidth=linewidth,
            )[0]
        )
        rim_lines.append(
            ax.plot(
                rim_outboard_pts[:, 0],
                rim_outboard_pts[:, 1],
                rim_outboard_pts[:, 2],
                color=self.wheel_config.color,
                alpha=self.wheel_config.alpha,
                linestyle=self.wheel_config.linestyle,
                linewidth=linewidth,
            )[0]
        )

        # Crown circles (tire radius).
        rim_lines.append(
            ax.plot(
                crown_center_pts[:, 0],
                crown_center_pts[:, 1],
                crown_center_pts[:, 2],
                color=self.wheel_config.color,
                alpha=self.wheel_config.alpha,
                linestyle=self.wheel_config.linestyle,
                linewidth=linewidth,
            )[0]
        )
        rim_lines.append(
            ax.plot(
                shoulder_inboard_pts[:, 0],
                shoulder_inboard_pts[:, 1],
                shoulder_inboard_pts[:, 2],
                color=self.wheel_config.color,
                alpha=0.5,
                linestyle=self.wheel_config.linestyle,
                linewidth=linewidth,
            )[0]
        )
        rim_lines.append(
            ax.plot(
                shoulder_outboard_pts[:, 0],
                shoulder_outboard_pts[:, 1],
                shoulder_outboard_pts[:, 2],
                color=self.wheel_config.color,
                alpha=0.5,
                linestyle=self.wheel_config.linestyle,
                linewidth=linewidth,
            )[0]
        )

        # Draw radial hoop lines that wrap around the tire.
        num_hoops = self.wheel_config.num_hoops
        hoop_angles = np.linspace(0, 2 * np.pi, num_hoops, endpoint=False)

        band_lines = []
        for theta in hoop_angles:
            hoop_pts = self._compute_hoop_points(
                wheel_inboard,
                wheel_outboard,
                wheel_center,
                e1,
                e2,
                e3,
                theta,
                tire_radius,
                rim_radius,
            )
            band_lines.append(
                ax.plot(
                    hoop_pts[:, 0],
                    hoop_pts[:, 1],
                    hoop_pts[:, 2],
                    color=self.wheel_config.color,
                    alpha=self.wheel_config.alpha,
                    linestyle=self.wheel_config.linestyle,
                    linewidth=linewidth,
                )[0]
            )

        return {"rims": rim_lines, "bands": band_lines}

    def update_wheel(
        self,
        artists: dict,
        positions: dict[PointID, Vec3],
    ) -> None:
        """
        Update the wheel artists with new geometry for animation.
        """
        wheel_center = positions[PointID.WHEEL_CENTER]
        wheel_inboard = positions[PointID.WHEEL_INBOARD]
        wheel_outboard = positions[PointID.WHEEL_OUTBOARD]

        e1, e2, e3 = self._compute_wheel_frame(positions)

        tire_radius = self.wheel_config.diameter / 2
        rim_radius = self.wheel_config.rim_diameter / 2

        # Compute axial positions for the 5 circles.
        half_width = np.linalg.norm(wheel_outboard - wheel_inboard) / 2
        crown_center = wheel_center
        shoulder_offset = self.wheel_config.shoulder_position * half_width
        shoulder_inboard = wheel_center - shoulder_offset * e1
        shoulder_outboard = wheel_center + shoulder_offset * e1

        # Compute circle points.
        rim_inboard_pts = self._compute_circle_points(wheel_inboard, rim_radius, e2, e3)
        rim_outboard_pts = self._compute_circle_points(
            wheel_outboard, rim_radius, e2, e3
        )
        crown_center_pts = self._compute_circle_points(
            crown_center, tire_radius, e2, e3
        )
        shoulder_inboard_pts = self._compute_circle_points(
            shoulder_inboard, tire_radius, e2, e3
        )
        shoulder_outboard_pts = self._compute_circle_points(
            shoulder_outboard, tire_radius, e2, e3
        )

        # Update rim edge circles.
        artists["rims"][0].set_data(rim_inboard_pts[:, 0], rim_inboard_pts[:, 1])
        artists["rims"][0].set_3d_properties(rim_inboard_pts[:, 2])
        artists["rims"][1].set_data(rim_outboard_pts[:, 0], rim_outboard_pts[:, 1])
        artists["rims"][1].set_3d_properties(rim_outboard_pts[:, 2])

        # Update crown circles.
        artists["rims"][2].set_data(crown_center_pts[:, 0], crown_center_pts[:, 1])
        artists["rims"][2].set_3d_properties(crown_center_pts[:, 2])
        artists["rims"][3].set_data(
            shoulder_inboard_pts[:, 0], shoulder_inboard_pts[:, 1]
        )
        artists["rims"][3].set_3d_properties(shoulder_inboard_pts[:, 2])
        artists["rims"][4].set_data(
            shoulder_outboard_pts[:, 0], shoulder_outboard_pts[:, 1]
        )
        artists["rims"][4].set_3d_properties(shoulder_outboard_pts[:, 2])

        # Update hoop lines.
        num_hoops = self.wheel_config.num_hoops
        hoop_angles = np.linspace(0, 2 * np.pi, num_hoops, endpoint=False)

        for i, theta in enumerate(hoop_angles):
            hoop_pts = self._compute_hoop_points(
                wheel_inboard,
                wheel_outboard,
                wheel_center,
                e1,
                e2,
                e3,
                theta,
                tire_radius,
                rim_radius,
            )
            artists["bands"][i].set_data(hoop_pts[:, 0], hoop_pts[:, 1])
            artists["bands"][i].set_3d_properties(hoop_pts[:, 2])

    @staticmethod
    def get_band_indices(num_points: int, num_bands: int) -> np.ndarray:
        """
        Return indices for equally spaced cross-tire bands (not spokes).

        Ensures bands are radially spaced regardless of num_points.
        """
        step = max(1, num_points // num_bands)
        indices = np.arange(0, num_points, step)
        # Ensure exactly num_bands bands (may need to trim or pad)
        if len(indices) > num_bands:
            indices = indices[:num_bands]
        elif len(indices) < num_bands:
            # Pad by repeating last index if needed.
            indices = np.pad(indices, (0, num_bands - len(indices)), "edge")
        return indices
