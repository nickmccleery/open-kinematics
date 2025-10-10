"""
Visualization utilities for geometric computations.
"""

from typing import Optional

import numpy as np

from kinematics.types import Vec3

PLOTTING_ENABLED = True


def should_plot() -> bool:
    """
    Check if plotting is enabled via environment variable (constant for now).
    """
    return PLOTTING_ENABLED


def plot_plane_from_points(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    normal: Optional[np.ndarray] = None,
    d: Optional[float] = None,
    title: str = "Plane from Three Points",
) -> None:
    """
    Plot three points and the plane they define.

    Args:
        a, b, c: The three points defining the plane.
        normal: Optional normal vector of the plane.
        d: Optional distance parameter of the plane.
        title: Title for the plot.
    """
    if not should_plot():
        return

    try:
        import matplotlib.pyplot as plt

        # Import needed for 3D plotting
        import mpl_toolkits.mplot3d  # noqa: F401
    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the three points
    points = np.array([a, b, c])
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=["red", "green", "blue"],
        s=[100, 100, 100],
        alpha=0.8,
    )

    # Label the points
    ax.text(a[0], a[1], a[2], s="A", fontsize=12)
    ax.text(b[0], b[1], b[2], s="B", fontsize=12)
    ax.text(c[0], c[1], c[2], s="C", fontsize=12)

    # Draw lines between points to show the triangle
    triangle = np.array([a, b, c, a])  # Close the triangle
    ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], "gray", alpha=0.5)

    if normal is not None and d is not None:
        # Plot normal vector from centroid
        centroid = (a + b + c) / 3
        span = max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
        normal_scale = span * 0.5

        ax.quiver(
            centroid[0],
            centroid[1],
            centroid[2],
            normal[0] * normal_scale,
            normal[1] * normal_scale,
            normal[2] * normal_scale,
            color="purple",
            arrow_length_ratio=0.1,
            linewidth=3,
        )

        ax.text(
            centroid[0] + normal[0] * normal_scale * 1.1,
            centroid[1] + normal[1] * normal_scale * 1.1,
            centroid[2] + normal[2] * normal_scale * 1.1,
            s="Normal",
            fontsize=10,
            color="purple",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.set_title(title)

    # Make axes equal
    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore

    plt.tight_layout()
    plt.show()


def plot_plane_intersection(
    n1: np.ndarray,
    d1: float,
    n2: np.ndarray,
    d2: float,
    line_point: Optional[np.ndarray] = None,
    line_direction: Optional[np.ndarray] = None,
    title: str = "Plane Intersection",
) -> None:
    """
    Plot two planes and their line of intersection.

    Args:
        n1, d1: Normal and distance of first plane.
        n2, d2: Normal and distance of second plane.
        line_point: Optional point on intersection line.
        line_direction: Optional direction of intersection line.
        title: Title for the plot.
    """
    if not should_plot():
        return

    try:
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d  # noqa: F401
    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot intersection line if provided
    if line_point is not None and line_direction is not None:
        plot_range = 5.0
        t_vals = np.linspace(-plot_range, plot_range, 100)
        line_points = line_point + t_vals[:, np.newaxis] * line_direction
        ax.plot(
            line_points[:, 0],
            line_points[:, 1],
            line_points[:, 2],
            "green",
            linewidth=4,
            label="Intersection Line",
        )

        # Plot direction vector at line point
        ax.quiver(
            line_point[0],
            line_point[1],
            line_point[2],
            line_direction[0],
            line_direction[1],
            line_direction[2],
            color="green",
            arrow_length_ratio=0.1,
            linewidth=2,
        )

    # Plot normal vectors from origin
    origin = np.array([0.0, 0.0, 0.0])
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        n1[0] * 2,
        n1[1] * 2,
        n1[2] * 2,
        color="red",
        arrow_length_ratio=0.1,
        linewidth=2,
        alpha=0.7,
        label="Normal 1",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        n2[0] * 2,
        n2[1] * 2,
        n2[2] * 2,
        color="blue",
        arrow_length_ratio=0.1,
        linewidth=2,
        alpha=0.7,
        label="Normal 2",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plot_range = 5.0
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_zlim(-plot_range, plot_range)

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_line_plane_intersection(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    plane_y: float,
    intersection: Optional[np.ndarray] = None,
    title: str = "Line-Plane Intersection",
) -> None:
    """
    Plot a line and vertical plane (Y = constant) with their intersection.

    Args:
        line_point: Point on the line.
        line_direction: Direction vector of the line.
        plane_y: Y-coordinate defining the vertical plane.
        intersection: Optional intersection point.
        title: Title for the plot.
    """
    if not should_plot():
        return

    try:
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d  # noqa: F401
    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the line
    plot_range = 10.0
    t_vals = np.linspace(-plot_range, plot_range, 100)
    line_points = line_point + t_vals[:, np.newaxis] * line_direction
    ax.plot(
        line_points[:, 0],
        line_points[:, 1],
        line_points[:, 2],
        "blue",
        linewidth=3,
        label="Line",
    )

    # Create vertical plane representation (just edges for clarity)
    x_edges = [-plot_range, plot_range, plot_range, -plot_range, -plot_range]
    z_edges = [-plot_range, -plot_range, plot_range, plot_range, -plot_range]
    y_edges = [plane_y] * len(x_edges)
    ax.plot(
        x_edges,
        y_edges,
        z_edges,
        "yellow",
        linewidth=2,
        alpha=0.7,
        label=f"Plane Y={plane_y}",
    )

    # Plot intersection point if provided
    if intersection is not None:
        ax.scatter(
            intersection[0],
            intersection[1],
            intersection[2],
            c="red",
            s=[100],
            label="Intersection",
        )
        ax.text(
            intersection[0],
            intersection[1],
            intersection[2],
            s=f"  ({intersection[0]:.1f}, {intersection[1]:.1f}, {intersection[2]:.1f})",
            fontsize=10,
        )

    # Plot line direction vector
    ax.quiver(
        line_point[0],
        line_point[1],
        line_point[2],
        line_direction[0],
        line_direction[1],
        line_direction[2],
        color="blue",
        arrow_length_ratio=0.1,
        linewidth=2,
        alpha=0.7,
    )

    # Plot line starting point
    ax.scatter(
        line_point[0],
        line_point[1],
        line_point[2],
        c="green",
        s=[80],
        label="Line Point",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set reasonable axis limits
    all_points = line_points
    if intersection is not None:
        all_points = np.vstack([all_points, intersection])

    x_range = [all_points[:, 0].min() - 2, all_points[:, 0].max() + 2]
    y_range = [
        min(all_points[:, 1].min() - 2, plane_y - 2),
        max(all_points[:, 1].max() + 2, plane_y + 2),
    ]
    z_range = [all_points[:, 2].min() - 2, all_points[:, 2].max() + 2]

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)

    ax.legend()
    plt.tight_layout()
    plt.show()
