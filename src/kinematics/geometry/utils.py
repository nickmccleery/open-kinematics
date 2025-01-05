from kinematics.geometry.points.base import Point3D
from kinematics.geometry.points.ids import PointID


def get_all_points(obj) -> list[Point3D]:
    points = []
    if isinstance(obj, Point3D):
        points.append(obj)
    elif hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            points.extend(get_all_points(value))
    return points


def build_point_map(points: list[Point3D]) -> dict[PointID, Point3D]:
    return {point.id: point for point in points}


def compute_midpoint(p1: Point3D, p2: Point3D) -> Point3D:
    return Point3D(
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2,
        (p1.z + p2.z) / 2,
    )
