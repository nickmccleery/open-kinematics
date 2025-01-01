from kinematics.geometry.loader import GeometryType
from kinematics.geometry.schemas import Point3D, PointID


def get_all_points(obj) -> list[Point3D]:
    points = []
    if isinstance(obj, Point3D):
        points.append(obj)
    elif hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            points.extend(get_all_points(value))
    return points


def build_point_lookup(geometry: GeometryType) -> dict[PointID, Point3D]:
    points = {}
    for component in geometry.hard_points.__dict__.values():
        for point in component.__dict__.values():
            if isinstance(point, Point3D):
                points[point.id] = point
    return points
