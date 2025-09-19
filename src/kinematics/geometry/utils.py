from kinematics.geometry.base import Point3D


def get_all_points(obj) -> list[Point3D]:
    points = []
    if isinstance(obj, Point3D):
        points.append(obj)
    elif hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            points.extend(get_all_points(value))
    return points
