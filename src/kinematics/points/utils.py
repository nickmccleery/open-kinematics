from kinematics.primitives import Point3D


def get_all_points(obj) -> list[Point3D]:
    """
    Recursively extracts all Point3D objects from an object or its attributes.

    Args:
        obj: Object to search for Point3D instances

    Returns:
        List of all Point3D objects found
    """
    points = []
    if isinstance(obj, Point3D):
        points.append(obj)
    elif hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            points.extend(get_all_points(value))
    return points
