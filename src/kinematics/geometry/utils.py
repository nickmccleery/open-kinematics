from typing import Dict

from kinematics.geometry.loader import GeometryType
from kinematics.geometry.schemas import Point3D, PointID


def build_point_lookup(geometry: GeometryType) -> Dict[PointID, Point3D]:
    points = {}
    for component in geometry.hard_points.__dict__.values():
        for point in component.__dict__.values():
            if isinstance(point, Point3D):
                points[point.id] = point
    return points
