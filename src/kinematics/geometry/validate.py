"""
Domain validation logic for suspension geometries.

Contains validation functions for geometry constraints and business rules.
"""

from typing import Union

from marshmallow.exceptions import ValidationError

from kinematics.points.ids import PointID
from kinematics.points.utils import get_all_points
from kinematics.suspensions.double_wishbone.model import DoubleWishboneModel
from kinematics.suspensions.macpherson.model import MacPhersonModel

GeometryType = Union[DoubleWishboneModel, MacPhersonModel]


def validate_geometry(geometry: GeometryType) -> None:
    """
    Validates a suspension geometry for domain-specific constraints.

    Args:
        geometry: The geometry instance to validate

    Raises:
        ValidationError: If validation fails
    """
    points = get_all_points(geometry.hard_points)
    for point in points:
        if point.id is PointID.NOT_ASSIGNED:
            raise ValidationError("Found unrecognized point ID in geometry.")
