"""
Domain validation logic for suspension geometries.

Contains validation functions for geometry constraints and business rules.
"""

from typing import Union

from kinematics.suspensions.double_wishbone.model import DoubleWishboneGeometry
from kinematics.suspensions.macpherson.model import MacPhersonGeometry

GeometryType = Union[DoubleWishboneGeometry, MacPhersonGeometry]


def validate_geometry(geometry: GeometryType) -> None:
    """
    Validates a suspension geometry for domain-specific constraints.

    Args:
        geometry: The geometry instance to validate

    Raises:
        ValueError: If validation fails
    """
    # With the new simplified structure, basic validation is handled
    # by the dataclass structure itself. Additional validation can be
    # added here as needed for specific business rules.
    if not geometry.validate():
        raise ValueError("Geometry validation failed.")
