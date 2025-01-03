from typing import Type, Union

from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.geometry.types.macpherson import MacPhersonGeometry

GeometryType = Union[DoubleWishboneGeometry, MacPhersonGeometry]

GEOMETRY_TYPES: dict[str, Type[GeometryType]] = {
    "DOUBLE_WISHBONE": DoubleWishboneGeometry,
    "MACPHERSON_STRUT": MacPhersonGeometry,
}
