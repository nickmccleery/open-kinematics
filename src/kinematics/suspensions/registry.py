from typing import Type, Union

from kinematics.suspensions.double_wishbone.geometry import DoubleWishboneGeometry
from kinematics.suspensions.macpherson.geometry import MacPhersonGeometry

GeometryType = Union[DoubleWishboneGeometry, MacPhersonGeometry]

GEOMETRY_TYPES: dict[str, Type[GeometryType]] = {
    "DOUBLE_WISHBONE": DoubleWishboneGeometry,
    "MACPHERSON_STRUT": MacPhersonGeometry,
}
