from typing import Dict, Type, Union

from kinematics.suspensions.base import SuspensionProvider
from kinematics.suspensions.double_wishbone.geometry import DoubleWishboneGeometry
from kinematics.suspensions.double_wishbone.provider import DoubleWishboneProvider
from kinematics.suspensions.macpherson.geometry import MacPhersonGeometry

# Maps a specific geometry class to the provider class
PROVIDER_REGISTRY: Dict[Type, Type[SuspensionProvider]] = {
    DoubleWishboneGeometry: DoubleWishboneProvider,
}

# Maps the key from the YAML file to the geometry class
GeometryType = Union[DoubleWishboneGeometry, MacPhersonGeometry]
GEOMETRY_TYPES: dict[str, Type[GeometryType]] = {
    "DOUBLE_WISHBONE": DoubleWishboneGeometry,
    "MACPHERSON_STRUT": MacPhersonGeometry,
}
