"""
Structured input schema.

Validated, transport-agnostic Pydantic models describing geometry and sweeps,
plus the coercion helpers that turn loose structured data (parsed YAML, JSON
request bodies) into core types. This package performs no file IO.
"""

from kinematics.schema.coercion import (
    CIAxis,
    CIPointID,
    CISide,
    CITargetPositionMode,
    CIUnits,
    PydanticDirection3,
    PydanticPoint3,
    coerce_direction3,
    coerce_enum,
    coerce_point3,
)
from kinematics.schema.geometry import (
    AxleHardpointsSpec,
    DoubleWishboneAxleGeometrySpec,
    DoubleWishboneCoiloverGeometrySpec,
    DoubleWishboneGeometrySpec,
    DoubleWishbonePushrodRockerArbGeometrySpec,
    DoubleWishbonePushrodRockerAxleGeometrySpec,
    DoubleWishbonePushrodRockerGeometrySpec,
    GeometrySpec,
    GeometrySpecBase,
    RockerSpringSpec,
    parse_geometry_spec,
)
from kinematics.schema.sweep import (
    DirectionSpec,
    SweepSpec,
    TargetSpec,
    build_sweep_config,
)

__all__ = [
    "AxleHardpointsSpec",
    "CIAxis",
    "CIPointID",
    "CISide",
    "CITargetPositionMode",
    "CIUnits",
    "DirectionSpec",
    "DoubleWishboneAxleGeometrySpec",
    "DoubleWishboneCoiloverGeometrySpec",
    "DoubleWishboneGeometrySpec",
    "DoubleWishbonePushrodRockerAxleGeometrySpec",
    "DoubleWishbonePushrodRockerArbGeometrySpec",
    "DoubleWishbonePushrodRockerGeometrySpec",
    "GeometrySpec",
    "GeometrySpecBase",
    "PydanticDirection3",
    "PydanticPoint3",
    "SweepSpec",
    "TargetSpec",
    "RockerSpringSpec",
    "build_sweep_config",
    "coerce_direction3",
    "coerce_enum",
    "coerce_point3",
    "parse_geometry_spec",
]
