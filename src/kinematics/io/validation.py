"""Shared Pydantic validation utilities for YAML loading."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, TypeVar

import numpy as np
from pydantic import BeforeValidator

from kinematics.core.enums import Axis, PointID, TargetPositionMode, Units
from kinematics.core.types import Vec3

E = TypeVar("E", bound=Enum)

# Input types that can be coerced to Vec3.
Vec3Like = Vec3 | dict[str, float] | list[float] | tuple[float, float, float]


def coerce_enum(enum_cls: type[E], value: str | int | E) -> E:
    """Case-insensitive enum coercion from name, value, or instance."""
    if isinstance(value, enum_cls):
        return value

    # Try by name (case-insensitive).
    if isinstance(value, str):
        for name, member in enum_cls.__members__.items():
            if name.lower() == value.lower():
                return member
        # Try by value for string-valued enums.
        for member in enum_cls:
            if isinstance(member.value, str) and member.value.lower() == value.lower():
                return member

    # Try by value directly (for IntEnum).
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        pass

    valid = ", ".join(enum_cls.__members__.keys())
    raise ValueError(f"Invalid {enum_cls.__name__}: {value!r}. Valid: {valid}")


# Case-insensitive enum type aliases.
CIAxis = Annotated[Axis, BeforeValidator(lambda v: coerce_enum(Axis, v))]
CIPointID = Annotated[PointID, BeforeValidator(lambda v: coerce_enum(PointID, v))]
CIUnits = Annotated[Units, BeforeValidator(lambda v: coerce_enum(Units, v))]
CITargetPositionMode = Annotated[
    TargetPositionMode, BeforeValidator(lambda v: coerce_enum(TargetPositionMode, v))
]


def coerce_vec3(value: Any) -> Vec3:
    """
    Coerce various input formats to a Vec3 (3D numpy array).

    Accepts:
        - [x, y, z] list/tuple
        - {x: ..., y: ..., z: ...} dict
        - numpy array
    """
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float64)
    elif isinstance(value, dict):
        arr = np.array([value["x"], value["y"], value["z"]], dtype=np.float64)
    else:
        arr = np.array(value, dtype=np.float64)

    if arr.shape != (3,):
        raise ValueError(f"Vec3 must have 3 components, got shape {arr.shape}")
    return arr


# Pydantic field type that accepts Vec3Like inputs and coerces to Vec3.
# Note: Type checkers see Vec3Like, but runtime value is always Vec3 (numpy array).
PydanticVec3 = Annotated[Vec3Like, BeforeValidator(coerce_vec3)]
