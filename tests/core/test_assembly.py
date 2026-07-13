"""Tests for suspension point and element assembly validation."""

import pytest

from kinematics.core.assembly import PointCatalog, SuspensionAssembly
from kinematics.core.elements import (
    RigidLinkElement,
    RigidLinkType,
)
from kinematics.core.points.derived.manager import DerivedPointsSpec
from kinematics.core.primitives.enums import PointID
from kinematics.core.primitives.geometry import Point3
from kinematics.core.state import SuspensionState

FIXED_POINT = PointID.UPPER_WISHBONE_INBOARD_FRONT
FREE_POINT = PointID.UPPER_WISHBONE_OUTBOARD
DERIVED_POINT = PointID.WHEEL_CENTER
UNKNOWN_POINT = PointID.CONTACT_PATCH_CENTER


def calculate_derived(positions):
    """Return a deterministic derived point for catalog construction."""
    return positions[FREE_POINT]


def make_state(*, derived_is_free: bool = False) -> SuspensionState:
    """Create one fixed, one free, and one derived point."""
    free_points = {FREE_POINT}
    if derived_is_free:
        free_points.add(DERIVED_POINT)
    return SuspensionState(
        positions={
            FIXED_POINT: Point3([0.0, 0.0, 0.0]),
            FREE_POINT: Point3([1.0, 0.0, 0.0]),
            DERIVED_POINT: Point3([1.0, 0.0, 0.0]),
        },
        free_points=free_points,
    )


def make_derived_spec(*, dependency=FREE_POINT) -> DerivedPointsSpec:
    """Create a single derived-point declaration."""
    return DerivedPointsSpec(
        functions={DERIVED_POINT: calculate_derived},
        dependencies={DERIVED_POINT: {dependency}},
    )


def test_point_catalog_classifies_identifiers_without_copying_positions() -> None:
    state = make_state()
    catalog = PointCatalog.from_state(state, make_derived_spec())

    assert catalog.fixed == frozenset({FIXED_POINT})
    assert catalog.free == frozenset({FREE_POINT})
    assert catalog.derived == frozenset({DERIVED_POINT})
    assert catalog.all == frozenset(state.positions)
    assert not hasattr(catalog, "positions")


def test_assembly_accepts_shared_element_points() -> None:
    state = make_state()
    link_a = RigidLinkElement(
        label="Link A",
        type=RigidLinkType.WISHBONE_LEG,
        point_a=FIXED_POINT,
        point_b=FREE_POINT,
    )
    link_b = RigidLinkElement(
        label="Link B",
        type=RigidLinkType.WISHBONE_LEG,
        point_a=FIXED_POINT,
        point_b=FREE_POINT,
    )

    assembly = SuspensionAssembly.from_state(
        state,
        make_derived_spec(),
        (link_a, link_b),
        (DERIVED_POINT,),
    )

    assert assembly.elements == (link_a, link_b)
    assert assembly.point_keys == (DERIVED_POINT, FIXED_POINT, FREE_POINT)


def test_point_catalog_rejects_derived_point_marked_free() -> None:
    with pytest.raises(ValueError, match="Free points must be non-derived"):
        PointCatalog.from_state(make_state(derived_is_free=True), make_derived_spec())


def test_point_catalog_rejects_unknown_derived_dependency() -> None:
    with pytest.raises(ValueError, match="dependencies are absent"):
        PointCatalog.from_state(
            make_state(),
            make_derived_spec(dependency=UNKNOWN_POINT),
        )


def test_assembly_rejects_unknown_element_point() -> None:
    invalid_link = RigidLinkElement(
        label="Invalid Link",
        type=RigidLinkType.WISHBONE_LEG,
        point_a=FIXED_POINT,
        point_b=UNKNOWN_POINT,
    )

    with pytest.raises(ValueError, match="elements reference unknown points"):
        SuspensionAssembly.from_state(
            make_state(),
            make_derived_spec(),
            (invalid_link,),
            (DERIVED_POINT,),
        )


def test_assembly_rejects_unknown_output_point() -> None:
    with pytest.raises(ValueError, match="output references unknown points"):
        SuspensionAssembly.from_state(
            make_state(),
            make_derived_spec(),
            (),
            (UNKNOWN_POINT,),
        )
