"""Tests for renderer-facing display topology normalization."""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.yaml import load_geometry
from kinematics.core.elements import (
    RackElement,
    RigidLinkElement,
    RockerElement,
    TorsionElement,
    UprightElement,
    VariableLengthLinkElement,
    WheelElement,
)
from kinematics.core.presentation import (
    AXIS_FOOT_SUFFIX,
    DisplayElementType,
    display_elements,
    display_point_keys,
    display_positions,
    rocker_display_groups,
)
from kinematics.core.primitives.enums import Axis, PointID
from kinematics.core.primitives.point_ref import PointRef, Side


def test_corner_rocker_fan_becomes_axis_and_perpendicular_arms(
    test_data_dir: Path,
) -> None:
    """Normalize the rocker fan without exposing renderer-side geometry math."""
    suspension = load_geometry(test_data_dir / "corner_strut_rocker_geometry.yaml")
    assembly = suspension.assembly()
    groups = rocker_display_groups(assembly)
    positions = display_positions(
        suspension.initial_state().positions,
        display_point_keys(assembly),
        groups,
    )
    elements = display_elements(assembly)

    assert len(groups) == 1
    assert any(isinstance(element, RigidLinkElement) for element in assembly.elements)
    assert any(isinstance(element, UprightElement) for element in assembly.elements)
    assert any(isinstance(element, WheelElement) for element in assembly.elements)
    assert any(
        isinstance(element, VariableLengthLinkElement) for element in assembly.elements
    )
    rocker = next(
        element for element in assembly.elements if isinstance(element, RockerElement)
    )
    assert rocker.rotation_axis == (
        PointID.ROCKER_AXIS_FRONT,
        PointID.ROCKER_AXIS_REAR,
    )
    assert all(isinstance(element.type, DisplayElementType) for element in elements)
    assert not any(hasattr(element, "color") for element in assembly.elements)
    assert not any(element.label == "Rocker" for element in elements)
    assert {element.label for element in elements if "Rocker" in element.label} == {
        "Rocker Axis",
        "Rocker Pushrod Arm",
    }

    group = groups[0]
    axis_start = np.asarray(positions[group.axis_start])
    axis_end = np.asarray(positions[group.axis_end])
    pickup_name = group.pickups[0]
    pickup = np.asarray(positions[pickup_name])
    foot = np.asarray(positions[f"{pickup_name}{AXIS_FOOT_SUFFIX}"])

    assert np.dot(axis_end - axis_start, pickup - foot) == pytest.approx(0.0)


def test_axle_display_includes_shared_arb_points_and_side_keyed_rockers(
    test_data_dir: Path,
) -> None:
    """Include fixed shared-link vertices and normalize both axle rockers."""
    suspension = load_geometry(test_data_dir / "axle_geometry_rocker.yaml")
    assembly = suspension.assembly()
    point_keys = display_point_keys(assembly)
    groups = rocker_display_groups(assembly)
    positions = display_positions(
        suspension.initial_state().positions,
        point_keys,
        groups,
    )

    assert PointRef(Side.CENTER, PointID.ARB_AXIS_A) in point_keys
    assert PointRef(Side.CENTER, PointID.ARB_AXIS_B) in point_keys
    assert {group.label for group in groups} == {"Left Rocker", "Right Rocker"}
    assert sum(isinstance(element, WheelElement) for element in assembly.elements) == 2
    assert sum(isinstance(element, RockerElement) for element in assembly.elements) == 2
    rack = next(
        element for element in assembly.elements if isinstance(element, RackElement)
    )
    assert rack.translation_axis is Axis.Y
    assert any(isinstance(element, TorsionElement) for element in assembly.elements)
    for group in groups:
        assert len(group.pickups) == 2
        for pickup in group.pickups:
            assert f"{pickup}{AXIS_FOOT_SUFFIX}" in positions
