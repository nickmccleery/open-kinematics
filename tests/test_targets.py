import numpy as np
import pytest

from kinematics.enums import Axis
from kinematics.targets import resolve_target
from kinematics.types import PointTargetAxis, PointTargetVector, WorldAxisSystem


def test_resolve_axis_targets_returns_unit_axes():
    np.testing.assert_allclose(
        resolve_target(PointTargetAxis(Axis.X)), WorldAxisSystem.X
    )
    np.testing.assert_allclose(
        resolve_target(PointTargetAxis(Axis.Y)), WorldAxisSystem.Y
    )
    np.testing.assert_allclose(
        resolve_target(PointTargetAxis(Axis.Z)), WorldAxisSystem.Z
    )


def test_resolve_vector_target_normalizes():
    direction = resolve_target(PointTargetVector(np.array([10.0, 0.0, 0.0])))

    np.testing.assert_allclose(direction, WorldAxisSystem.X)
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_resolve_vector_target_zero_raises():
    with pytest.raises(ValueError):
        resolve_target(PointTargetVector(np.array([0.0, 0.0, 0.0])))
