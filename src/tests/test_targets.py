import numpy as np
import pytest

from kinematics.targets import AxisFrame, resolve_target
from kinematics.types import Axis, AxisTarget, VectorTarget


def test_resolve_axis_targets_returns_unit_axes():
    frame = AxisFrame(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        resolve_target(AxisTarget(Axis.X), frame), [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        resolve_target(AxisTarget(Axis.Y), frame), [0.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(
        resolve_target(AxisTarget(Axis.Z), frame), [0.0, 0.0, 1.0]
    )


def test_resolve_vector_target_normalizes():
    frame = AxisFrame(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    direction = resolve_target(VectorTarget(np.array([10.0, 0.0, 0.0])), frame)

    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_resolve_vector_target_zero_raises():
    frame = AxisFrame(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    with pytest.raises(ValueError):
        resolve_target(VectorTarget(np.array([0.0, 0.0, 0.0])), frame)
