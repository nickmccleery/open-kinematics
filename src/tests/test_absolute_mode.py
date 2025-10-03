import numpy as np

from kinematics.core import PointID, SuspensionState
from kinematics.solver import resolve_targets_to_absolute, solve_sweep
from kinematics.types import (
    Axis,
    PointTarget,
    PointTargetAxis,
    SweepConfig,
    TargetPositionMode,
)


def test_resolve_targets_to_absolute():
    initial_positions = {
        PointID.WHEEL_CENTER: np.array([0.0, 0.0, 150.0]),
    }
    initial_state = SuspensionState(positions=initial_positions, free_points=set())

    # Relative conversion -> absolute
    relative_target = PointTarget(
        PointID.WHEEL_CENTER,
        PointTargetAxis(Axis.Z),
        50.0,
        TargetPositionMode.RELATIVE,
    )

    resolved = resolve_targets_to_absolute([relative_target], initial_state)

    assert resolved[0].mode == TargetPositionMode.ABSOLUTE
    assert resolved[0].value == 200.0  # 150 + 50

    # Absolute passthrough
    absolute_target = PointTarget(
        PointID.WHEEL_CENTER,
        PointTargetAxis(Axis.Z),
        400.0,
        TargetPositionMode.ABSOLUTE,
    )

    resolved2 = resolve_targets_to_absolute([absolute_target], initial_state)

    assert resolved2[0].mode == TargetPositionMode.ABSOLUTE
    assert resolved2[0].value == 400.0


def test_default_relative_mode():
    target = PointTarget(PointID.WHEEL_CENTER, PointTargetAxis(Axis.Z), 50)
    assert target.mode == TargetPositionMode.RELATIVE


def test_absolute_mode_solve():
    # simple identity derived function
    def identity(positions):
        return positions

    positions = {PointID.LOWER_WISHBONE_OUTBOARD: np.array([0.0, 0.0, 0.0])}
    free = {PointID.LOWER_WISHBONE_OUTBOARD}
    initial_state = SuspensionState(positions=positions, free_points=free)

    # Fully determined with 3 absolute targets (X, Y, Z) applied simultaneously.
    x_sweep = [
        PointTarget(
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointTargetAxis(Axis.X),
            10.0,
            TargetPositionMode.ABSOLUTE,
        )
    ]
    y_sweep = [
        PointTarget(
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointTargetAxis(Axis.Y),
            -5.0,
            TargetPositionMode.ABSOLUTE,
        )
    ]
    z_sweep = [
        PointTarget(
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointTargetAxis(Axis.Z),
            100.0,
            TargetPositionMode.ABSOLUTE,
        )
    ]

    states = solve_sweep(
        initial_state=initial_state,
        constraints=[],
        sweep_config=SweepConfig([x_sweep, y_sweep, z_sweep]),
        compute_derived_points_func=identity,
    )

    assert len(states) == 1
    assert np.allclose(
        states[0].positions[PointID.LOWER_WISHBONE_OUTBOARD],
        np.array([10.0, -5.0, 100.0]),
    )
