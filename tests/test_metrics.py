import numpy as np

from kinematics.core.enums import PointID
from kinematics.io.geometry_loader import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import solve_sweep
from kinematics.metrics.catalog import get_default_corner_metrics
from kinematics.metrics.main import compute_metrics_for_state_from_suspension
from kinematics.suspensions.double_wishbone import DoubleWishboneSuspension


def _shift_x(vec3: object, delta_x: float) -> tuple[float, float, float]:
    """
    Shift a 3D point along the world X axis by a fixed amount.
    """
    shifted = np.asarray(vec3, dtype=np.float64).copy()
    shifted[0] += delta_x
    return (float(shifted[0]), float(shifted[1]), float(shifted[2]))


def _translate_double_wishbone_x(
    suspension: DoubleWishboneSuspension, delta_x: float
) -> DoubleWishboneSuspension:
    """
    Build a rigidly translated copy of a double wishbone suspension.

    Hardpoints and any configuration points that live in world coordinates
    are shifted together so the translated suspension is geometrically
    identical to the original one.
    """
    hardpoints = {
        point_id: position.copy() + np.array([delta_x, 0.0, 0.0], dtype=np.float64)
        for point_id, position in suspension.hardpoints.items()
    }

    config = suspension.config
    translated_config = None
    if config is not None:
        config_updates: dict[str, object] = {
            "cg_position": _shift_x(config.cg_position, delta_x)
        }

        if config.camber_shim is not None:
            translated_shim = config.camber_shim.model_copy(
                update={
                    "shim_face_point_a": _shift_x(
                        config.camber_shim.shim_face_point_a, delta_x
                    ),
                    "shim_face_point_b": _shift_x(
                        config.camber_shim.shim_face_point_b, delta_x
                    ),
                }
            )
            config_updates["camber_shim"] = translated_shim

        translated_config = config.model_copy(update=config_updates)

    return DoubleWishboneSuspension(
        name=suspension.name,
        version=suspension.version,
        units=suspension.units,
        hardpoints=hardpoints,
        config=translated_config,
    )


def test_front_view_metrics_are_invariant_to_rigid_x_translation(
    double_wishbone_geometry_file,
    test_data_dir,
) -> None:
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)

    sweep_config = parse_sweep_file(test_data_dir / "sweep.yaml")
    states, _ = solve_sweep(suspension, sweep_config)

    translated = _translate_double_wishbone_x(suspension, 100.0)
    translated_states, _ = solve_sweep(translated, sweep_config)

    original_metrics = [
        compute_metrics_for_state_from_suspension(state, suspension) for state in states
    ]
    translated_metrics = [
        compute_metrics_for_state_from_suspension(state, translated)
        for state in translated_states
    ]

    comparison_index = next(
        index
        for index, metrics in enumerate(original_metrics)
        if metrics["fvic_y_mm"] is not None
    )

    for column_name in ("fvic_y_mm", "fvic_z_mm", "fvsa_length_mm"):
        original_value = original_metrics[comparison_index][column_name]
        translated_value = translated_metrics[comparison_index][column_name]
        assert original_value is not None, f"{column_name} is None in original"
        assert translated_value is not None, f"{column_name} is None in translated"
        np.testing.assert_allclose(
            original_value,
            translated_value,
            atol=5e-4,
            rtol=0.0,
            err_msg=f"{column_name} changed under rigid X translation",
        )


def test_parallel_wishbone_planes_produce_null_ic_metrics(
    double_wishbone_geometry_file,
) -> None:
    suspension = load_geometry(double_wishbone_geometry_file)
    assert isinstance(suspension, DoubleWishboneSuspension)

    state = suspension.initial_state().copy()
    plane_offset = np.array([0.0, 0.0, 300.0], dtype=np.float64)

    # Make the upper wishbone plane a translated copy of the lower
    # wishbone plane so the planes are parallel and have no unique
    # instant-axis intersection.
    state[PointID.UPPER_WISHBONE_INBOARD_FRONT] = (
        state[PointID.LOWER_WISHBONE_INBOARD_FRONT] + plane_offset
    )
    state[PointID.UPPER_WISHBONE_INBOARD_REAR] = (
        state[PointID.LOWER_WISHBONE_INBOARD_REAR] + plane_offset
    )
    state[PointID.UPPER_WISHBONE_OUTBOARD] = (
        state[PointID.LOWER_WISHBONE_OUTBOARD] + plane_offset
    )

    assert suspension.compute_instant_axis(state) is None
    assert suspension.compute_side_view_instant_center(state) is None
    assert suspension.compute_front_view_instant_center(state) is None

    metrics = compute_metrics_for_state_from_suspension(state, suspension)

    assert metrics["svic_x_mm"] is None
    assert metrics["svic_z_mm"] is None
    assert metrics["svsa_length_mm"] is None
    assert metrics["fvic_y_mm"] is None
    assert metrics["fvic_z_mm"] is None
    assert metrics["fvsa_length_mm"] is None


def test_default_corner_metric_catalog_includes_anti_metrics() -> None:
    column_names = [metric.column_name for metric in get_default_corner_metrics()]

    assert "anti_dive_pct" in column_names
    assert "anti_squat_pct" in column_names
