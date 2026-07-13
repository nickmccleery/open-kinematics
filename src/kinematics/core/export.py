"""Pure helpers for flattening structured analysis at transport boundaries."""

from collections.abc import Mapping, Sequence

from kinematics.core.metrics.main import AxleMetricRows, MetricRow
from kinematics.core.metrics.main import flatten_metric_rows as flatten_metric_rows
from kinematics.core.metrics.registry import (
    flat_specs_for_suspension as flat_specs_for_suspension,
)
from kinematics.core.primitives.geometry import extract_array
from kinematics.core.primitives.point_ref import PointKey
from kinematics.core.primitives.point_ref import point_key_name as point_key_name


def flatten_positions(
    positions: Mapping[PointKey, object],
    output_points: Sequence[PointKey],
) -> dict[str, tuple[float, float, float]]:
    """Flatten selected typed positions to public point names and tuples."""
    flattened: dict[str, tuple[float, float, float]] = {}
    for point in output_points:
        position = positions.get(point)
        if position is None:
            continue
        raw = extract_array(position)
        flattened[point_key_name(point)] = (
            float(raw[0]),
            float(raw[1]),
            float(raw[2]),
        )
    return flattened


def flatten_metric_result(
    row: MetricRow | AxleMetricRows,
) -> dict[str, float | None]:
    """Flatten either corner or structural axle metric rows for export."""
    if isinstance(row, AxleMetricRows):
        return flatten_metric_rows(row.axle, row.corners)
    return dict(row)
