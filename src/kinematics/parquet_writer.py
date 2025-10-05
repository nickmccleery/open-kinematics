"""
Parquet writer for sweep results.

This module converts a list of SuspensionState objects into a columnar Arrow
Table and writes a Parquet file with stable schema and rich metadata.

Schema v1.0:
  - step: int32                  # 0..N-1 sweep index
  - point_id: int32              # enums.PointID value
  - point_name: string           # enums.PointID name
  - x: float64
  - y: float64
  - z: float64

File metadata keys (key -> value):
  - kinematics.schema_version -> "1.0"
  - kinematics.geometry_type   -> str
  - kinematics.provider_class  -> str
  - kinematics.n_steps         -> str(int)
  - kinematics.points_count    -> str(int)
  - kinematics.sweep_summary   -> JSON string (compact)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from kinematics.enums import PointID
from kinematics.state import SuspensionState
from kinematics.sweep_io import describe_sweep
from kinematics.types import SweepConfig

SCHEMA_VERSION = "1.0"


def _states_to_batches(states: Sequence[SuspensionState]) -> pa.RecordBatch:
    rows: List[int] = []
    pids: List[int] = []
    pnames: List[str] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    for step, st in enumerate(states):
        for pid, pos in st.items():
            rows.append(step)
            pids.append(int(pid))
            pnames.append(PointID(pid).name)
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
            zs.append(float(pos[2]))

    arrays = [
        pa.array(rows, type=pa.int32()),
        pa.array(pids, type=pa.int32()),
        pa.array(pnames, type=pa.string()),
        pa.array(xs, type=pa.float64()),
        pa.array(ys, type=pa.float64()),
        pa.array(zs, type=pa.float64()),
    ]
    names = ["step", "point_id", "point_name", "x", "y", "z"]
    return pa.RecordBatch.from_arrays(arrays, names=names)


def states_to_table(states: Sequence[SuspensionState]) -> pa.Table:
    batch = _states_to_batches(states)
    return pa.Table.from_batches([batch])


def write_parquet(
    states: Sequence[SuspensionState],
    out_path: Path,
    *,
    geometry_type: str,
    provider_class: str,
    sweep: SweepConfig | None = None,
) -> None:
    table = states_to_table(states)

    # Build metadata
    md = {
        "kinematics.schema_version": SCHEMA_VERSION,
        "kinematics.geometry_type": geometry_type,
        "kinematics.provider_class": provider_class,
        "kinematics.n_steps": str(len(states)),
        "kinematics.points_count": str(len(states[0].positions) if states else 0),
    }
    if sweep is not None:
        md["kinematics.sweep_summary"] = json.dumps(
            describe_sweep(sweep), separators=(",", ":")
        )

    # Attach metadata to schema
    schema_with_md = table.schema.with_metadata(
        {k.encode(): v.encode() for k, v in md.items()}
    )
    table = table.cast(schema_with_md)

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
