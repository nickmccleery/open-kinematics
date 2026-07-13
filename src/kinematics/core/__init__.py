"""
Public, transport-independent suspension solver workflows.

Value types live in their defining modules. This package facade exposes only
the high-level operations used to build, solve, evaluate, and analyze suspensions.
"""

from kinematics.core.analysis import (
    analyze_evaluated_sweep,
    analyze_solved_sweep,
    analyze_sweep,
    initial_pose,
)
from kinematics.core.schema.geometry import parse_geometry_spec
from kinematics.core.schema.sweep import build_sweep_config
from kinematics.core.suspensions.build import build_suspension
from kinematics.core.suspensions.registry import list_supported_types
from kinematics.core.sweep import solve_evaluated_sweep, solve_sweep
