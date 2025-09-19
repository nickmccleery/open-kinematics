from functools import partial
from typing import Dict

from kinematics.constraints.types import PointOnLine, PointPointDistance, VectorAngle
from kinematics.constraints.utils import make_point_point_distance, make_vector_angle
from kinematics.geometry.constants import Direction
from kinematics.geometry.points.ids import PointID
from kinematics.points.derived import (
    get_axle_midpoint,
    get_wheel_center,
    get_wheel_inboard,
    get_wheel_outboard,
)
from kinematics.solver.derived_points import DerivedPointDefinition, DerivedPointManager
from kinematics.solvers.core import Constraint, PointTargetSet, solve_sweep
from kinematics.suspensions.double_wishbone.geometry import DoubleWishboneGeometry
from kinematics.types.state import Positions

FREE_POINTS: set = {
    PointID.UPPER_WISHBONE_OUTBOARD,
    PointID.LOWER_WISHBONE_OUTBOARD,
    PointID.AXLE_INBOARD,
    PointID.AXLE_OUTBOARD,
    PointID.TRACKROD_OUTBOARD,
    PointID.TRACKROD_INBOARD,
}


def get_dw_derived_point_definitions(
    geometry: DoubleWishboneGeometry,
) -> Dict[PointID, DerivedPointDefinition]:
    """
    Creates the specific set of derived point definitions for a double wishbone suspension.
    """
    wheel_cfg = geometry.configuration.wheel

    definitions: Dict[PointID, DerivedPointDefinition] = {
        PointID.AXLE_MIDPOINT: (
            get_axle_midpoint,
            {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        ),
        PointID.WHEEL_CENTER: (
            partial(get_wheel_center, wheel_offset=wheel_cfg.offset),
            {PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD},
        ),
        PointID.WHEEL_INBOARD: (
            partial(get_wheel_inboard, wheel_width=wheel_cfg.width),
            {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        ),
        PointID.WHEEL_OUTBOARD: (
            partial(get_wheel_outboard, wheel_width=wheel_cfg.width),
            {PointID.WHEEL_CENTER, PointID.AXLE_INBOARD},
        ),
    }
    return definitions


def create_length_constraints(positions: Positions) -> list[PointPointDistance]:
    constraints = []

    def constrain(p1: PointID, p2: PointID):
        constraints.append(make_point_point_distance(positions, p1, p2))

    constrain(PointID.UPPER_WISHBONE_INBOARD_FRONT, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.UPPER_WISHBONE_INBOARD_REAR, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD)
    constrain(PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD)
    constrain(PointID.UPPER_WISHBONE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD)
    constrain(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD)
    constrain(PointID.AXLE_INBOARD, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD)
    constrain(PointID.AXLE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD)
    constrain(PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD)
    constrain(PointID.UPPER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD)
    constrain(PointID.LOWER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD)
    constrain(PointID.AXLE_INBOARD, PointID.TRACKROD_OUTBOARD)
    constrain(PointID.AXLE_OUTBOARD, PointID.TRACKROD_OUTBOARD)
    return constraints


def create_angle_constraints(positions: Positions) -> list[VectorAngle]:
    return [
        make_vector_angle(
            positions,
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        )
    ]


def create_linear_constraints(positions: Positions) -> list[PointOnLine]:
    return [
        PointOnLine(
            point_id=PointID.TRACKROD_INBOARD,
            line_point=positions[PointID.TRACKROD_INBOARD],
            line_direction=Direction.y,
        )
    ]


def create_constraints(positions: Positions) -> list[Constraint]:
    constraints: list[Constraint] = []
    constraints.extend(create_length_constraints(positions))
    constraints.extend(create_angle_constraints(positions))
    constraints.extend(create_linear_constraints(positions))
    return constraints


def create_initial_positions(geometry: DoubleWishboneGeometry) -> Positions:
    from kinematics.geometry.utils import get_all_points

    points = get_all_points(geometry.hard_points)
    return {p.id: p.as_array() for p in points}


def solve_suspension(
    geometry: DoubleWishboneGeometry, point_targets: list[PointTargetSet]
) -> list[Positions]:
    derived_point_definitions = get_dw_derived_point_definitions(geometry)
    derived_point_manager = DerivedPointManager(derived_point_definitions)

    initial_positions = create_initial_positions(geometry)
    initial_positions_with_derived = derived_point_manager.update(initial_positions)

    constraints = create_constraints(positions=initial_positions_with_derived)

    return solve_sweep(
        initial_positions=initial_positions_with_derived,
        constraints=constraints,
        free_points=FREE_POINTS,
        targets=point_targets,
        compute_derived_points=derived_point_manager.update,
    )
