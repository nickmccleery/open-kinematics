from functools import partial

import numpy as np
from numpy.typing import NDArray

from kinematics.constraints.types import PointOnLine, PointPointDistance, VectorAngle
from kinematics.constraints.utils import make_point_point_distance, make_vector_angle
from kinematics.geometry.config.wheel import WheelConfig
from kinematics.geometry.constants import CoordinateAxis, Direction
from kinematics.geometry.points.ids import PointID
from kinematics.geometry.types.double_wishbone import DoubleWishboneGeometry
from kinematics.solvers.core import MotionTarget, solve_sweep
from kinematics.types.state import Positions


def calculate_wheel_center(positions: Positions, wheel_offset: float) -> NDArray:
    p1 = positions[PointID.AXLE_OUTBOARD]
    p2 = positions[PointID.AXLE_INBOARD]
    v = p2 - p1
    v = v / np.linalg.norm(v)
    return p1 + v * wheel_offset


def calculate_wheel_inboard(positions: Positions, wheel_width: float) -> NDArray:
    center = positions[PointID.WHEEL_CENTER]
    axle = positions[PointID.AXLE_INBOARD]
    v = center - axle
    v = v / np.linalg.norm(v)
    return center - v * (wheel_width / 2)


def calculate_wheel_outboard(positions: Positions, wheel_width: float) -> NDArray:
    center = positions[PointID.WHEEL_CENTER]
    axle = positions[PointID.AXLE_INBOARD]
    v = center - axle
    v = v / np.linalg.norm(v)
    return center + v * (wheel_width / 2)


def update_wheel_positions(positions: Positions, config: WheelConfig) -> Positions:
    result = positions.copy()
    result[PointID.WHEEL_CENTER] = calculate_wheel_center(result, config.offset)
    result[PointID.WHEEL_INBOARD] = calculate_wheel_inboard(result, config.width)
    result[PointID.WHEEL_OUTBOARD] = calculate_wheel_outboard(result, config.width)
    return result


def create_length_constraints(positions: Positions) -> list[PointPointDistance]:
    constraints = []

    def add_distance(p1: PointID, p2: PointID):
        constraints.append(make_point_point_distance(positions, p1, p2))

    add_distance(PointID.UPPER_WISHBONE_INBOARD_FRONT, PointID.UPPER_WISHBONE_OUTBOARD)
    add_distance(PointID.UPPER_WISHBONE_INBOARD_REAR, PointID.UPPER_WISHBONE_OUTBOARD)
    add_distance(PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD)
    add_distance(PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD)
    add_distance(PointID.UPPER_WISHBONE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD)
    add_distance(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD)
    add_distance(PointID.AXLE_INBOARD, PointID.UPPER_WISHBONE_OUTBOARD)
    add_distance(PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD)
    add_distance(PointID.AXLE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD)
    add_distance(PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD)
    add_distance(PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD)
    add_distance(PointID.UPPER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD)
    add_distance(PointID.LOWER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD)
    add_distance(PointID.AXLE_INBOARD, PointID.TRACKROD_OUTBOARD)
    add_distance(PointID.AXLE_OUTBOARD, PointID.TRACKROD_OUTBOARD)

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
            line_point=PointID.TRACKROD_INBOARD,
            line_direction=Direction.y,
        )
    ]


def get_free_points(geometry: DoubleWishboneGeometry) -> set[PointID]:
    return {
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
        PointID.TRACKROD_OUTBOARD,
        PointID.TRACKROD_INBOARD,
    }


def create_initial_positions(geometry: DoubleWishboneGeometry) -> Positions:
    positions = {}

    positions[PointID.UPPER_WISHBONE_INBOARD_FRONT] = (
        geometry.hard_points.upper_wishbone.inboard_front.as_array()
    )
    positions[PointID.UPPER_WISHBONE_INBOARD_REAR] = (
        geometry.hard_points.upper_wishbone.inboard_rear.as_array()
    )
    positions[PointID.UPPER_WISHBONE_OUTBOARD] = (
        geometry.hard_points.upper_wishbone.outboard.as_array()
    )

    positions[PointID.LOWER_WISHBONE_INBOARD_FRONT] = (
        geometry.hard_points.lower_wishbone.inboard_front.as_array()
    )
    positions[PointID.LOWER_WISHBONE_INBOARD_REAR] = (
        geometry.hard_points.lower_wishbone.inboard_rear.as_array()
    )
    positions[PointID.LOWER_WISHBONE_OUTBOARD] = (
        geometry.hard_points.lower_wishbone.outboard.as_array()
    )

    positions[PointID.AXLE_INBOARD] = geometry.hard_points.wheel_axle.inner.as_array()
    positions[PointID.AXLE_OUTBOARD] = geometry.hard_points.wheel_axle.outer.as_array()

    positions[PointID.TRACKROD_INBOARD] = (
        geometry.hard_points.track_rod.inner.as_array()
    )
    positions[PointID.TRACKROD_OUTBOARD] = (
        geometry.hard_points.track_rod.outer.as_array()
    )

    return positions


def create_target(positions: Positions) -> MotionTarget:
    return MotionTarget(
        point_id=PointID.WHEEL_CENTER,
        axis=CoordinateAxis.Z,
        reference_position=positions[PointID.WHEEL_CENTER],
    )


def solve_suspension(
    geometry: DoubleWishboneGeometry, displacements: list[float]
) -> list[Positions]:
    initial_positions = create_initial_positions(geometry)
    wheel_config = WheelConfig(
        width=geometry.configuration.wheel.width,
        offset=geometry.configuration.wheel.offset,
        diameter=geometry.configuration.wheel.diameter,
    )
    derived_updater = partial(update_wheel_positions, config=wheel_config)

    positions = update_wheel_positions(initial_positions, wheel_config)
    free_points = get_free_points(geometry)

    constraints = []
    constraints.extend(create_length_constraints(positions))
    constraints.extend(create_angle_constraints(positions))
    constraints.extend(create_linear_constraints(positions))

    target = create_target(positions)

    return solve_sweep(
        positions=positions,
        free_points=free_points,
        constraints=constraints,
        target=target,
        displacements=displacements,
        derived_updater=derived_updater,
    )
