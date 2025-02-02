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


def compute_derived_points(positions: Positions, config: WheelConfig) -> Positions:
    result = positions.copy()
    result[PointID.WHEEL_CENTER] = calculate_wheel_center(result, config.offset)
    result[PointID.WHEEL_INBOARD] = calculate_wheel_inboard(result, config.width)
    result[PointID.WHEEL_OUTBOARD] = calculate_wheel_outboard(result, config.width)
    return result


def create_length_constraints(positions: Positions) -> list[PointPointDistance]:
    constraints = []

    def constrain(p1: PointID, p2: PointID):
        constraints.append(make_point_point_distance(positions, p1, p2))

    # Wishbone inboard to outboard constraints.
    constrain(PointID.UPPER_WISHBONE_INBOARD_FRONT, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.UPPER_WISHBONE_INBOARD_REAR, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.LOWER_WISHBONE_INBOARD_FRONT, PointID.LOWER_WISHBONE_OUTBOARD)
    constrain(PointID.LOWER_WISHBONE_INBOARD_REAR, PointID.LOWER_WISHBONE_OUTBOARD)

    # Upright length constraint.
    constrain(PointID.UPPER_WISHBONE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD)

    # Axle length constraint.
    constrain(PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD)

    # Axle to balljoint constraints.
    constrain(PointID.AXLE_INBOARD, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.AXLE_INBOARD, PointID.LOWER_WISHBONE_OUTBOARD)
    constrain(PointID.AXLE_OUTBOARD, PointID.UPPER_WISHBONE_OUTBOARD)
    constrain(PointID.AXLE_OUTBOARD, PointID.LOWER_WISHBONE_OUTBOARD)

    # Track rod length constraint.
    constrain(PointID.TRACKROD_INBOARD, PointID.TRACKROD_OUTBOARD)

    # Track rod constraints.
    constrain(PointID.UPPER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD)
    constrain(PointID.LOWER_WISHBONE_OUTBOARD, PointID.TRACKROD_OUTBOARD)

    # Axle to TRE constraints.
    constrain(PointID.AXLE_INBOARD, PointID.TRACKROD_OUTBOARD)
    constrain(PointID.AXLE_OUTBOARD, PointID.TRACKROD_OUTBOARD)

    return constraints


def create_angle_constraints(positions: Positions) -> list[VectorAngle]:
    constraints = []

    def constrain(
        v1_start: PointID, v1_end: PointID, v2_start: PointID, v2_end: PointID
    ):
        constraints.append(
            make_vector_angle(positions, v1_start, v1_end, v2_start, v2_end)
        )

    # Kingpin axis to axle orientation constraint.
    constrain(
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.AXLE_INBOARD,
        PointID.AXLE_OUTBOARD,
    )

    return constraints


def create_linear_constraints(positions: Positions) -> list[PointOnLine]:
    constraints = []

    def constrain(point_id: PointID, line_point: PointID, line_direction: NDArray):
        constraints.append(
            PointOnLine(
                point_id=point_id, line_point=line_point, line_direction=line_direction
            )
        )

    # Track rod inner point should only move in Y direction. Note that this
    # could also be achieved with two PointFixedAxisConstraints, but this is
    # more concise.
    constrain(PointID.TRACKROD_INBOARD, PointID.TRACKROD_INBOARD, Direction.y)

    return constraints


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
    # Pull wheel config.
    wheel_config = WheelConfig(
        width=geometry.configuration.wheel.width,
        offset=geometry.configuration.wheel.offset,
        diameter=geometry.configuration.wheel.diameter,
    )

    # Compute initial positions dict.
    initial_positions = create_initial_positions(geometry)

    # Compute derived points and create motion target.
    compute_derived_points_ptl = partial(compute_derived_points, config=wheel_config)
    positions = compute_derived_points_ptl(initial_positions)
    target = create_target(positions)

    # Create constraints.
    constraints = []
    constraints.extend(create_length_constraints(positions))
    constraints.extend(create_angle_constraints(positions))
    constraints.extend(create_linear_constraints(positions))

    # Get free points.
    free_points = get_free_points(geometry)

    return solve_sweep(
        positions=positions,
        free_points=free_points,
        constraints=constraints,
        target=target,
        displacements=displacements,
        compute_derived_points=compute_derived_points_ptl,
    )
