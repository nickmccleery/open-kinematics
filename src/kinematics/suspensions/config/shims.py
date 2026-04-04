"""
Camber shim geometry calculations.

This module solves suspension pose for the specified split body (outboard) camber shim.
The upper shim block rotates about the upper ball joint, the lower upright body rotates
about the lower ball joint, and the two shim faces remain separated by the requested
setup thickness.

The solver uses a 9 variable overdetermined least-squares formulation:
    - UBJ_xyz (3): Upper ball joint position, constrained to the upper wishbone arc.
    - upper_rotvec (3): Rotation vector for the upper shim block about UBJ.
    - lower_rotvec (3): Rotation vector for the lower upright body about LBJ.

With 12 residuals:
    - 2 scalar upper arm distance constraints (UBJ to each inboard pickup).
    - 3 scalar datum A closure (lower face A - upper face A = thickness * normal).
    - 3 scalar datum B closure (lower face B - upper face B = thickness * normal).
    - 3 scalar normal alignment (upper and lower face normals must match).
    - 1 scalar trackrod length (preserves design trackrod length through shim change)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.core.constants import EPS_GEOMETRIC, EPS_NUMERICAL
from kinematics.core.enums import PointID
from kinematics.core.types import Vec3, make_vec3
from kinematics.core.vector_utils.generic import normalize_vector
from kinematics.core.vector_utils.geometric import rodrigues_rotate_vector
from kinematics.solver import SolverConfig, solve_least_squares_problem
from kinematics.suspensions.config.settings import CamberShimConfig

CAMBER_SHIM_N_VARS = 9
CAMBER_SHIM_N_RESIDUALS = 12


@dataclass(frozen=True)
class CamberShimAssemblySolution:
    """
    Solved split-body shim assembly state.

    The local assembly solve determines how both the upper shim block and lower
    upright body rotate to accommodate the setup shim thickness. The suspension
    integration consumes the solved UBJ position and lower-body rotation to update
    the global suspension state.
    """

    solved_ubj: Vec3
    upper_rotation_vector: Vec3
    lower_rotation_vector: Vec3
    upper_face_normal: Vec3
    lower_face_normal: Vec3
    lower_rotation_axis: Vec3
    lower_rotation_angle_rad: float
    constraint_residual_norm: float


@dataclass(frozen=True)
class CamberShimAssemblyContext:
    """
    Fixed geometry and invariants for a single shim assembly solve.

    This is the local equivalent of the main solver's pre-built residual context:
    all design-state offsets and invariant lengths are computed once, then reused
    on every residual evaluation.
    """

    lower_ball_joint: Vec3
    upper_wishbone_inboard_front: Vec3
    upper_wishbone_inboard_rear: Vec3
    trackrod_inboard: Vec3
    design_face_normal: Vec3
    upper_datum_a_offset: Vec3
    upper_datum_b_offset: Vec3
    lower_datum_a_offset: Vec3
    lower_datum_b_offset: Vec3
    trackrod_offset: Vec3
    setup_thickness: float
    upper_arm_length_front: float
    upper_arm_length_rear: float
    trackrod_length: float


def compute_camber_shim_assembly_residuals(
    x: np.ndarray,
    assembly_context: CamberShimAssemblyContext,
) -> np.ndarray:
    """
    Compute the residual vector for the local camber shim assembly solve.

    Variables (9):
        x[0:3] - solved UBJ position
        x[3:6] - upper block rotation vector about UBJ
        x[6:9] - lower body rotation vector about LBJ

    Residuals (12):
        [0]    - |UBJ - UWB_IF| - L_front
        [1]    - |UBJ - UWB_IR| - L_rear
        [2:5]  - datum A closure: p_lA - p_uA - t * n_u
        [5:8]  - datum B closure: p_lB - p_uB - t * n_u
        [8:11] - normal alignment: n_l - n_u
        [11]   - trackrod length: |rotated_tro - tri| - L_trackrod
    """
    solved_upper_ball_joint = x[:3]
    upper_rotation_vector = x[3:6]
    lower_rotation_vector = x[6:9]

    # Upper arm distance constraints keep UBJ on the articulation locus defined
    # by the two rigid upper wishbone legs.
    upper_arm_front_residual = (
        np.linalg.norm(
            solved_upper_ball_joint - assembly_context.upper_wishbone_inboard_front
        )
        - assembly_context.upper_arm_length_front
    )
    upper_arm_rear_residual = (
        np.linalg.norm(
            solved_upper_ball_joint - assembly_context.upper_wishbone_inboard_rear
        )
        - assembly_context.upper_arm_length_rear
    )

    # Rotate design-state local offsets on each rigid half of the split upright.
    rotated_upper_datum_a = rodrigues_rotate_vector(
        assembly_context.upper_datum_a_offset,
        upper_rotation_vector,
    )
    rotated_upper_datum_b = rodrigues_rotate_vector(
        assembly_context.upper_datum_b_offset,
        upper_rotation_vector,
    )
    rotated_lower_datum_a = rodrigues_rotate_vector(
        assembly_context.lower_datum_a_offset,
        lower_rotation_vector,
    )
    rotated_lower_datum_b = rodrigues_rotate_vector(
        assembly_context.lower_datum_b_offset,
        lower_rotation_vector,
    )
    solved_upper_face_normal = rodrigues_rotate_vector(
        assembly_context.design_face_normal,
        upper_rotation_vector,
    )
    solved_lower_face_normal = rodrigues_rotate_vector(
        assembly_context.design_face_normal,
        lower_rotation_vector,
    )

    # Reconstruct the world positions of the A/B interface datums from the solved
    # rigid-body pose of each half.
    solved_upper_datum_a = solved_upper_ball_joint + rotated_upper_datum_a
    solved_upper_datum_b = solved_upper_ball_joint + rotated_upper_datum_b
    solved_lower_datum_a = assembly_context.lower_ball_joint + rotated_lower_datum_a
    solved_lower_datum_b = assembly_context.lower_ball_joint + rotated_lower_datum_b

    # Each datum pair must close to the requested shim gap along the shared upper
    # face normal. Two ordered dowel datums provide the interface clocking.
    datum_a_closure = (
        solved_lower_datum_a
        - solved_upper_datum_a
        - assembly_context.setup_thickness * solved_upper_face_normal
    )
    datum_b_closure = (
        solved_lower_datum_b
        - solved_upper_datum_b
        - assembly_context.setup_thickness * solved_upper_face_normal
    )

    # The two faces must keep the same orientation, not just be parallel up to a
    # sign flip. Using the vector difference rejects the anti-parallel branch.
    face_normal_alignment = solved_lower_face_normal - solved_upper_face_normal

    # The trackrod remains a rigid link while its outboard pickup rotates with the
    # lower body about LBJ.
    rotated_trackrod_offset = rodrigues_rotate_vector(
        assembly_context.trackrod_offset,
        lower_rotation_vector,
    )
    solved_trackrod_outboard = (
        assembly_context.lower_ball_joint + rotated_trackrod_offset
    )
    trackrod_length_residual = (
        np.linalg.norm(solved_trackrod_outboard - assembly_context.trackrod_inboard)
        - assembly_context.trackrod_length
    )

    return np.concatenate(
        [
            np.array([upper_arm_front_residual, upper_arm_rear_residual]),
            datum_a_closure,
            datum_b_closure,
            face_normal_alignment,
            np.array([trackrod_length_residual]),
        ]
    )


# These are the points we need to run the solve.
REQUIRED_POINT_IDS = frozenset(
    {
        PointID.UPPER_WISHBONE_OUTBOARD,
        PointID.LOWER_WISHBONE_OUTBOARD,
        PointID.UPPER_WISHBONE_INBOARD_FRONT,
        PointID.UPPER_WISHBONE_INBOARD_REAR,
        PointID.TRACKROD_OUTBOARD,
        PointID.TRACKROD_INBOARD,
        PointID.CAMBER_SHIM_FACE_POINT_A,
        PointID.CAMBER_SHIM_FACE_POINT_B,
        PointID.CAMBER_SHIM_FACE_NORMAL,
    }
)


def solve_camber_shim_assembly(
    positions: dict[PointID, Vec3],
    shim_config: CamberShimConfig,
    solver_config: SolverConfig = SolverConfig(),
) -> CamberShimAssemblySolution:
    """
    Solve the suspension pose for the specified split body (outboard) camber shim.

    Finds the configuration where the upper shim block (rotating about the UBJ) and the
    lower upright body (rotating about the LBJ) produce parallel shim faces separated by
    the setup thickness, while the UBJ remains on the upper wishbone arc, with trackrod
    length remaining equal to design condition.

    Args:
        positions: Dict mapping PointID to Vec3 positions.
        shim_config: Shim thickness configuration (design and setup thicknesses).
        solver_config: Solver configuration (tolerances, verbosity, etc.).

    Returns:
        Solved assembly state with UBJ position, rotation vectors, and convergence
        info.

    Raises:
        RuntimeError: If the solver fails to converge.
        KeyError: If a required PointID is missing from positions.
    """
    missing = REQUIRED_POINT_IDS - positions.keys()
    if missing:
        names = sorted(p.name for p in missing)
        raise KeyError(f"Missing required PointIDs: {names}")

    upper_ball_joint_design = make_vec3(positions[PointID.UPPER_WISHBONE_OUTBOARD])
    lower_ball_joint = make_vec3(positions[PointID.LOWER_WISHBONE_OUTBOARD])
    upper_wishbone_pickup_front = make_vec3(
        positions[PointID.UPPER_WISHBONE_INBOARD_FRONT]
    )
    upper_wishbone_pickup_rear = make_vec3(
        positions[PointID.UPPER_WISHBONE_INBOARD_REAR]
    )
    trackrod_outboard_design = make_vec3(positions[PointID.TRACKROD_OUTBOARD])
    trackrod_inboard_fixed = make_vec3(positions[PointID.TRACKROD_INBOARD])
    shim_face_datum_a = make_vec3(positions[PointID.CAMBER_SHIM_FACE_POINT_A])
    shim_face_datum_b = make_vec3(positions[PointID.CAMBER_SHIM_FACE_POINT_B])
    design_face_normal = normalize_vector(
        make_vec3(positions[PointID.CAMBER_SHIM_FACE_NORMAL])
    )

    # Early exit when there is no shim thickness change.
    if abs(shim_config.setup_thickness - shim_config.design_thickness) < EPS_GEOMETRIC:
        return CamberShimAssemblySolution(
            solved_ubj=make_vec3(upper_ball_joint_design.copy()),
            upper_rotation_vector=make_vec3(np.zeros(3)),
            lower_rotation_vector=make_vec3(np.zeros(3)),
            upper_face_normal=make_vec3(design_face_normal.copy()),
            lower_face_normal=make_vec3(design_face_normal.copy()),
            lower_rotation_axis=make_vec3(np.array([0.0, 0.0, 1.0])),
            lower_rotation_angle_rad=0.0,
            constraint_residual_norm=0.0,
        )

    half_design = 0.5 * shim_config.design_thickness

    # Design-state face datum positions. The upper face is on the inboard side
    # (toward UBJ), the lower face on the outboard side (toward the upright body).
    # "Upper" and "lower" refer to the shim block attached to UBJ and the upright
    # body attached to LBJ respectively, not vertical position.
    upper_a_design = shim_face_datum_a - half_design * design_face_normal
    upper_b_design = shim_face_datum_b - half_design * design_face_normal
    lower_a_design = shim_face_datum_a + half_design * design_face_normal
    lower_b_design = shim_face_datum_b + half_design * design_face_normal

    # Design-state upper wishbone arm lengths (invariant under UBJ articulation).
    arm_length_front = float(
        np.linalg.norm(upper_ball_joint_design - upper_wishbone_pickup_front)
    )
    arm_length_rear = float(
        np.linalg.norm(upper_ball_joint_design - upper_wishbone_pickup_rear)
    )

    # Design-state trackrod length. The trackrod is a rigid link so this distance
    # must be preserved through the shim change.
    trackrod_length = float(
        np.linalg.norm(trackrod_outboard_design - trackrod_inboard_fixed)
    )

    # Design-state local offsets from each ball joint. These are the vectors that
    # get rotated by the respective rotation vectors during the solve.
    d_upper_a = upper_a_design - upper_ball_joint_design
    d_upper_b = upper_b_design - upper_ball_joint_design
    d_lower_a = lower_a_design - lower_ball_joint
    d_lower_b = lower_b_design - lower_ball_joint

    # Trackrod outboard offset from LBJ (rotates with the lower body).
    d_trackrod = trackrod_outboard_design - lower_ball_joint

    assembly_context = CamberShimAssemblyContext(
        lower_ball_joint=make_vec3(lower_ball_joint.copy()),
        upper_wishbone_inboard_front=make_vec3(upper_wishbone_pickup_front.copy()),
        upper_wishbone_inboard_rear=make_vec3(upper_wishbone_pickup_rear.copy()),
        trackrod_inboard=make_vec3(trackrod_inboard_fixed.copy()),
        design_face_normal=make_vec3(design_face_normal.copy()),
        upper_datum_a_offset=make_vec3(d_upper_a.copy()),
        upper_datum_b_offset=make_vec3(d_upper_b.copy()),
        lower_datum_a_offset=make_vec3(d_lower_a.copy()),
        lower_datum_b_offset=make_vec3(d_lower_b.copy()),
        trackrod_offset=make_vec3(d_trackrod.copy()),
        setup_thickness=shim_config.setup_thickness,
        upper_arm_length_front=arm_length_front,
        upper_arm_length_rear=arm_length_rear,
        trackrod_length=trackrod_length,
    )

    # Seed from design condition: design UBJ position, zero rotations.
    x_0 = np.zeros(CAMBER_SHIM_N_VARS)
    x_0[:3] = upper_ball_joint_design

    result = solve_least_squares_problem(
        residual_function=compute_camber_shim_assembly_residuals,
        x_0=x_0,
        args=(assembly_context,),
        solver_config=solver_config,
        n_residuals=CAMBER_SHIM_N_RESIDUALS,
    )

    if not result.success:
        raise RuntimeError(
            f"Camber shim assembly solve failed to converge.\nMessage: {result.message}"
        )

    # Extract solution.
    solved_ubj_pos = make_vec3(result.x[:3])
    upper_rv = make_vec3(result.x[3:6])
    lower_rv = make_vec3(result.x[6:9])

    # Compute solved face normals.
    solved_n_upper = make_vec3(
        rodrigues_rotate_vector(design_face_normal, result.x[3:6])
    )
    solved_n_lower = make_vec3(
        rodrigues_rotate_vector(design_face_normal, result.x[6:9])
    )

    # Extract lower rotation axis and angle for suspension integration.
    lower_angle = float(np.linalg.norm(lower_rv))
    if lower_angle > EPS_NUMERICAL:
        lower_axis = make_vec3(lower_rv / lower_angle)
    else:
        lower_axis = make_vec3(np.array([0.0, 0.0, 1.0]))

    return CamberShimAssemblySolution(
        solved_ubj=solved_ubj_pos,
        upper_rotation_vector=upper_rv,
        lower_rotation_vector=lower_rv,
        upper_face_normal=solved_n_upper,
        lower_face_normal=solved_n_lower,
        lower_rotation_axis=lower_axis,
        lower_rotation_angle_rad=lower_angle,
        constraint_residual_norm=float(np.linalg.norm(result.fun)),
    )
