"""
Camber shim geometry calculations.

This module solves suspension pose for the specified split body (outboard) camber shim.
The upper shim block rotates about the upper ball joint, the lower upright body rotates
about the lower ball joint, and the two shim faces remain separated by the requested
setup thickness.

The solver uses a 9 variable overdetermined least-squares formulation:
    - UBJ_xyz (3): Upper ball joint position, constrained to the upper wishbone arc.
    - camber_block_rotvec (3): Rotation vector for the camber block about UBJ.
    - upright_body_rotvec (3): Rotation vector for the upright body about LBJ.

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
from kinematics.core.vector_utils.geometric import rotate_vector_rodrigues
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

    ubj_position: Vec3
    camber_block_rot_vec: Vec3
    upright_body_rot_vec: Vec3
    camber_block_face_normal: Vec3
    upright_body_face_normal: Vec3
    upright_body_rot_axis: Vec3
    upright_body_rot_angle_rad: float
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
    camber_block_to_datum_a: Vec3
    camber_block_to_datum_b: Vec3
    upright_body_to_datum_a: Vec3
    upright_body_to_datum_b: Vec3
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
        x[0:3] - Solved UBJ position.
        x[3:6] - Camber block rotation vector about UBJ.
        x[6:9] - Upright body rotation vector about LBJ.

    Residuals (12):
        [0]    - |UBJ - UWB_IF| - L_front (top wishbone, front leg).
        [1]    - |UBJ - UWB_IR| - L_rear (top wishbone, rear leg).
        [2:5]  - Shim contact/datum A closure: p_lA - p_uA - t * n_u.
        [5:8]  - Shim contact/datum B closure: p_lB - p_uB - t * n_u.
        [8:11] - Normal alignment: n_l - n_u.
        [11]   - Trackrod length: |rotated_tro - tri| - L_trackrod.
    """
    solved_upper_ball_joint = x[:3]
    camber_block_rot_vec = x[3:6]
    upright_body_rot_vec = x[6:9]

    # Wishbone leg length constraints keep UBJ on a prescribed arc.
    uwb_fwd_residual = (
        np.linalg.norm(
            solved_upper_ball_joint - assembly_context.upper_wishbone_inboard_front
        )
        - assembly_context.upper_arm_length_front
    )
    uwb_rwd_residual = (
        np.linalg.norm(
            solved_upper_ball_joint - assembly_context.upper_wishbone_inboard_rear
        )
        - assembly_context.upper_arm_length_rear
    )

    # Rotate design-state local offsets on each rigid half of the split upright.
    rotated_camber_block_datum_a = rotate_vector_rodrigues(
        assembly_context.camber_block_to_datum_a,
        camber_block_rot_vec,
    )
    rotated_camber_block_datum_b = rotate_vector_rodrigues(
        assembly_context.camber_block_to_datum_b,
        camber_block_rot_vec,
    )
    solved_camber_block_face_normal = rotate_vector_rodrigues(
        assembly_context.design_face_normal,
        camber_block_rot_vec,
    )

    rotated_upright_body_datum_a = rotate_vector_rodrigues(
        assembly_context.upright_body_to_datum_a,
        upright_body_rot_vec,
    )
    rotated_upright_body_datum_b = rotate_vector_rodrigues(
        assembly_context.upright_body_to_datum_b,
        upright_body_rot_vec,
    )

    solved_upright_body_face_normal = rotate_vector_rodrigues(
        assembly_context.design_face_normal,
        upright_body_rot_vec,
    )

    # Reconstruct the world positions of the A/B interface datums from the solved
    # rigid-body pose of each half.
    solved_camber_block_datum_a = solved_upper_ball_joint + rotated_camber_block_datum_a
    solved_camber_block_datum_b = solved_upper_ball_joint + rotated_camber_block_datum_b
    solved_upright_body_datum_a = (
        assembly_context.lower_ball_joint + rotated_upright_body_datum_a
    )
    solved_upright_body_datum_b = (
        assembly_context.lower_ball_joint + rotated_upright_body_datum_b
    )

    # Closure residual: the opposing datum points on each shim face must have coaxial
    # normals (parallel with the main face normal) and separated by exactly the setup
    # shim thickness. Two dowel datums (A, B) clock the interface orientation; no
    # relative rotation is allowed.
    datum_a_closure_residual = (
        solved_upright_body_datum_a
        - solved_camber_block_datum_a
        - assembly_context.setup_thickness * solved_camber_block_face_normal
    )
    datum_b_closure_residual = (
        solved_upright_body_datum_b
        - solved_camber_block_datum_b
        - assembly_context.setup_thickness * solved_camber_block_face_normal
    )

    # The two faces must keep the same orientation, not just be parallel up to a
    # sign flip. Using the vector difference rejects the anti-parallel branch.
    face_normal_alignment_residual = (
        solved_upright_body_face_normal - solved_camber_block_face_normal
    )

    # The trackrod remains a rigid link while its outboard pickup rotates with the
    # lower body about LBJ.
    rotated_trackrod_offset = rotate_vector_rodrigues(
        assembly_context.trackrod_offset,
        upright_body_rot_vec,
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
            np.array([uwb_fwd_residual, uwb_rwd_residual]),
            datum_a_closure_residual,
            datum_b_closure_residual,
            face_normal_alignment_residual,
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
            ubj_position=make_vec3(upper_ball_joint_design.copy()),
            camber_block_rot_vec=make_vec3(np.zeros(3)),
            upright_body_rot_vec=make_vec3(np.zeros(3)),
            camber_block_face_normal=make_vec3(design_face_normal.copy()),
            upright_body_face_normal=make_vec3(design_face_normal.copy()),
            upright_body_rot_axis=make_vec3(np.array([0.0, 0.0, 1.0])),
            upright_body_rot_angle_rad=0.0,
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
        camber_block_to_datum_a=make_vec3(d_upper_a.copy()),
        camber_block_to_datum_b=make_vec3(d_upper_b.copy()),
        upright_body_to_datum_a=make_vec3(d_lower_a.copy()),
        upright_body_to_datum_b=make_vec3(d_lower_b.copy()),
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
    camber_block_rv = make_vec3(result.x[3:6])
    upright_body_rv = make_vec3(result.x[6:9])

    # Compute solved face normals.
    solved_n_camber_block = make_vec3(
        rotate_vector_rodrigues(design_face_normal, result.x[3:6])
    )
    solved_n_upright_body = make_vec3(
        rotate_vector_rodrigues(design_face_normal, result.x[6:9])
    )

    # Extract upright body rotation axis and angle for suspension integration.
    upright_body_angle = float(np.linalg.norm(upright_body_rv))
    if upright_body_angle > EPS_NUMERICAL:
        upright_body_axis = make_vec3(upright_body_rv / upright_body_angle)
    else:
        upright_body_axis = make_vec3(np.array([0.0, 0.0, 1.0]))

    return CamberShimAssemblySolution(
        ubj_position=solved_ubj_pos,
        camber_block_rot_vec=camber_block_rv,
        upright_body_rot_vec=upright_body_rv,
        camber_block_face_normal=solved_n_camber_block,
        upright_body_face_normal=solved_n_upright_body,
        upright_body_rot_axis=upright_body_axis,
        upright_body_rot_angle_rad=upright_body_angle,
        constraint_residual_norm=float(np.linalg.norm(result.fun)),
    )
