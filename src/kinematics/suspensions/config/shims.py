"""
Camber shim geometry calculations.

This module solves the local split-body camber shim assembly. The upper shim block
rotates about the upper ball joint, the lower upright body rotates about the lower
ball joint, and the two shim faces remain separated by the requested setup thickness.

The solver uses a 9-variable overdetermined least-squares formulation:
    - UBJ_xyz (3): upper ball joint position, constrained to the upper wishbone arc
    - upper_rotvec (3): rotation vector for the upper shim block about UBJ
    - lower_rotvec (3): rotation vector for the lower upright body about LBJ

with 12 residuals:
    - 2 scalar upper-arm distance constraints (UBJ to each inboard pickup)
    - 3 scalar datum A closure (lower face A - upper face A = thickness * normal)
    - 3 scalar datum B closure (lower face B - upper face B = thickness * normal)
    - 3 scalar parallelism (cross product of upper and lower face normals)
    - 1 scalar trackrod length (preserves design trackrod length through shim change)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from kinematics.core.types import Vec3, make_vec3
from kinematics.core.vector_utils.generic import normalize_vector
from kinematics.core.vector_utils.geometric import rodrigues_rotate_vector
from kinematics.io.validation import Vec3Like, coerce_vec3


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


def solve_camber_shim_assembly(
    upper_ball_joint: Vec3Like,
    lower_ball_joint: Vec3Like,
    upper_wishbone_inboard_front: Vec3Like,
    upper_wishbone_inboard_rear: Vec3Like,
    trackrod_outboard: Vec3Like,
    trackrod_inboard: Vec3Like,
    shim_face_point_a: Vec3Like,
    shim_face_point_b: Vec3Like,
    shim_face_normal: Vec3Like,
    design_thickness: float,
    setup_thickness: float,
) -> CamberShimAssemblySolution:
    """
    Solve the local split-body shim assembly.

    Finds the configuration where the upper shim block (rotating about UBJ) and the
    lower upright body (rotating about LBJ) produce parallel shim faces separated by
    the setup thickness, while UBJ remains on the upper wishbone arc and the trackrod
    length is preserved.

    Args:
        upper_ball_joint: Design UBJ position in world coordinates.
        lower_ball_joint: Fixed LBJ position in world coordinates.
        upper_wishbone_inboard_front: Fixed upper wishbone front pickup.
        upper_wishbone_inboard_rear: Fixed upper wishbone rear pickup.
        trackrod_outboard: Design trackrod outboard position (on the upright body).
        trackrod_inboard: Fixed trackrod inboard position (on the chassis/rack).
        shim_face_point_a: First dowel datum on the design mid-thickness plane.
        shim_face_point_b: Second dowel datum on the design mid-thickness plane.
        shim_face_normal: Design shim face normal direction.
        design_thickness: Design shim stack thickness in mm.
        setup_thickness: Setup shim stack thickness in mm.

    Returns:
        Solved assembly state with UBJ position, rotation vectors, and convergence
        info.

    Raises:
        RuntimeError: If the solver fails to converge.
    """
    ubj = coerce_vec3(upper_ball_joint)
    lbj = coerce_vec3(lower_ball_joint)
    uwb_if = coerce_vec3(upper_wishbone_inboard_front)
    uwb_ir = coerce_vec3(upper_wishbone_inboard_rear)
    tro = coerce_vec3(trackrod_outboard)
    tri = coerce_vec3(trackrod_inboard)
    pt_a = coerce_vec3(shim_face_point_a)
    pt_b = coerce_vec3(shim_face_point_b)
    normal = normalize_vector(coerce_vec3(shim_face_normal))

    # Early exit when there is no shim thickness change.
    if abs(setup_thickness - design_thickness) < 1e-10:
        return CamberShimAssemblySolution(
            solved_ubj=make_vec3(ubj.copy()),
            upper_rotation_vector=make_vec3(np.zeros(3)),
            lower_rotation_vector=make_vec3(np.zeros(3)),
            upper_face_normal=make_vec3(normal.copy()),
            lower_face_normal=make_vec3(normal.copy()),
            lower_rotation_axis=make_vec3(np.array([0.0, 0.0, 1.0])),
            lower_rotation_angle_rad=0.0,
            constraint_residual_norm=0.0,
        )

    half_design = 0.5 * design_thickness

    # Design-state face datum positions. The upper face is on the inboard side
    # (toward UBJ), the lower face on the outboard side (toward the upright body).
    # "Upper" and "lower" refer to the shim block attached to UBJ and the upright
    # body attached to LBJ respectively, not vertical position.
    upper_a_design = pt_a - half_design * normal
    upper_b_design = pt_b - half_design * normal
    lower_a_design = pt_a + half_design * normal
    lower_b_design = pt_b + half_design * normal

    # Design-state upper wishbone arm lengths (invariant under UBJ articulation).
    arm_length_front = float(np.linalg.norm(ubj - uwb_if))
    arm_length_rear = float(np.linalg.norm(ubj - uwb_ir))

    # Design-state trackrod length. The trackrod is a rigid link so this distance
    # must be preserved through the shim change.
    trackrod_length = float(np.linalg.norm(tro - tri))

    # Design-state local offsets from each ball joint. These are the vectors that
    # get rotated by the respective rotation vectors during the solve.
    d_upper_a = upper_a_design - ubj
    d_upper_b = upper_b_design - ubj
    d_lower_a = lower_a_design - lbj
    d_lower_b = lower_b_design - lbj

    # Trackrod outboard offset from LBJ (rotates with the lower body).
    d_trackrod = tro - lbj

    def residuals(x: np.ndarray) -> np.ndarray:
        """
        Compute the 12 residuals for the shim assembly.

        Variables (9):
            x[0:3] - UBJ position
            x[3:6] - upper block rotation vector about UBJ
            x[6:9] - lower body rotation vector about LBJ

        Residuals (12):
            [0]    - |UBJ - UWB_IF| - L_front
            [1]    - |UBJ - UWB_IR| - L_rear
            [2:5]  - datum A closure: p_lA - p_uA - t * n_u
            [5:8]  - datum B closure: p_lB - p_uB - t * n_u
            [8:11] - parallelism: cross(n_u, n_l)
            [11]   - trackrod length: |rotated_tro - tri| - L_trackrod
        """
        ubj_pos = x[:3]
        upper_rv = x[3:6]
        lower_rv = x[6:9]

        # Upper arm distance constraints. These keep UBJ on the upper wishbone arc.
        dist_front = np.linalg.norm(ubj_pos - uwb_if) - arm_length_front
        dist_rear = np.linalg.norm(ubj_pos - uwb_ir) - arm_length_rear

        # Rotate design offsets by the respective rotation vectors.
        rot_upper_a = rodrigues_rotate_vector(d_upper_a, upper_rv)
        rot_upper_b = rodrigues_rotate_vector(d_upper_b, upper_rv)
        rot_lower_a = rodrigues_rotate_vector(d_lower_a, lower_rv)
        rot_lower_b = rodrigues_rotate_vector(d_lower_b, lower_rv)
        n_u = rodrigues_rotate_vector(normal, upper_rv)
        n_l = rodrigues_rotate_vector(normal, lower_rv)

        # World positions of face datums after rotation.
        p_upper_a = ubj_pos + rot_upper_a
        p_upper_b = ubj_pos + rot_upper_b
        p_lower_a = lbj + rot_lower_a
        p_lower_b = lbj + rot_lower_b

        # Datum closure: the gap from upper to lower face at each datum must equal
        # the setup thickness along the upper face normal direction.
        closure_a = p_lower_a - p_upper_a - setup_thickness * n_u
        closure_b = p_lower_b - p_upper_b - setup_thickness * n_u

        # Parallelism: upper and lower face normals must remain parallel.
        parallel = np.cross(n_u, n_l)

        # Trackrod length: the trackrod outboard pickup rotates with the lower body
        # about LBJ, but the trackrod is a rigid link so its length must match the
        # design-state distance to the fixed inboard pickup.
        rot_trackrod = rodrigues_rotate_vector(d_trackrod, lower_rv)
        tro_solved = lbj + rot_trackrod
        trackrod_residual = np.linalg.norm(tro_solved - tri) - trackrod_length

        return np.concatenate(
            [
                np.array([dist_front, dist_rear]),
                closure_a,
                closure_b,
                parallel,
                np.array([trackrod_residual]),
            ]
        )

    # Seed from design condition: design UBJ position, zero rotations.
    x0 = np.zeros(9)
    x0[:3] = ubj

    result = least_squares(
        residuals,
        x0,
        method="lm",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    if not result.success:
        raise RuntimeError(
            f"Camber shim assembly solve failed to converge.\n"
            f"Message: {result.message}"
        )

    # Extract solution.
    solved_ubj_pos = make_vec3(result.x[:3])
    upper_rv = make_vec3(result.x[3:6])
    lower_rv = make_vec3(result.x[6:9])

    # Compute solved face normals.
    solved_n_upper = make_vec3(rodrigues_rotate_vector(normal, result.x[3:6]))
    solved_n_lower = make_vec3(rodrigues_rotate_vector(normal, result.x[6:9]))

    # Extract lower rotation axis and angle for suspension integration.
    lower_angle = float(np.linalg.norm(lower_rv))
    if lower_angle > 1e-15:
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
