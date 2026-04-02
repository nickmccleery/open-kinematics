"""
Camber shim geometry calculations.

This module solves the local split-body camber shim assembly. The upper shim block
rotates about the upper ball joint, the lower upright body rotates about the lower
ball joint, and the two shim faces remain separated by the requested setup thickness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.constraints import DistanceConstraint
from kinematics.core.enums import PointID
from kinematics.core.types import Vec3, make_vec3
from kinematics.core.vector_utils.generic import normalize_vector
from kinematics.io.validation import Vec3Like, coerce_vec3


@dataclass(frozen=True)
class ShimFaceCenters:
    """
    Design-state shim face centers derived from the configured shim mid-plane.
    """

    upper_face_center_design: Vec3
    lower_face_center_design: Vec3


@dataclass(frozen=True)
class CamberShimAssemblySolution:
    """
    Solved split-body shim assembly state.

    The local assembly solve tracks the upper and lower rigid-body rotations, but the
    suspension model only consumes the lower-body rotation when rotating the upright-
    mounted points in the global suspension state.
    """

    upper_rotation_vector: Vec3
    lower_rotation_vector: Vec3
    upper_face_center: Vec3
    lower_face_center: Vec3
    upper_face_normal: Vec3
    lower_face_normal: Vec3
    lower_rotation_axis: Vec3
    lower_rotation_angle_rad: float
    constraint_residual_norm: float


@dataclass(frozen=True)
class ShimAssemblyState:
    """
    Internal state derived from a pair of upper/lower rotation vectors.
    """

    upper_face_center: Vec3
    upper_face_normal: Vec3

    lower_face_center: Vec3
    lower_face_normal: Vec3


def compute_shim_face_centers(
    shim_face_center: Vec3Like,
    shim_normal: Vec3Like,
    design_thickness: float,
) -> ShimFaceCenters:
    """
    Derive the design-state shim face centers from the configured shim mid-plane.

    Args:
        shim_face_center: Design shim mid-plane center in world coordinates.
        shim_normal: Design shim-face normal.
        design_thickness: Design shim stack thickness in mm.

    Returns:
        Design upper and lower shim face centers.
    """
    shim_center = coerce_vec3(shim_face_center)
    normal_unit = normalize_vector(coerce_vec3(shim_normal))

    # The configured point is the mid-plane center, so each face sits half a design
    # thickness away from that point along the face normal.
    half_thickness_offset = 0.5 * design_thickness * normal_unit

    return ShimFaceCenters(
        upper_face_center_design=make_vec3(shim_center - half_thickness_offset),
        lower_face_center_design=make_vec3(shim_center + half_thickness_offset),
    )


def solve_camber_shim_assembly(
    upper_ball_joint: Vec3Like,
    lower_ball_joint: Vec3Like,
    shim_face_center: Vec3Like,
    shim_normal: Vec3Like,
    design_thickness: float,
    setup_thickness: float,
) -> CamberShimAssemblySolution:
    """
    Solve the local split-body shim assembly.

    The solve enforces three physical constraints:

    1. The upper face center remains on the rigid orbit about the upper ball joint.
    2. The lower face center remains on the rigid orbit about the lower ball joint.
    3. The two shim face centre points are separated by the setup shim thickness.

    Args:
        upper_ball_joint: Upper ball joint position in world coordinates.
        lower_ball_joint: Lower ball joint position in world coordinates.
        shim_face_center: Design shim mid-plane center in world coordinates.
        shim_normal: Design shim-face normal.
        design_thickness: Design shim stack thickness in mm.
        setup_thickness: Setup shim stack thickness in mm.

    Returns:
        Solved assembly state, including the lower-body axis/angle consumed by the
        suspension model.

    Raises:
        ValueError: If the requested shim geometry is infeasible for the split-body
            assembly model.
    """
    # Constraints here:
    # - UBJ mounted camber block is free to rotate. One constraint: shim face centre
    #   must remain a Euclidean fixed distance from the UBJ.
    # - Upright body is free to rotate about LBJ. Upright rigid body (including axle
    #   points, trackrod pickup etc.) points remain fixed distance from the LBJ.
    # - Distance between shim face centre points must match the setup shim thickness.
    #
    # So I think we end up with four degrees of freedom:
    # - Azimuth angle, camber block.
    # - Elevation angle, camber block.
    # - Azimuth angle, upright body.
    # - Elevation angle, upright body.
    # We'll need to come up with axis system for the ball joints; maybe global
    # axis aligned but centred at each ball joint?
    #
    # Then our residual terms will just be:
    # - Camber block shim face centroid distance from UBJ.
    # - Upright body shim face centroid distance from LBJ.
    # - Shim face centroid to centroid distance - setup shim thickness.

    # Note: azimuth/elevation isn't enough. I think we'll need Euler angles.

    shim_face_center = make_vec3(shim_face_center)
    upper_ball_joint = make_vec3(upper_ball_joint)
    lower_ball_joint = make_vec3(lower_ball_joint)

    t_shim_half = design_thickness / 2
    position_inboard_sfc = shim_face_center + t_shim_half * (
        upper_ball_joint - shim_face_center
    )
    position_outboard_sfc = shim_face_center + t_shim_half * (
        lower_ball_joint - shim_face_center
    )

    l_inboard_arm = float(np.linalg.norm(upper_ball_joint - position_inboard_sfc))
    l_outboard_arm = float(np.linalg.norm(lower_ball_joint - position_outboard_sfc))

    constraints = [
        DistanceConstraint(
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.CAMBER_SHIM_CENTROID_INBOARD,
            l_inboard_arm,
        ),
        DistanceConstraint(
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.CAMBER_SHIM_CENTROID_OUTBOARD,
            l_outboard_arm,
        ),
        DistanceConstraint(
            PointID.CAMBER_SHIM_CENTROID_INBOARD,
            PointID.CAMBER_SHIM_CENTROID_OUTBOARD,
            setup_thickness,
        ),
    ]

    # Now we need to set up a solve with our four angle variables...
    azimuth_inboard = 0
    elevation_inboard = 0
    azimuth_outboard = 0
    elevation_outboard = 0

    x0 = [azimuth_inboard, elevation_inboard, azimuth_outboard, elevation_outboard]

    def compute_residuals(x):
        # From x (vector of angles):
        # - 1: Apply azimuth rotation to upper arm (+ve CCW about Z transposed to UBJ
        #      position)
        # - 2: Apply elevation rotation to upper arm (+ve CCW about X transposed to
        #      UBJ position)
        #  ...
        # - Establish the overall transformation we want to apply to each shim face centroid.
        # - Evaluate the residual on each constraint.
        # - Bundle into the residuals array.

        dummy_vec = make_vec3([0, 0, 0])

        positions = {
            PointID.UPPER_WISHBONE_OUTBOARD: dummy_vec,  # Should be initial position; fixed.
            PointID.LOWER_WISHBONE_INBOARD_FRONT: dummy_vec,  # Should be initial position; fixed.
            PointID.CAMBER_SHIM_CENTROID_INBOARD: dummy_vec,  # To be computed...
            PointID.CAMBER_SHIM_CENTROID_OUTBOARD: dummy_vec,  # To be computed...
        }

        r = []
        for constraint in constraints:
            r_local = constraint.residual(positions)
            r.append(r_local)

        return r

    # TODO:
    # - Patch in Euler angles.
    # - Get transformation setup up and running, maybe a Rodrigues rotation about each
    #   Euler angle axis, or just a global transformation matrix.
    # - Get least_squares solve going.
    # - Try to reuse exciting `solver.py` machinery; we have a much smaller solve here,
    #   but we're doing basically the same thing so factoring out the common approach
    #   to fixed points vs. variables etc. would be good.
