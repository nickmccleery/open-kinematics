"""
Constraint helpers for points rigidly attached to moving corner bodies.
"""

from kinematics.constraints import (
    Constraint,
    DistanceConstraint,
    ScalarTripleProductConstraint,
)
from kinematics.core.constants import MIN_CHIRALITY_VOLUME
from kinematics.core.enums import PointID
from kinematics.core.vector_utils.geometric import (
    compute_point_point_distance,
    compute_scalar_triple_product,
)
from kinematics.state import SuspensionState


def rigid_point_constraints(
    initial_state: SuspensionState,
    point: PointID,
    references: tuple[PointID, PointID, PointID],
) -> list[Constraint]:
    """
    Hold a point rigidly to a body defined by three reference points.

    Three design-length distance constraints locate the point relative to the
    body. A signed-volume constraint selects the design side of the reference
    plane so the solver cannot converge onto the reflected assembly branch.
    """
    positions = initial_state.positions
    constraints: list[Constraint] = [
        DistanceConstraint(
            point,
            reference,
            compute_point_point_distance(positions[point], positions[reference]),
        )
        for reference in references
    ]

    ref_a, ref_b, ref_c = references
    design_triple = compute_scalar_triple_product(
        positions[ref_b] - positions[ref_a],
        positions[ref_c] - positions[ref_a],
        positions[point] - positions[ref_a],
    )
    if abs(design_triple) >= MIN_CHIRALITY_VOLUME:
        constraints.append(
            ScalarTripleProductConstraint(
                ref_a,
                ref_b,
                ref_c,
                point,
                target_volume=design_triple,
                scale=max(abs(design_triple), 1.0),
            )
        )
    return constraints
