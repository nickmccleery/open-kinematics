import numpy as np
from numpy.typing import NDArray

from kinematics.constants import EPSILON


def normalize_vector(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize a vector of any dimension to a unit vector.
    """
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        raise ValueError("Cannot normalize zero-length vector")
    return (v / norm).astype(np.float64)
