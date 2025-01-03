from enum import IntEnum

import numpy as np


# Cardinal directions.
class Direction:
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])


class CoordinateAxis(IntEnum):
    X = 0
    Y = 1
    Z = 2
