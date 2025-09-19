from typing import Dict, TypeAlias

import numpy as np

from kinematics.geometry.points.ids import PointID

# --- Core Data Types ---
Position: TypeAlias = np.ndarray
Positions: TypeAlias = Dict[PointID, Position]
