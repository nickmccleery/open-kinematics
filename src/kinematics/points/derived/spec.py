"""""""""

Derived point specifications for explicit derived-point definitions.

Derived point specifications for explicit derived-point definitions.Derived point specifications for explicit derived-point definitions.

Provides DerivedSpec class to make derived-point definitions explicit

and self-describing, replacing loose dicts/tuples.

"""

Provides DerivedSpec class to make derived-point definitions explicitProvides DerivedSpec class to make derived-point definitions explicit

from dataclasses import dataclass

from typing import Callable, Dict, Setand self-describing, replacing loose dicts/tuples.and self-describing, replacing loose dicts/tuples.



import numpy as np""""""



from kinematics.core import PointID



# Function signature for computing a derived point positionfrom dataclasses import dataclassfrom dataclasses import dataclass

PositionFn = Callable[[Dict[PointID, np.ndarray]], np.ndarray]

from typing import Callable, Dict, Setfrom typing import Callable, Dict, Set



@dataclass(frozen=True)

class DerivedSpec:

    """import numpy as npimport numpy as np

    Specification for derived point calculations.



    Contains the functions to compute derived points and their dependencies

    in a self-describing format that can be validated and sorted.from kinematics.core import PointIDfrom kinematics.core import PointID

    """



    functions: Dict[PointID, PositionFn]

    dependencies: Dict[PointID, Set[PointID]]# Function signature for computing a derived point position# Function signature for computing a derived point position



    def all_points(self) -> Set[PointID]:PositionFn = Callable[[Dict[PointID, np.ndarray]], np.ndarray]PositionFn = Callable[[Dict[PointID, np.ndarray]], np.ndarray]

        """Get all derived point IDs defined in this spec."""

        return set(self.functions.keys())



    def validate(self) -> None:

        """

        Validate the spec for consistency.@dataclass(frozen=True)@dataclass(frozen=True)



        Raises:class DerivedSpec:class DerivedSpec:

            ValueError: If spec is inconsistent

        """    """    """

        # Check that all points in functions have dependencies defined

        function_points = set(self.functions.keys())    Specification for derived point calculations.    Specification for derived point calculations.

        dependency_points = set(self.dependencies.keys())



        if function_points != dependency_points:

            missing_deps = function_points - dependency_points    Contains the functions to compute derived points and their dependencies    Contains the functions to compute derived points and their dependencies

            extra_deps = dependency_points - function_points

    in a self-describing format that can be validated and sorted.    in a self-describing format that can be validated and sorted.

            msg_parts = []

            if missing_deps:    """    """

                msg_parts.append(f"Missing dependencies for: {missing_deps}")

            if extra_deps:

                msg_parts.append(f"Extra dependencies for: {extra_deps}")

    functions: Dict[PointID, PositionFn]    functions: Dict[PointID, PositionFn]

            raise ValueError("; ".join(msg_parts))

    dependencies: Dict[PointID, Set[PointID]]    dependencies: Dict[PointID, Set[PointID]]

    def __post_init__(self):

        """Validate the spec after initialization."""

        self.validate()
    def all_points(self) -> Set[PointID]:    def all_points(self) -> Set[PointID]:

        """Get all derived point IDs defined in this spec."""        """Get all derived point IDs defined in this spec."""

        return set(self.functions.keys())        return set(self.functions.keys())



    def validate(self) -> None:    def validate(self) -> None:

        """        """

        Validate the spec for consistency.        Validate the spec for consistency.



        Raises:        Raises:

            ValueError: If spec is inconsistent            ValueError: If spec is inconsistent

        """        """

        # Check that all points in functions have dependencies defined        # Check that all points in functions have dependencies defined

        function_points = set(self.functions.keys())        function_points = set(self.functions.keys())

        dependency_points = set(self.dependencies.keys())        dependency_points = set(self.dependencies.keys())



        if function_points != dependency_points:        if function_points != dependency_points:

            missing_deps = function_points - dependency_points            missing_deps = function_points - dependency_points

            extra_deps = dependency_points - function_points            extra_deps = dependency_points - function_points



            msg_parts = []            msg_parts = []

            if missing_deps:            if missing_deps:

                msg_parts.append(f"Missing dependencies for: {missing_deps}")                msg_parts.append(f"Missing dependencies for: {missing_deps}")

            if extra_deps:            if extra_deps:

                msg_parts.append(f"Extra dependencies for: {extra_deps}")                msg_parts.append(f"Extra dependencies for: {extra_deps}")



            raise ValueError("; ".join(msg_parts))            raise ValueError("; ".join(msg_parts))



    def __post_init__(self):    def __post_init__(self):

        """Validate the spec after initialization."""        """Validate the spec after initialization."""

        self.validate()        self.validate()
