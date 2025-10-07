# Used for equality checks to avoid floating point issues.
EPSILON = 1e-6

# Solve tolerances.
SOLVE_TOLERANCE_VALUE = 1e-4  # 0.1um for mm units.
SOLVE_TOLERANCE_STEP = 1e-7
SOLVE_TOLERANCE_GRAD = 1e-7

# Tolerance for tests; has headroom over solve tolerances.
TEST_TOLERANCE = 1e-3
