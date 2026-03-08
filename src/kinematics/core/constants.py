# Used for equality checks to avoid floating point issues.
EPSILON = 1e-6

# Solve tolerances.
SOLVE_TOLERANCE_VALUE = 1e-5  # 0.01um for mm units.
SOLVE_TOLERANCE_STEP = 1e-8
SOLVE_TOLERANCE_GRAD = 1e-8

# Tolerance for tests; has headroom over solve tolerances.
TEST_TOLERANCE = 1e-3

# Because rims are still spec'd in freedom units.
MM_PER_INCH = 25.4
