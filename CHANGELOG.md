# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - Unreleased

### Changed
- Adopted ISO/SAE wheel offset (ET) convention in `get_wheel_center`.
- Positive `wheel.offset` now places the wheel centerline inboard of the hub face (reduced track for larger positive ET).
- Updated wheel offset configuration docs to explicitly describe ET sign convention.
- Updated derived-point expectations to match ISO/SAE offset behavior for wheel center, wheel inboard, and wheel outboard.
- `ResidualComputer` now uses a fixed-size residual vector and Jacobian matrix, removing per-step trimming. The target count is validated once at each evaluation rather than allowing variable-length slicing.
- `ResidualComputer` internals are no longer private: `_n_vars` → `n_vars`, `_jac_buffer` → `jac_buffer`, `_jac_plans` → `jac_plan`, `_validate_target_count` → `validate_target_count`.
- Renamed Jacobian "scatter" operations to "distribute" in `ResidualComputer._build_jac_plan`.
- Moved underdetermined-system check out of the per-step loop in `solve_suspension_sweep` (both `n_vars` and `m_res` are constant across a sweep).
- Simplified `DoubleWishboneSuspension._apply_camber_shim` docstring.

### Added
- `ResidualComputer.validate_target_count` enforces that every evaluation receives the same number of targets configured at init time.
- Test for Jacobian shape consistency (`test_residual_computer_rejects_target_count_changes`).
- Front-view (Y-Z) comparison plot in `visualize_camber_shim.py` overlaying design and setup suspensions with distinct colours.

### Fixed
- Clarified `get_wheel_center_on_ground` docstring to describe the wheel-center line/ground-plane intersection behavior.
- Clarified `get_contact_patch_center` docstring as the lowest point on an ideal tire circle in the wheel center plane.



