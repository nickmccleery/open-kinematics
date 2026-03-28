# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - Unreleased

### Changed
- Adopted ISO/SAE wheel offset (ET) convention in `get_wheel_center`.
- Positive `wheel.offset` now places the wheel centerline inboard of the hub face (reduced track for larger positive ET).
- Updated wheel offset configuration docs to explicitly describe ET sign convention.
- Updated derived-point expectations to match ISO/SAE offset behavior for wheel center, wheel inboard, and wheel outboard.

### Fixed
- Clarified `get_wheel_center_on_ground` docstring to describe the wheel-center line/ground-plane intersection behavior.
- Clarified `get_contact_patch_center` docstring as the lowest point on an ideal tire circle in the wheel center plane.



