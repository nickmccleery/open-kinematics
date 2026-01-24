"""
Tests for the suspension template system.

Tests cover:
- Template validation with helpful error messages
- Template geometry creation and conversion
- Template provider functionality
- YAML loading with the new format
"""

from pathlib import Path

import numpy as np
import pytest

from kinematics.enums import PointID, Units
from kinematics.io.geometry_loader import LoadedSuspension, load_geometry
from kinematics.suspensions.core.settings import (
    CamberShimConfigOutboard,
    SuspensionConfig,
    TireConfig,
    WheelConfig,
)
from kinematics.suspensions.core.template_geometry import (
    TemplateGeometry,
    create_template_geometry,
)
from kinematics.suspensions.implementations.template_provider import (
    TemplateSuspensionProvider,
)
from kinematics.suspensions.templates.base import ComponentSpec, SuspensionTemplate
from kinematics.suspensions.templates.library import (
    DOUBLE_WISHBONE_TEMPLATE,
    build_template_registry,
    get_template,
)
from kinematics.suspensions.templates.validation import (
    ValidationError,
    find_closest_matches,
    format_validation_errors,
    levenshtein_distance,
    validate_hardpoints,
    validate_shim_config,
)

# Test fixtures


@pytest.fixture
def valid_hardpoints() -> dict[str, list[float]]:
    """
    Valid hardpoints for double wishbone suspension.
    """
    return {
        "LOWER_WISHBONE_INBOARD_FRONT": [250, 400, 200],
        "LOWER_WISHBONE_INBOARD_REAR": [-250, 450, 200],
        "LOWER_WISHBONE_OUTBOARD": [0, 900, 200],
        "UPPER_WISHBONE_INBOARD_FRONT": [225, 350, 500],
        "UPPER_WISHBONE_INBOARD_REAR": [-275, 350, 500],
        "UPPER_WISHBONE_OUTBOARD": [-25, 750, 500],
        "TRACKROD_INBOARD": [50, 200, 250],
        "TRACKROD_OUTBOARD": [150, 800, 275],
        "AXLE_INBOARD": [-20, 800, 308.426],
        "AXLE_OUTBOARD": [-20, 950, 313.426],
    }


@pytest.fixture
def valid_config() -> SuspensionConfig:
    """
    Valid suspension configuration.
    """
    return SuspensionConfig(
        steered=True,
        wheel=WheelConfig(
            offset=0,
            tire=TireConfig(
                aspect_ratio=0.55,
                section_width=270,
                rim_diameter=13,
            ),
        ),
        cg_position={"x": 1250, "y": 0, "z": 450},
        wheelbase=2500.0,
        camber_shim=CamberShimConfigOutboard(
            shim_face_center={"x": -25.0, "y": 750.0, "z": 500.0},
            shim_normal={"x": 0.0, "y": 1.0, "z": 0.0},
            design_thickness=30.0,
            setup_thickness=30.0,
        ),
    )


# Test template base classes


class TestSuspensionTemplate:
    """
    Tests for SuspensionTemplate class.
    """

    def test_template_creation(self):
        """
        Test basic template creation.
        """
        template = SuspensionTemplate(
            key="test_template",
            required_point_ids=frozenset({PointID.LOWER_WISHBONE_OUTBOARD}),
            optional_point_ids=frozenset({PointID.PUSHROD_OUTBOARD}),
        )
        assert template.key == "test_template"
        assert PointID.LOWER_WISHBONE_OUTBOARD in template.required_point_ids

    def test_template_rejects_overlap(self):
        """
        Test that template rejects overlapping required/optional sets.
        """
        with pytest.raises(ValueError, match="both required and optional"):
            SuspensionTemplate(
                key="bad_template",
                required_point_ids=frozenset({PointID.LOWER_WISHBONE_OUTBOARD}),
                optional_point_ids=frozenset({PointID.LOWER_WISHBONE_OUTBOARD}),
            )

    def test_all_valid_point_ids(self):
        """
        Test all_valid_point_ids combines required and optional.
        """
        template = SuspensionTemplate(
            key="test",
            required_point_ids=frozenset({PointID.LOWER_WISHBONE_OUTBOARD}),
            optional_point_ids=frozenset({PointID.PUSHROD_OUTBOARD}),
        )
        valid = template.all_valid_point_ids
        assert PointID.LOWER_WISHBONE_OUTBOARD in valid
        assert PointID.PUSHROD_OUTBOARD in valid

    def test_point_id_from_name(self):
        """
        Test converting name strings to PointID.
        """
        template = DOUBLE_WISHBONE_TEMPLATE
        assert (
            template.point_id_from_name("UPPER_WISHBONE_OUTBOARD")
            == PointID.UPPER_WISHBONE_OUTBOARD
        )
        assert (
            template.point_id_from_name("upper_wishbone_outboard")
            == PointID.UPPER_WISHBONE_OUTBOARD
        )
        assert template.point_id_from_name("INVALID_POINT") is None


class TestComponentSpec:
    """
    Tests for ComponentSpec class.
    """

    def test_component_spec_creation(self):
        """
        Test basic component spec creation.
        """
        spec = ComponentSpec(
            name="upright",
            mount_roles={
                "upper_ball_joint": PointID.UPPER_WISHBONE_OUTBOARD,
                "lower_ball_joint": PointID.LOWER_WISHBONE_OUTBOARD,
            },
            attachment_point_ids=[PointID.AXLE_INBOARD, PointID.AXLE_OUTBOARD],
        )
        assert spec.name == "upright"
        assert len(spec.mount_roles) == 2
        assert len(spec.attachment_point_ids) == 2


# Test template library


class TestTemplateLibrary:
    """
    Tests for template library functions.
    """

    def test_build_template_registry(self):
        """
        Test building the template registry.
        """
        registry = build_template_registry()
        assert "double_wishbone" in registry
        assert "double_wishbone_front" in registry  # Alias

    def test_get_template(self):
        """
        Test getting a template by key.
        """
        template = get_template("double_wishbone")
        assert template is not None
        assert template.key == "double_wishbone"

        # Test case insensitivity
        template2 = get_template("DOUBLE_WISHBONE")
        assert template2 is not None
        assert template2.key == "double_wishbone"

    def test_get_template_not_found(self):
        """
        Test getting a non-existent template.
        """
        assert get_template("nonexistent") is None

    def test_double_wishbone_template_completeness(self):
        """
        Test that double wishbone template has all required components.
        """
        template = DOUBLE_WISHBONE_TEMPLATE

        # Check required points
        assert PointID.LOWER_WISHBONE_INBOARD_FRONT in template.required_point_ids
        assert PointID.LOWER_WISHBONE_INBOARD_REAR in template.required_point_ids
        assert PointID.LOWER_WISHBONE_OUTBOARD in template.required_point_ids
        assert PointID.UPPER_WISHBONE_INBOARD_FRONT in template.required_point_ids
        assert PointID.UPPER_WISHBONE_INBOARD_REAR in template.required_point_ids
        assert PointID.UPPER_WISHBONE_OUTBOARD in template.required_point_ids
        assert PointID.TRACKROD_INBOARD in template.required_point_ids
        assert PointID.TRACKROD_OUTBOARD in template.required_point_ids
        assert PointID.AXLE_INBOARD in template.required_point_ids
        assert PointID.AXLE_OUTBOARD in template.required_point_ids

        # Check components
        assert template.get_upright_component() is not None
        assert template.shim_support is True


# Test validation


class TestLevenshteinDistance:
    """
    Tests for Levenshtein distance calculation.
    """

    def test_identical_strings(self):
        """
        Test distance between identical strings is 0.
        """
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_string(self):
        """
        Test distance with empty string.
        """
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "world") == 5

    def test_single_character_difference(self):
        """
        Test single character differences.
        """
        assert levenshtein_distance("hello", "hallo") == 1  # Substitution
        assert levenshtein_distance("hello", "helloo") == 1  # Insertion
        assert levenshtein_distance("hello", "helo") == 1  # Deletion


class TestFindClosestMatches:
    """
    Tests for finding closest matching keys.
    """

    def test_finds_exact_match(self):
        """
        Test finding exact matches.
        """
        valid = {"UPPER_WISHBONE_OUTBOARD", "LOWER_WISHBONE_OUTBOARD"}
        matches = find_closest_matches("UPPER_WISHBONE_OUTBOARD", valid)
        assert "UPPER_WISHBONE_OUTBOARD" in matches

    def test_finds_typo_matches(self):
        """
        Test finding matches for typos.
        """
        valid = {"UPPER_WISHBONE_OUTBOARD", "LOWER_WISHBONE_OUTBOARD"}
        matches = find_closest_matches("UPPER_WISHBONE_OUTBAORD", valid)  # Typo
        assert "UPPER_WISHBONE_OUTBOARD" in matches

    def test_respects_max_distance(self):
        """
        Test that max_distance is respected.
        """
        valid = {"ABCDEF"}
        matches = find_closest_matches("XYZ", valid, max_distance=2)
        assert len(matches) == 0


class TestValidateHardpoints:
    """
    Tests for hardpoint validation.
    """

    def test_valid_hardpoints(self, valid_hardpoints):
        """
        Test validation passes with valid hardpoints.
        """
        errors = validate_hardpoints(valid_hardpoints, DOUBLE_WISHBONE_TEMPLATE)
        assert len(errors) == 0

    def test_missing_required_point(self, valid_hardpoints):
        """
        Test validation catches missing required points.
        """
        del valid_hardpoints["UPPER_WISHBONE_OUTBOARD"]
        errors = validate_hardpoints(valid_hardpoints, DOUBLE_WISHBONE_TEMPLATE)
        assert len(errors) == 1
        assert "Missing required hardpoints" in errors[0].message
        assert "UPPER_WISHBONE_OUTBOARD" in errors[0].message

    def test_unknown_point_with_suggestion(self, valid_hardpoints):
        """
        Test validation suggests corrections for typos.
        """
        valid_hardpoints["UPPER_WISHBONE_OUTBAORD"] = valid_hardpoints.pop(
            "UPPER_WISHBONE_OUTBOARD"
        )
        errors = validate_hardpoints(valid_hardpoints, DOUBLE_WISHBONE_TEMPLATE)

        # Should have both unknown key error and missing required error
        unknown_errors = [e for e in errors if "Unknown" in e.message]
        assert len(unknown_errors) == 1
        assert unknown_errors[0].suggestion is not None
        assert "UPPER_WISHBONE_OUTBOARD" in unknown_errors[0].suggestion

    def test_invalid_triplet_format(self, valid_hardpoints):
        """
        Test validation catches invalid coordinate format.
        """
        valid_hardpoints["UPPER_WISHBONE_OUTBOARD"] = [1, 2]  # Wrong length
        errors = validate_hardpoints(valid_hardpoints, DOUBLE_WISHBONE_TEMPLATE)
        assert any("exactly 3 coordinates" in e.message for e in errors)

    def test_non_numeric_coordinates(self, valid_hardpoints):
        """
        Test validation catches non-numeric coordinates.
        """
        valid_hardpoints["UPPER_WISHBONE_OUTBOARD"] = [1, 2, "three"]
        errors = validate_hardpoints(valid_hardpoints, DOUBLE_WISHBONE_TEMPLATE)
        assert any("must be numeric" in e.message for e in errors)

    def test_dict_format_coordinates(self, valid_hardpoints):
        """
        Test validation accepts dict format coordinates.
        """
        valid_hardpoints["UPPER_WISHBONE_OUTBOARD"] = {"x": 1, "y": 2, "z": 3}
        errors = validate_hardpoints(valid_hardpoints, DOUBLE_WISHBONE_TEMPLATE)
        assert len(errors) == 0


class TestValidateShimConfig:
    """
    Tests for shim configuration validation.
    """

    def test_valid_shim_config(self):
        """
        Test validation passes with valid shim config.
        """
        config = CamberShimConfigOutboard(
            shim_face_center={"x": 0, "y": 1, "z": 2},
            shim_normal={"x": 0, "y": 1, "z": 0},
            design_thickness=30.0,
            setup_thickness=30.0,
        )
        errors = validate_shim_config(config, DOUBLE_WISHBONE_TEMPLATE)
        assert len(errors) == 0

    def test_none_shim_config_ok(self):
        """
        Test that None shim config is valid (shims are optional).
        """
        errors = validate_shim_config(None, DOUBLE_WISHBONE_TEMPLATE)
        assert len(errors) == 0

    def test_near_zero_normal_rejected(self):
        """
        Test that near-zero shim normal is rejected.
        """
        config = {
            "shim_face_center": {"x": 0, "y": 0, "z": 0},
            "shim_normal": {"x": 0, "y": 0, "z": 0},
            "design_thickness": 30.0,
            "setup_thickness": 30.0,
        }
        errors = validate_shim_config(config, DOUBLE_WISHBONE_TEMPLATE)
        assert any("near-zero" in e.message for e in errors)


class TestFormatValidationErrors:
    """
    Tests for error message formatting.
    """

    def test_format_empty_errors(self):
        """
        Test formatting empty error list.
        """
        assert format_validation_errors([]) == ""

    def test_format_single_error(self):
        """
        Test formatting single error.
        """
        errors = [ValidationError("Test error")]
        formatted = format_validation_errors(errors)
        assert "Validation failed" in formatted
        assert "Test error" in formatted

    def test_format_error_with_suggestion(self):
        """
        Test formatting error with suggestion.
        """
        errors = [ValidationError("Unknown key", suggestion="Did you mean X?")]
        formatted = format_validation_errors(errors)
        assert "Did you mean X?" in formatted


# Test template geometry


class TestTemplateGeometry:
    """
    Tests for TemplateGeometry class.
    """

    def test_create_template_geometry(self, valid_hardpoints, valid_config):
        """
        Test creating template geometry.
        """
        geometry = TemplateGeometry(
            name="test",
            version="1.0.0",
            units=Units.MILLIMETERS,
            configuration=valid_config,
            hardpoints=valid_hardpoints,
            template_key="double_wishbone",
        )
        assert geometry.name == "test"
        assert len(geometry.hardpoints) == 10

    def test_get_hardpoints_dict(self, valid_hardpoints, valid_config):
        """
        Test converting hardpoints to hardpoints dict.
        """
        geometry = TemplateGeometry(
            name="test",
            version="1.0.0",
            units=Units.MILLIMETERS,
            configuration=valid_config,
            hardpoints=valid_hardpoints,
            template_key="double_wishbone",
        )
        hardpoints_dict = geometry.get_hardpoints_dict(DOUBLE_WISHBONE_TEMPLATE)

        assert PointID.UPPER_WISHBONE_OUTBOARD in hardpoints_dict
        assert hardpoints_dict[PointID.UPPER_WISHBONE_OUTBOARD].shape == (3,)
        np.testing.assert_array_equal(
            hardpoints_dict[PointID.UPPER_WISHBONE_OUTBOARD],
            np.array([-25, 750, 500]),
        )

    def test_create_template_geometry_factory(self, valid_hardpoints, valid_config):
        """
        Test the factory function for creating geometry.
        """
        geometry = create_template_geometry(
            template=DOUBLE_WISHBONE_TEMPLATE,
            hardpoints=valid_hardpoints,
            configuration=valid_config,
            name="factory_test",
        )
        assert geometry.name == "factory_test"
        assert geometry.template_key == "double_wishbone"


# Test template provider


class TestTemplateSuspensionProvider:
    """
    Tests for TemplateSuspensionProvider class.
    """

    @pytest.fixture
    def provider(self, valid_hardpoints, valid_config) -> TemplateSuspensionProvider:
        """
        Create a template provider for testing.
        """
        geometry = TemplateGeometry(
            name="test",
            version="1.0.0",
            units=Units.MILLIMETERS,
            configuration=valid_config,
            hardpoints=valid_hardpoints,
            template_key="double_wishbone",
        )
        return TemplateSuspensionProvider(geometry, DOUBLE_WISHBONE_TEMPLATE)

    def test_build_upright(self, provider):
        """
        Test building upright from template.
        """
        upright = provider.build_upright()
        assert upright is not None
        assert upright.mount_ids is not None

    def test_initial_state(self, provider):
        """
        Test generating initial state.
        """
        state = provider.initial_state()
        assert state is not None
        assert PointID.UPPER_WISHBONE_OUTBOARD in state.positions
        assert PointID.WHEEL_CENTER in state.positions  # Derived point

    def test_free_points(self, provider):
        """
        Test getting free points.
        """
        free = provider.free_points()
        assert PointID.UPPER_WISHBONE_OUTBOARD in free
        assert PointID.LOWER_WISHBONE_OUTBOARD in free

    def test_constraints(self, provider):
        """
        Test building constraints.
        """
        constraints = provider.constraints()
        assert len(constraints) > 0

    def test_derived_spec(self, provider):
        """
        Test derived point specification.
        """
        spec = provider.derived_spec()
        assert PointID.WHEEL_CENTER in spec.functions
        assert PointID.CONTACT_PATCH_CENTER in spec.functions

    def test_visualization_links(self, provider):
        """
        Test visualization link generation.
        """
        links = provider.get_visualization_links()
        assert len(links) > 0
        # Check for expected link labels
        labels = [link.label for link in links]
        assert "Upper Wishbone" in labels
        assert "Lower Wishbone" in labels


# Test YAML loading


class TestYAMLLoading:
    """
    Tests for loading geometry from YAML files.
    """

    def test_load_template_format_yaml(self, tmp_path):
        """
        Test loading YAML in new template format.
        """
        yaml_content = """
type: double_wishbone
name: "Test"
version: "1.0.0"
units: MILLIMETERS

hardpoints:
  LOWER_WISHBONE_INBOARD_FRONT: [250, 400, 200]
  LOWER_WISHBONE_INBOARD_REAR: [-250, 450, 200]
  LOWER_WISHBONE_OUTBOARD: [0, 900, 200]
  UPPER_WISHBONE_INBOARD_FRONT: [225, 350, 500]
  UPPER_WISHBONE_INBOARD_REAR: [-275, 350, 500]
  UPPER_WISHBONE_OUTBOARD: [-25, 750, 500]
  TRACKROD_INBOARD: [50, 200, 250]
  TRACKROD_OUTBOARD: [150, 800, 275]
  AXLE_INBOARD: [-20, 800, 308.426]
  AXLE_OUTBOARD: [-20, 950, 313.426]

config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 1250, y: 0, z: 450}
  wheelbase: 2500.0
"""
        yaml_file = tmp_path / "test_geometry.yaml"
        yaml_file.write_text(yaml_content)

        result = load_geometry(yaml_file)

        assert isinstance(result, LoadedSuspension)
        assert isinstance(result.geometry, TemplateGeometry)
        assert isinstance(result.provider, TemplateSuspensionProvider)

    def test_load_with_camber_shim(self, tmp_path):
        """
        Test loading YAML with camber shim configuration.
        """
        yaml_content = """
type: double_wishbone
name: "With Shim"
units: MILLIMETERS

hardpoints:
  LOWER_WISHBONE_INBOARD_FRONT: [250, 400, 200]
  LOWER_WISHBONE_INBOARD_REAR: [-250, 450, 200]
  LOWER_WISHBONE_OUTBOARD: [0, 900, 200]
  UPPER_WISHBONE_INBOARD_FRONT: [225, 350, 500]
  UPPER_WISHBONE_INBOARD_REAR: [-275, 350, 500]
  UPPER_WISHBONE_OUTBOARD: [-25, 750, 500]
  TRACKROD_INBOARD: [50, 200, 250]
  TRACKROD_OUTBOARD: [150, 800, 275]
  AXLE_INBOARD: [-20, 800, 308.426]
  AXLE_OUTBOARD: [-20, 950, 313.426]

config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 1250, y: 0, z: 450}
  wheelbase: 2500.0
  camber_shim:
    shim_face_center: {x: -25.0, y: 750.0, z: 500.0}
    shim_normal: {x: 0.0, y: 1.0, z: 0.0}
    design_thickness: 30.0
    setup_thickness: 35.0  # 5mm more than design
"""
        yaml_file = tmp_path / "test_shim.yaml"
        yaml_file.write_text(yaml_content)

        result = load_geometry(yaml_file)
        assert result.geometry.configuration.camber_shim is not None
        assert result.geometry.configuration.camber_shim.setup_thickness == 35.0

    def test_load_rejects_unknown_type(self, tmp_path):
        """
        Test that unknown geometry types are rejected.
        """
        yaml_content = """
type: unknown_suspension
hardpoints: {}
config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 0, y: 0, z: 0}
  wheelbase: 2500.0
"""
        yaml_file = tmp_path / "unknown.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Unsupported geometry type"):
            load_geometry(yaml_file)

    def test_load_rejects_missing_hardpoints(self, tmp_path):
        """
        Test that missing required hardpoints are rejected.
        """
        yaml_content = """
type: double_wishbone
hardpoints:
  LOWER_WISHBONE_INBOARD_FRONT: [250, 400, 200]
  # Missing most required points!

config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 0, y: 0, z: 0}
  wheelbase: 2500.0
"""
        yaml_file = tmp_path / "missing.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Missing required hardpoints"):
            load_geometry(yaml_file)

    def test_load_file_not_found(self):
        """
        Test that FileNotFoundError is raised for missing files.
        """
        with pytest.raises(FileNotFoundError):
            load_geometry(Path("/nonexistent/path.yaml"))


# Integration tests


class TestIntegration:
    """
    Integration tests for the complete template system.
    """

    def test_full_workflow(self, valid_hardpoints, valid_config):
        """
        Test complete workflow from hardpoints to solved state.
        """
        # Create geometry
        geometry = create_template_geometry(
            template=DOUBLE_WISHBONE_TEMPLATE,
            hardpoints=valid_hardpoints,
            configuration=valid_config,
        )

        # Create provider
        provider = TemplateSuspensionProvider(geometry, DOUBLE_WISHBONE_TEMPLATE)

        # Get initial state
        state = provider.initial_state()

        # Verify all required points present
        for point_id in DOUBLE_WISHBONE_TEMPLATE.required_point_ids:
            assert point_id in state.positions

        # Verify derived points calculated
        assert PointID.WHEEL_CENTER in state.positions
        assert PointID.CONTACT_PATCH_CENTER in state.positions

        # Verify constraints can be built
        constraints = provider.constraints()
        assert len(constraints) > 0

    def test_shim_application_workflow(self, valid_hardpoints, valid_config):
        """
        Test workflow with camber shim application.
        """
        # Modify config to have shim effect
        valid_config.camber_shim.setup_thickness = 35.0  # 5mm more than design

        geometry = create_template_geometry(
            template=DOUBLE_WISHBONE_TEMPLATE,
            hardpoints=valid_hardpoints,
            configuration=valid_config,
        )

        provider = TemplateSuspensionProvider(geometry, DOUBLE_WISHBONE_TEMPLATE)
        state = provider.initial_state()

        # Axle points should have moved due to shim
        original_axle = np.array(valid_hardpoints["AXLE_OUTBOARD"])
        new_axle = state.positions[PointID.AXLE_OUTBOARD]

        # Should not be identical (shim rotates attachments)
        assert not np.allclose(original_axle, new_axle)
