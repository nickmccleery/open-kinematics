"""
Visualisation script for camber shim effects.

Generates side-by-side comparisons of stock vs. shimmed suspension geometry,
demonstrating how camber shims rotate the upright about the lower ball joint.
"""

from pathlib import Path

from kinematics.io.geometry_loader import load_geometry
from kinematics.suspensions.core.settings import CamberShimConfig
from kinematics.visualization.api import visualize_geometry

# Shim configuration constants
DESIGN_SHIM_THICKNESS = 30.0  # mm - design/baseline shim stack thickness
SETUP_SHIM_THICKNESS = 1.0  # mm - as-built/setup shim stack thickness


def main():
    # Load the base geometry
    geometry_path = Path("tests/data/geometry.yaml")
    loaded = load_geometry(geometry_path)

    # Visualise design configuration
    print("\nGenerating design (baseline) visualisation...")
    design_output = Path("camber_shim_design.png")
    visualize_geometry(loaded.provider, design_output)
    print(f"Design visualisation saved to: {design_output}")

    # Create setup configuration with shim change
    # The shim sits between chassis and upper wishbone bracket
    # The shim face center should be at/near the upper ball joint for maximum effect
    # This represents the mounting face of the upright-side bracket
    # Normal points outboard (positive Y)
    shim_config = CamberShimConfig(
        shim_face_center={
            "x": -25.0,  # Near upper ball joint X
            "y": 750.0,  # Near upper ball joint Y (outboard of inboards)
            "z": 500.0,  # At upper ball joint Z
        },
        shim_normal={
            "x": 0.0,
            "y": 1.0,  # Unit vector pointing outboard
            "z": 0.0,
        },
        design_thickness=DESIGN_SHIM_THICKNESS,
        setup_thickness=SETUP_SHIM_THICKNESS,
    )

    setup_geometry = loaded.geometry
    setup_geometry.configuration.camber_shim = shim_config

    # Re-initialize provider with setup geometry
    provider_class = type(loaded.provider)
    setup_provider = provider_class(setup_geometry)

    # 3. Visualize setup configuration
    shim_delta = SETUP_SHIM_THICKNESS - DESIGN_SHIM_THICKNESS
    print(f"Generating setup ({shim_delta:+.1f}mm shim change) visualization...")
    setup_output = Path("camber_shim_setup.png")
    visualize_geometry(setup_provider, setup_output)
    print(f"Setup visualization saved to: {setup_output}")


if __name__ == "__main__":
    main()
