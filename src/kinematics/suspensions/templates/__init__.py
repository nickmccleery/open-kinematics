"""
Suspension templates module.

Templates define the topology, ownership, and validation rules for suspension types,
enabling a unified YAML format while keeping type-specific logic in Python.
"""

from .base import ComponentSpec, SuspensionTemplate
from .library import (
    DOUBLE_WISHBONE_TEMPLATE,
    build_template_registry,
    get_template,
)
from .validation import (
    ValidationError,
    validate_hardpoints,
    validate_shim_config,
)

__all__ = [
    # Base classes.
    "ComponentSpec",
    "SuspensionTemplate",
    # Library.
    "DOUBLE_WISHBONE_TEMPLATE",
    "build_template_registry",
    "get_template",
    # Validation.
    "ValidationError",
    "validate_hardpoints",
    "validate_shim_config",
]
