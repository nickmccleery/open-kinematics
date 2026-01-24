"""
Suspension type registry.

Provides the template registry mapping type keys to SuspensionTemplate instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinematics.suspensions.templates.library import build_template_registry

if TYPE_CHECKING:
    from kinematics.suspensions.templates.base import SuspensionTemplate

TemplateRegistry = dict[str, "SuspensionTemplate"]


def get_template_registry() -> TemplateRegistry:
    """
    Return the template registry mapping type keys to templates.

    Returns:
        Dictionary mapping lowercase type keys to SuspensionTemplate instances.
    """
    return build_template_registry()
