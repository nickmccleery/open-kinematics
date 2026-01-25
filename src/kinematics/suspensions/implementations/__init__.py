"""
Suspension provider implementations.

TemplateSuspensionProvider is the canonical entrypoint for template-first workflows.
"""

from kinematics.suspensions.implementations.template_provider import (
    TemplateSuspensionProvider,
)

__all__ = ["TemplateSuspensionProvider"]
