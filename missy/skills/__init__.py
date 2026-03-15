"""Skills subsystem — domain-specific capabilities for the agent.

Provides the base class, result container, and registry for registering
and discovering skills.
"""

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult
from missy.skills.registry import SkillRegistry

__all__ = [
    "BaseSkill",
    "SkillPermissions",
    "SkillRegistry",
    "SkillResult",
]
