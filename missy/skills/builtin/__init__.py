"""Built-in skills shipped with Missy.

All concrete :class:`~missy.skills.base.BaseSkill` subclasses defined in
this package are exported here so that the
:class:`~missy.skills.registry.SkillRegistry` (or any other consumer) can
import them from a single, stable location::

    from missy.skills.builtin import (
        SystemInfoSkill,
        DateTimeSkill,
        ConfigShowSkill,
        HealthCheckSkill,
        SummarizeSessionSkill,
    )
"""

from missy.skills.builtin.config_show import ConfigShowSkill
from missy.skills.builtin.datetime_info import DateTimeSkill
from missy.skills.builtin.health_check import HealthCheckSkill
from missy.skills.builtin.summarize_session import SummarizeSessionSkill
from missy.skills.builtin.system_info import SystemInfoSkill

__all__ = [
    "ConfigShowSkill",
    "DateTimeSkill",
    "HealthCheckSkill",
    "SummarizeSessionSkill",
    "SystemInfoSkill",
]
