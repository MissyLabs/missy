"""Built-in skill: system information.

Reports basic system information (OS, hostname, Python version).
Safe local automation with no network access and no write access.
"""
from __future__ import annotations

import platform
import socket
import sys
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult


class SystemInfoSkill(BaseSkill):
    """Reports safe read-only system information."""

    name = "system_info"
    description = "Report basic system information: OS, hostname, Python version."
    version = "1.0.0"
    permissions = SkillPermissions()  # No special permissions required.

    def execute(self, **kwargs: Any) -> SkillResult:
        """Return a formatted system information string."""
        info = {
            "hostname": socket.gethostname(),
            "os": platform.system(),
            "os_release": platform.release(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        }
        lines = [f"{k}: {v}" for k, v in info.items()]
        return SkillResult(success=True, output="\n".join(lines))
