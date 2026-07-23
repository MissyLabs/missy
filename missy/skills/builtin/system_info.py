"""Built-in skill: system information.

Reports basic system information (OS, hostname, Python version).
Safe local automation with no network access and no write access.
"""

from __future__ import annotations

import platform
import socket
import sys
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult, reject_unknown_arguments

_MAX_FIELD_CHARS = 256


def _safe_probe(probe: Any) -> str:
    """Run and censor one local metadata probe without affecting other fields."""
    try:
        value = str(probe())
    except Exception:
        return "unavailable"
    try:
        from missy.security.censor import censor_response

        safe = censor_response(value)
        if not isinstance(safe, str):
            raise TypeError("censor returned a non-string")
    except Exception:
        return "unavailable (redaction failed)"
    safe = " ".join(safe.split())
    return safe[:_MAX_FIELD_CHARS] or "unavailable"


class SystemInfoSkill(BaseSkill):
    """Reports safe read-only system information."""

    name = "system_info"
    description = "Report basic system information: OS, hostname, Python version."
    version = "1.0.0"
    permissions = SkillPermissions()  # No special permissions required.

    def execute(self, **kwargs: Any) -> SkillResult:
        """Return a formatted system information string."""
        if error := reject_unknown_arguments(kwargs):
            return error
        info = {
            "scope": "process-visible namespace; not physical-host attestation",
            "hostname": _safe_probe(socket.gethostname),
            "os": _safe_probe(platform.system),
            "os_release": _safe_probe(platform.release),
            "python": _safe_probe(lambda: sys.version.split()[0]),
            "machine": _safe_probe(platform.machine),
        }
        lines = [f"{k}: {v}" for k, v in info.items()]
        return SkillResult(success=True, output="\n".join(lines))
