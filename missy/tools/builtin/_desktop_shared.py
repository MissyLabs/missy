"""Shared helpers for the desktop/OBS/VTube Studio tool family.

Not a tool module itself -- ``obs_tools.py``, ``vtube_tools.py``, and
``desktop_tools.py`` all need the same two things: a way to load the
operator's real config (since built-in tools get no constructor injection;
see ``missy/tools/builtin/__init__.py``'s ``register_builtin_tools()``) and
a consistent, fail-closed human-confirmation check for destructive/public
actions. Centralized here so the security-relevant fail-closed behavior
can't drift between the three tool files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_missy_config():
    """Return the operator's real :class:`~missy.config.settings.MissyConfig`.

    Uses the same ``MISSY_CONFIG`` env var / ``~/.missy/config.yaml``
    default precedence as the CLI (see ``missy/cli/main.py``'s
    ``DEFAULT_CONFIG``). Returns ``None`` on any failure (missing file,
    invalid YAML) so callers can fail closed rather than raise.
    """
    try:
        from missy.config.settings import load_config

        path = os.environ.get("MISSY_CONFIG", "~/.missy/config.yaml")
        return load_config(str(Path(path).expanduser()))
    except Exception:
        logger.debug("desktop/obs/vtube tools: could not load Missy config", exc_info=True)
        return None


def require_approval(action: str, reason: str, risk: str = "high") -> str | None:
    """Block on :class:`~missy.agent.approval.ApprovalGate` confirmation.

    Fails closed -- returns a denial message (never ``None``) when no gate
    is configured for this session, mirroring
    ``McpManager``'s ``requires_approval`` handling exactly (see
    ``missy/mcp/manager.py``) rather than silently allowing the action
    when confirmation infrastructure is absent.

    Returns:
        ``None`` when approved (proceed); an error string to return in a
        failed :class:`~missy.tools.base.ToolResult` otherwise.
    """
    from missy.agent.approval import get_shared_approval_gate

    gate = get_shared_approval_gate()
    if gate is None:
        return (
            f"Action {action!r} requires human approval but no approval gate is "
            "configured for this session. Run this from an interactive channel "
            "(e.g. Discord) where approvals can be granted."
        )
    try:
        gate.request(action=action, reason=reason, risk=risk)
    except Exception as exc:
        return f"Approval for {action!r} was not granted: {exc}"
    return None
