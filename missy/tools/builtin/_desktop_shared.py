"""Shared helpers for the desktop/OBS/VTube Studio tool family.

Not a tool module itself -- ``obs_tools.py``, ``vtube_tools.py``,
``desktop_tools.py``, ``x11_tools.py``, and ``audio_route.py`` all need the
same handful of things: a way to load the operator's real config (since
built-in tools get no constructor injection; see
``missy/tools/builtin/__init__.py``'s ``register_builtin_tools()``), a
consistent fail-closed human-confirmation check for destructive/public
actions, a per-tool call-rate limiter, and a window-name allowlist check.
Centralized here so this security-relevant behavior can't drift between
the tool files.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
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


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
#
# A minimal in-memory sliding-window limiter, process-wide, keyed by an
# arbitrary string (tool name). Deliberately not the provider-facing
# missy.providers.rate_limiter.RateLimiter -- that class paces outbound API
# call *throughput* (RPM/TPM budgets, retry-after handling) for a paid
# provider connection; this exists purely as an abuse guardrail against a
# runaway loop hammering a local desktop/OBS/VTube Studio action, and fails
# closed (rejects the call) rather than blocking/queuing, since these are
# synchronous tool calls where silently delaying would just look like a
# hang to the caller.

_rate_limit_lock = threading.Lock()
_rate_limit_calls: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(key: str, max_calls: int, window_seconds: float = 60.0) -> str | None:
    """Return an error string if *key* has exceeded *max_calls* in the window.

    Args:
        key: Arbitrary rate-limit bucket identifier -- callers use the tool
            name, so each tool has its own independent budget.
        max_calls: Maximum calls allowed within *window_seconds*. Values
            ``<= 0`` disable the limit entirely (treated as "unlimited",
            matching ``RateLimiter``'s ``0 = unlimited`` convention
            elsewhere in this codebase).
        window_seconds: Rolling window length in seconds.

    Returns:
        ``None`` when the call is within budget (and is recorded as having
        happened); an error string to return in a failed
        :class:`~missy.tools.base.ToolResult` when the limit is exceeded
        (in which case the call is *not* recorded, since it didn't happen).
    """
    if max_calls <= 0:
        return None
    now = time.monotonic()
    with _rate_limit_lock:
        calls = _rate_limit_calls[key]
        while calls and now - calls[0] > window_seconds:
            calls.popleft()
        if len(calls) >= max_calls:
            retry_after = window_seconds - (now - calls[0])
            return (
                f"Rate limit exceeded for {key!r}: {max_calls} calls per "
                f"{int(window_seconds)}s. Try again in {max(retry_after, 0):.0f}s."
            )
        calls.append(now)
        return None


def reset_rate_limits() -> None:
    """Clear all recorded call history. Test-only convenience."""
    with _rate_limit_lock:
        _rate_limit_calls.clear()


# ---------------------------------------------------------------------------
# Window allowlist
# ---------------------------------------------------------------------------


def check_window_allowed(window_name: str) -> str | None:
    """Gate targeting a window by name against ``desktop.window_allowlist``.

    Used by ``desktop_focus_window`` and (when a caller-supplied
    ``window_name`` is given) ``x11_click``/``x11_type``/``x11_key`` --
    those three tools primarily act on whatever window is already focused,
    but their optional ``window_name`` parameter lets a call target an
    *arbitrary* window by name pattern, which is exactly the capability
    ``desktop.window_allowlist`` is meant to bound.

    Matching is case-insensitive substring matching (an allowlist entry
    matches if it appears anywhere in *window_name*), since window titles
    are often dynamic -- e.g. a browser's title includes the current page.

    Unlike ``obs_tools.py``/``vtube_tools.py`` (brand-new capabilities that
    fail closed with no config at all), this check only activates when the
    operator has explicitly set ``desktop.enabled: true`` --
    ``x11_click``/``x11_type``/``x11_key`` predate this guardrail and are
    relied on unrestricted by every deployment that never configured a
    ``desktop:`` section at all; retroactively fail-closing their existing
    ``window_name`` targeting the moment this guardrail shipped would be a
    breaking change for anyone not opting into it, not an additive safety
    net. Once ``desktop.enabled`` is set, the same allowlist/unrestricted
    logic ``desktop_focus_window`` always used applies here too.

    Returns:
        ``None`` when allowed (desktop config disabled/absent, matches the
        allowlist, or ``unrestricted`` is set); an error string when it
        requires -- and doesn't get -- approval.
    """
    cfg = load_missy_config()
    desktop = cfg.desktop if cfg is not None else None
    if desktop is None or not desktop.enabled or desktop.unrestricted:
        return None
    lowered = window_name.lower()
    if any(entry.lower() in lowered for entry in desktop.window_allowlist):
        return None
    return require_approval(
        action=f"Target window {window_name!r}",
        reason=f"{window_name!r} does not match any entry in desktop.window_allowlist.",
        risk="medium",
    )
