"""HTML helpers for the local operator console.

Compatibility facade over :mod:`missy.api.webui`, which now renders the
console as a multipage app (dashboard, memory browser, audit trail,
diagnostics, providers, scheduler, sessions). ``render_console`` keeps
serving the dashboard for existing callers.
"""

from __future__ import annotations

from missy.api.webui import render_page
from missy.api.webui.layout import (
    console_css,
    render_login,
    render_message,
    shared_script,
)


def render_console(*, csrf_token: str) -> str:
    """Render the authenticated operator console (dashboard page)."""
    return render_page("dashboard", csrf_token=csrf_token)


def console_script() -> str:
    """Return the dashboard's embedded JavaScript (shared helpers + page)."""
    from missy.api.webui import dashboard

    return shared_script() + dashboard.script()


__all__ = [
    "console_css",
    "console_script",
    "render_console",
    "render_login",
    "render_message",
]
