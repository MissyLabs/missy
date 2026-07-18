"""Multipage browser operator console (Web TUI).

Each page module renders one full operator page inside the shared
rack-console layout (:mod:`missy.api.webui.layout`). Pages are served by
:class:`~missy.api.server.ApiServer` at ``/`` (dashboard) and
``/<slug>`` for every other entry in :data:`PAGES`, all behind the same
cookie-session auth + CSRF posture as the JSON API.
"""

from __future__ import annotations

from missy.api.webui import (
    audit,
    dashboard,
    diagnostics,
    logs,
    memory,
    providers,
    scheduler,
    sessions,
)
from missy.api.webui.layout import render_login, render_message, render_shell

# Ordered page registry: slug -> (nav label, module).
PAGES = {
    "dashboard": ("Dashboard", dashboard),
    "memory": ("Memory", memory),
    "audit": ("Audit", audit),
    "logs": ("Logs", logs),
    "diagnostics": ("Diagnostics", diagnostics),
    "providers": ("Providers", providers),
    "scheduler": ("Scheduler", scheduler),
    "sessions": ("Sessions", sessions),
}


def page_exists(slug: str) -> bool:
    """Return True when *slug* names a servable operator page."""
    return slug in PAGES


def render_page(slug: str, *, csrf_token: str) -> str:
    """Render the full HTML document for the page at *slug*."""
    label, module = PAGES[slug]
    return render_shell(
        active=slug,
        title=label,
        csrf_token=csrf_token,
        content=module.content(),
        script=module.script(),
    )


__all__ = [
    "PAGES",
    "page_exists",
    "render_login",
    "render_message",
    "render_page",
]
