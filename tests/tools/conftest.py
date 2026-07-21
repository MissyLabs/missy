"""Shared fixtures for tests/tools/.

The desktop/OBS/VTube Studio tool family's rate limiter
(``missy.tools.builtin._desktop_shared``) is process-wide, in-memory state
keyed by tool name -- exactly what makes it effective as a real guardrail
in production, but it means leftover call history from one test would
otherwise leak into the next test's rate-limit budget within the same
pytest process. Reset it before every test in this directory so each test
starts with a clean budget regardless of run order.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_desktop_rate_limits():
    from missy.tools.builtin._desktop_shared import reset_rate_limits

    reset_rate_limits()
    yield
    reset_rate_limits()
