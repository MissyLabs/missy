"""Suite-wide isolation from the operator's live ``~/.missy`` state.

Production defaults resolve under ``~/.missy`` (memory, checkpoints, audit
log, jobs). Tests run on the same machine as a live gateway, and before this
fixture existed a parallel ``pytest -n auto`` run wrote into the live
``checkpoints.db`` and corrupted it within seconds (premortem item 9).

``HOME`` is redirected here, at import time, so it takes effect before any
test module (or any module-level ``expanduser`` default) is imported. Each
xdist worker is its own process and gets its own throwaway home.
"""

from __future__ import annotations

import os
import shutil
import site
import tempfile
from pathlib import Path

import pytest

_REAL_HOME = os.environ.get("HOME", "")
# Child processes derive the user site-packages from HOME; pin it so they can
# still import the installed packages while every state path is redirected.
os.environ.setdefault("PYTHONUSERBASE", site.getuserbase())
_TEST_HOME = Path(tempfile.mkdtemp(prefix="missy-test-home-"))
# Children that sanitize their environment (dropping PYTHONUSERBASE) still
# derive user site-packages from HOME, so expose only the installed
# libraries; no state directory is shared with the real home.
_real_user_lib = Path(_REAL_HOME, ".local", "lib")
if _REAL_HOME and _real_user_lib.is_dir():
    (_TEST_HOME / ".local").mkdir()
    (_TEST_HOME / ".local" / "lib").symlink_to(_real_user_lib, target_is_directory=True)
os.environ["HOME"] = str(_TEST_HOME)
os.environ.pop("MISSY_CONFIG", None)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    os.environ["HOME"] = _REAL_HOME
    shutil.rmtree(_TEST_HOME, ignore_errors=True)


@pytest.fixture(autouse=True)
def _guard_live_missy_home() -> None:
    """Fail loudly if a test re-points HOME back at the real home directory."""
    yield
    # Compare HOME itself: with HOME unset, expanduser falls back to the passwd
    # entry, which tests that strip the environment trigger legitimately.
    if _REAL_HOME and os.environ.get("HOME") == _REAL_HOME:
        pytest.fail("test restored HOME to the operator's real home directory")
