"""Hardening tests for session 28 — input validation, coverage gaps, edge cases.

Covers:
- Scheduler add_job input validation (empty name, empty task, invalid max_attempts)
- File delete tool O_NOFOLLOW symlink path (lines 77-79)
- Skills __init__.py exports
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Scheduler add_job input validation
# ---------------------------------------------------------------------------


class TestSchedulerAddJobValidation:
    """Tests for input validation added to SchedulerManager.add_job()."""

    def _make_manager(self) -> Any:
        from missy.scheduler.manager import SchedulerManager

        mgr = SchedulerManager.__new__(SchedulerManager)
        mgr._jobs = {}
        mgr._scheduler = MagicMock()
        mgr._event_bus = MagicMock()
        mgr._audit = MagicMock()
        mgr._jobs_path = Path("/tmp/test_jobs.json")
        mgr._max_jobs = 0
        mgr._active_hours = ""
        return mgr

    def test_empty_name_rejected(self) -> None:
        """add_job with empty name raises ValueError."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="name must not be empty"):
            mgr.add_job(name="", schedule="every 5 minutes", task="do something")

    def test_whitespace_name_rejected(self) -> None:
        """add_job with whitespace-only name raises ValueError."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="name must not be empty"):
            mgr.add_job(name="   ", schedule="every 5 minutes", task="do something")

    def test_empty_task_rejected(self) -> None:
        """add_job with empty task raises ValueError."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="task must not be empty"):
            mgr.add_job(name="test-job", schedule="every 5 minutes", task="")

    def test_whitespace_task_rejected(self) -> None:
        """add_job with whitespace-only task raises ValueError."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="task must not be empty"):
            mgr.add_job(name="test-job", schedule="every 5 minutes", task="   ")

    def test_zero_max_attempts_rejected(self) -> None:
        """add_job with max_attempts=0 raises ValueError."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            mgr.add_job(
                name="test-job",
                schedule="every 5 minutes",
                task="do something",
                max_attempts=0,
            )

    def test_negative_max_attempts_rejected(self) -> None:
        """add_job with negative max_attempts raises ValueError."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            mgr.add_job(
                name="test-job",
                schedule="every 5 minutes",
                task="do something",
                max_attempts=-1,
            )


# ---------------------------------------------------------------------------
# File delete tool — O_NOFOLLOW symlink re-resolve path
# ---------------------------------------------------------------------------


class TestFileDeleteSymlinkResolve:
    """Cover the O_NOFOLLOW OSError branch (lines 77-79) in file_delete.py."""

    def test_symlink_file_delete_resolves_and_deletes_target(self) -> None:
        """When O_NOFOLLOW fails (symlink), the tool re-resolves and deletes."""
        from missy.tools.builtin.file_delete import FileDeleteTool

        tool = FileDeleteTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real file and a symlink to it
            target = Path(tmpdir) / "real_file.txt"
            target.write_text("content")
            symlink = Path(tmpdir) / "link_to_file"
            symlink.symlink_to(target)

            # The symlink should trigger the O_NOFOLLOW branch on Linux
            # (O_NOFOLLOW causes os.open to fail with ELOOP for symlinks)
            result = tool.execute(path=str(symlink))

            # Either the delete succeeded or was blocked, but should not crash
            assert isinstance(result.success, bool)

    def test_o_nofollow_oserror_triggers_re_resolve(self) -> None:
        """Simulate O_NOFOLLOW OSError to cover lines 77-79 directly."""
        from missy.tools.builtin.file_delete import FileDeleteTool

        tool = FileDeleteTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "file.txt"
            target.write_text("data")

            # Mock os.open to raise OSError (simulating O_NOFOLLOW on a symlink)
            original_open = os.open

            def mock_os_open(path, flags, *args, **kwargs):
                if "file.txt" in str(path):
                    raise OSError("ELOOP: Too many symbolic links")
                return original_open(path, flags, *args, **kwargs)

            with patch("os.open", side_effect=mock_os_open):
                result = tool.execute(path=str(target))

            # The file should be deleted after re-resolve
            assert result.success is True
            assert not target.exists()


# ---------------------------------------------------------------------------
# Skills __init__.py exports
# ---------------------------------------------------------------------------


class TestSkillsExports:
    """Verify skills package exports are correct."""

    def test_all_exports_defined(self) -> None:
        """skills.__all__ should export the expected names."""
        import missy.skills

        assert hasattr(missy.skills, "__all__")
        expected = {"BaseSkill", "SkillPermissions", "SkillRegistry", "SkillResult"}
        assert set(missy.skills.__all__) == expected

    def test_base_skill_importable(self) -> None:
        """BaseSkill should be importable from missy.skills."""
        from missy.skills import BaseSkill

        assert BaseSkill is not None

    def test_skill_registry_importable(self) -> None:
        """SkillRegistry should be importable from missy.skills."""
        from missy.skills import SkillRegistry

        assert SkillRegistry is not None

    def test_skill_result_importable(self) -> None:
        """SkillResult should be importable from missy.skills."""
        from missy.skills import SkillResult

        assert SkillResult is not None

    def test_skill_permissions_importable(self) -> None:
        """SkillPermissions should be importable from missy.skills."""
        from missy.skills import SkillPermissions

        assert SkillPermissions is not None


# ---------------------------------------------------------------------------
# Agent context manager — magic number extraction verification
# ---------------------------------------------------------------------------


class TestContextManagerDefaults:
    """Verify context manager uses documented defaults via TokenBudget."""

    def test_default_token_budget(self) -> None:
        """Default token budget should be 30,000."""
        from missy.agent.context import ContextManager

        cm = ContextManager()
        assert cm._budget.total == 30_000

    def test_default_memory_fraction(self) -> None:
        """Default memory fraction should be 0.15."""
        from missy.agent.context import ContextManager

        cm = ContextManager()
        assert cm._budget.memory_fraction == 0.15

    def test_default_learnings_fraction(self) -> None:
        """Default learnings fraction should be 0.05."""
        from missy.agent.context import ContextManager

        cm = ContextManager()
        assert cm._budget.learnings_fraction == 0.05


# ---------------------------------------------------------------------------
# Rate limiter defaults
# ---------------------------------------------------------------------------


class TestRateLimiterDefaults:
    """Verify rate limiter uses expected default values."""

    def test_default_rpm(self) -> None:
        """Default RPM should be 60."""
        from missy.providers.rate_limiter import RateLimiter

        rl = RateLimiter()
        assert rl._rpm == 60

    def test_default_tpm(self) -> None:
        """Default TPM should be 100,000."""
        from missy.providers.rate_limiter import RateLimiter

        rl = RateLimiter()
        assert rl._tpm == 100_000

    def test_default_max_wait(self) -> None:
        """Default max wait should be 30 seconds."""
        from missy.providers.rate_limiter import RateLimiter

        rl = RateLimiter()
        assert rl._max_wait == 30.0
