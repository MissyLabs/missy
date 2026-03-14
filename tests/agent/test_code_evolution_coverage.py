"""Coverage gap tests for missy.agent.code_evolution.

Targets uncovered lines: 58-67, 474-479, 573-581, 610, 630-631,
708-711, 753-754, 776-777.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.code_evolution import (
    CodeEvolutionManager,
    EvolutionStatus,
    FileDiff,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path):
    """Minimal git repo with a fake Missy source file."""
    repo = tmp_path / "repo"
    repo.mkdir()

    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )

    pkg = repo / "missy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "example.py").write_text("def greet():\n    return 'hello'\n")

    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    return repo


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "evolutions.json")


@pytest.fixture
def mgr(tmp_repo, store_path):
    return CodeEvolutionManager(
        store_path=store_path,
        repo_root=str(tmp_repo),
        test_command="true",
    )


# ---------------------------------------------------------------------------
# Lines 58-67: restart_process
# ---------------------------------------------------------------------------


class TestRestartProcess:
    def test_restart_process_calls_execv(self):
        """restart_process calls os.execv with the current interpreter and argv."""
        from missy.agent.code_evolution import restart_process

        with (
            patch("os.execv") as mock_execv,
            patch("sys.argv", ["missy", "run"]),
            patch("sys.executable", "/usr/bin/python3"),
        ):
            restart_process()

        mock_execv.assert_called_once_with("/usr/bin/python3", ["/usr/bin/python3", "missy", "run"])

    def test_restart_process_falls_back_to_sys_exit_on_oserror(self):
        """When os.execv raises OSError, restart_process calls sys.exit(75)."""
        from missy.agent.code_evolution import restart_process

        with (
            patch("os.execv", side_effect=OSError("permission denied")),
            patch("sys.argv", ["missy"]),
            patch("sys.executable", "/usr/bin/python3"),
            pytest.raises(SystemExit) as exc_info,
        ):
            restart_process()

        assert exc_info.value.code == 75


# ---------------------------------------------------------------------------
# Lines 474-479: apply() — diff validation fails after approval
# ---------------------------------------------------------------------------


class TestApplyDiffValidationFailsAfterApproval:
    def test_apply_returns_failure_when_diff_validation_fails(self, mgr, tmp_repo):
        """If the source file changes between approve and apply, validation fails."""
        prop = mgr.propose(
            title="Fix greeting",
            description="change hello",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)

        # Mutate the file so original_code no longer matches
        (tmp_repo / "missy" / "example.py").write_text("def greet():\n    return 'changed'\n")

        result = mgr.apply(prop.id)

        assert result["success"] is False
        assert "Diff validation failed" in result["message"]
        assert mgr.get(prop.id).status == EvolutionStatus.FAILED


# ---------------------------------------------------------------------------
# Lines 573-581: apply() — general exception handler
# ---------------------------------------------------------------------------


class TestApplyGeneralExceptionHandler:
    def test_apply_returns_failure_on_unexpected_exception(self, mgr, tmp_repo):
        """Unexpected exceptions during apply are caught and returned as failure."""
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)

        # Patch _stash_if_dirty to avoid filesystem work, then make subprocess.run raise
        with (
            patch.object(mgr, "_stash_if_dirty", return_value=False),
            patch("subprocess.run", side_effect=RuntimeError("disk on fire")),
        ):
            result = mgr.apply(prop.id)

        assert result["success"] is False
        assert "Application failed" in result["message"]
        assert "disk on fire" in result["test_output"]
        assert mgr.get(prop.id).status == EvolutionStatus.FAILED

    def test_apply_reverts_diffs_on_unexpected_exception(self, mgr, tmp_repo):
        """_revert_diffs is called when apply hits an unexpected exception."""
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)

        revert_called = []

        original_revert = mgr._revert_diffs

        def tracking_revert(diffs):
            revert_called.append(True)
            original_revert(diffs)

        with (
            patch.object(mgr, "_stash_if_dirty", return_value=False),
            patch.object(mgr, "_revert_diffs", side_effect=tracking_revert),
            patch("subprocess.run", side_effect=RuntimeError("unexpected")),
        ):
            mgr.apply(prop.id)

        assert revert_called, "_revert_diffs must be called on exception"


# ---------------------------------------------------------------------------
# Line 610: rollback() — no commit SHA recorded
# ---------------------------------------------------------------------------


class TestRollbackNoCommitSha:
    def test_rollback_returns_failure_when_no_commit_sha(self, mgr, tmp_repo):
        """rollback returns failure message when git_commit_sha is empty."""
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)

        # Force applied status without a real commit sha
        prop.status = EvolutionStatus.APPLIED
        prop.git_commit_sha = ""
        mgr._save()

        result = mgr.rollback(prop.id)

        assert result["success"] is False
        assert "No commit SHA" in result["message"]


# ---------------------------------------------------------------------------
# Lines 630-631: rollback() — CalledProcessError from git revert
# ---------------------------------------------------------------------------


class TestRollbackGitRevertFails:
    def test_rollback_returns_failure_on_git_error(self, mgr, tmp_repo):
        """When git revert fails (CalledProcessError), rollback returns failure."""
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        result_apply = mgr.apply(prop.id)
        assert result_apply["success"]

        # Now rollback but make git revert fail
        error = subprocess.CalledProcessError(1, ["git", "revert"], stderr="conflict")
        with patch.object(mgr, "_git", side_effect=error):
            result = mgr.rollback(prop.id)

        assert result["success"] is False
        assert "git revert failed" in result["message"]


# ---------------------------------------------------------------------------
# Lines 708-711: analyze_error_for_evolution — ValueError from relative_to
# ---------------------------------------------------------------------------


class TestRejectAlreadyApplied:
    def test_reject_returns_false_for_applied_proposal(self, mgr, tmp_repo):
        """reject() returns False when the proposal is in APPLIED status."""
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        mgr.apply(prop.id)
        # Now it's APPLIED — can't reject
        result = mgr.reject(prop.id)
        assert result is False

    def test_reject_returns_false_for_failed_proposal(self, mgr, tmp_repo):
        """reject() returns False when the proposal is in FAILED status."""
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        # Mutate source file so apply fails validation
        (tmp_repo / "missy" / "example.py").write_text("def greet():\n    return 'changed'\n")
        mgr.apply(prop.id)  # FAILED because diff validation fails
        assert mgr.get(prop.id).status == EvolutionStatus.FAILED
        result = mgr.reject(prop.id)
        assert result is False


class TestAnalyzeErrorRelativeToValueError:
    def test_analyze_error_skips_path_when_relative_to_raises(self, mgr, tmp_repo):
        """If Path.relative_to raises ValueError, the file is silently skipped."""
        # Construct a traceback that looks like it's in missy/ but the path
        # will not be relative_to the repo_root (simulated via mock)
        traceback = (
            "Traceback (most recent call last):\n"
            '  File "/some/other/root/missy/module.py", line 10, in func\n'
            '    raise ValueError("oops")\n'
            "ValueError: oops"
        )

        # Make relative_to always raise ValueError
        with patch(
            "missy.agent.code_evolution.Path.relative_to", side_effect=ValueError("not relative")
        ):
            result = mgr.analyze_error_for_evolution(
                error_message="ValueError: oops",
                traceback_text=traceback,
                tool_name="my_tool",
                failure_count=5,
            )

        # All paths were skipped → no Missy files found → None
        assert result is None

    def test_analyze_error_skips_invalid_path_gracefully(self, mgr, tmp_repo):
        """IndexError/ValueError in path extraction is silently skipped."""
        # Traceback line that has 'missy/' but malformed so split fails
        traceback = (
            "Traceback (most recent call last):\n"
            "  File missy/broken_no_quotes, line 1\n"
            "    code\n"
            "TypeError: bad"
        )
        result = mgr.analyze_error_for_evolution(
            error_message="TypeError: bad",
            traceback_text=traceback,
            tool_name="tool",
            failure_count=5,
        )
        # The malformed line is skipped; no valid Missy file found
        assert result is None

    def test_analyze_error_handles_indexerror_in_path_split(self, mgr, tmp_repo):
        """Lines with 'File "' but no closing quote trigger IndexError, which is swallowed."""
        # This line has 'File "' but splits in a way that produces IndexError
        # on the second split operation.
        traceback = 'Traceback (most recent call last):\n  File "missy/noclose\nTypeError: trouble'
        result = mgr.analyze_error_for_evolution(
            error_message="TypeError: trouble",
            traceback_text=traceback,
            tool_name="tool",
            failure_count=5,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Lines 753-754: _revert_diffs — exception in git checkout is swallowed
# ---------------------------------------------------------------------------


class TestRevertDiffsExceptionSwallowed:
    def test_revert_diffs_continues_on_git_exception(self, mgr, tmp_repo):
        """If git checkout fails for a diff, the exception is logged but not raised."""
        diffs = [
            FileDiff("missy/example.py", "return 'hello'", "return 'hi'"),
        ]

        with patch.object(mgr, "_git", side_effect=RuntimeError("checkout failed")):
            # Must not raise
            mgr._revert_diffs(diffs)

    def test_revert_diffs_processes_all_diffs_despite_exception(self, mgr, tmp_repo):
        """All diffs are attempted even if earlier ones raise."""
        (tmp_repo / "missy" / "second.py").write_text("x = 1\n")
        diffs = [
            FileDiff("missy/example.py", "return 'hello'", "return 'hi'"),
            FileDiff("missy/second.py", "x = 1", "x = 2"),
        ]

        call_count = [0]

        def git_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first checkout failed")
            # Second call succeeds
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(mgr, "_git", side_effect=git_side_effect):
            mgr._revert_diffs(diffs)

        # Both diffs were attempted
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Lines 776-777: _emit_event — exception is silently swallowed
# ---------------------------------------------------------------------------


class TestEmitEventExceptionSwallowed:
    def test_emit_event_swallows_exception(self, mgr):
        """_emit_event must not propagate exceptions from event_bus.publish."""
        # event_bus is imported lazily inside _emit_event, so patch via the
        # missy.core.events module that the function imports from.
        mock_bus = MagicMock()
        mock_bus.publish.side_effect = RuntimeError("bus broken")

        mock_events_module = MagicMock()
        mock_events_module.event_bus = mock_bus
        mock_events_module.AuditEvent = MagicMock()
        mock_events_module.AuditEvent.now.return_value = MagicMock()

        with patch.dict("sys.modules", {"missy.core.events": mock_events_module}):
            # Should not raise
            mgr._emit_event("test.event", "allow", detail={"key": "val"})

    def test_emit_event_swallows_import_error(self, mgr):
        """If the event import itself fails, _emit_event swallows it."""
        with patch.dict("sys.modules", {"missy.core.events": None}):
            # Should not raise even if the import breaks
            try:
                mgr._emit_event("test.event", "error", detail={})
            except Exception as exc:
                pytest.fail(f"_emit_event raised unexpectedly: {exc}")
