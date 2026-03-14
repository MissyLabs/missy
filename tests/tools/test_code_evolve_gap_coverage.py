"""Gap coverage tests for missy/tools/builtin/code_evolve.py.

Targets remaining uncovered lines:
  165-166   : execute() — CodeEvolutionManager init raises
  247       : _propose_multi — ValueError from mgr.propose_multi
  285-286   : _propose_multi — ValueError already covered, ensure it hits line 285-286
  340       : _show — prop.diffs has description (so "Why:" line is appended)
  345       : _show — prop.error_pattern is set
  348       : _show — prop.test_output is set
  390       : _approve — mgr.approve returns False
  460-466   : _apply — success path with restart_process call; apply returns failure dict
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.tools.builtin.code_evolve import CodeEvolveTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal git repo with a fake Missy source file."""
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
def tool():
    return CodeEvolveTool()


def _make_mgr(store_path, repo_root):
    from missy.agent.code_evolution import CodeEvolutionManager

    return CodeEvolutionManager(
        store_path=store_path,
        repo_root=repo_root,
        test_command="true",
    )


# ---------------------------------------------------------------------------
# execute() — CodeEvolutionManager init failure  (lines 165-166)
# ---------------------------------------------------------------------------


class TestExecuteManagerInitFailure:
    def test_manager_init_failure_returns_error_result(self, tool):
        """Lines 165-166: CodeEvolutionManager() raises → ToolResult with error."""
        # The import happens inside execute() as:
        # from missy.agent.code_evolution import CodeEvolutionManager
        # So we patch it at the source module.
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            side_effect=RuntimeError("cannot init"),
        ):
            result = tool.execute(action="list")

        assert not result.success
        assert "Failed to initialize CodeEvolutionManager" in result.error
        assert "cannot init" in result.error


# ---------------------------------------------------------------------------
# _propose_multi — missing required fields  (line 247)
# ---------------------------------------------------------------------------


class TestProposeMultiMissingFields:
    def test_propose_multi_missing_fields_returns_failure(self, tool, tmp_repo, store_path):
        """Line 247: propose_multi called with missing required fields → error ToolResult."""
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=MagicMock(),
        ):
            result = tool.execute(
                action="propose_multi",
                title="Only title provided",
                # description and diffs are missing
            )

        assert not result.success
        assert "Missing required fields" in result.error
        assert "propose_multi" in result.error


# ---------------------------------------------------------------------------
# _propose_multi — ValueError from mgr.propose_multi  (lines 285-286)
# ---------------------------------------------------------------------------


class TestProposeMultiValueError:
    def test_propose_multi_value_error_returns_failure(self, tool, tmp_repo, store_path):
        """Lines 285-286: mgr.propose_multi raises ValueError → ToolResult error."""
        mock_mgr = MagicMock()
        mock_mgr.propose_multi.side_effect = ValueError("file not found in repo")

        diffs_json = json.dumps(
            [
                {
                    "file_path": "missy/example.py",
                    "original_code": "return 'hello'",
                    "proposed_code": "return 'hi'",
                }
            ]
        )

        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            result = tool.execute(
                action="propose_multi",
                title="Test",
                description="desc",
                diffs=diffs_json,
            )

        assert not result.success
        assert "file not found in repo" in result.error


# ---------------------------------------------------------------------------
# _show — diffs with description, error_pattern, test_output  (lines 340, 345, 348)
# ---------------------------------------------------------------------------


class TestShowWithOptionalFields:
    def test_show_diff_with_description_includes_why_line(self, tool, tmp_repo, store_path):
        """Line 340: diff.description is set → 'Why: ...' appended to output."""
        from missy.agent.code_evolution import FileDiff

        mgr = _make_mgr(store_path, str(tmp_repo))
        # Propose with a diff description by using propose_multi.
        prop = mgr.propose_multi(
            title="Show with desc",
            description="Full description",
            diffs=[
                FileDiff(
                    file_path="missy/example.py",
                    original_code="return 'hello'",
                    proposed_code="return 'hi'",
                    description="Changed greeting to hi",
                )
            ],
        )

        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="show", proposal_id=prop.id)

        assert result.success
        assert "Why: Changed greeting to hi" in result.output

    def test_show_includes_error_pattern_when_set(self, tool, tmp_repo, store_path):
        """Line 345: prop.error_pattern is set → included in output."""
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="Error pattern prop",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
            error_pattern="AttributeError: 'NoneType'",
        )

        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="show", proposal_id=prop.id)

        assert result.success
        assert "AttributeError: 'NoneType'" in result.output

    def test_show_includes_test_output_when_set(self, tool, tmp_repo, store_path):
        """Line 348: prop.test_output is set → last 500 chars appended."""
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="Test output prop",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        # Manually set test_output on the proposal via its _proposals list.
        for p in mgr._proposals:
            if p.id == prop.id:
                p.test_output = "PASSED 42 tests"
                break

        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="show", proposal_id=prop.id)

        assert result.success
        assert "PASSED 42 tests" in result.output


# ---------------------------------------------------------------------------
# _approve — mgr.approve returns False  (line 390)
# ---------------------------------------------------------------------------


class TestApproveReturnsFalse:
    def test_approve_returns_false_when_manager_approve_fails(self, tool, tmp_repo, store_path):
        """Line 390: mgr.approve(proposal_id) returns False → failure ToolResult."""

        mgr = MagicMock()
        mock_prop = MagicMock()
        # Status must have a .value attribute equal to "proposed".
        mock_prop.status = MagicMock()
        mock_prop.status.value = "proposed"
        mgr.get.return_value = mock_prop
        mgr.approve.return_value = False  # simulate failure

        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="approve", proposal_id="pid-123")

        assert not result.success
        assert "Failed to approve" in result.error


# ---------------------------------------------------------------------------
# _apply — apply returns failure dict; apply ValueError  (lines 460-466)
# ---------------------------------------------------------------------------


class TestApplyFailurePaths:
    def test_apply_failure_dict_returns_failure_result(self, tool, tmp_repo, store_path):
        """Lines 460-463: mgr.apply returns {'success': False, 'message': '...'} → failure."""
        from missy.agent.code_evolution import EvolutionStatus

        mock_mgr = MagicMock()
        mock_prop = MagicMock()
        # Status must equal the string "approved" for the inequality check to pass.
        mock_prop.status = EvolutionStatus.APPROVED
        mock_mgr.get.return_value = mock_prop
        mock_mgr.apply.return_value = {"success": False, "message": "Tests failed"}

        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            result = tool.execute(action="apply", proposal_id="pid-abc")

        assert not result.success
        assert "Tests failed" in result.error

    def test_apply_value_error_returns_error_result(self, tool, tmp_repo, store_path):
        """Lines 465-466: mgr.apply raises ValueError → ToolResult error."""
        from missy.agent.code_evolution import EvolutionStatus

        mock_mgr = MagicMock()
        mock_prop = MagicMock()
        mock_prop.status = EvolutionStatus.APPROVED
        mock_mgr.get.return_value = mock_prop
        mock_mgr.apply.side_effect = ValueError("proposal not in approved state")

        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mock_mgr,
        ):
            result = tool.execute(action="apply", proposal_id="pid-abc")

        assert not result.success
        assert "proposal not in approved state" in result.error

    def test_apply_success_calls_restart_process(self, tool, tmp_repo, store_path):
        """Lines 453-459: apply succeeds → restart_process called, output contains message."""
        from missy.agent.code_evolution import EvolutionStatus

        mock_mgr = MagicMock()
        mock_prop = MagicMock()
        mock_prop.status = EvolutionStatus.APPROVED
        mock_mgr.get.return_value = mock_prop
        mock_mgr.apply.return_value = {"success": True, "message": "Applied and committed sha123"}

        with (
            patch(
                "missy.agent.code_evolution.CodeEvolutionManager",
                return_value=mock_mgr,
            ),
            patch("missy.agent.code_evolution.restart_process") as mock_restart,
        ):
            result = tool.execute(action="apply", proposal_id="pid-abc")

        assert result.success
        assert "Restarting" in result.output
        mock_restart.assert_called_once()
