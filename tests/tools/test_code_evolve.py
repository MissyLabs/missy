"""Tests for missy.tools.builtin.code_evolve — the agent-facing code evolution tool."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from missy.tools.builtin.code_evolve import CodeEvolveTool


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal git repo with a fake Missy source file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True, check=True,
    )
    pkg = repo / "missy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "example.py").write_text("def greet():\n    return 'hello'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(repo), capture_output=True, check=True,
    )
    return repo


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "evolutions.json")


@pytest.fixture
def tool():
    return CodeEvolveTool()


def _patch_mgr(store_path, repo_root):
    """Patch CodeEvolutionManager to use test paths."""
    return patch(
        "missy.agent.code_evolution.CodeEvolutionManager",
        return_value=_make_mgr(store_path, repo_root),
    )


def _make_mgr(store_path, repo_root):
    from missy.agent.code_evolution import CodeEvolutionManager
    return CodeEvolutionManager(
        store_path=store_path,
        repo_root=repo_root,
        test_command="true",
    )


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


class TestPropose:
    def test_propose_success(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(
                action="propose",
                title="Fix greeting",
                description="Change hello to hi",
                file_path="missy/example.py",
                original_code="return 'hello'",
                proposed_code="return 'hi'",
            )
        assert result.success
        assert "Evolution proposed" in result.output
        assert "code_evolve(action='approve'" in result.output

    def test_propose_missing_fields(self, tool):
        result = tool.execute(action="propose", title="T")
        assert not result.success
        assert "Missing required fields" in result.error

    def test_propose_invalid_path(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(
                action="propose",
                title="Bad",
                description="test",
                file_path="missy/nonexistent.py",
                original_code="foo",
                proposed_code="bar",
            )
        assert not result.success


# ---------------------------------------------------------------------------
# propose_multi
# ---------------------------------------------------------------------------


class TestProposeMulti:
    def test_propose_multi_success(self, tool, tmp_repo, store_path):
        (tmp_repo / "missy" / "second.py").write_text("val = 42\n")
        subprocess.run(["git", "add", "."], cwd=str(tmp_repo), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add second"],
            cwd=str(tmp_repo), capture_output=True,
        )
        diffs_json = json.dumps([
            {
                "file_path": "missy/example.py",
                "original_code": "return 'hello'",
                "proposed_code": "return 'hi'",
            },
            {
                "file_path": "missy/second.py",
                "original_code": "val = 42",
                "proposed_code": "val = 99",
            },
        ])
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(
                action="propose_multi",
                title="Multi change",
                description="test",
                diffs=diffs_json,
            )
        assert result.success
        assert "Multi-file evolution proposed" in result.output

    def test_propose_multi_bad_json(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(
                action="propose_multi",
                title="T",
                description="D",
                diffs="NOT JSON",
            )
        assert not result.success
        assert "Invalid diffs JSON" in result.error


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_empty(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(action="list")
        assert result.success
        assert "No evolution proposals" in result.output

    def test_list_with_proposals(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        mgr.propose(
            title="Test prop",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="list")
        assert result.success
        assert "Test prop" in result.output


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


class TestShow:
    def test_show_missing_id(self, tool):
        result = tool.execute(action="show")
        assert not result.success

    def test_show_not_found(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(action="show", proposal_id="nope")
        assert not result.success

    def test_show_success(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="Show me",
            description="details here",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="show", proposal_id=prop.id)
        assert result.success
        assert "Show me" in result.output
        assert "details here" in result.output


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------


class TestApprove:
    def test_approve_missing_id(self, tool):
        result = tool.execute(action="approve")
        assert not result.success
        assert "proposal_id is required" in result.error

    def test_approve_not_found(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(action="approve", proposal_id="nope")
        assert not result.success

    def test_approve_success(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="Approve me",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="approve", proposal_id=prop.id)
        assert result.success
        assert "approved" in result.output
        assert "code_evolve(action='apply'" in result.output

    def test_approve_already_approved(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="approve", proposal_id=prop.id)
        assert not result.success
        assert "approved" in result.error


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


class TestReject:
    def test_reject_missing_id(self, tool):
        result = tool.execute(action="reject")
        assert not result.success

    def test_reject_success(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="Reject me",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="reject", proposal_id=prop.id)
        assert result.success
        assert "rejected" in result.output

    def test_reject_not_found(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(action="reject", proposal_id="nope")
        assert not result.success


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_missing_id(self, tool):
        result = tool.execute(action="apply")
        assert not result.success

    def test_apply_not_found(self, tool, tmp_repo, store_path):
        with _patch_mgr(store_path, str(tmp_repo)):
            result = tool.execute(action="apply", proposal_id="nope")
        assert not result.success

    def test_apply_not_approved(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="apply", proposal_id=prop.id)
        assert not result.success
        assert "Approve it first" in result.error

    def test_apply_approved_success(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ), patch(
            "missy.agent.code_evolution.restart_process",
        ):
            result = tool.execute(action="apply", proposal_id=prop.id)
        assert result.success
        assert "committed" in result.output


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_missing_id(self, tool):
        result = tool.execute(action="rollback")
        assert not result.success

    def test_rollback_not_applied(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ):
            result = tool.execute(action="rollback", proposal_id=prop.id)
        assert not result.success

    def test_rollback_success(self, tool, tmp_repo, store_path):
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        mgr.apply(prop.id)
        with patch(
            "missy.agent.code_evolution.CodeEvolutionManager",
            return_value=mgr,
        ), patch(
            "missy.agent.code_evolution.restart_process",
        ):
            result = tool.execute(action="rollback", proposal_id=prop.id)
        assert result.success
        assert "Reverted" in result.output


# ---------------------------------------------------------------------------
# unknown action
# ---------------------------------------------------------------------------


class TestUnknownAction:
    def test_unknown_action(self, tool):
        result = tool.execute(action="delete")
        assert not result.success
        assert "Unknown action" in result.error
