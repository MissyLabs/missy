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
        assert "missy evolve approve" in result.output

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
            cwd=str(tmp_repo),
            capture_output=True,
        )
        diffs_json = json.dumps(
            [
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
            ]
        )
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
# resolve_filesystem_targets / ToolRegistry permission-check integration
# ---------------------------------------------------------------------------


class TestResolveFilesystemTargets:
    """propose/propose_multi only ever READ the target file(s) -- they
    never write to them (only apply(), unreachable from this tool per
    SR-1.2/1.3, actually mutates source). resolve_filesystem_targets()
    must declare every real file this tool touches as a read path, and
    never as a write path.
    """

    def test_propose_declares_file_path_as_read_only(self, tool):
        read_paths, write_paths = tool.resolve_filesystem_targets({"file_path": "missy/example.py"})
        assert read_paths == ["missy/example.py"]
        assert write_paths == []

    def test_propose_multi_declares_every_diff_path_as_read_only(self, tool):
        diffs_json = json.dumps(
            [
                {"file_path": "missy/a.py", "original_code": "x", "proposed_code": "y"},
                {"file_path": "missy/b.py", "original_code": "x", "proposed_code": "y"},
            ]
        )
        read_paths, write_paths = tool.resolve_filesystem_targets({"diffs": diffs_json})
        assert read_paths == ["missy/a.py", "missy/b.py"]
        assert write_paths == []

    def test_no_kwargs_returns_empty(self, tool):
        assert tool.resolve_filesystem_targets({}) == ([], [])

    def test_malformed_diffs_json_does_not_raise(self, tool):
        read_paths, write_paths = tool.resolve_filesystem_targets({"diffs": "NOT JSON"})
        assert read_paths == []
        assert write_paths == []


class TestRegistryPermissionEnforcement:
    """Regression: CodeEvolveTool previously declared filesystem_write=True,
    so ToolRegistry's generic kwarg-name heuristic ran an unnecessary
    check_write(file_path) against a file the tool never actually writes
    to -- denying the tool's own documented primary use case (proposing a
    single-file fix) under any config where the target file is readable
    but not in allowed_write_paths (e.g. arbitrary repo source, gated
    behind the human-only `missy evolve apply` CLI).
    """

    def _init_engine(self, tmp_repo):
        from missy.config.settings import (
            FilesystemPolicy,
            MissyConfig,
            NetworkPolicy,
            PluginPolicy,
            ShellPolicy,
        )
        from missy.policy import engine as engine_module
        from missy.policy.engine import init_policy_engine

        engine_module._engine = None
        cfg = MissyConfig(
            network=NetworkPolicy(default_deny=True),
            filesystem=FilesystemPolicy(
                allowed_read_paths=[str(tmp_repo)],
                allowed_write_paths=["~/workspace", "~/.missy"],
            ),
            shell=ShellPolicy(),
            plugins=PluginPolicy(),
            providers={},
            workspace_path="/tmp",
            audit_log_path="/tmp/audit.log",
        )
        init_policy_engine(cfg)

    def test_propose_allowed_when_file_readable_but_not_writable(self, tmp_repo, store_path):
        from missy.tools.registry import ToolRegistry

        self._init_engine(tmp_repo)
        registry = ToolRegistry()
        registry.register(CodeEvolveTool())

        with _patch_mgr(store_path, str(tmp_repo)):
            result = registry.execute(
                "code_evolve",
                session_id="s1",
                task_id="t1",
                action="propose",
                title="Fix greeting",
                description="Change hello to hi",
                file_path=str(tmp_repo / "missy" / "example.py"),
                original_code="return 'hello'",
                proposed_code="return 'hi'",
            )
        assert result.success, result.error

    def test_propose_still_denied_when_file_not_readable(self, tmp_repo, store_path):
        """The read-side policy check must still apply -- this fix only
        removes the incorrect write-side check, not enforcement entirely.
        """
        from missy.tools.registry import ToolRegistry

        self._init_engine(tmp_repo)
        registry = ToolRegistry()
        registry.register(CodeEvolveTool())

        with _patch_mgr(store_path, str(tmp_repo)):
            result = registry.execute(
                "code_evolve",
                session_id="s1",
                task_id="t1",
                action="propose",
                title="Fix",
                description="test",
                file_path="/etc/shadow",
                original_code="x",
                proposed_code="y",
            )
        assert not result.success
        assert result.policy_denied

    def test_propose_multi_paths_enforced_not_silently_skipped(self, tmp_repo, store_path):
        """Regression: propose_multi's per-file paths live inside a
        JSON-encoded 'diffs' string, which the registry's generic
        kwarg-name heuristic can't see at all -- pre-fix, this meant NO
        filesystem policy check ever ran for propose_multi. Confirm an
        unreadable path is now actually denied.
        """
        from missy.tools.registry import ToolRegistry

        self._init_engine(tmp_repo)
        registry = ToolRegistry()
        registry.register(CodeEvolveTool())

        diffs_json = json.dumps(
            [{"file_path": "/etc/shadow", "original_code": "x", "proposed_code": "y"}]
        )
        with _patch_mgr(store_path, str(tmp_repo)):
            result = registry.execute(
                "code_evolve",
                session_id="s1",
                task_id="t1",
                action="propose_multi",
                title="T",
                description="D",
                diffs=diffs_json,
            )
        assert not result.success
        assert result.policy_denied


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
# approve / apply / rollback — SR-1.2/1.3: refused unconditionally.
#
# A model must never approve or apply its own code change. These actions
# are only reachable through the human-operator `missy evolve` CLI
# (missy/cli/main.py), which calls CodeEvolutionManager directly and does
# not go through this tool at all. No CodeEvolutionManager patching is
# needed for these tests since the tool refuses before ever constructing
# a manager.
# ---------------------------------------------------------------------------


class TestHumanOperatorOnlyActionsRefused:
    def test_approve_refused_without_touching_manager(self, tool):
        with patch("missy.agent.code_evolution.CodeEvolutionManager") as mock_mgr_cls:
            result = tool.execute(action="approve", proposal_id="abc123")
        assert not result.success
        assert "authenticated human operator" in result.error
        assert "missy evolve approve abc123" in result.error
        mock_mgr_cls.assert_not_called()

    def test_apply_refused_without_touching_manager(self, tool):
        with patch("missy.agent.code_evolution.CodeEvolutionManager") as mock_mgr_cls:
            result = tool.execute(action="apply", proposal_id="abc123")
        assert not result.success
        assert "authenticated human operator" in result.error
        assert "missy evolve apply abc123" in result.error
        mock_mgr_cls.assert_not_called()

    def test_rollback_refused_without_touching_manager(self, tool):
        with patch("missy.agent.code_evolution.CodeEvolutionManager") as mock_mgr_cls:
            result = tool.execute(action="rollback", proposal_id="abc123")
        assert not result.success
        assert "authenticated human operator" in result.error
        assert "missy evolve rollback abc123" in result.error
        mock_mgr_cls.assert_not_called()

    def test_approve_refused_even_without_proposal_id(self, tool):
        result = tool.execute(action="approve")
        assert not result.success
        assert "authenticated human operator" in result.error

    def test_refusal_does_not_suggest_bypass(self, tool):
        result = tool.execute(action="apply", proposal_id="xyz")
        assert not result.success
        assert "do not write" in result.error.lower()

    def test_real_manager_approve_still_has_no_gate_itself(self, tmp_repo, store_path):
        # Documents the actual trust boundary: CodeEvolutionManager.approve()
        # performs no authentication of its own -- the gate lives entirely
        # in the tool's refusal to expose these actions. The `missy evolve`
        # CLI is trusted because it requires a human's own shell session on
        # the host, the same boundary every other local-first Missy control
        # surface relies on.
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        assert mgr.approve(prop.id) is True


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

    def test_reject_is_not_a_human_operator_only_action(self, tool, tmp_repo, store_path):
        # reject only narrows scope (marks a proposal rejected); it does
        # not mutate source or restart the process, so it stays available
        # to the agent unlike approve/apply/rollback.
        mgr = _make_mgr(store_path, str(tmp_repo))
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mgr):
            result = tool.execute(action="reject", proposal_id=prop.id)
        assert result.success


# ---------------------------------------------------------------------------
# unknown action
# ---------------------------------------------------------------------------


class TestUnknownAction:
    def test_unknown_action(self, tool):
        result = tool.execute(action="delete")
        assert not result.success
        assert "Unknown action" in result.error
