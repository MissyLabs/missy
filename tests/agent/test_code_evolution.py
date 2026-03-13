"""Tests for missy.agent.code_evolution — the self-evolving code engine."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from missy.agent.code_evolution import (
    CodeEvolutionManager,
    EvolutionProposal,
    EvolutionStatus,
    EvolutionTrigger,
    FileDiff,
    _deserialize_proposal,
    _serialize_proposal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal git repo with a fake Missy source file."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Init git
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True, check=True,
    )

    # Create a missy package structure
    pkg = repo / "missy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "example.py").write_text(
        "def greet():\n    return 'hello'\n"
    )

    # Initial commit
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
def mgr(tmp_repo, store_path):
    """Manager pointing at the temp repo with a no-op test command."""
    return CodeEvolutionManager(
        store_path=store_path,
        repo_root=str(tmp_repo),
        test_command="true",  # always passes
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self):
        diff = FileDiff("missy/example.py", "old", "new", "fix")
        prop = EvolutionProposal(
            id="abc12345",
            title="Test",
            description="A test proposal",
            diffs=[diff],
            trigger=EvolutionTrigger.USER_REQUEST,
            confidence=0.8,
        )
        serialized = _serialize_proposal(prop)
        restored = _deserialize_proposal(serialized)
        assert restored.id == "abc12345"
        assert restored.title == "Test"
        assert len(restored.diffs) == 1
        assert restored.diffs[0].original_code == "old"
        assert restored.trigger == EvolutionTrigger.USER_REQUEST
        assert restored.status == EvolutionStatus.PROPOSED


# ---------------------------------------------------------------------------
# CodeEvolutionManager — propose
# ---------------------------------------------------------------------------


class TestPropose:
    def test_propose_single_diff(self, mgr, tmp_repo):
        prop = mgr.propose(
            title="Fix greeting",
            description="Change hello to hi",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        assert prop.id
        assert prop.status == EvolutionStatus.PROPOSED
        assert len(prop.diffs) == 1
        assert prop.diffs[0].file_path == "missy/example.py"

    def test_propose_validates_path_inside_package(self, mgr, tmp_repo):
        # Create a file outside missy/
        (tmp_repo / "outside.py").write_text("x = 1")
        with pytest.raises(ValueError, match="outside the Missy package"):
            mgr.propose(
                title="Bad path",
                description="test",
                file_path="outside.py",
                original_code="x = 1",
                proposed_code="x = 2",
            )

    def test_propose_validates_original_code_exists(self, mgr, tmp_repo):
        with pytest.raises(ValueError, match="Original code not found"):
            mgr.propose(
                title="Missing code",
                description="test",
                file_path="missy/example.py",
                original_code="THIS DOES NOT EXIST",
                proposed_code="replacement",
            )

    def test_propose_validates_file_exists(self, mgr, tmp_repo):
        with pytest.raises(ValueError, match="does not exist"):
            mgr.propose(
                title="No file",
                description="test",
                file_path="missy/nonexistent.py",
                original_code="foo",
                proposed_code="bar",
            )

    def test_propose_persists_to_file(self, mgr, store_path):
        mgr.propose(
            title="Persisted",
            description="test",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        data = json.loads(Path(store_path).read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Persisted"

    def test_propose_multi(self, mgr, tmp_repo):
        # Create a second file
        (tmp_repo / "missy" / "second.py").write_text("val = 42\n")
        prop = mgr.propose_multi(
            title="Multi-file change",
            description="test",
            diffs=[
                FileDiff("missy/example.py", "return 'hello'", "return 'hi'"),
                FileDiff("missy/second.py", "val = 42", "val = 99"),
            ],
        )
        assert len(prop.diffs) == 2

    def test_propose_capacity_limit(self, mgr, tmp_repo):
        mgr.MAX_PROPOSALS = 2
        mgr.propose(
            title="1", description="t",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.propose(
            title="2", description="t",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with pytest.raises(ValueError, match="capacity"):
            mgr.propose(
                title="3", description="t",
                file_path="missy/example.py",
                original_code="return 'hello'",
                proposed_code="return 'hi'",
            )


# ---------------------------------------------------------------------------
# CodeEvolutionManager — approve / reject
# ---------------------------------------------------------------------------


class TestApproveReject:
    def test_approve(self, mgr):
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        assert mgr.approve(prop.id)
        assert mgr.get(prop.id).status == EvolutionStatus.APPROVED

    def test_approve_nonexistent(self, mgr):
        assert not mgr.approve("nope")

    def test_approve_already_applied(self, mgr):
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        mgr.apply(prop.id)
        # Can't approve again
        assert not mgr.approve(prop.id)

    def test_reject(self, mgr):
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        assert mgr.reject(prop.id)
        assert mgr.get(prop.id).status == EvolutionStatus.REJECTED

    def test_reject_approved(self, mgr):
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        assert mgr.reject(prop.id)
        assert mgr.get(prop.id).status == EvolutionStatus.REJECTED


# ---------------------------------------------------------------------------
# CodeEvolutionManager — apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_success(self, mgr, tmp_repo):
        prop = mgr.propose(
            title="Fix greeting",
            description="change hello to hi",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        result = mgr.apply(prop.id)
        assert result["success"]
        assert result["commit_sha"]
        # Verify file was modified
        content = (tmp_repo / "missy" / "example.py").read_text()
        assert "return 'hi'" in content
        # Verify commit exists
        git_log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(tmp_repo), capture_output=True, text=True,
        )
        assert "[missy-evolve]" in git_log.stdout

    def test_apply_not_approved(self, mgr):
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with pytest.raises(ValueError, match="not approved"):
            mgr.apply(prop.id)

    def test_apply_tests_fail_reverts(self, tmp_repo, store_path):
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="false",  # always fails
        )
        prop = mgr.propose(
            title="Will fail tests",
            description="test",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'BROKEN'",
        )
        mgr.approve(prop.id)
        result = mgr.apply(prop.id)
        assert not result["success"]
        assert "Tests failed" in result["message"]
        # File should be reverted
        content = (tmp_repo / "missy" / "example.py").read_text()
        assert "return 'hello'" in content
        assert mgr.get(prop.id).status == EvolutionStatus.FAILED

    def test_apply_stashes_dirty_work(self, mgr, tmp_repo):
        # Make uncommitted changes to an unrelated file
        (tmp_repo / "missy" / "__init__.py").write_text("# dirty\n")
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        result = mgr.apply(prop.id)
        assert result["success"]
        # Dirty changes should be restored
        content = (tmp_repo / "missy" / "__init__.py").read_text()
        assert "# dirty" in content

    def test_apply_nonexistent_proposal(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.apply("nope1234")


# ---------------------------------------------------------------------------
# CodeEvolutionManager — rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_success(self, mgr, tmp_repo):
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        mgr.apply(prop.id)
        # Verify change was made
        assert "return 'hi'" in (tmp_repo / "missy" / "example.py").read_text()

        result = mgr.rollback(prop.id)
        assert result["success"]
        # Verify change was reverted
        assert "return 'hello'" in (tmp_repo / "missy" / "example.py").read_text()
        assert mgr.get(prop.id).status == EvolutionStatus.ROLLED_BACK

    def test_rollback_not_applied(self, mgr):
        prop = mgr.propose(
            title="T", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        result = mgr.rollback(prop.id)
        assert not result["success"]

    def test_rollback_nonexistent(self, mgr):
        result = mgr.rollback("nope")
        assert not result["success"]


# ---------------------------------------------------------------------------
# CodeEvolutionManager — queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_list_all(self, mgr):
        mgr.propose(
            title="A", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.propose(
            title="B", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        assert len(mgr.list_all()) == 2

    def test_list_pending(self, mgr):
        p1 = mgr.propose(
            title="A", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        p2 = mgr.propose(
            title="B", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(p1.id)
        mgr.apply(p1.id)
        pending = mgr.list_pending()
        assert len(pending) == 1
        assert pending[0].id == p2.id

    def test_list_applied(self, mgr):
        prop = mgr.propose(
            title="A", description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        mgr.apply(prop.id)
        applied = mgr.list_applied()
        assert len(applied) == 1

    def test_get_nonexistent(self, mgr):
        assert mgr.get("nope") is None


# ---------------------------------------------------------------------------
# CodeEvolutionManager — error analysis
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    def test_ignores_low_failure_count(self, mgr):
        result = mgr.analyze_error_for_evolution(
            "some error", "traceback with missy/example.py",
            tool_name="shell_exec", failure_count=2,
        )
        assert result is None

    def test_ignores_non_missy_traceback(self, mgr):
        result = mgr.analyze_error_for_evolution(
            "some error",
            'File "/usr/lib/python3/something.py", line 1\nError',
            tool_name="shell_exec",
            failure_count=5,
        )
        assert result is None

    def test_creates_skeleton_for_missy_error(self, mgr, tmp_repo):
        missy_path = str(tmp_repo / "missy" / "example.py")
        traceback = (
            f'Traceback (most recent call last):\n'
            f'  File "{missy_path}", line 2, in greet\n'
            f'    return "hello"\n'
            f'TypeError: bad return type'
        )
        result = mgr.analyze_error_for_evolution(
            "TypeError: bad return type",
            traceback,
            tool_name="shell_exec",
            failure_count=5,
        )
        assert result is not None
        assert result.trigger == EvolutionTrigger.REPEATED_ERROR
        assert "example.py" in result.title
        assert result.confidence == 0.0  # Skeleton


# ---------------------------------------------------------------------------
# Persistence reload
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_reload_from_file(self, mgr, store_path, tmp_repo):
        mgr.propose(
            title="Persisted",
            description="test",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        # Create a new manager from the same store
        mgr2 = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="true",
        )
        assert len(mgr2.list_all()) == 1
        assert mgr2.list_all()[0].title == "Persisted"

    def test_handles_corrupt_file(self, store_path, tmp_repo):
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        Path(store_path).write_text("NOT JSON")
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
        )
        assert len(mgr.list_all()) == 0
