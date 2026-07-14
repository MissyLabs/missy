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

    # Create a missy package structure
    pkg = repo / "missy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "example.py").write_text("def greet():\n    return 'hello'\n")

    # Initial commit
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
            title="1",
            description="t",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.propose(
            title="2",
            description="t",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        with pytest.raises(ValueError, match="capacity"):
            mgr.propose(
                title="3",
                description="t",
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
            title="T",
            description="D",
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
            title="T",
            description="D",
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
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        assert mgr.reject(prop.id)
        assert mgr.get(prop.id).status == EvolutionStatus.REJECTED

    def test_reject_approved(self, mgr):
        prop = mgr.propose(
            title="T",
            description="D",
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
            cwd=str(tmp_repo),
            capture_output=True,
            text=True,
        )
        assert "[missy-evolve]" in git_log.stdout

    def test_apply_not_approved(self, mgr):
        prop = mgr.propose(
            title="T",
            description="D",
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

    def test_apply_tests_fail_reverts_untracked_file(self, tmp_repo, store_path):
        """Regression: _revert_diffs() used `git checkout -- <path>` alone,
        which only restores files git already has a committed version of.
        For a file that was never committed (created earlier in the same
        session but not yet `git add`/`git commit`ed), that checkout is a
        silent no-op -- with check=False it doesn't even raise -- so the
        broken proposed content was permanently left in place while apply()
        still reported "Tests failed. Changes reverted." Must actually
        restore the original content for an untracked file too.
        """
        (tmp_repo / "missy" / "new_untracked.py").write_text("def foo():\n    return 'ORIGINAL'\n")
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="false",  # always fails
        )
        prop = mgr.propose(
            title="Will fail tests on an untracked file",
            description="test",
            file_path="missy/new_untracked.py",
            original_code="return 'ORIGINAL'",
            proposed_code="return 'BROKEN_SHOULD_BE_REVERTED'",
        )
        mgr.approve(prop.id)
        result = mgr.apply(prop.id)
        assert not result["success"]
        assert "Tests failed" in result["message"]
        content = (tmp_repo / "missy" / "new_untracked.py").read_text()
        assert "ORIGINAL" in content
        assert "BROKEN_SHOULD_BE_REVERTED" not in content

    def test_apply_tests_fail_reverts_untracked_file_multi_diff_same_file(
        self, tmp_repo, store_path
    ):
        """Regression: apply() captured `original_contents[diff.file_path]`
        inside the diff-application loop, keyed only by file_path. When a
        single proposal has two FileDiff entries against the SAME untracked
        file, the second diff's iteration reads the file *after* the first
        diff was already written, overwriting original_contents with that
        intermediate (already-patched) state instead of the true pre-edit
        original. _revert_diffs()'s untracked-file fallback then restores
        that corrupted "original," permanently leaving the first diff's
        edit in place while apply() still reports "Tests failed. Changes
        reverted."
        """
        (tmp_repo / "missy" / "new_untracked.py").write_text(
            "def foo():\n    return 'ORIGINAL_A'\n\ndef bar():\n    return 'ORIGINAL_B'\n"
        )
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="false",  # always fails
        )
        prop = mgr.propose_multi(
            title="Multi-diff same untracked file",
            description="test",
            diffs=[
                FileDiff("missy/new_untracked.py", "return 'ORIGINAL_A'", "return 'BROKEN_A'"),
                FileDiff("missy/new_untracked.py", "return 'ORIGINAL_B'", "return 'BROKEN_B'"),
            ],
        )
        mgr.approve(prop.id)
        result = mgr.apply(prop.id)
        assert not result["success"]
        assert "Tests failed" in result["message"]
        content = (tmp_repo / "missy" / "new_untracked.py").read_text()
        assert "ORIGINAL_A" in content
        assert "ORIGINAL_B" in content
        assert "BROKEN" not in content

    def test_apply_stashes_dirty_work(self, mgr, tmp_repo):
        # Make uncommitted changes to an unrelated file
        (tmp_repo / "missy" / "__init__.py").write_text("# dirty\n")
        prop = mgr.propose(
            title="T",
            description="D",
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

    def test_apply_pops_correct_stash_despite_concurrent_unrelated_stash(self, mgr, tmp_repo):
        """A stash pushed by another process/session between our push and
        pop must not be disturbed, and our own safety stash must still be
        restored correctly (SHA-identity lookup, not position-based
        stash@{0})."""
        # Make uncommitted changes to an unrelated file -- this is what
        # apply() will stash.
        (tmp_repo / "missy" / "__init__.py").write_text("# dirty\n")
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)

        real_stash_if_dirty = mgr._stash_if_dirty

        def stash_then_simulate_concurrent_push():
            sha = real_stash_if_dirty()
            # Simulate a concurrent/interleaved stash from an unrelated
            # process landing on top of ours before we pop.
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "unrelated concurrent commit"],
                cwd=str(tmp_repo),
                capture_output=True,
                check=True,
            )
            (tmp_repo / "concurrent.txt").write_text("unrelated concurrent work\n")
            subprocess.run(
                ["git", "add", "concurrent.txt"], cwd=str(tmp_repo), capture_output=True, check=True
            )
            subprocess.run(
                ["git", "stash", "push", "-m", "unrelated-concurrent-stash"],
                cwd=str(tmp_repo),
                capture_output=True,
                check=True,
            )
            return sha

        mgr._stash_if_dirty = stash_then_simulate_concurrent_push
        try:
            result = mgr.apply(prop.id)
        finally:
            mgr._stash_if_dirty = real_stash_if_dirty

        assert result["success"]
        # Our stashed dirty change was restored correctly.
        content = (tmp_repo / "missy" / "__init__.py").read_text()
        assert "# dirty" in content
        # The unrelated concurrent stash was left untouched on the stack --
        # not popped, not merged, not corrupted.
        stash_list = subprocess.run(
            ["git", "stash", "list"], cwd=str(tmp_repo), capture_output=True, text=True
        ).stdout
        assert "unrelated-concurrent-stash" in stash_list
        # The unrelated stash's content was never applied to the working tree.
        assert not (tmp_repo / "concurrent.txt").exists()

        # Clean up the unrelated stash left on the stack by this test.
        subprocess.run(["git", "stash", "drop"], cwd=str(tmp_repo), capture_output=True, check=True)


class TestApplyTestTimeout:
    """FX-apply-timeout: the previous hardcoded 300s subprocess timeout
    guaranteed every apply() using the real default test_command (the
    entire Missy test suite, observed taking 9-13+ minutes) would always
    time out and fail regardless of whether the proposed change was
    good. Live-confirmed blocking a real Discord-triggered apply.
    test_timeout_seconds is now a real, configurable constructor
    parameter threaded into the subprocess.run() call.
    """

    def test_default_timeout_is_generous(self):
        mgr = CodeEvolutionManager(store_path="/nonexistent/never/written.json")
        assert mgr._test_timeout_seconds == 1200

    def test_custom_timeout_stored(self, tmp_repo, store_path):
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="true",
            test_timeout_seconds=45,
        )
        assert mgr._test_timeout_seconds == 45

    def test_custom_timeout_actually_used_by_apply(self, tmp_repo, store_path):
        """Prove the configured value reaches subprocess.run(), not just
        that the attribute is set -- a command that sleeps longer than a
        short configured timeout must be killed and reported as a
        timeout-flavored failure, not silently ignored."""
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="sleep 5",
            test_timeout_seconds=1,
        )
        prop = mgr.propose(
            title="T",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.approve(prop.id)
        result = mgr.apply(prop.id)
        assert not result["success"]
        assert "timed out" in result["message"].lower() or "timeout" in result["message"].lower()
        # File must still be reverted even on a timeout-triggered failure.
        content = (tmp_repo / "missy" / "example.py").read_text()
        assert "return 'hello'" in content


class TestAutoScopeTests:
    """FX-evolve-scoping: apply()'s only verification gate defaulted to
    running the *entire* Missy test suite on every single-file change,
    even a one-line diff to a leaf module -- correct, but needlessly
    slow (the same 9-13 minute full-suite run FX-apply-timeout above
    exists to survive). auto_scope_tests narrows apply()'s test run to
    just the tests directly associated with the changed file(s),
    falling back to the full suite whenever nothing can be confidently
    associated with a diff.
    """

    def test_auto_scope_tests_default_is_true(self):
        mgr = CodeEvolutionManager(store_path="/nonexistent/never/written.json")
        assert mgr._auto_scope_tests is True

    def test_scoped_test_paths_empty_when_no_tests_dir(self, tmp_repo, store_path):
        mgr = CodeEvolutionManager(store_path=store_path, repo_root=str(tmp_repo))
        diffs = [FileDiff("missy/example.py", "return 'hello'", "return 'hi'")]
        assert mgr._scoped_test_paths(diffs) == []

    def test_scoped_test_paths_naming_convention_match(self, tmp_repo, store_path):
        tests_dir = tmp_repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_example.py").write_text("def test_ok():\n    assert True\n")
        (tests_dir / "test_unrelated.py").write_text("def test_ok():\n    assert True\n")
        mgr = CodeEvolutionManager(store_path=store_path, repo_root=str(tmp_repo))
        diffs = [FileDiff("missy/example.py", "return 'hello'", "return 'hi'")]
        assert mgr._scoped_test_paths(diffs) == ["tests/test_example.py"]

    def test_scoped_test_paths_import_scan_match(self, tmp_repo, store_path):
        """A test module named after something else entirely is still
        picked up if it actually imports the changed module."""
        tests_dir = tmp_repo / "tests" / "tools"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_wrapper.py").write_text(
            "from missy.example import greet\n\ndef test_greet():\n    assert greet()\n"
        )
        mgr = CodeEvolutionManager(store_path=store_path, repo_root=str(tmp_repo))
        diffs = [FileDiff("missy/example.py", "return 'hello'", "return 'hi'")]
        assert mgr._scoped_test_paths(diffs) == ["tests/tools/test_wrapper.py"]

    def test_scoped_test_paths_split_import_form_match(self, tmp_repo, store_path):
        """``from missy import example`` doesn't contain the contiguous
        dotted string "missy.example" -- the parent-dotted + bare-stem
        fallback must still catch it."""
        tests_dir = tmp_repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_wrapper.py").write_text(
            "from missy import example\n\ndef test_greet():\n    assert example.greet()\n"
        )
        mgr = CodeEvolutionManager(store_path=store_path, repo_root=str(tmp_repo))
        diffs = [FileDiff("missy/example.py", "return 'hello'", "return 'hi'")]
        assert mgr._scoped_test_paths(diffs) == ["tests/test_wrapper.py"]

    def test_scoped_test_paths_non_python_diff_contributes_nothing(self, tmp_repo, store_path):
        tests_dir = tmp_repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_example.py").write_text("def test_ok():\n    assert True\n")
        mgr = CodeEvolutionManager(store_path=store_path, repo_root=str(tmp_repo))
        diffs = [FileDiff("missy/config.yaml", "a: 1", "a: 2")]
        assert mgr._scoped_test_paths(diffs) == []

    def test_scoped_test_argv_splices_scoped_paths(self, tmp_repo, store_path):
        tests_dir = tmp_repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_example.py").write_text("def test_ok():\n    assert True\n")
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="python3 -m pytest tests/ -x -q --tb=short",
        )
        diffs = [FileDiff("missy/example.py", "return 'hello'", "return 'hi'")]
        argv = mgr._scoped_test_argv(diffs)
        assert argv == [
            "python3",
            "-m",
            "pytest",
            "tests/test_example.py",
            "-x",
            "-q",
            "--tb=short",
        ]

    def test_scoped_test_argv_falls_back_when_auto_scope_disabled(self, tmp_repo, store_path):
        tests_dir = tmp_repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_example.py").write_text("def test_ok():\n    assert True\n")
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="python3 -m pytest tests/ -x -q --tb=short",
            auto_scope_tests=False,
        )
        diffs = [FileDiff("missy/example.py", "return 'hello'", "return 'hi'")]
        argv = mgr._scoped_test_argv(diffs)
        assert argv == ["python3", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"]

    def test_scoped_test_argv_falls_back_when_nothing_matches(self, tmp_repo, store_path):
        # No tests/ dir at all -> _scoped_test_paths() returns [].
        mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="python3 -m pytest tests/ -x -q --tb=short",
        )
        diffs = [FileDiff("missy/example.py", "return 'hello'", "return 'hi'")]
        argv = mgr._scoped_test_argv(diffs)
        assert argv == ["python3", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"]

    def test_apply_scoped_succeeds_while_unrelated_full_suite_test_would_fail(
        self, tmp_repo, store_path
    ):
        """End-to-end proof, not just argv computation: a genuinely
        failing, unrelated test elsewhere under tests/ must NOT stop a
        scoped apply() from succeeding, because it's never selected to
        run -- while the exact same kind of proposal against the
        unscoped full suite (auto_scope_tests=False) must still fail,
        proving scoping is what made the difference.
        """
        tests_dir = tmp_repo / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_example.py").write_text(
            "def test_greet_is_a_string():\n    assert True\n"
        )
        other_dir = tests_dir / "other"
        other_dir.mkdir()
        (other_dir / "test_unrelated.py").write_text(
            "def test_always_fails():\n    assert False, 'unrelated failure'\n"
        )

        scoped_mgr = CodeEvolutionManager(
            store_path=store_path,
            repo_root=str(tmp_repo),
            test_command="python3 -m pytest tests/ -x -q --tb=short",
            test_timeout_seconds=30,
        )
        prop = scoped_mgr.propose(
            title="Fix greeting",
            description="change hello to hi",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        scoped_mgr.approve(prop.id)
        result = scoped_mgr.apply(prop.id)
        assert result["success"], result["test_output"]
        content = (tmp_repo / "missy" / "example.py").read_text()
        assert "return 'hi'" in content

        # Same kind of proposal, unscoped: the unrelated failing test
        # now runs too and must sink it.
        unscoped_store = str(Path(store_path).parent / "evolutions_unscoped.json")
        unscoped_mgr = CodeEvolutionManager(
            store_path=unscoped_store,
            repo_root=str(tmp_repo),
            test_command="python3 -m pytest tests/ -x -q --tb=short",
            test_timeout_seconds=30,
            auto_scope_tests=False,
        )
        prop2 = unscoped_mgr.propose(
            title="Fix greeting again",
            description="change hi back to hi2",
            file_path="missy/example.py",
            original_code="return 'hi'",
            proposed_code="return 'hi2'",
        )
        unscoped_mgr.approve(prop2.id)
        result2 = unscoped_mgr.apply(prop2.id)
        assert not result2["success"]
        assert "unrelated failure" in result2["test_output"] or "Tests failed" in result2["message"]


class TestStashIdentity:
    """Unit coverage for the SHA-identity-based stash helpers."""

    def test_stash_if_dirty_returns_none_when_clean(self, mgr):
        assert mgr._stash_if_dirty() is None

    def test_stash_if_dirty_returns_none_for_untracked_only_dirty_state(self, mgr, tmp_repo):
        """Regression: `git status --porcelain` reports untracked files as
        dirty, but a plain `git stash push` (no -u) never actually stashes
        them -- it's a no-op. The old code then ran a bare
        `git rev-parse stash@{0}` against the nonexistent stash, which
        writes its "fatal: ambiguous argument..." recovery hint to *stdout*
        ending in the literal text "stash@{0}" -- truthy, and easily
        mistaken by `.strip() or None` for a real stash SHA. Must return
        None (no stash was actually created), not a bogus SHA-shaped string.
        """
        (tmp_repo / "missy" / "brand_new_untracked.py").write_text("# new file\n")
        sha = mgr._stash_if_dirty()
        assert sha is None

    def test_stash_if_dirty_returns_commit_sha(self, mgr, tmp_repo):
        (tmp_repo / "missy" / "__init__.py").write_text("# dirty\n")
        sha = mgr._stash_if_dirty()
        assert sha
        assert len(sha) == 40  # full git commit SHA
        # Working tree should be clean again after the stash push.
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(tmp_repo), capture_output=True, text=True
        ).stdout
        assert status.strip() == ""
        mgr._stash_pop(sha)

    def test_stash_pop_with_none_is_a_no_op(self, mgr, tmp_repo):
        # Should not raise and should not touch any existing stash.
        mgr._stash_pop(None)

    def test_stash_pop_with_unknown_sha_leaves_stack_untouched(self, mgr, tmp_repo):
        (tmp_repo / "missy" / "__init__.py").write_text("# dirty\n")
        subprocess.run(
            ["git", "stash", "push", "-m", "real stash"],
            cwd=str(tmp_repo),
            capture_output=True,
            check=True,
        )
        before = subprocess.run(
            ["git", "stash", "list"], cwd=str(tmp_repo), capture_output=True, text=True
        ).stdout

        mgr._stash_pop("0" * 40)  # SHA that doesn't exist on the stack

        after = subprocess.run(
            ["git", "stash", "list"], cwd=str(tmp_repo), capture_output=True, text=True
        ).stdout
        assert before == after
        subprocess.run(["git", "stash", "drop"], cwd=str(tmp_repo), capture_output=True, check=True)


# ---------------------------------------------------------------------------
# CodeEvolutionManager — rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_success(self, mgr, tmp_repo):
        prop = mgr.propose(
            title="T",
            description="D",
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
            title="T",
            description="D",
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
            title="A",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        mgr.propose(
            title="B",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        assert len(mgr.list_all()) == 2

    def test_list_pending(self, mgr):
        p1 = mgr.propose(
            title="A",
            description="D",
            file_path="missy/example.py",
            original_code="return 'hello'",
            proposed_code="return 'hi'",
        )
        p2 = mgr.propose(
            title="B",
            description="D",
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
            title="A",
            description="D",
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
            "some error",
            "traceback with missy/example.py",
            tool_name="shell_exec",
            failure_count=2,
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
            f"Traceback (most recent call last):\n"
            f'  File "{missy_path}", line 2, in greet\n'
            f'    return "hello"\n'
            f"TypeError: bad return type"
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
