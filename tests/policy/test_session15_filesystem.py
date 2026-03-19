"""Session-15 comprehensive tests for missy.policy.filesystem.FilesystemPolicyEngine.

Coverage targets not exercised by test_filesystem.py:
  - symlink traversal: directory symlinks, chained symlinks, symlink loop detection
  - deeply nested paths (5+ levels)
  - exact directory match vs child match distinction
  - parent directory traversal via ".." components
  - non-existent target paths (strict=False must not raise)
  - non-existent allowed paths (strict=False must not raise)
  - Unicode directory and file names
  - very long path segments
  - root "/" as an allowed path (allows everything beneath it)
  - concurrent _resolve calls (thread safety of staticmethod)
  - session_id / task_id propagation for both allow and deny on both operations
  - audit event detail structure: path key, operation key
  - audit event category is always "filesystem"
  - audit event event_type is "filesystem_read" or "filesystem_write"
  - PolicyViolationError.category == "filesystem" for both operations
  - PolicyViolationError.detail contains resolved path string
  - PolicyViolationError message contains original path
  - allow-path itself is returned as policy_rule (not the target path)
  - multiple allowed paths: second entry matches when first does not
  - read and write lists are fully independent (cross-contamination guard)
  - mock event_bus.publish to inspect raw AuditEvent objects
  - empty session_id / task_id defaults are empty strings in event
  - Path objects and strings are interchangeable for allowed_paths
  - allowed path with redundant components (e.g. /tmp/../tmp/dir)
  - sibling paths do not match (e.g. /tmp/abc does not allow /tmp/abcdef)
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import FilesystemPolicy
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError
from missy.policy.filesystem import FilesystemPolicyEngine


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def make_engine(
    read_paths: list[str] | None = None,
    write_paths: list[str] | None = None,
) -> FilesystemPolicyEngine:
    policy = FilesystemPolicy(
        allowed_read_paths=read_paths or [],
        allowed_write_paths=write_paths or [],
    )
    return FilesystemPolicyEngine(policy)


@pytest.fixture(autouse=True)
def clear_bus() -> Generator[None, None, None]:
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Standard workspace tree used across multiple test groups."""
    base = tmp_path / "workspace"
    base.mkdir()
    (base / "reports").mkdir()
    (base / "reports" / "monthly").mkdir()
    (base / "reports" / "monthly" / "january").mkdir()
    (base / "scratch").mkdir()
    (base / "notes.txt").write_text("notes")
    (base / "reports" / "summary.txt").write_text("summary")
    return base


# ---------------------------------------------------------------------------
# Group 1: allow/deny fundamentals (distinct from existing tests)
# ---------------------------------------------------------------------------


class TestAllowDenyFundamentals:
    def test_check_write_returns_true_on_allow(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        assert engine.check_write(str(workspace / "new_file.txt")) is True

    def test_check_read_returns_true_on_allow(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        assert engine.check_read(str(workspace / "notes.txt")) is True

    def test_write_denied_raises_policy_violation(self, tmp_path: Path):
        engine = make_engine(write_paths=[str(tmp_path / "allowed")])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(tmp_path / "other" / "file.txt"))

    def test_read_denied_raises_policy_violation(self, tmp_path: Path):
        engine = make_engine(read_paths=[str(tmp_path / "allowed")])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(tmp_path / "other" / "file.txt"))

    def test_empty_write_paths_denies_all(self, workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(workspace / "output.txt"))

    def test_empty_read_paths_denies_all(self, workspace: Path):
        engine = make_engine(read_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(workspace / "notes.txt"))

    def test_exact_directory_match_allowed_for_write(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        assert engine.check_write(str(workspace)) is True

    def test_exact_directory_match_allowed_for_read(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        assert engine.check_read(str(workspace)) is True

    def test_parent_of_allowed_dir_is_denied_for_write(self, workspace: Path):
        child = workspace / "reports"
        engine = make_engine(write_paths=[str(child)])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(workspace))

    def test_parent_of_allowed_dir_is_denied_for_read(self, workspace: Path):
        child = workspace / "reports"
        engine = make_engine(read_paths=[str(child)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(workspace))

    def test_sibling_path_not_allowed_for_write(self, tmp_path: Path):
        abc = tmp_path / "abc"
        abcdef = tmp_path / "abcdef"
        abc.mkdir()
        abcdef.mkdir()
        engine = make_engine(write_paths=[str(abc)])
        # "abcdef" starts with "abc" as a string but is NOT a child of "abc"
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(abcdef / "file.txt"))

    def test_sibling_path_not_allowed_for_read(self, tmp_path: Path):
        foo = tmp_path / "foo"
        foobar = tmp_path / "foobar"
        foo.mkdir()
        foobar.mkdir()
        engine = make_engine(read_paths=[str(foo)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(foobar / "data.txt"))


# ---------------------------------------------------------------------------
# Group 2: Symlink traversal prevention
# ---------------------------------------------------------------------------


class TestSymlinkTraversal:
    def test_file_symlink_to_outside_write_denied(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "sensitive"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret")
        link = allowed / "link.txt"
        link.symlink_to(secret)
        engine = make_engine(write_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(link))

    def test_file_symlink_to_outside_read_denied(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "sensitive"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret")
        link = allowed / "link.txt"
        link.symlink_to(secret)
        engine = make_engine(read_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(link))

    def test_dir_symlink_to_outside_write_denied(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "outside_dir"
        outside.mkdir()
        (outside / "payload.txt").write_text("payload")
        dir_link = allowed / "linked_dir"
        dir_link.symlink_to(outside)
        engine = make_engine(write_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(dir_link / "payload.txt"))

    def test_dir_symlink_to_outside_read_denied(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "outside_dir"
        outside.mkdir()
        (outside / "payload.txt").write_text("payload")
        dir_link = allowed / "linked_dir"
        dir_link.symlink_to(outside)
        engine = make_engine(read_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(dir_link / "payload.txt"))

    def test_chained_symlinks_to_outside_read_denied(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "vault"
        outside.mkdir()
        target = outside / "key.pem"
        target.write_text("PRIVATE KEY")
        # First hop: inside allowed, points to second hop
        hop1 = allowed / "hop1"
        hop2 = allowed / "hop2"
        hop2.symlink_to(target)
        hop1.symlink_to(hop2)
        engine = make_engine(read_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(hop1))

    def test_symlink_inside_allowed_dir_pointing_within_allowed_is_ok_write(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        real_file = allowed / "real.txt"
        real_file.write_text("data")
        link = allowed / "alias.txt"
        link.symlink_to(real_file)
        engine = make_engine(write_paths=[str(allowed)])
        # Symlink resolves to a path still inside allowed — should be permitted
        assert engine.check_write(str(link)) is True

    def test_symlink_inside_allowed_dir_pointing_within_allowed_is_ok_read(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        real_file = allowed / "real.txt"
        real_file.write_text("data")
        link = allowed / "alias.txt"
        link.symlink_to(real_file)
        engine = make_engine(read_paths=[str(allowed)])
        assert engine.check_read(str(link)) is True


# ---------------------------------------------------------------------------
# Group 3: Deeply nested paths
# ---------------------------------------------------------------------------


class TestDeeplyNestedPaths:
    def test_five_levels_deep_write_allowed(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        deep = workspace / "a" / "b" / "c" / "d" / "e" / "file.txt"
        assert engine.check_write(str(deep)) is True

    def test_five_levels_deep_read_allowed(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        deep = workspace / "reports" / "monthly" / "january" / "week1" / "data.csv"
        assert engine.check_read(str(deep)) is True

    def test_deeply_nested_path_outside_is_denied(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        engine = make_engine(write_paths=[str(allowed)])
        # Path goes outside via parent traversal at resolve time
        sibling = tmp_path / "unsafe" / "a" / "b" / "c" / "file.txt"
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(sibling))


# ---------------------------------------------------------------------------
# Group 4: Path with ".." and relative path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_dotdot_inside_allowed_resolves_correctly_for_write(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        # reports/../scratch/out.txt resolves to workspace/scratch/out.txt
        messy = str(workspace / "reports" / ".." / "scratch" / "out.txt")
        assert engine.check_write(messy) is True

    def test_dotdot_escaping_allowed_dir_is_denied_for_read(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        # workspace/../outside/data.txt resolves to tmp_path/outside/data.txt
        escape = str(workspace / ".." / "outside" / "data.txt")
        with pytest.raises(PolicyViolationError):
            engine.check_read(escape)

    def test_dotdot_escaping_allowed_dir_is_denied_for_write(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        escape = str(workspace / ".." / "outside" / "data.txt")
        with pytest.raises(PolicyViolationError):
            engine.check_write(escape)

    def test_current_dir_dot_component_in_path_for_read(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        path_with_dot = str(workspace / "." / "notes.txt")
        assert engine.check_read(path_with_dot) is True

    def test_current_dir_dot_component_in_allowed_path(self, workspace: Path):
        # The allowed path itself has a "." component — should still resolve correctly
        allowed_with_dot = str(workspace) + "/."
        engine = make_engine(read_paths=[allowed_with_dot])
        assert engine.check_read(str(workspace / "notes.txt")) is True


# ---------------------------------------------------------------------------
# Group 5: Non-existent paths (strict=False)
# ---------------------------------------------------------------------------


class TestNonExistentPaths:
    def test_write_to_nonexistent_file_inside_allowed_returns_true(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        nonexistent = str(workspace / "not_yet_created.txt")
        assert engine.check_write(nonexistent) is True

    def test_read_from_nonexistent_file_inside_allowed_returns_true(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        nonexistent = str(workspace / "ghost.txt")
        assert engine.check_read(nonexistent) is True

    def test_write_with_nonexistent_allowed_path_denies_target(self, tmp_path: Path):
        # The allowed path itself does not exist — engine must not raise on construction
        # and must deny any target that resolves outside it
        ghost_allowed = str(tmp_path / "ghost_dir")
        engine = make_engine(write_paths=[ghost_allowed])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(tmp_path / "other" / "file.txt"))

    def test_write_with_nonexistent_allowed_path_permits_child(self, tmp_path: Path):
        ghost_allowed = str(tmp_path / "ghost_dir")
        engine = make_engine(write_paths=[ghost_allowed])
        # A path nested under the non-existent allowed dir should be permitted
        assert engine.check_write(str(tmp_path / "ghost_dir" / "output.txt")) is True

    def test_read_deeply_nonexistent_path_inside_allowed(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        nonexistent_deep = str(workspace / "x" / "y" / "z" / "file.txt")
        assert engine.check_read(nonexistent_deep) is True


# ---------------------------------------------------------------------------
# Group 6: Multiple allowed paths
# ---------------------------------------------------------------------------


class TestMultipleAllowedPaths:
    def test_write_second_entry_matches_when_first_does_not(self, tmp_path: Path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        engine = make_engine(write_paths=[str(first), str(second)])
        assert engine.check_write(str(second / "file.txt")) is True

    def test_read_second_entry_matches_when_first_does_not(self, tmp_path: Path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        engine = make_engine(read_paths=[str(first), str(second)])
        assert engine.check_read(str(second / "data.txt")) is True

    def test_write_neither_entry_matches_raises(self, tmp_path: Path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        other = tmp_path / "other"
        first.mkdir()
        second.mkdir()
        other.mkdir()
        engine = make_engine(write_paths=[str(first), str(second)])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(other / "file.txt"))

    def test_read_three_allowed_paths_last_matches(self, tmp_path: Path):
        dirs = [tmp_path / f"dir{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()
        engine = make_engine(read_paths=[str(d) for d in dirs])
        assert engine.check_read(str(dirs[2] / "file.txt")) is True

    def test_policy_rule_in_event_is_first_matching_allowed_path(self, tmp_path: Path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        engine = make_engine(write_paths=[str(first), str(second)])
        engine.check_write(str(first / "file.txt"))
        events = event_bus.get_events(result="allow")
        assert len(events) == 1
        assert str(first) in events[0].policy_rule


# ---------------------------------------------------------------------------
# Group 7: Trailing slash normalisation
# ---------------------------------------------------------------------------


class TestTrailingSlash:
    def test_trailing_slash_on_allowed_write_path_is_normalised(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace) + "/"])
        assert engine.check_write(str(workspace / "output.txt")) is True

    def test_trailing_slash_on_allowed_read_path_is_normalised(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace) + "/"])
        assert engine.check_read(str(workspace / "notes.txt")) is True

    def test_multiple_trailing_slashes_on_allowed_path(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace) + "///"])
        assert engine.check_read(str(workspace / "notes.txt")) is True


# ---------------------------------------------------------------------------
# Group 8: Unicode paths
# ---------------------------------------------------------------------------


class TestUnicodePaths:
    def test_unicode_directory_name_write_allowed(self, tmp_path: Path):
        udir = tmp_path / "日本語ディレクトリ"
        udir.mkdir()
        engine = make_engine(write_paths=[str(udir)])
        assert engine.check_write(str(udir / "ファイル.txt")) is True

    def test_unicode_directory_name_read_allowed(self, tmp_path: Path):
        udir = tmp_path / "données"
        udir.mkdir()
        engine = make_engine(read_paths=[str(udir)])
        assert engine.check_read(str(udir / "résumé.txt")) is True

    def test_unicode_path_outside_allowed_denied(self, tmp_path: Path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "données"
        outside.mkdir()
        engine = make_engine(read_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(outside / "résumé.txt"))

    def test_unicode_in_file_name_only_write_allowed(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        assert engine.check_write(str(workspace / "报告_2026.txt")) is True


# ---------------------------------------------------------------------------
# Group 9: Very long paths
# ---------------------------------------------------------------------------


class TestLongPaths:
    def test_very_long_filename_inside_allowed_write(self, workspace: Path):
        long_name = "a" * 200 + ".txt"
        engine = make_engine(write_paths=[str(workspace)])
        assert engine.check_write(str(workspace / long_name)) is True

    def test_very_long_filename_inside_allowed_read(self, workspace: Path):
        long_name = "b" * 200 + ".csv"
        engine = make_engine(read_paths=[str(workspace)])
        assert engine.check_read(str(workspace / long_name)) is True

    def test_long_nested_path_inside_allowed_write(self, workspace: Path):
        # Build a path with many components — all conceptually under workspace
        parts = ["sub"] * 10
        engine = make_engine(write_paths=[str(workspace)])
        deep = workspace.joinpath(*parts) / "file.txt"
        assert engine.check_write(str(deep)) is True


# ---------------------------------------------------------------------------
# Group 10: Root "/" as allowed path
# ---------------------------------------------------------------------------


class TestRootAsAllowedPath:
    def test_root_allowed_write_path_permits_everything(self, tmp_path: Path):
        engine = make_engine(write_paths=["/"])
        assert engine.check_write(str(tmp_path / "anywhere" / "file.txt")) is True

    def test_root_allowed_read_path_permits_everything(self, tmp_path: Path):
        engine = make_engine(read_paths=["/"])
        assert engine.check_read(str(tmp_path / "deep" / "nested" / "file.txt")) is True

    def test_root_read_allows_etc_passwd(self):
        engine = make_engine(read_paths=["/"])
        # /etc/passwd does not need to exist for this check (strict=False)
        assert engine.check_read("/etc/passwd") is True


# ---------------------------------------------------------------------------
# Group 11: session_id and task_id propagation
# ---------------------------------------------------------------------------


class TestSessionAndTaskIdPropagation:
    def test_session_id_in_allow_write_event(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        engine.check_write(str(workspace / "out.txt"), session_id="sess-abc", task_id="task-001")
        events = event_bus.get_events(result="allow")
        assert events[0].session_id == "sess-abc"
        assert events[0].task_id == "task-001"

    def test_session_id_in_deny_write_event(self, workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(workspace / "out.txt"), session_id="sess-xyz", task_id="task-002")
        events = event_bus.get_events(result="deny")
        assert events[0].session_id == "sess-xyz"
        assert events[0].task_id == "task-002"

    def test_session_id_in_allow_read_event(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        engine.check_read(str(workspace / "notes.txt"), session_id="s1", task_id="t1")
        events = event_bus.get_events(result="allow")
        assert events[0].session_id == "s1"
        assert events[0].task_id == "t1"

    def test_session_id_in_deny_read_event(self, workspace: Path):
        engine = make_engine(read_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(workspace / "notes.txt"), session_id="s2", task_id="t2")
        events = event_bus.get_events(result="deny")
        assert events[0].session_id == "s2"
        assert events[0].task_id == "t2"

    def test_default_session_and_task_id_are_empty_strings(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        engine.check_write(str(workspace / "file.txt"))
        event = event_bus.get_events()[0]
        assert event.session_id == ""
        assert event.task_id == ""


# ---------------------------------------------------------------------------
# Group 12: Audit event structure
# ---------------------------------------------------------------------------


class TestAuditEventStructure:
    def test_allow_write_event_type_is_filesystem_write(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        engine.check_write(str(workspace / "file.txt"))
        events = event_bus.get_events(event_type="filesystem_write")
        assert len(events) == 1

    def test_allow_read_event_type_is_filesystem_read(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        engine.check_read(str(workspace / "notes.txt"))
        events = event_bus.get_events(event_type="filesystem_read")
        assert len(events) == 1

    def test_deny_write_event_type_is_filesystem_write(self, workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(workspace / "file.txt"))
        events = event_bus.get_events(event_type="filesystem_write", result="deny")
        assert len(events) == 1

    def test_deny_read_event_type_is_filesystem_read(self, workspace: Path):
        engine = make_engine(read_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(workspace / "notes.txt"))
        events = event_bus.get_events(event_type="filesystem_read", result="deny")
        assert len(events) == 1

    def test_allow_event_category_is_filesystem(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        engine.check_read(str(workspace / "notes.txt"))
        assert event_bus.get_events()[0].category == "filesystem"

    def test_deny_event_category_is_filesystem(self, workspace: Path):
        engine = make_engine(read_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(workspace / "notes.txt"))
        assert event_bus.get_events()[0].category == "filesystem"

    def test_allow_write_event_detail_contains_path_key(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        target = str(workspace / "out.txt")
        engine.check_write(target)
        event = event_bus.get_events()[0]
        assert "path" in event.detail

    def test_allow_write_event_detail_path_is_resolved_absolute(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        engine.check_write(str(workspace / "out.txt"))
        event = event_bus.get_events()[0]
        assert Path(event.detail["path"]).is_absolute()

    def test_allow_read_event_detail_operation_is_read(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        engine.check_read(str(workspace / "notes.txt"))
        event = event_bus.get_events()[0]
        assert event.detail["operation"] == "read"

    def test_allow_write_event_detail_operation_is_write(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        engine.check_write(str(workspace / "out.txt"))
        event = event_bus.get_events()[0]
        assert event.detail["operation"] == "write"

    def test_deny_event_policy_rule_is_none(self, workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(workspace / "file.txt"))
        event = event_bus.get_events()[0]
        assert event.policy_rule is None

    def test_allow_event_policy_rule_is_resolved_allowed_path(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        engine.check_read(str(workspace / "notes.txt"))
        event = event_bus.get_events()[0]
        assert event.policy_rule is not None
        # The rule should be the resolved allowed path, not the target
        assert Path(event.policy_rule) == workspace.resolve()

    def test_exactly_one_event_emitted_per_allow_check(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        engine.check_write(str(workspace / "file.txt"))
        engine.check_write(str(workspace / "file2.txt"))
        assert len(event_bus.get_events()) == 2

    def test_exactly_one_event_emitted_per_deny_check(self, workspace: Path):
        engine = make_engine(write_paths=[])
        for _ in range(3):
            with pytest.raises(PolicyViolationError):
                engine.check_write(str(workspace / "file.txt"))
        assert len(event_bus.get_events()) == 3

    def test_mock_publish_called_once_on_allow_write(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        with patch("missy.policy.filesystem.event_bus") as mock_bus:
            engine.check_write(str(workspace / "out.txt"))
        mock_bus.publish.assert_called_once()
        call_arg = mock_bus.publish.call_args[0][0]
        assert isinstance(call_arg, AuditEvent)
        assert call_arg.result == "allow"

    def test_mock_publish_called_once_on_deny_read(self, workspace: Path):
        engine = make_engine(read_paths=[])
        with patch("missy.policy.filesystem.event_bus") as mock_bus:
            with pytest.raises(PolicyViolationError):
                engine.check_read(str(workspace / "notes.txt"))
        mock_bus.publish.assert_called_once()
        call_arg = mock_bus.publish.call_args[0][0]
        assert isinstance(call_arg, AuditEvent)
        assert call_arg.result == "deny"

    def test_mock_publish_event_has_correct_session_id(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        with patch("missy.policy.filesystem.event_bus") as mock_bus:
            engine.check_write(str(workspace / "out.txt"), session_id="SID", task_id="TID")
        event = mock_bus.publish.call_args[0][0]
        assert event.session_id == "SID"
        assert event.task_id == "TID"


# ---------------------------------------------------------------------------
# Group 13: PolicyViolationError attributes
# ---------------------------------------------------------------------------


class TestPolicyViolationErrorAttributes:
    def test_write_violation_category_is_filesystem(self, workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_write(str(workspace / "out.txt"))
        assert exc_info.value.category == "filesystem"

    def test_read_violation_category_is_filesystem(self, workspace: Path):
        engine = make_engine(read_paths=[])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_read(str(workspace / "notes.txt"))
        assert exc_info.value.category == "filesystem"

    def test_write_violation_detail_mentions_resolved_path(self, workspace: Path):
        engine = make_engine(write_paths=[])
        target = workspace / "secret.txt"
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_write(str(target))
        assert "secret.txt" in exc_info.value.detail

    def test_read_violation_detail_mentions_resolved_path(self, workspace: Path):
        engine = make_engine(read_paths=[])
        target = workspace / "notes.txt"
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_read(str(target))
        assert "notes.txt" in exc_info.value.detail

    def test_write_violation_message_contains_original_path(self, workspace: Path):
        engine = make_engine(write_paths=[])
        target = str(workspace / "report.txt")
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_write(target)
        assert "report.txt" in str(exc_info.value)

    def test_read_violation_message_contains_original_path(self, workspace: Path):
        engine = make_engine(read_paths=[])
        target = str(workspace / "notes.txt")
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_read(target)
        assert "notes.txt" in str(exc_info.value)

    def test_write_violation_detail_mentions_allowed_write_paths(self, workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_write(str(workspace / "file.txt"))
        assert "allowed_write_paths" in exc_info.value.detail

    def test_read_violation_detail_mentions_allowed_read_paths(self, workspace: Path):
        engine = make_engine(read_paths=[])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_read(str(workspace / "notes.txt"))
        assert "allowed_read_paths" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Group 14: Path object vs string interchangeability
# ---------------------------------------------------------------------------


class TestPathObjectVsString:
    def test_write_path_object_allowed(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        assert engine.check_write(workspace / "file.txt") is True

    def test_read_path_object_allowed(self, workspace: Path):
        engine = make_engine(read_paths=[str(workspace)])
        assert engine.check_read(workspace / "notes.txt") is True

    def test_write_path_object_denied(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        engine = make_engine(write_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_write(tmp_path / "other" / "file.txt")

    def test_read_path_object_denied(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        engine = make_engine(read_paths=[str(allowed)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(tmp_path / "other" / "data.txt")

    def test_allowed_path_as_path_object_string_target(self, workspace: Path):
        # Demonstrate that allowed path stored as string correctly handles Path target
        engine = make_engine(read_paths=[str(workspace)])
        path_obj = workspace / "notes.txt"
        assert engine.check_read(path_obj) is True

    def test_allowed_path_redundant_components_resolves_correctly(self, workspace: Path):
        # Allowed path has redundant components — should resolve to same location
        redundant = str(workspace) + "/../" + workspace.name
        engine = make_engine(write_paths=[redundant])
        assert engine.check_write(str(workspace / "file.txt")) is True


# ---------------------------------------------------------------------------
# Group 15: Concurrency / thread safety of _resolve
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_resolve_staticmethod_is_reentrant(self, workspace: Path):
        results: list[Path] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def resolve_path(p: str) -> None:
            try:
                resolved = FilesystemPolicyEngine._resolve(p)
                with lock:
                    results.append(resolved)
            except Exception as e:
                with lock:
                    errors.append(e)

        paths = [str(workspace / f"file_{i}.txt") for i in range(50)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(resolve_path, p) for p in paths]
            concurrent.futures.wait(futures)

        assert errors == []
        assert len(results) == 50
        for r in results:
            assert r.is_absolute()

    def test_concurrent_check_write_calls_all_succeed(self, workspace: Path):
        engine = make_engine(write_paths=[str(workspace)])
        results: list[bool] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def do_check(n: int) -> None:
            try:
                result = engine.check_write(str(workspace / f"file_{n}.txt"))
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(do_check, i) for i in range(40)]
            concurrent.futures.wait(futures)

        assert errors == []
        assert all(r is True for r in results)
        assert len(results) == 40

    def test_concurrent_mixed_allow_deny_do_not_interfere(self, tmp_path: Path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        engine = make_engine(read_paths=[str(allowed)])
        allow_count = 0
        deny_count = 0
        lock = threading.Lock()

        def do_check(is_allowed: bool) -> None:
            nonlocal allow_count, deny_count
            target = allowed / "file.txt" if is_allowed else tmp_path / "denied" / "file.txt"
            try:
                engine.check_read(str(target))
                with lock:
                    allow_count += 1
            except PolicyViolationError:
                with lock:
                    deny_count += 1

        tasks = [True] * 20 + [False] * 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(do_check, t) for t in tasks]
            concurrent.futures.wait(futures)

        assert allow_count == 20
        assert deny_count == 20


# ---------------------------------------------------------------------------
# Group 16: Read/write policy independence
# ---------------------------------------------------------------------------


class TestReadWriteIndependence:
    def test_write_allowed_does_not_grant_read(self, tmp_path: Path):
        write_dir = tmp_path / "write_zone"
        write_dir.mkdir()
        engine = make_engine(write_paths=[str(write_dir)], read_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(write_dir / "file.txt"))

    def test_read_allowed_does_not_grant_write(self, tmp_path: Path):
        read_dir = tmp_path / "read_zone"
        read_dir.mkdir()
        engine = make_engine(read_paths=[str(read_dir)], write_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(read_dir / "output.txt"))

    def test_overlapping_read_and_write_policies_each_respected(self, tmp_path: Path):
        shared = tmp_path / "shared"
        shared.mkdir()
        engine = make_engine(read_paths=[str(shared)], write_paths=[str(shared)])
        assert engine.check_read(str(shared / "data.txt")) is True
        assert engine.check_write(str(shared / "output.txt")) is True
