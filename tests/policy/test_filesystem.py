"""Tests for missy.policy.filesystem.FilesystemPolicyEngine."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from missy.config.settings import FilesystemPolicy
from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.policy.filesystem import FilesystemPolicyEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_event_bus() -> Generator[None, None, None]:
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    """Return a temporary directory tree for path tests."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = workspace / "output"
    output.mkdir()
    (workspace / "notes.txt").write_text("hello")
    return tmp_path


def make_engine(
    read_paths: list[str] | None = None,
    write_paths: list[str] | None = None,
) -> FilesystemPolicyEngine:
    policy = FilesystemPolicy(
        allowed_read_paths=read_paths or [],
        allowed_write_paths=write_paths or [],
    )
    return FilesystemPolicyEngine(policy)


# ---------------------------------------------------------------------------
# check_write
# ---------------------------------------------------------------------------


class TestCheckWrite:
    def test_write_inside_allowed_path(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace" / "output")
        engine = make_engine(write_paths=[allowed])
        target = str(tmp_workspace / "workspace" / "output" / "result.txt")
        assert engine.check_write(target) is True

    def test_write_to_allowed_dir_itself(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace" / "output")
        engine = make_engine(write_paths=[allowed])
        assert engine.check_write(allowed) is True

    def test_write_outside_allowed_path_denied(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace" / "output")
        engine = make_engine(write_paths=[allowed])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_write(str(tmp_workspace / "workspace" / "notes.txt"))
        assert exc_info.value.category == "filesystem"

    def test_write_to_parent_of_allowed_denied(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace" / "output")
        engine = make_engine(write_paths=[allowed])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(tmp_workspace / "workspace"))

    def test_write_no_allowed_paths_always_denied(self, tmp_workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(tmp_workspace / "workspace" / "output" / "x.txt"))

    def test_write_accepts_path_object(self, tmp_workspace: Path):
        allowed = tmp_workspace / "workspace" / "output"
        engine = make_engine(write_paths=[str(allowed)])
        assert engine.check_write(allowed / "file.txt") is True

    def test_write_allow_emits_event(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace" / "output")
        engine = make_engine(write_paths=[allowed])
        engine.check_write(allowed + "/x.txt")
        events = event_bus.get_events(event_type="filesystem_write", result="allow")
        assert len(events) == 1
        assert events[0].detail["operation"] == "write"

    def test_write_deny_emits_event(self, tmp_workspace: Path):
        engine = make_engine(write_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(tmp_workspace / "notes.txt"))
        events = event_bus.get_events(event_type="filesystem_write", result="deny")
        assert len(events) == 1

    def test_write_policy_rule_in_event(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace" / "output")
        engine = make_engine(write_paths=[allowed])
        engine.check_write(allowed + "/x.txt")
        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule is not None
        assert "output" in event.policy_rule

    def test_write_session_task_in_event(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace" / "output")
        engine = make_engine(write_paths=[allowed])
        engine.check_write(allowed + "/x.txt", session_id="s9", task_id="t3")
        event = event_bus.get_events()[0]
        assert event.session_id == "s9"
        assert event.task_id == "t3"

    def test_write_symlink_resolved(self, tmp_workspace: Path):
        """A symlink outside allowed_write_paths must be denied even if the
        symlink itself lives inside an allowed directory."""
        allowed_dir = tmp_workspace / "workspace" / "output"
        outside_dir = tmp_workspace / "secret"
        outside_dir.mkdir()
        (outside_dir / "data.txt").write_text("secret")

        link_path = allowed_dir / "evil_link"
        link_path.symlink_to(outside_dir / "data.txt")

        engine = make_engine(write_paths=[str(allowed_dir)])
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(link_path))

    def test_write_multiple_allowed_paths(self, tmp_workspace: Path):
        p1 = str(tmp_workspace / "workspace" / "output")
        p2 = str(tmp_workspace / "workspace")
        engine = make_engine(write_paths=[p1, p2])
        # Allowed via p2
        assert engine.check_write(str(tmp_workspace / "workspace" / "notes.txt")) is True


# ---------------------------------------------------------------------------
# check_read
# ---------------------------------------------------------------------------


class TestCheckRead:
    def test_read_inside_allowed_path(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace")
        engine = make_engine(read_paths=[allowed])
        assert engine.check_read(str(tmp_workspace / "workspace" / "notes.txt")) is True

    def test_read_outside_allowed_path_denied(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace")
        engine = make_engine(read_paths=[allowed])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_read(str(tmp_workspace / "other.txt"))
        assert exc_info.value.category == "filesystem"

    def test_read_allows_path_object(self, tmp_workspace: Path):
        allowed = tmp_workspace / "workspace"
        engine = make_engine(read_paths=[str(allowed)])
        assert engine.check_read(allowed / "notes.txt") is True

    def test_read_emits_allow_event(self, tmp_workspace: Path):
        allowed = str(tmp_workspace / "workspace")
        engine = make_engine(read_paths=[allowed])
        engine.check_read(str(tmp_workspace / "workspace" / "notes.txt"))
        events = event_bus.get_events(event_type="filesystem_read")
        assert len(events) == 1
        assert events[0].result == "allow"
        assert events[0].detail["operation"] == "read"

    def test_read_emits_deny_event(self, tmp_workspace: Path):
        engine = make_engine(read_paths=[])
        with pytest.raises(PolicyViolationError):
            engine.check_read("/etc/passwd")
        events = event_bus.get_events(event_type="filesystem_read", result="deny")
        assert len(events) == 1

    def test_read_write_policies_are_independent(self, tmp_path: Path):
        """Allowed write path must not grant read access and vice versa.

        The two allowed trees are siblings (no overlap) to avoid the case where
        the write directory is nested inside the read directory.
        """
        write_only = tmp_path / "write_only"
        read_only = tmp_path / "read_only"
        write_only.mkdir()
        read_only.mkdir()

        engine = FilesystemPolicyEngine(
            FilesystemPolicy(
                allowed_write_paths=[str(write_only)],
                allowed_read_paths=[str(read_only)],
            )
        )
        # Reading a file inside write-only dir is denied (not in read list).
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(write_only / "result.txt"))
        # Writing a file inside read-only dir is denied (not in write list).
        with pytest.raises(PolicyViolationError):
            engine.check_write(str(read_only / "notes.txt"))

    def test_read_symlink_outside_allowed_denied(self, tmp_workspace: Path):
        allowed_dir = tmp_workspace / "workspace"
        outside_dir = tmp_workspace / "secret"
        outside_dir.mkdir()
        (outside_dir / "data.txt").write_text("secret")
        link_path = allowed_dir / "link_to_secret"
        link_path.symlink_to(outside_dir / "data.txt")

        engine = make_engine(read_paths=[str(allowed_dir)])
        with pytest.raises(PolicyViolationError):
            engine.check_read(str(link_path))

    def test_read_absolute_path_resolution(self, tmp_workspace: Path):
        """Relative path components should be resolved before comparison."""
        allowed = str(tmp_workspace / "workspace")
        engine = make_engine(read_paths=[allowed])
        # Build a path with redundant components
        messy = str(tmp_workspace / "workspace" / "." / "notes.txt")
        assert engine.check_read(messy) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFilesystemEdgeCases:
    def test_error_message_contains_path(self, tmp_path: Path):
        engine = make_engine()
        target = str(tmp_path / "forbidden.txt")
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_write(target)
        assert "forbidden.txt" in str(exc_info.value) or "forbidden.txt" in exc_info.value.detail

    def test_allowed_path_trailing_slash_normalised(self, tmp_workspace: Path):
        """Trailing slash on the configured entry should not prevent matching."""
        allowed = str(tmp_workspace / "workspace") + "/"
        engine = make_engine(read_paths=[allowed])
        assert engine.check_read(str(tmp_workspace / "workspace" / "notes.txt")) is True

    def test_no_events_emitted_before_check(self):
        assert event_bus.get_events() == []
