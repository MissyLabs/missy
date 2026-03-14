"""Coverage gap tests for built-in file tools and list_files.

Targets uncovered lines:
  list_files.py:
    68-69  : Path() constructor raises (invalid path)
    96-97  : entry.stat() raises OSError → falls back to no-size format
    105-108: PermissionError on iterdir, generic Exception on iterdir

  file_read.py:
    67-68  : Path() constructor raises (invalid path)
    89-90  : generic Exception in execute

  file_write.py:
    76-77  : Path() constructor raises (invalid path)
    90-91  : generic Exception in execute

  file_delete.py:
    57-58  : Path() constructor raises (invalid path)
    73-74  : generic Exception in execute
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.tools.builtin.file_delete import FileDeleteTool
from missy.tools.builtin.file_read import FileReadTool
from missy.tools.builtin.file_write import FileWriteTool
from missy.tools.builtin.list_files import ListFilesTool

# ---------------------------------------------------------------------------
# ListFilesTool — uncovered lines
# ---------------------------------------------------------------------------


class TestListFilesToolCoverageGaps:
    """Target lines 68-69, 96-97, 105-108 in list_files.py."""

    # Lines 68-69: Path() constructor raises
    def test_invalid_path_construction_returns_error(self):
        """When Path() raises on the given path, a descriptive error is returned."""
        with patch("missy.tools.builtin.list_files.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("bad path bytes")
            result = ListFilesTool().execute(path="\x00invalid")

        assert result.success is False
        assert "Invalid path" in result.error

    # Lines 96-97: entry.stat() raises OSError
    def test_stat_oserror_falls_back_to_no_size_format(self, tmp_path: Path):
        """When stat() raises OSError for a file entry, [file] line has no size."""
        target = tmp_path / "unstat_file.txt"
        target.write_text("content")

        # Build a fake entry that reports is_file()=True but raises OSError on stat().
        # We intercept iterdir() so the tool processes our mock entry instead.
        fake_entry = MagicMock(spec=Path)
        fake_entry.is_dir.return_value = False
        fake_entry.is_file.return_value = True
        fake_entry.name = "unstat_file.txt"
        fake_entry.relative_to.return_value = Path("unstat_file.txt")
        fake_entry.stat.side_effect = OSError("permission denied on stat")

        with patch.object(Path, "iterdir", return_value=[fake_entry]):
            result = ListFilesTool().execute(path=str(tmp_path))

        assert result.success is True
        # File should appear but without the byte count
        assert "unstat_file.txt" in result.output
        assert "bytes" not in result.output

    # Lines 105-106: PermissionError on directory iteration
    def test_permission_error_on_iterdir(self, tmp_path: Path):
        """PermissionError while iterating directory → failure result."""
        target_dir = tmp_path / "restricted"
        target_dir.mkdir()

        with patch.object(Path, "iterdir", side_effect=PermissionError("no access")):
            result = ListFilesTool().execute(path=str(target_dir))

        assert result.success is False
        assert "Permission denied" in result.error

    # Lines 107-108: generic Exception on directory iteration
    def test_generic_exception_on_iterdir(self, tmp_path: Path):
        """Any non-permission exception while iterating → failure result."""
        target_dir = tmp_path / "broken_dir"
        target_dir.mkdir()

        with patch.object(Path, "iterdir", side_effect=OSError("I/O error")):
            result = ListFilesTool().execute(path=str(target_dir))

        assert result.success is False
        assert "I/O error" in result.error

    def test_generic_exception_on_rglob(self, tmp_path: Path):
        """Any exception while calling rglob → failure result."""
        target_dir = tmp_path / "rglob_broken"
        target_dir.mkdir()

        with patch.object(Path, "rglob", side_effect=OSError("rglob failed")):
            result = ListFilesTool().execute(path=str(target_dir), recursive=True)

        assert result.success is False
        assert "rglob failed" in result.error


# ---------------------------------------------------------------------------
# FileReadTool — uncovered lines
# ---------------------------------------------------------------------------


class TestFileReadToolCoverageGaps:
    """Target lines 67-68 and 89-90 in file_read.py."""

    # Lines 67-68: Path() constructor raises
    def test_invalid_path_construction_returns_error(self):
        """When Path() raises on the given path string, a descriptive error is returned."""
        with patch("missy.tools.builtin.file_read.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("null bytes in path")
            result = FileReadTool().execute(path="\x00bad")

        assert result.success is False
        assert "Invalid path" in result.error

    # Lines 89-90: generic Exception in read block
    def test_generic_exception_during_read(self, tmp_path: Path):
        """Non-permission exceptions during stat → failure result (generic except)."""
        target = tmp_path / "readable.txt"
        target.write_text("content")

        # Patch Path.stat to raise a non-PermissionError so the generic
        # except clause (lines 89-90) is exercised
        with patch.object(Path, "stat", side_effect=OSError("disk read error")):
            result = FileReadTool().execute(path=str(target))

        assert result.success is False
        assert "disk read error" in result.error

    def test_permission_error_during_read(self, tmp_path: Path):
        """PermissionError during open → 'Permission denied' in error."""
        target = tmp_path / "perm_file.txt"
        target.write_text("secret")
        target.chmod(0o000)

        try:
            result = FileReadTool().execute(path=str(target))
            assert result.success is False
            assert "Permission denied" in result.error
        finally:
            target.chmod(0o644)


# ---------------------------------------------------------------------------
# FileWriteTool — uncovered lines
# ---------------------------------------------------------------------------


class TestFileWriteToolCoverageGaps:
    """Target lines 76-77 and 90-91 in file_write.py."""

    # Lines 76-77: Path() constructor raises
    def test_invalid_path_construction_returns_error(self):
        """When Path() raises on the given path, a descriptive error is returned."""
        with patch("missy.tools.builtin.file_write.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("invalid path encoding")
            result = FileWriteTool().execute(path="\x00bad", content="data")

        assert result.success is False
        assert "Invalid path" in result.error

    # Lines 90-91: generic Exception in write block
    def test_generic_exception_during_write(self, tmp_path: Path):
        """Non-permission exceptions during Path.open → failure result (generic except)."""
        target = tmp_path / "out.txt"

        # Patch Path.open to raise a non-PermissionError so the generic
        # except clause (lines 90-91) is exercised
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            result = FileWriteTool().execute(path=str(target), content="data")

        assert result.success is False
        assert "disk full" in result.error

    def test_permission_error_during_write(self, tmp_path: Path):
        """PermissionError during write → 'Permission denied' in error."""
        readonly_dir = tmp_path / "ro"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)

        try:
            result = FileWriteTool().execute(path=str(readonly_dir / "blocked.txt"), content="x")
            assert result.success is False
            assert "Permission denied" in result.error
        finally:
            readonly_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# FileDeleteTool — uncovered lines
# ---------------------------------------------------------------------------


class TestFileDeleteToolCoverageGaps:
    """Target lines 57-58 and 73-74 in file_delete.py."""

    # Lines 57-58: Path() constructor raises
    def test_invalid_path_construction_returns_error(self):
        """When Path() raises on the given path, a descriptive error is returned."""
        with patch("missy.tools.builtin.file_delete.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("null byte in path")
            result = FileDeleteTool().execute(path="\x00bad")

        assert result.success is False
        assert "Invalid path" in result.error

    # Lines 73-74: generic Exception in delete block
    def test_generic_exception_during_unlink(self, tmp_path: Path):
        """Non-permission exceptions during unlink → failure result."""
        target = tmp_path / "target.txt"
        target.write_text("data")

        with patch.object(Path, "unlink", side_effect=OSError("filesystem error")):
            result = FileDeleteTool().execute(path=str(target))

        assert result.success is False
        assert "filesystem error" in result.error

    def test_permission_error_during_unlink(self, tmp_path: Path):
        """PermissionError during unlink → 'Permission denied' in error."""
        target = tmp_path / "locked.txt"
        target.write_text("data")
        tmp_path.chmod(0o555)

        try:
            result = FileDeleteTool().execute(path=str(target))
            assert result.success is False
            assert "Permission denied" in result.error
        finally:
            tmp_path.chmod(0o755)
