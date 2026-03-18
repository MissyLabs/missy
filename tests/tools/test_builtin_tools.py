"""Comprehensive unit tests for all built-in tools.

Covers:
    - FileReadTool    (missy/tools/builtin/file_read.py)
    - FileWriteTool   (missy/tools/builtin/file_write.py)
    - FileDeleteTool  (missy/tools/builtin/file_delete.py)
    - ListFilesTool   (missy/tools/builtin/list_files.py)
    - WebFetchTool    (missy/tools/builtin/web_fetch.py)
    - SelfCreateTool  (missy/tools/builtin/self_create_tool.py)
    - DiscordUploadTool (missy/tools/builtin/discord_upload.py)
    - TTSSpeakTool / AudioListDevicesTool / AudioSetVolumeTool
      (missy/tools/builtin/tts_speak.py)

Design notes:
    - No real network calls, subprocesses, or audio hardware are invoked.
    - Every external dependency is mocked at the narrowest possible scope.
    - tmp_path is used for all filesystem fixtures; permissions are always
      restored in finally blocks so pytest cleanup can remove the tree.
    - Each test class targets exactly one tool or one logical group of helpers.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Imports — keep at module level so import errors surface immediately.
# ---------------------------------------------------------------------------
from missy.tools.builtin.discord_upload import DiscordUploadTool
from missy.tools.builtin.file_delete import FileDeleteTool
from missy.tools.builtin.file_read import _DEFAULT_ENCODING, _DEFAULT_MAX_BYTES, FileReadTool
from missy.tools.builtin.file_write import FileWriteTool
from missy.tools.builtin.list_files import _DEFAULT_MAX_ENTRIES, ListFilesTool
from missy.tools.builtin.self_create_tool import ALLOWED_LANGUAGES, SelfCreateTool
from missy.tools.builtin.tts_speak import (
    AudioListDevicesTool,
    AudioSetVolumeTool,
    TTSSpeakTool,
    _ensure_runtime_dir,
    _find_piper_model,
    _piper_env,
    _synth_piper,
)
from missy.tools.builtin.web_fetch import _DEFAULT_TIMEOUT, _MAX_RESPONSE_BYTES, WebFetchTool
from missy.tools.base import ToolPermissions, ToolResult


# ===========================================================================
# FileReadTool
# ===========================================================================


class TestFileReadTool:
    """Tests for FileReadTool.execute() and FileReadTool.get_schema()."""

    # ------------------------------------------------------------------
    # Happy-path reads
    # ------------------------------------------------------------------

    def test_read_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "hello.txt"
        target.write_text("hello world", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))

        assert result.success is True
        assert result.output == "hello world"
        assert result.error is None

    def test_read_returns_full_content_under_limit(self, tmp_path: Path) -> None:
        content = "a" * 100
        target = tmp_path / "file.txt"
        target.write_text(content, encoding="utf-8")

        result = FileReadTool().execute(path=str(target), max_bytes=200)

        assert result.success is True
        assert result.output == content

    def test_read_with_encoding_parameter(self, tmp_path: Path) -> None:
        target = tmp_path / "latin.txt"
        target.write_bytes("caf\xe9".encode("latin-1"))

        result = FileReadTool().execute(path=str(target), encoding="latin-1")

        assert result.success is True
        assert "caf" in result.output

    def test_read_file_with_special_characters(self, tmp_path: Path) -> None:
        special = "line1\ttab\nline2\x00null\nline3"
        target = tmp_path / "special.txt"
        # Write bytes directly so the null byte survives.
        target.write_bytes(special.encode("utf-8", errors="replace"))

        result = FileReadTool().execute(path=str(target))

        assert result.success is True
        assert "line1" in result.output
        assert "line3" in result.output

    def test_read_uses_tilde_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "home_file.txt"
        target.write_text("home content", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))

        result = FileReadTool().execute(path="~/home_file.txt")

        assert result.success is True
        assert result.output == "home content"

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def test_read_truncates_at_max_bytes(self, tmp_path: Path) -> None:
        content = "x" * 200
        target = tmp_path / "big.txt"
        target.write_text(content, encoding="utf-8")

        result = FileReadTool().execute(path=str(target), max_bytes=50)

        assert result.success is True
        assert "Truncated" in result.output
        assert "200" in result.output
        assert "50" in result.output

    def test_read_no_truncation_notice_when_within_limit(self, tmp_path: Path) -> None:
        target = tmp_path / "small.txt"
        target.write_text("short", encoding="utf-8")

        result = FileReadTool().execute(path=str(target), max_bytes=1000)

        assert result.success is True
        assert "Truncated" not in result.output

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_read_non_existent_file_returns_error(self, tmp_path: Path) -> None:
        result = FileReadTool().execute(path=str(tmp_path / "ghost.txt"))

        assert result.success is False
        assert result.output is None
        assert "not found" in result.error.lower()

    def test_read_directory_rejected(self, tmp_path: Path) -> None:
        result = FileReadTool().execute(path=str(tmp_path))

        assert result.success is False
        assert result.output is None
        assert "not a file" in result.error.lower()

    def test_read_permission_denied(self, tmp_path: Path) -> None:
        target = tmp_path / "secret.txt"
        target.write_text("classified", encoding="utf-8")
        target.chmod(0o000)
        try:
            result = FileReadTool().execute(path=str(target))
            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            target.chmod(0o644)

    def test_read_path_traversal_attempt_resolved_safely(self, tmp_path: Path) -> None:
        """Path.resolve() normalises traversal sequences; the result is a real path."""
        safe_file = tmp_path / "safe.txt"
        safe_file.write_text("safe", encoding="utf-8")
        # Construct a path with traversal that resolves to the same safe file.
        traversal = str(tmp_path / "subdir" / ".." / "safe.txt")

        result = FileReadTool().execute(path=traversal)

        assert result.success is True
        assert result.output == "safe"

    def test_read_invalid_path_construction_returns_error(self) -> None:
        with patch("missy.tools.builtin.file_read.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("null bytes in path")
            result = FileReadTool().execute(path="\x00bad")

        assert result.success is False
        assert "Invalid path" in result.error

    def test_read_generic_oserror_during_read(self, tmp_path: Path) -> None:
        target = tmp_path / "readable.txt"
        target.write_text("content")

        with patch.object(Path, "stat", side_effect=OSError("disk read error")):
            result = FileReadTool().execute(path=str(target))

        assert result.success is False
        assert "disk read error" in result.error

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def test_get_schema_returns_valid_dict(self) -> None:
        schema = FileReadTool().get_schema()

        assert isinstance(schema, dict)
        assert schema["name"] == "file_read"
        assert isinstance(schema["description"], str)
        assert schema["parameters"]["type"] == "object"

    def test_get_schema_required_fields(self) -> None:
        schema = FileReadTool().get_schema()

        assert schema["parameters"]["required"] == ["path"]

    def test_get_schema_properties(self) -> None:
        props = FileReadTool().get_schema()["parameters"]["properties"]

        assert "path" in props
        assert "encoding" in props
        assert "max_bytes" in props

    def test_get_schema_defaults_documented(self) -> None:
        schema = FileReadTool().get_schema()
        max_bytes_desc = schema["parameters"]["properties"]["max_bytes"]["description"]
        encoding_desc = schema["parameters"]["properties"]["encoding"]["description"]

        assert str(_DEFAULT_MAX_BYTES) in max_bytes_desc
        assert _DEFAULT_ENCODING in encoding_desc

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def test_permissions_declare_filesystem_read(self) -> None:
        perms = FileReadTool().permissions
        assert perms.filesystem_read is True
        assert perms.filesystem_write is False
        assert perms.network is False
        assert perms.shell is False


# ===========================================================================
# FileWriteTool
# ===========================================================================


class TestFileWriteTool:
    """Tests for FileWriteTool.execute() and FileWriteTool.get_schema()."""

    # ------------------------------------------------------------------
    # Overwrite mode
    # ------------------------------------------------------------------

    def test_write_new_file_with_overwrite_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "new.txt"

        result = FileWriteTool().execute(path=str(target), content="brand new")

        assert result.success is True
        assert target.read_text() == "brand new"

    def test_overwrite_replaces_existing_content(self, tmp_path: Path) -> None:
        target = tmp_path / "existing.txt"
        target.write_text("old content", encoding="utf-8")

        result = FileWriteTool().execute(
            path=str(target), content="new content", mode="overwrite"
        )

        assert result.success is True
        assert target.read_text() == "new content"

    def test_write_reports_char_count(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        content = "hello"

        result = FileWriteTool().execute(path=str(target), content=content)

        assert result.success is True
        assert str(len(content)) in result.output

    # ------------------------------------------------------------------
    # Append mode
    # ------------------------------------------------------------------

    def test_write_with_append_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "log.txt"
        target.write_text("line1\n", encoding="utf-8")

        result = FileWriteTool().execute(path=str(target), content="line2\n", mode="append")

        assert result.success is True
        assert target.read_text() == "line1\nline2\n"

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------

    def test_write_to_non_existent_directory_creates_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "file.txt"

        result = FileWriteTool().execute(path=str(target), content="nested")

        assert result.success is True
        assert target.read_text() == "nested"

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def test_write_respects_encoding(self, tmp_path: Path) -> None:
        target = tmp_path / "encoded.txt"

        result = FileWriteTool().execute(
            path=str(target), content="caf\u00e9", encoding="utf-8"
        )

        assert result.success is True
        assert target.read_bytes() == "caf\u00e9".encode("utf-8")

    def test_write_uses_tilde_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        result = FileWriteTool().execute(path="~/tilde_write.txt", content="via tilde")

        assert result.success is True
        assert (tmp_path / "tilde_write.txt").read_text() == "via tilde"

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_invalid_mode_handling(self, tmp_path: Path) -> None:
        result = FileWriteTool().execute(
            path=str(tmp_path / "f.txt"), content="x", mode="truncate"
        )

        assert result.success is False
        assert result.output is None
        assert "invalid mode" in result.error.lower()
        assert "truncate" in result.error

    def test_write_permission_denied(self, tmp_path: Path) -> None:
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)
        try:
            result = FileWriteTool().execute(
                path=str(readonly_dir / "blocked.txt"), content="nope"
            )
            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            readonly_dir.chmod(0o755)

    def test_write_invalid_path_construction_returns_error(self) -> None:
        with patch("missy.tools.builtin.file_write.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("invalid path encoding")
            result = FileWriteTool().execute(path="\x00bad", content="data")

        assert result.success is False
        assert "Invalid path" in result.error

    def test_write_generic_oserror_during_write(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"

        with patch("os.open", side_effect=OSError("disk full")):
            result = FileWriteTool().execute(path=str(target), content="data")

        assert result.success is False
        assert "disk full" in result.error

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def test_get_schema_returns_valid_dict(self) -> None:
        schema = FileWriteTool().get_schema()

        assert isinstance(schema, dict)
        assert schema["name"] == "file_write"
        assert schema["parameters"]["type"] == "object"

    def test_get_schema_required_fields(self) -> None:
        required = FileWriteTool().get_schema()["parameters"]["required"]

        assert "path" in required
        assert "content" in required

    def test_get_schema_mode_enum(self) -> None:
        mode_schema = FileWriteTool().get_schema()["parameters"]["properties"]["mode"]

        assert "overwrite" in mode_schema["enum"]
        assert "append" in mode_schema["enum"]

    def test_get_schema_all_properties_present(self) -> None:
        props = FileWriteTool().get_schema()["parameters"]["properties"]

        assert "path" in props
        assert "content" in props
        assert "mode" in props
        assert "encoding" in props

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def test_permissions_declare_filesystem_write(self) -> None:
        perms = FileWriteTool().permissions
        assert perms.filesystem_write is True
        assert perms.filesystem_read is False
        assert perms.network is False
        assert perms.shell is False


# ===========================================================================
# FileDeleteTool
# ===========================================================================


class TestFileDeleteTool:
    """Tests for FileDeleteTool.execute() and FileDeleteTool.get_schema()."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_delete_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "to_delete.txt"
        target.write_text("bye", encoding="utf-8")

        result = FileDeleteTool().execute(path=str(target))

        assert result.success is True
        assert not target.exists()
        assert "Deleted" in result.output

    def test_delete_output_includes_path(self, tmp_path: Path) -> None:
        target = tmp_path / "named.txt"
        target.write_text("x")
        path_str = str(target)

        result = FileDeleteTool().execute(path=path_str)

        assert path_str in result.output

    def test_delete_uses_tilde_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "tilde_del.txt"
        target.write_text("delete me")

        result = FileDeleteTool().execute(path="~/tilde_del.txt")

        assert result.success is True
        assert not target.exists()

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_delete_non_existent_file_returns_error(self, tmp_path: Path) -> None:
        result = FileDeleteTool().execute(path=str(tmp_path / "no_such_file.txt"))

        assert result.success is False
        assert result.output is None
        assert "not found" in result.error.lower()

    def test_delete_directory_rejected(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = FileDeleteTool().execute(path=str(subdir))

        assert result.success is False
        assert result.output is None
        assert "directories will not be deleted" in result.error.lower()
        assert subdir.exists()

    def test_delete_permission_denied(self, tmp_path: Path) -> None:
        target = tmp_path / "protected.txt"
        target.write_text("locked")
        tmp_path.chmod(0o555)
        try:
            result = FileDeleteTool().execute(path=str(target))
            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            tmp_path.chmod(0o755)

    def test_delete_invalid_path_construction_returns_error(self) -> None:
        with patch("missy.tools.builtin.file_delete.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("null byte in path")
            result = FileDeleteTool().execute(path="\x00bad")

        assert result.success is False
        assert "Invalid path" in result.error

    def test_delete_generic_oserror_during_unlink(self, tmp_path: Path) -> None:
        target = tmp_path / "target.txt"
        target.write_text("data")

        with patch.object(Path, "unlink", side_effect=OSError("filesystem error")):
            result = FileDeleteTool().execute(path=str(target))

        assert result.success is False
        assert "filesystem error" in result.error

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def test_get_schema_returns_valid_dict(self) -> None:
        schema = FileDeleteTool().get_schema()

        assert isinstance(schema, dict)
        assert schema["name"] == "file_delete"
        assert schema["parameters"]["type"] == "object"

    def test_get_schema_required_fields(self) -> None:
        assert FileDeleteTool().get_schema()["parameters"]["required"] == ["path"]

    def test_get_schema_path_property_present(self) -> None:
        props = FileDeleteTool().get_schema()["parameters"]["properties"]
        assert "path" in props

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def test_permissions_declare_filesystem_write(self) -> None:
        perms = FileDeleteTool().permissions
        assert perms.filesystem_write is True
        assert perms.filesystem_read is False


# ===========================================================================
# ListFilesTool
# ===========================================================================


class TestListFilesTool:
    """Tests for ListFilesTool.execute() and ListFilesTool.get_schema()."""

    # ------------------------------------------------------------------
    # Non-recursive listing
    # ------------------------------------------------------------------

    def test_list_directory_contents(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()

        result = ListFilesTool().execute(path=str(tmp_path))

        assert result.success is True
        assert "a.txt" in result.output
        assert "b.txt" in result.output
        assert "subdir" in result.output

    def test_list_shows_dir_and_file_markers(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "mydir").mkdir()

        result = ListFilesTool().execute(path=str(tmp_path))

        assert "[dir]" in result.output
        assert "[file]" in result.output

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        result = ListFilesTool().execute(path=str(empty))

        assert result.success is True
        assert "empty directory" in result.output.lower()

    def test_list_shows_file_sizes(self, tmp_path: Path) -> None:
        (tmp_path / "sized.txt").write_text("hello world", encoding="utf-8")

        result = ListFilesTool().execute(path=str(tmp_path))

        assert result.success is True
        assert "bytes" in result.output

    # ------------------------------------------------------------------
    # Recursive listing
    # ------------------------------------------------------------------

    def test_list_with_recursive_true(self, tmp_path: Path) -> None:
        nested = tmp_path / "level1" / "level2"
        nested.mkdir(parents=True)
        (nested / "deep.txt").write_text("deep content")

        result = ListFilesTool().execute(path=str(tmp_path), recursive=True)

        assert result.success is True
        assert "deep.txt" in result.output

    # ------------------------------------------------------------------
    # max_entries cap
    # ------------------------------------------------------------------

    def test_max_entries_limit(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"file{i:02d}.txt").write_text(str(i))

        result = ListFilesTool().execute(path=str(tmp_path), max_entries=3)

        assert result.success is True
        assert "more entries" in result.output
        assert "7" in result.output  # 10 - 3 = 7 overflow

    def test_no_truncation_notice_within_limit(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(str(i))

        result = ListFilesTool().execute(path=str(tmp_path), max_entries=100)

        assert result.success is True
        assert "more entries" not in result.output

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_list_non_existent_directory_returns_error(self, tmp_path: Path) -> None:
        result = ListFilesTool().execute(path=str(tmp_path / "nonexistent"))

        assert result.success is False
        assert result.output is None
        assert "not found" in result.error.lower()

    def test_list_file_path_rejected_as_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "notadir.txt"
        target.write_text("text")

        result = ListFilesTool().execute(path=str(target))

        assert result.success is False
        assert "not a directory" in result.error.lower()

    def test_list_uses_tilde_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "home_file.txt").write_text("here")

        result = ListFilesTool().execute(path="~")

        assert result.success is True
        assert "home_file.txt" in result.output

    def test_list_permission_error_on_iterdir(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "restricted"
        target_dir.mkdir()

        with patch.object(Path, "iterdir", side_effect=PermissionError("no access")):
            result = ListFilesTool().execute(path=str(target_dir))

        assert result.success is False
        assert "Permission denied" in result.error

    def test_list_generic_exception_on_iterdir(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "broken_dir"
        target_dir.mkdir()

        with patch.object(Path, "iterdir", side_effect=OSError("I/O error")):
            result = ListFilesTool().execute(path=str(target_dir))

        assert result.success is False
        assert "I/O error" in result.error

    def test_list_stat_oserror_falls_back_to_no_size_format(self, tmp_path: Path) -> None:
        """When stat() raises OSError for an entry, the [file] line omits the size."""
        target = tmp_path / "unstat_file.txt"
        target.write_text("content")

        fake_entry = MagicMock(spec=Path)
        fake_entry.is_dir.return_value = False
        fake_entry.is_file.return_value = True
        fake_entry.name = "unstat_file.txt"
        fake_entry.relative_to.return_value = Path("unstat_file.txt")
        fake_entry.stat.side_effect = OSError("permission denied on stat")

        with patch.object(Path, "iterdir", return_value=[fake_entry]):
            result = ListFilesTool().execute(path=str(tmp_path))

        assert result.success is True
        assert "unstat_file.txt" in result.output
        assert "bytes" not in result.output

    def test_list_invalid_path_construction_returns_error(self) -> None:
        with patch("missy.tools.builtin.list_files.Path") as mock_path_cls:
            mock_path_cls.side_effect = ValueError("bad path bytes")
            result = ListFilesTool().execute(path="\x00invalid")

        assert result.success is False
        assert "Invalid path" in result.error

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def test_get_schema_returns_valid_dict(self) -> None:
        schema = ListFilesTool().get_schema()

        assert isinstance(schema, dict)
        assert schema["name"] == "list_files"
        assert schema["parameters"]["type"] == "object"

    def test_get_schema_required_fields(self) -> None:
        assert ListFilesTool().get_schema()["parameters"]["required"] == ["path"]

    def test_get_schema_all_properties_present(self) -> None:
        props = ListFilesTool().get_schema()["parameters"]["properties"]
        assert "path" in props
        assert "recursive" in props
        assert "max_entries" in props

    def test_get_schema_max_entries_default_documented(self) -> None:
        desc = ListFilesTool().get_schema()["parameters"]["properties"]["max_entries"]["description"]
        assert str(_DEFAULT_MAX_ENTRIES) in desc


# ===========================================================================
# WebFetchTool
# ===========================================================================


class TestWebFetchTool:
    """Tests for WebFetchTool.execute() and WebFetchTool.get_schema().

    All tests mock missy.gateway.client.create_client to avoid real network
    calls and policy-engine initialisation.
    """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_response(*, text: str, status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.status_code = status_code
        return resp

    @staticmethod
    def _make_client(response: MagicMock) -> MagicMock:
        client = MagicMock()
        client.get.return_value = response
        return client

    # ------------------------------------------------------------------
    # Successful fetches
    # ------------------------------------------------------------------

    def test_execute_with_mocked_httpx_successful_fetch(self) -> None:
        mock_resp = self._make_response(text="<html>ok</html>")
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is True
        assert result.output == "<html>ok</html>"
        assert result.error is None

    def test_2xx_boundary_status_299_is_success(self) -> None:
        mock_resp = self._make_response(text="partial content", status_code=299)
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is True
        assert result.error is None

    # ------------------------------------------------------------------
    # HTTP error statuses
    # ------------------------------------------------------------------

    def test_execute_with_invalid_url_returns_failure(self) -> None:
        """A URL that causes the client to raise is treated as failure."""
        mock_client = MagicMock()
        mock_client.get.side_effect = ValueError("Invalid URL: not-a-url")

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="not-a-url")

        assert result.success is False
        assert result.output is None

    def test_http_404_returns_failure(self) -> None:
        mock_resp = self._make_response(text="not found", status_code=404)
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com/missing")

        assert result.success is False
        assert "404" in result.error

    def test_http_500_returns_failure(self) -> None:
        mock_resp = self._make_response(text="server error", status_code=500)
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is False
        assert "500" in result.error

    def test_redirect_301_is_not_success(self) -> None:
        mock_resp = self._make_response(text="moved", status_code=301)
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is False
        assert "301" in result.error

    def test_network_exception_returns_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("network unreachable")

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is False
        assert result.output is None
        assert "network unreachable" in result.error

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def test_response_truncated_at_max_bytes(self) -> None:
        long_text = "x" * (_MAX_RESPONSE_BYTES + 100)
        mock_resp = self._make_response(text=long_text)
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is True
        assert "truncated" in result.output.lower()
        assert len(result.output) <= _MAX_RESPONSE_BYTES + 50

    # ------------------------------------------------------------------
    # Header filtering
    # ------------------------------------------------------------------

    def test_blocked_headers_are_filtered(self) -> None:
        """Authorization, Host, Cookie and other sensitive headers must be stripped."""
        mock_resp = self._make_response(text="ok")
        mock_client = self._make_client(mock_resp)

        headers = {
            "Authorization": "Bearer secret",
            "Host": "evil.com",
            "Cookie": "session=abc",
            "x-forwarded-for": "1.2.3.4",
            "proxy-authorization": "Basic x",
            "X-Custom": "safe",
        }
        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://api.example.com", headers=headers)

        assert result.success is True
        call_kwargs = mock_client.get.call_args.kwargs
        passed_headers = call_kwargs.get("headers", {})
        for blocked in ("Authorization", "Host", "Cookie", "x-forwarded-for", "proxy-authorization"):
            assert blocked not in passed_headers
        assert passed_headers.get("X-Custom") == "safe"

    def test_all_blocked_headers_omits_headers_kwarg(self) -> None:
        """When all supplied headers are blocked the 'headers' kwarg is omitted entirely."""
        mock_resp = self._make_response(text="ok")
        mock_client = self._make_client(mock_resp)

        headers = {"Authorization": "Bearer secret", "Host": "evil.com"}
        with patch("missy.gateway.client.create_client", return_value=mock_client):
            WebFetchTool().execute(url="https://example.com", headers=headers)

        call_kwargs = mock_client.get.call_args.kwargs
        assert "headers" not in call_kwargs

    def test_no_headers_argument_omits_headers_kwarg(self) -> None:
        mock_resp = self._make_response(text="ok")
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            WebFetchTool().execute(url="https://example.com")

        assert "headers" not in mock_client.get.call_args.kwargs

    # ------------------------------------------------------------------
    # Timeout
    # ------------------------------------------------------------------

    def test_timeout_handling_passed_to_create_client(self) -> None:
        mock_resp = self._make_response(text="ok")
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client) as mock_cc:
            WebFetchTool().execute(url="https://example.com", timeout=60)

        mock_cc.assert_called_once_with(
            session_id="web_fetch_tool", task_id="fetch", timeout=60
        )

    def test_default_timeout_is_used_when_not_specified(self) -> None:
        mock_resp = self._make_response(text="ok")
        mock_client = self._make_client(mock_resp)

        with patch("missy.gateway.client.create_client", return_value=mock_client) as mock_cc:
            WebFetchTool().execute(url="https://example.com")

        mock_cc.assert_called_once_with(
            session_id="web_fetch_tool", task_id="fetch", timeout=_DEFAULT_TIMEOUT
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def test_get_schema_returns_valid_dict(self) -> None:
        schema = WebFetchTool().get_schema()

        assert isinstance(schema, dict)
        assert schema["name"] == "web_fetch"
        assert schema["parameters"]["type"] == "object"

    def test_get_schema_required_fields(self) -> None:
        assert WebFetchTool().get_schema()["parameters"]["required"] == ["url"]

    def test_get_schema_all_properties_present(self) -> None:
        props = WebFetchTool().get_schema()["parameters"]["properties"]
        assert "url" in props
        assert "timeout" in props
        assert "headers" in props

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def test_permissions_declare_network(self) -> None:
        perms = WebFetchTool().permissions
        assert perms.network is True
        assert perms.filesystem_read is False
        assert perms.shell is False


# ===========================================================================
# SelfCreateTool
# ===========================================================================


class TestSelfCreateTool:
    """Tests for SelfCreateTool.execute() with a redirected tools directory."""

    # ------------------------------------------------------------------
    # Create action
    # ------------------------------------------------------------------

    def test_create_tool_action_python(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="my_tool",
                language="python",
                script="print('hello')",
                tool_description="A test tool",
            )

        assert result.success is True
        assert "my_tool" in result.output
        assert (tmp_path / "my_tool.py").exists()
        assert (tmp_path / "my_tool.py").read_text() == "print('hello')"
        assert (tmp_path / "my_tool.json").exists()

    def test_create_tool_action_bash(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="bash_tool",
                language="bash",
                script="#!/bin/bash\necho hi",
                tool_description="Bash helper",
            )

        assert result.success is True
        assert (tmp_path / "bash_tool.sh").exists()

    def test_create_tool_action_node(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="node_tool",
                language="node",
                script="console.log('hi')",
                tool_description="Node helper",
            )

        assert result.success is True
        assert (tmp_path / "node_tool.js").exists()

    def test_create_script_is_executable(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            mod.SelfCreateTool().execute(
                action="create",
                tool_name="exec_check",
                language="python",
                script="pass",
                tool_description="",
            )

        file_stat = (tmp_path / "exec_check.py").stat()
        assert file_stat.st_mode & stat.S_IXUSR

    def test_create_stores_metadata_json(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            mod.SelfCreateTool().execute(
                action="create",
                tool_name="meta_tool",
                language="python",
                script="pass",
                tool_description="Checks metadata",
            )

        meta = json.loads((tmp_path / "meta_tool.json").read_text())
        assert meta["name"] == "meta_tool"
        assert meta["description"] == "Checks metadata"
        assert meta["language"] == "python"

    def test_create_invalid_language_rejected(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="lang_test",
                language="ruby",
                script="puts 'hi'",
                tool_description="",
            )

        assert result.success is False
        assert "language" in result.error.lower()

    def test_create_invalid_tool_name_rejected(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="bad name!",
                language="python",
                script="pass",
                tool_description="",
            )

        assert result.success is False
        assert "alphanumeric" in result.error.lower()

    def test_create_empty_tool_name_rejected(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="",
                language="python",
                script="pass",
                tool_description="",
            )

        assert result.success is False
        assert "alphanumeric" in result.error.lower()

    def test_create_missing_script_rejected(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="valid_name",
                language="python",
                script="",
                tool_description="",
            )

        assert result.success is False
        assert "script" in result.error.lower()

    def test_create_dangerous_script_pattern_rejected(self, tmp_path: Path) -> None:
        """Scripts containing dangerous patterns like os.system( are blocked."""
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="create",
                tool_name="dangerous",
                language="python",
                script="os.system('rm -rf /')",
                tool_description="",
            )

        assert result.success is False
        assert "dangerous" in result.error.lower()

    # ------------------------------------------------------------------
    # List action
    # ------------------------------------------------------------------

    def test_list_tools_action_empty(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path / "no_tools"):
            result = mod.SelfCreateTool().execute(action="list")

        assert result.success is True
        assert "no custom tools" in result.output.lower()

    def test_list_tools_action_shows_existing(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        (tmp_path / "tool_a.json").write_text(
            json.dumps({"name": "tool_a", "description": "Alpha"})
        )
        (tmp_path / "tool_b.json").write_text(
            json.dumps({"name": "tool_b", "description": "Beta"})
        )

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="list")

        assert result.success is True
        assert "tool_a" in result.output
        assert "Alpha" in result.output
        assert "tool_b" in result.output

    def test_list_tools_skips_malformed_json_silently(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        (tmp_path / "broken.json").write_text("not json {{{{")
        (tmp_path / "good.json").write_text('{"name": "good", "description": "OK"}')

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="list")

        assert result.success is True
        assert "good" in result.output

    # ------------------------------------------------------------------
    # Delete action
    # ------------------------------------------------------------------

    def test_delete_tool_action(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        script = tmp_path / "to_remove.py"
        script.write_text("pass")
        meta = tmp_path / "to_remove.json"
        meta.write_text('{"name": "to_remove"}')

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="delete", tool_name="to_remove")

        assert result.success is True
        assert "to_remove" in result.output
        assert not script.exists()
        assert not meta.exists()

    def test_delete_tool_missing_name_returns_error(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="delete", tool_name="")

        assert result.success is False
        assert "tool_name" in result.error.lower()

    def test_delete_non_existent_tool_returns_error(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="delete", tool_name="does_not_exist"
            )

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_delete_removes_all_script_extensions(self, tmp_path: Path) -> None:
        """If a tool has multiple extension files, all are removed."""
        import missy.tools.builtin.self_create_tool as mod

        (tmp_path / "multi.py").write_text("pass")
        (tmp_path / "multi.sh").write_text("#!/bin/bash")

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="delete", tool_name="multi")

        assert result.success is True
        assert not (tmp_path / "multi.py").exists()
        assert not (tmp_path / "multi.sh").exists()

    # ------------------------------------------------------------------
    # Unknown action
    # ------------------------------------------------------------------

    def test_unknown_action_returns_error(self, tmp_path: Path) -> None:
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="inspect")

        assert result.success is False
        assert "unknown action" in result.error.lower()
        assert "inspect" in result.error

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def test_get_schema_returns_valid_dict(self) -> None:
        schema = SelfCreateTool().get_schema()

        assert isinstance(schema, dict)
        assert schema["name"] == "self_create_tool"

    def test_get_schema_all_properties_present(self) -> None:
        props = SelfCreateTool().get_schema()["parameters"]["properties"]
        assert "action" in props
        assert "tool_name" in props
        assert "language" in props
        assert "script" in props
        assert "tool_description" in props

    def test_get_schema_action_in_required(self) -> None:
        assert "action" in SelfCreateTool().get_schema()["parameters"]["required"]


# ===========================================================================
# TTSSpeakTool / AudioListDevicesTool / AudioSetVolumeTool — module helpers
# ===========================================================================


class TestEnsureRuntimeDir:
    """Tests for the _ensure_runtime_dir() module-level helper."""

    def test_ensure_runtime_dir_returns_dict(self) -> None:
        result = _ensure_runtime_dir()
        assert isinstance(result, dict)

    def test_preserves_existing_xdg_runtime_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/already/set")
        env = _ensure_runtime_dir()
        assert env["XDG_RUNTIME_DIR"] == "/run/user/already/set"

    def test_sets_xdg_runtime_dir_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        uid = os.getuid()
        env = _ensure_runtime_dir()
        assert env["XDG_RUNTIME_DIR"] == f"/run/user/{uid}"

    def test_returned_env_contains_only_safe_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_SECRET_KEY", "super-secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

        env = _ensure_runtime_dir()

        assert env.get("HOME") == "/home/test"
        assert "MY_SECRET_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env

    def test_works_with_env_containing_no_safe_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with a completely stripped environment the function returns a dict."""
        from missy.tools.builtin.tts_speak import _SAFE_TTS_ENV_VARS

        for key in list(_SAFE_TTS_ENV_VARS):
            monkeypatch.delenv(key, raising=False)

        env = _ensure_runtime_dir()
        assert isinstance(env, dict)
        assert "XDG_RUNTIME_DIR" in env


class TestPiperEnv:
    """Tests for the _piper_env() module-level helper."""

    def test_piper_env_returns_dict(self) -> None:
        result = _piper_env()
        assert isinstance(result, dict)

    def test_piper_env_contains_ld_library_path(self) -> None:
        env = _piper_env()
        assert "LD_LIBRARY_PATH" in env

    def test_piper_env_includes_piper_bin_parent_in_library_path(self) -> None:
        from missy.tools.builtin.tts_speak import _PIPER_BIN

        env = _piper_env()
        expected_lib = str(_PIPER_BIN.parent)
        assert expected_lib in env["LD_LIBRARY_PATH"]

    def test_piper_env_does_not_duplicate_piper_bin_parent(self) -> None:
        """Calling _piper_env twice should not duplicate the path segment."""
        from missy.tools.builtin.tts_speak import _PIPER_BIN

        env1 = _piper_env()
        # Force an existing LD_LIBRARY_PATH that already contains the piper dir.
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": env1["LD_LIBRARY_PATH"]}):
            env2 = _piper_env()

        piper_lib = str(_PIPER_BIN.parent)
        # Should appear exactly once.
        assert env2["LD_LIBRARY_PATH"].count(piper_lib) == 1

    def test_piper_env_xdg_runtime_dir_present(self) -> None:
        env = _piper_env()
        assert "XDG_RUNTIME_DIR" in env


# ===========================================================================
# TTSSpeakTool
# ===========================================================================


class TestTTSSpeakTool:
    """Tests for TTSSpeakTool.execute() with all subprocesses mocked."""

    # ------------------------------------------------------------------
    # Construction and schema
    # ------------------------------------------------------------------

    def test_construction(self) -> None:
        tool = TTSSpeakTool()
        assert tool.name == "tts_speak"
        assert isinstance(tool.description, str)
        assert tool.permissions.shell is True

    def test_get_schema_via_base(self) -> None:
        schema = TTSSpeakTool().get_schema()
        assert schema["name"] == "tts_speak"
        props = schema["parameters"]["properties"]
        assert "text" in props
        assert "speed" in props
        assert "voice" in props

    # ------------------------------------------------------------------
    # Guard: empty text
    # ------------------------------------------------------------------

    def test_execute_empty_text_returns_failure(self) -> None:
        result = TTSSpeakTool().execute(text="   ")
        assert result.success is False
        assert "No text provided" in result.error

    # ------------------------------------------------------------------
    # Happy path via mocked subprocess
    # ------------------------------------------------------------------

    def test_execute_with_mocked_subprocess_success(self) -> None:
        """Happy path: piper succeeds + playback succeeds."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world", speed=1.0, voice="en_US-lessac-medium")

        assert result.success is True
        assert "2" in result.output  # "hello world" = 2 words
        assert "engine=piper" in result.output
        assert "voice=en_US-lessac-medium" in result.output

    def test_execute_espeak_fallback_success(self) -> None:
        """When piper fails but espeak + playback succeed, output mentions espeak-ng."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value="piper not found"),
            patch("missy.tools.builtin.tts_speak._synth_espeak", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="test text here", voice="en")

        assert result.success is True
        assert "engine=espeak-ng" in result.output

    # ------------------------------------------------------------------
    # Failure paths
    # ------------------------------------------------------------------

    def test_execute_piper_and_espeak_both_fail(self) -> None:
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value="piper error"),
            patch("missy.tools.builtin.tts_speak._synth_espeak", return_value="espeak error"),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello")

        assert result.success is False
        assert "TTS synthesis failed" in result.error

    def test_execute_piper_fails_espeak_succeeds_play_fails(self) -> None:
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value="piper not found"),
            patch("missy.tools.builtin.tts_speak._synth_espeak", return_value=None),
            patch(
                "missy.tools.builtin.tts_speak._play_wav",
                return_value="gst-launch-1.0 not installed",
            ),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world")

        assert result.success is False
        assert "gst-launch" in result.error

    def test_execute_timeout_returns_failure(self) -> None:
        tool = TTSSpeakTool()
        with (
            patch(
                "missy.tools.builtin.tts_speak._synth_piper",
                side_effect=subprocess.TimeoutExpired("cmd", 60),
            ),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_execute_file_not_found_returns_failure(self) -> None:
        tool = TTSSpeakTool()
        with (
            patch(
                "missy.tools.builtin.tts_speak._synth_piper",
                side_effect=FileNotFoundError("piper"),
            ),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world")

        assert result.success is False
        assert "not found" in result.error.lower()

    # ------------------------------------------------------------------
    # Speed clamping
    # ------------------------------------------------------------------

    def test_execute_clamps_speed_above_max(self) -> None:
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="fast", speed=99.0)

        assert result.success is True

    def test_execute_clamps_speed_below_min(self) -> None:
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="slow", speed=0.0)

        assert result.success is True


# ===========================================================================
# AudioListDevicesTool
# ===========================================================================


class TestAudioListDevicesTool:
    """Tests for AudioListDevicesTool.execute() with subprocess mocked."""

    def test_construction(self) -> None:
        tool = AudioListDevicesTool()
        assert tool.name == "audio_list_devices"
        assert isinstance(tool.description, str)
        assert tool.permissions.shell is True

    def test_execute_wpctl_success(self) -> None:
        """When wpctl returns audio section, output contains it."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "PipeWire\nAudio\n├── Sink: Built-in Audio\n└── Source: Mic\n"

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = AudioListDevicesTool().execute()

        assert result.success is True
        assert "Audio" in result.output

    def test_execute_wpctl_not_found_falls_back_to_aplay(self) -> None:
        """When wpctl raises FileNotFoundError, aplay is tried instead."""
        aplay_result = MagicMock()
        aplay_result.returncode = 0
        aplay_result.stdout = "card 0: PCH [HDA Intel PCH], device 0"

        def run_side_effect(cmd, **kwargs):
            if "wpctl" in cmd:
                raise FileNotFoundError("wpctl")
            return aplay_result

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=run_side_effect),
        ):
            result = AudioListDevicesTool().execute()

        assert result.success is True
        assert "PCH" in result.output

    def test_execute_no_audio_tools_returns_failure(self) -> None:
        """When both wpctl and aplay are unavailable, return a failure result."""
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch(
                "subprocess.run",
                side_effect=FileNotFoundError("no audio tools"),
            ),
        ):
            result = AudioListDevicesTool().execute()

        assert result.success is False
        assert "wpctl" in result.error.lower() or "aplay" in result.error.lower()

    def test_execute_wpctl_nonzero_returncode_falls_through(self) -> None:
        """If wpctl exits non-zero, the tool falls back to aplay."""
        wpctl_fail = MagicMock()
        wpctl_fail.returncode = 1
        wpctl_fail.stdout = ""
        aplay_ok = MagicMock()
        aplay_ok.returncode = 0
        aplay_ok.stdout = "aplay output"

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=[wpctl_fail, aplay_ok]),
        ):
            result = AudioListDevicesTool().execute()

        assert result.success is True
        assert "aplay output" in result.output

    def test_get_schema_structure(self) -> None:
        schema = AudioListDevicesTool().get_schema()
        assert schema["name"] == "audio_list_devices"
        assert isinstance(schema["parameters"]["properties"], dict)


# ===========================================================================
# AudioSetVolumeTool
# ===========================================================================


class TestAudioSetVolumeTool:
    """Tests for AudioSetVolumeTool.execute() with subprocess mocked."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_construction(self) -> None:
        tool = AudioSetVolumeTool()
        assert tool.name == "audio_set_volume"
        assert tool.permissions.shell is True

    # ------------------------------------------------------------------
    # Happy paths — wpctl mocked to succeed
    # ------------------------------------------------------------------

    def _ok_run_pair(self, current_vol: str = "Volume: 0.75") -> list[MagicMock]:
        """Return two MagicMock subprocess results: set + get."""
        set_ok = MagicMock()
        set_ok.returncode = 0
        set_ok.stderr = ""
        set_ok.stdout = ""
        get_ok = MagicMock()
        get_ok.returncode = 0
        get_ok.stdout = current_vol
        return [set_ok, get_ok]

    def test_execute_with_mocked_subprocess_absolute_volume(self) -> None:
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=self._ok_run_pair("Volume: 0.75")),
        ):
            result = AudioSetVolumeTool().execute(volume="75%")

        assert result.success is True
        assert "0.75" in result.output

    def test_mute_command(self) -> None:
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=self._ok_run_pair("Volume: 0.50 [MUTED]")) as mock_run,
        ):
            result = AudioSetVolumeTool().execute(volume="mute")

        assert result.success is True
        first_cmd = mock_run.call_args_list[0].args[0]
        assert "set-mute" in first_cmd
        assert "1" in first_cmd

    def test_unmute_command(self) -> None:
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=self._ok_run_pair()) as mock_run,
        ):
            result = AudioSetVolumeTool().execute(volume="unmute")

        assert result.success is True
        first_cmd = mock_run.call_args_list[0].args[0]
        assert "set-mute" in first_cmd
        assert "0" in first_cmd

    def test_relative_volume_increase(self) -> None:
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=self._ok_run_pair("Volume: 0.80")) as mock_run,
        ):
            result = AudioSetVolumeTool().execute(volume="+10%")

        assert result.success is True
        first_cmd = mock_run.call_args_list[0].args[0]
        assert any(arg.endswith("+") for arg in first_cmd)

    def test_relative_volume_decrease(self) -> None:
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=self._ok_run_pair("Volume: 0.65")) as mock_run,
        ):
            result = AudioSetVolumeTool().execute(volume="-5%")

        assert result.success is True
        first_cmd = mock_run.call_args_list[0].args[0]
        assert any(arg.endswith("-") for arg in first_cmd)

    # ------------------------------------------------------------------
    # Failure paths
    # ------------------------------------------------------------------

    def test_wpctl_failure_returns_error(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error msg"
        mock_result.stdout = ""

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = AudioSetVolumeTool().execute(volume="50%")

        assert result.success is False
        assert "wpctl failed" in result.error

    def test_file_not_found_returns_install_hint(self) -> None:
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=FileNotFoundError("wpctl")),
        ):
            result = AudioSetVolumeTool().execute(volume="50%")

        assert result.success is False
        assert "wpctl not found" in result.error

    def test_timeout_returns_error(self) -> None:
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("wpctl", 5),
            ),
        ):
            result = AudioSetVolumeTool().execute(volume="50%")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_invalid_volume_string_returns_error(self) -> None:
        """A non-numeric volume string that triggers ValueError is handled."""
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=ValueError("bad volume conversion")),
        ):
            result = AudioSetVolumeTool().execute(volume="75%")

        assert result.success is False
        assert "Invalid volume value" in result.error
        assert "75%" in result.error

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def test_get_schema_structure(self) -> None:
        schema = AudioSetVolumeTool().get_schema()
        assert schema["name"] == "audio_set_volume"
        props = schema["parameters"]["properties"]
        assert "volume" in props
        assert "device_id" in props


# ===========================================================================
# DiscordUploadTool
# ===========================================================================


class TestDiscordUploadTool:
    """Tests for DiscordUploadTool.execute()."""

    def test_no_bot_token_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)

        result = DiscordUploadTool().execute(file_path="/tmp/file.png", channel_id="123456")

        assert result.success is False
        assert result.output is None
        assert "DISCORD_BOT_TOKEN" in result.error

    def test_successful_upload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot xyzzy")
        target = tmp_path / "image.png"
        target.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {"id": "9999888877776666"}

        with patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest):
            result = DiscordUploadTool().execute(
                file_path=str(target),
                channel_id="111222333",
                caption="Look at this!",
            )

        assert result.success is True
        assert "9999888877776666" in result.output
        mock_rest.upload_file.assert_called_once_with(
            channel_id="111222333",
            file_path=str(target),
            caption="Look at this!",
        )

    def test_upload_with_empty_caption(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot abc")
        target = tmp_path / "doc.pdf"
        target.write_bytes(b"%PDF")

        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {"id": "42"}

        with patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest):
            result = DiscordUploadTool().execute(file_path=str(target), channel_id="777")

        assert result.success is True
        mock_rest.upload_file.assert_called_once_with(
            channel_id="777", file_path=str(target), caption=""
        )

    def test_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot token123")
        mock_rest = MagicMock()
        mock_rest.upload_file.side_effect = FileNotFoundError("No such file")

        with patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest):
            result = DiscordUploadTool().execute(
                file_path="/nonexistent/path.png", channel_id="123"
            )

        assert result.success is False
        assert result.output is None

    def test_general_exception_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot token456")
        mock_rest = MagicMock()
        mock_rest.upload_file.side_effect = RuntimeError("Discord API exploded")

        with patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest):
            result = DiscordUploadTool().execute(
                file_path="/some/file.txt", channel_id="321"
            )

        assert result.success is False
        assert "Upload failed" in result.error
        assert "Discord API exploded" in result.error

    def test_bot_token_passed_to_rest_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = "Bot supersecret"
        monkeypatch.setenv("DISCORD_BOT_TOKEN", token)
        target = tmp_path / "f.txt"
        target.write_text("x")
        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {"id": "1"}

        with patch(
            "missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest
        ) as mock_cls:
            DiscordUploadTool().execute(file_path=str(target), channel_id="1")

        mock_cls.assert_called_once_with(bot_token=token)

    def test_message_id_fallback_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot x")
        target = tmp_path / "f.txt"
        target.write_text("x")
        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {}  # no 'id' key

        with patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest):
            result = DiscordUploadTool().execute(file_path=str(target), channel_id="1")

        assert result.success is True
        assert "?" in result.output
