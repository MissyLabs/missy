"""Comprehensive tests for low-coverage built-in tools.

Covers:
    - FileReadTool   (missy/tools/builtin/file_read.py)
    - FileWriteTool  (missy/tools/builtin/file_write.py)
    - FileDeleteTool (missy/tools/builtin/file_delete.py)
    - ListFilesTool  (missy/tools/builtin/list_files.py)
    - WebFetchTool   (missy/tools/builtin/web_fetch.py)
    - DiscordUploadTool (missy/tools/builtin/discord_upload.py)
    - SelfCreateTool (missy/tools/builtin/self_create_tool.py)
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.tools.base import ToolResult
from missy.tools.builtin.file_delete import FileDeleteTool
from missy.tools.builtin.file_read import FileReadTool, _DEFAULT_ENCODING, _DEFAULT_MAX_BYTES
from missy.tools.builtin.file_write import FileWriteTool
from missy.tools.builtin.discord_upload import DiscordUploadTool
from missy.tools.builtin.list_files import ListFilesTool, _DEFAULT_MAX_ENTRIES
from missy.tools.builtin.web_fetch import WebFetchTool, _MAX_RESPONSE_BYTES


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------


class TestFileReadTool:
    """Tests for FileReadTool.execute and FileReadTool.get_schema."""

    def test_read_existing_file(self, tmp_path: Path):
        target = tmp_path / "hello.txt"
        target.write_text("hello world", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))

        assert result.success is True
        assert result.output == "hello world"
        assert result.error is None

    def test_read_returns_full_content_under_limit(self, tmp_path: Path):
        content = "a" * 100
        target = tmp_path / "file.txt"
        target.write_text(content, encoding="utf-8")

        result = FileReadTool().execute(path=str(target), max_bytes=200)

        assert result.success is True
        assert result.output == content

    def test_read_truncates_at_max_bytes(self, tmp_path: Path):
        content = "x" * 200
        target = tmp_path / "big.txt"
        target.write_text(content, encoding="utf-8")

        result = FileReadTool().execute(path=str(target), max_bytes=50)

        assert result.success is True
        assert "Truncated" in result.output
        assert "200" in result.output  # total size mentioned
        assert "50" in result.output   # limit mentioned

    def test_read_no_truncation_notice_when_within_limit(self, tmp_path: Path):
        target = tmp_path / "small.txt"
        target.write_text("short", encoding="utf-8")

        result = FileReadTool().execute(path=str(target), max_bytes=1000)

        assert "Truncated" not in result.output

    def test_read_file_not_found(self, tmp_path: Path):
        result = FileReadTool().execute(path=str(tmp_path / "ghost.txt"))

        assert result.success is False
        assert result.output is None
        assert "not found" in result.error.lower()

    def test_read_directory_rejected(self, tmp_path: Path):
        result = FileReadTool().execute(path=str(tmp_path))

        assert result.success is False
        assert result.output is None
        assert "not a file" in result.error.lower()

    def test_read_permission_denied(self, tmp_path: Path):
        target = tmp_path / "secret.txt"
        target.write_text("classified", encoding="utf-8")
        target.chmod(0o000)

        try:
            result = FileReadTool().execute(path=str(target))
            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            # Restore so tmp_path cleanup can delete it.
            target.chmod(0o644)

    def test_read_respects_encoding_parameter(self, tmp_path: Path):
        target = tmp_path / "latin.txt"
        target.write_bytes("caf\xe9".encode("latin-1"))

        result = FileReadTool().execute(path=str(target), encoding="latin-1")

        assert result.success is True
        assert "caf" in result.output

    def test_read_uses_tilde_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Point HOME at tmp_path so ~ expands there.
        target = tmp_path / "home_file.txt"
        target.write_text("home content", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))

        result = FileReadTool().execute(path="~/home_file.txt")

        assert result.success is True
        assert result.output == "home content"

    def test_get_schema_structure(self):
        schema = FileReadTool().get_schema()

        assert schema["name"] == "file_read"
        assert isinstance(schema["description"], str)
        props = schema["parameters"]["properties"]
        assert "path" in props
        assert "encoding" in props
        assert "max_bytes" in props
        assert schema["parameters"]["required"] == ["path"]

    def test_get_schema_defaults_documented(self):
        schema = FileReadTool().get_schema()
        max_bytes_desc = schema["parameters"]["properties"]["max_bytes"]["description"]
        encoding_desc = schema["parameters"]["properties"]["encoding"]["description"]

        assert str(_DEFAULT_MAX_BYTES) in max_bytes_desc
        assert _DEFAULT_ENCODING in encoding_desc


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------


class TestFileWriteTool:
    """Tests for FileWriteTool.execute and FileWriteTool.get_schema."""

    def test_write_new_file(self, tmp_path: Path):
        target = tmp_path / "new.txt"

        result = FileWriteTool().execute(path=str(target), content="brand new")

        assert result.success is True
        assert target.read_text() == "brand new"

    def test_write_reports_char_count(self, tmp_path: Path):
        target = tmp_path / "out.txt"
        content = "hello"

        result = FileWriteTool().execute(path=str(target), content=content)

        assert result.success is True
        assert str(len(content)) in result.output

    def test_overwrite_replaces_existing_content(self, tmp_path: Path):
        target = tmp_path / "existing.txt"
        target.write_text("old content", encoding="utf-8")

        result = FileWriteTool().execute(path=str(target), content="new content", mode="overwrite")

        assert result.success is True
        assert target.read_text() == "new content"

    def test_append_adds_after_existing_content(self, tmp_path: Path):
        target = tmp_path / "log.txt"
        target.write_text("line1\n", encoding="utf-8")

        result = FileWriteTool().execute(path=str(target), content="line2\n", mode="append")

        assert result.success is True
        assert target.read_text() == "line1\nline2\n"

    def test_invalid_mode_rejected(self, tmp_path: Path):
        result = FileWriteTool().execute(
            path=str(tmp_path / "f.txt"), content="x", mode="truncate"
        )

        assert result.success is False
        assert result.output is None
        assert "invalid mode" in result.error.lower()
        assert "truncate" in result.error

    def test_parent_directories_created(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "file.txt"

        result = FileWriteTool().execute(path=str(target), content="nested")

        assert result.success is True
        assert target.read_text() == "nested"

    def test_write_uses_tilde_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        result = FileWriteTool().execute(path="~/tilde_write.txt", content="via tilde")

        assert result.success is True
        assert (tmp_path / "tilde_write.txt").read_text() == "via tilde"

    def test_write_permission_denied(self, tmp_path: Path):
        # Make the directory read-only so writing into it fails.
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

    def test_write_respects_encoding(self, tmp_path: Path):
        target = tmp_path / "encoded.txt"

        result = FileWriteTool().execute(
            path=str(target), content="caf\u00e9", encoding="utf-8"
        )

        assert result.success is True
        assert target.read_bytes() == "caf\u00e9".encode("utf-8")

    def test_get_schema_structure(self):
        schema = FileWriteTool().get_schema()

        assert schema["name"] == "file_write"
        props = schema["parameters"]["properties"]
        assert "path" in props
        assert "content" in props
        assert "mode" in props
        assert "encoding" in props
        required = schema["parameters"]["required"]
        assert "path" in required
        assert "content" in required

    def test_get_schema_mode_enum(self):
        schema = FileWriteTool().get_schema()
        mode_schema = schema["parameters"]["properties"]["mode"]

        assert "overwrite" in mode_schema["enum"]
        assert "append" in mode_schema["enum"]


# ---------------------------------------------------------------------------
# FileDeleteTool
# ---------------------------------------------------------------------------


class TestFileDeleteTool:
    """Tests for FileDeleteTool.execute and FileDeleteTool.get_schema."""

    def test_delete_existing_file(self, tmp_path: Path):
        target = tmp_path / "to_delete.txt"
        target.write_text("bye", encoding="utf-8")

        result = FileDeleteTool().execute(path=str(target))

        assert result.success is True
        assert not target.exists()
        assert "Deleted" in result.output

    def test_delete_output_includes_path(self, tmp_path: Path):
        target = tmp_path / "named.txt"
        target.write_text("x")
        path_str = str(target)

        result = FileDeleteTool().execute(path=path_str)

        assert path_str in result.output

    def test_delete_file_not_found(self, tmp_path: Path):
        result = FileDeleteTool().execute(path=str(tmp_path / "no_such_file.txt"))

        assert result.success is False
        assert result.output is None
        assert "not found" in result.error.lower()

    def test_delete_directory_rejected(self, tmp_path: Path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = FileDeleteTool().execute(path=str(subdir))

        assert result.success is False
        assert result.output is None
        assert "directories will not be deleted" in result.error.lower()
        assert subdir.exists()  # untouched

    def test_delete_uses_tilde_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "tilde_del.txt"
        target.write_text("delete me")

        result = FileDeleteTool().execute(path="~/tilde_del.txt")

        assert result.success is True
        assert not target.exists()

    def test_delete_permission_denied(self, tmp_path: Path):
        target = tmp_path / "protected.txt"
        target.write_text("locked")
        # Make the parent directory non-writable so unlink is denied.
        tmp_path.chmod(0o555)

        try:
            result = FileDeleteTool().execute(path=str(target))
            assert result.success is False
            assert "permission denied" in result.error.lower()
        finally:
            tmp_path.chmod(0o755)

    def test_get_schema_structure(self):
        schema = FileDeleteTool().get_schema()

        assert schema["name"] == "file_delete"
        props = schema["parameters"]["properties"]
        assert "path" in props
        assert schema["parameters"]["required"] == ["path"]


# ---------------------------------------------------------------------------
# ListFilesTool
# ---------------------------------------------------------------------------


class TestListFilesTool:
    """Tests for ListFilesTool.execute and ListFilesTool.get_schema."""

    def test_list_non_recursive(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = ListFilesTool().execute(path=str(tmp_path))

        assert result.success is True
        assert "a.txt" in result.output
        assert "b.txt" in result.output
        assert "subdir" in result.output

    def test_list_shows_dir_and_file_markers(self, tmp_path: Path):
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "mydir").mkdir()

        result = ListFilesTool().execute(path=str(tmp_path))

        assert "[dir]" in result.output
        assert "[file]" in result.output

    def test_list_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()

        result = ListFilesTool().execute(path=str(empty))

        assert result.success is True
        assert "empty directory" in result.output.lower()

    def test_list_recursive(self, tmp_path: Path):
        nested = tmp_path / "level1" / "level2"
        nested.mkdir(parents=True)
        (nested / "deep.txt").write_text("deep content")

        result = ListFilesTool().execute(path=str(tmp_path), recursive=True)

        assert result.success is True
        assert "deep.txt" in result.output

    def test_list_max_entries_truncates(self, tmp_path: Path):
        for i in range(10):
            (tmp_path / f"file{i:02d}.txt").write_text(str(i))

        result = ListFilesTool().execute(path=str(tmp_path), max_entries=3)

        assert result.success is True
        assert "more entries" in result.output
        assert "7" in result.output  # 10 - 3 = 7 overflow

    def test_list_no_truncation_notice_within_limit(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(str(i))

        result = ListFilesTool().execute(path=str(tmp_path), max_entries=100)

        assert result.success is True
        assert "more entries" not in result.output

    def test_list_path_not_found(self, tmp_path: Path):
        result = ListFilesTool().execute(path=str(tmp_path / "nonexistent"))

        assert result.success is False
        assert result.output is None
        assert "not found" in result.error.lower()

    def test_list_file_rejected_as_directory(self, tmp_path: Path):
        target = tmp_path / "notadir.txt"
        target.write_text("text")

        result = ListFilesTool().execute(path=str(target))

        assert result.success is False
        assert result.output is None
        assert "not a directory" in result.error.lower()

    def test_list_shows_file_sizes(self, tmp_path: Path):
        content = "hello world"
        (tmp_path / "sized.txt").write_text(content, encoding="utf-8")

        result = ListFilesTool().execute(path=str(tmp_path))

        assert result.success is True
        assert "bytes" in result.output

    def test_list_uses_tilde_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "home_file.txt").write_text("here")

        result = ListFilesTool().execute(path="~")

        assert result.success is True
        assert "home_file.txt" in result.output

    def test_get_schema_structure(self):
        schema = ListFilesTool().get_schema()

        assert schema["name"] == "list_files"
        props = schema["parameters"]["properties"]
        assert "path" in props
        assert "recursive" in props
        assert "max_entries" in props
        assert schema["parameters"]["required"] == ["path"]

    def test_get_schema_max_entries_default_documented(self):
        schema = ListFilesTool().get_schema()
        desc = schema["parameters"]["properties"]["max_entries"]["description"]

        assert str(_DEFAULT_MAX_ENTRIES) in desc


# ---------------------------------------------------------------------------
# WebFetchTool
# ---------------------------------------------------------------------------


class TestWebFetchTool:
    """Tests for WebFetchTool.execute and WebFetchTool.get_schema.

    All tests mock missy.gateway.client.create_client to avoid real network
    calls and policy-engine initialisation.
    """

    def _make_response(self, *, text: str, status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.status_code = status_code
        return resp

    def test_successful_fetch(self):
        mock_resp = self._make_response(text="<html>ok</html>")
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is True
        assert result.output == "<html>ok</html>"
        assert result.error is None

    def test_http_error_status_returns_failure(self):
        mock_resp = self._make_response(text="not found", status_code=404)
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com/missing")

        assert result.success is False
        assert "404" in result.error
        assert result.output == "not found"

    def test_http_500_error(self):
        mock_resp = self._make_response(text="server error", status_code=500)
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is False
        assert "500" in result.error

    def test_network_exception_returns_failure(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("network unreachable")

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is False
        assert result.output is None
        assert "network unreachable" in result.error

    def test_response_truncated_at_max_bytes(self):
        # Build a string that will exceed _MAX_RESPONSE_BYTES when utf-8 encoded.
        long_text = "x" * (_MAX_RESPONSE_BYTES + 100)
        mock_resp = self._make_response(text=long_text)
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is True
        assert "truncated" in result.output.lower()
        assert len(result.output) <= _MAX_RESPONSE_BYTES + 50  # truncated + notice

    def test_custom_headers_forwarded(self):
        mock_resp = self._make_response(text="auth ok")
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        headers = {"Authorization": "Bearer mytoken", "X-Custom": "value"}
        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://api.example.com", headers=headers)

        assert result.success is True
        mock_client.get.assert_called_once_with(
            "https://api.example.com", headers=headers
        )

    def test_no_headers_argument_omitted(self):
        """When headers=None, no 'headers' kwarg should be passed to get()."""
        mock_resp = self._make_response(text="ok")
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            WebFetchTool().execute(url="https://example.com")

        call_kwargs = mock_client.get.call_args.kwargs
        assert "headers" not in call_kwargs

    def test_timeout_passed_to_create_client(self):
        mock_resp = self._make_response(text="ok")
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client) as mock_cc:
            WebFetchTool().execute(url="https://example.com", timeout=60)

        mock_cc.assert_called_once_with(
            session_id="web_fetch_tool", task_id="fetch", timeout=60
        )

    def test_get_schema_structure(self):
        schema = WebFetchTool().get_schema()

        assert schema["name"] == "web_fetch"
        props = schema["parameters"]["properties"]
        assert "url" in props
        assert "timeout" in props
        assert "headers" in props
        assert schema["parameters"]["required"] == ["url"]

    def test_get_schema_2xx_boundary_success(self):
        """299 is the last 2xx code; verify it is treated as success."""
        mock_resp = self._make_response(text="partial content", status_code=299)
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is True
        assert result.error is None

    def test_redirect_300_is_not_success(self):
        mock_resp = self._make_response(text="moved", status_code=301)
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("missy.gateway.client.create_client", return_value=mock_client):
            result = WebFetchTool().execute(url="https://example.com")

        assert result.success is False
        assert "301" in result.error


# ---------------------------------------------------------------------------
# DiscordUploadTool
# ---------------------------------------------------------------------------


class TestDiscordUploadTool:
    """Tests for DiscordUploadTool.execute."""

    def test_no_bot_token_returns_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)

        result = DiscordUploadTool().execute(
            file_path="/tmp/file.png", channel_id="123456"
        )

        assert result.success is False
        assert result.output is None
        assert "DISCORD_BOT_TOKEN" in result.error

    def test_successful_upload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot xyzzy")
        target = tmp_path / "image.png"
        target.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG-like header

        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {"id": "9999888877776666"}

        with patch(
            "missy.channels.discord.rest.DiscordRestClient",
            return_value=mock_rest,
        ):
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

    def test_upload_with_empty_caption(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot abc")
        target = tmp_path / "doc.pdf"
        target.write_bytes(b"%PDF")

        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {"id": "42"}

        with patch(
            "missy.channels.discord.rest.DiscordRestClient",
            return_value=mock_rest,
        ):
            result = DiscordUploadTool().execute(
                file_path=str(target), channel_id="777"
            )

        assert result.success is True
        mock_rest.upload_file.assert_called_once_with(
            channel_id="777", file_path=str(target), caption=""
        )

    def test_file_not_found(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot token123")

        mock_rest = MagicMock()
        mock_rest.upload_file.side_effect = FileNotFoundError(
            "No such file: /nonexistent/path.png"
        )

        with patch(
            "missy.channels.discord.rest.DiscordRestClient",
            return_value=mock_rest,
        ):
            result = DiscordUploadTool().execute(
                file_path="/nonexistent/path.png", channel_id="123"
            )

        assert result.success is False
        assert result.output is None
        assert "nonexistent" in result.error

    def test_general_exception_wrapped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot token456")

        mock_rest = MagicMock()
        mock_rest.upload_file.side_effect = RuntimeError("Discord API exploded")

        with patch(
            "missy.channels.discord.rest.DiscordRestClient",
            return_value=mock_rest,
        ):
            result = DiscordUploadTool().execute(
                file_path="/some/file.txt", channel_id="321"
            )

        assert result.success is False
        assert result.output is None
        assert "Upload failed" in result.error
        assert "Discord API exploded" in result.error

    def test_bot_token_passed_to_rest_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        token = "Bot supersecret"
        monkeypatch.setenv("DISCORD_BOT_TOKEN", token)
        target = tmp_path / "f.txt"
        target.write_text("x")

        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {"id": "1"}

        with patch(
            "missy.channels.discord.rest.DiscordRestClient",
            return_value=mock_rest,
        ) as mock_cls:
            DiscordUploadTool().execute(file_path=str(target), channel_id="1")

        mock_cls.assert_called_once_with(bot_token=token)

    def test_message_id_fallback_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When the REST response has no 'id' key, output should show '?'."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "Bot x")
        target = tmp_path / "f.txt"
        target.write_text("x")

        mock_rest = MagicMock()
        mock_rest.upload_file.return_value = {}  # no 'id' key

        with patch(
            "missy.channels.discord.rest.DiscordRestClient",
            return_value=mock_rest,
        ):
            result = DiscordUploadTool().execute(
                file_path=str(target), channel_id="1"
            )

        assert result.success is True
        assert "?" in result.output


# ---------------------------------------------------------------------------
# SelfCreateTool
# ---------------------------------------------------------------------------


class TestSelfCreateTool:
    """Tests for SelfCreateTool.execute with a redirected tools directory."""

    def _tool(self, tools_dir: Path) -> "SelfCreateTool":
        """Return a SelfCreateTool patched to use *tools_dir* as CUSTOM_TOOLS_DIR."""
        from missy.tools.builtin import self_create_tool as mod

        tool = mod.SelfCreateTool()
        # Patch the module-level constant so the tool uses our tmp directory.
        with patch.object(mod, "CUSTOM_TOOLS_DIR", tools_dir):
            return tool

    # -- create --

    def test_create_python_tool(self, tmp_path: Path):
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
        script_file = tmp_path / "my_tool.py"
        assert script_file.exists()
        assert script_file.read_text() == "print('hello')"
        meta_file = tmp_path / "my_tool.json"
        assert meta_file.exists()

    def test_create_bash_tool(self, tmp_path: Path):
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

    def test_create_node_tool(self, tmp_path: Path):
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

    def test_create_script_is_executable(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            mod.SelfCreateTool().execute(
                action="create",
                tool_name="exec_check",
                language="python",
                script="pass",
                tool_description="",
            )

        script_file = tmp_path / "exec_check.py"
        file_stat = script_file.stat()
        assert file_stat.st_mode & stat.S_IXUSR

    def test_create_invalid_tool_name_rejected(self, tmp_path: Path):
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

    def test_create_empty_tool_name_rejected(self, tmp_path: Path):
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

    def test_create_missing_script_rejected(self, tmp_path: Path):
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

    def test_create_unsupported_language_rejected(self, tmp_path: Path):
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

    def test_create_stores_metadata(self, tmp_path: Path):
        import json
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

    # -- list --

    def test_list_empty_returns_no_custom_tools(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        empty = tmp_path / "no_tools"
        with patch.object(mod, "CUSTOM_TOOLS_DIR", empty):
            result = mod.SelfCreateTool().execute(action="list")

        assert result.success is True
        assert "no custom tools" in result.output.lower()

    def test_list_when_dir_does_not_exist(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path / "missing_dir"):
            result = mod.SelfCreateTool().execute(action="list")

        assert result.success is True
        assert "no custom tools" in result.output.lower()

    def test_list_shows_existing_tools(self, tmp_path: Path):
        import json
        import missy.tools.builtin.self_create_tool as mod

        tmp_path.mkdir(parents=True, exist_ok=True)
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
        assert "Beta" in result.output

    def test_list_skips_malformed_json_silently(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "broken.json").write_text("not json {{{{")
        (tmp_path / "good.json").write_text('{"name": "good", "description": "OK"}')

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="list")

        assert result.success is True
        assert "good" in result.output

    # -- delete --

    def test_delete_existing_tool(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        script = tmp_path / "to_remove.py"
        script.write_text("pass")
        meta = tmp_path / "to_remove.json"
        meta.write_text('{"name": "to_remove"}')

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="delete", tool_name="to_remove"
            )

        assert result.success is True
        assert "to_remove" in result.output
        assert not script.exists()
        assert not meta.exists()

    def test_delete_removes_all_script_extensions(self, tmp_path: Path):
        """If a tool has multiple extension files, all are removed."""
        import missy.tools.builtin.self_create_tool as mod

        (tmp_path / "multi.py").write_text("pass")
        (tmp_path / "multi.sh").write_text("#!/bin/bash")

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="delete", tool_name="multi"
            )

        assert result.success is True
        assert not (tmp_path / "multi.py").exists()
        assert not (tmp_path / "multi.sh").exists()

    def test_delete_missing_tool_returns_error(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(
                action="delete", tool_name="does_not_exist"
            )

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_delete_missing_tool_name_returns_error(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="delete", tool_name="")

        assert result.success is False
        assert "tool_name" in result.error.lower()

    # -- unknown action --

    def test_unknown_action_returns_error(self, tmp_path: Path):
        import missy.tools.builtin.self_create_tool as mod

        with patch.object(mod, "CUSTOM_TOOLS_DIR", tmp_path):
            result = mod.SelfCreateTool().execute(action="inspect")

        assert result.success is False
        assert "unknown action" in result.error.lower()
        assert "inspect" in result.error

    # -- schema --

    def test_get_schema_structure(self):
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        schema = SelfCreateTool().get_schema()

        assert schema["name"] == "self_create_tool"
        props = schema["parameters"]["properties"]
        assert "action" in props
        assert "tool_name" in props
        assert "language" in props
        assert "script" in props
        assert "tool_description" in props
        assert "action" in schema["parameters"]["required"]
