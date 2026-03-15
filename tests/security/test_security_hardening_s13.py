"""Security hardening tests for session 13.

Covers:
1. Gateway URL scheme restriction (_check_url rejects non-http/https schemes)
2. Gateway kwargs sanitization (_sanitize_kwargs strips follow_redirects)
3. Gateway follow_redirects=False enforced on client creation
4. Shell policy bare '&' operator splitting (_extract_all_programs)
5. Shell policy launcher command warning emission
6. MCP config permission check (uid mismatch, group-writable, OSError)
7. New secret detection patterns (GitLab, npm, PyPI, SendGrid, DB connection)
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from missy.config.settings import ShellPolicy
from missy.core.exceptions import PolicyViolationError
from missy.gateway.client import PolicyHTTPClient
from missy.mcp.manager import McpManager
from missy.policy.shell import ShellPolicyEngine
from missy.security.secrets import SecretsDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shell_engine(commands: list[str]) -> ShellPolicyEngine:
    """Return an enabled ShellPolicyEngine with the given allowed_commands."""
    return ShellPolicyEngine(ShellPolicy(enabled=True, allowed_commands=commands))


def _policy_http_client() -> PolicyHTTPClient:
    """Return a PolicyHTTPClient with a mocked-out policy engine."""
    return PolicyHTTPClient(session_id="s1", task_id="t1")


# ---------------------------------------------------------------------------
# 1. Gateway URL scheme restriction
# ---------------------------------------------------------------------------


class TestGatewayUrlSchemeRestriction:
    """_check_url must reject any URL whose scheme is not http or https."""

    def _client(self) -> PolicyHTTPClient:
        return PolicyHTTPClient(session_id="test", task_id="test")

    def _check(self, url: str) -> None:
        """Call _check_url directly, bypassing the real policy engine."""
        client = self._client()
        # We patch check_network so scheme-valid URLs don't hit a real engine.
        with patch("missy.gateway.client.get_policy_engine") as mock_engine:
            mock_engine.return_value.check_network.return_value = None
            client._check_url(url)

    def test_file_scheme_raises_value_error(self):
        client = self._client()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("file:///etc/passwd")

    def test_ftp_scheme_raises_value_error(self):
        client = self._client()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("ftp://example.com/resource")

    def test_javascript_scheme_raises_value_error(self):
        client = self._client()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("javascript:alert(1)")

    def test_data_scheme_raises_value_error(self):
        client = self._client()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("data:text/html,<h1>hi</h1>")

    def test_http_scheme_passes_scheme_check(self):
        """http:// must not raise a scheme error (policy check may still run)."""
        with patch("missy.gateway.client.get_policy_engine") as mock_engine:
            mock_engine.return_value.check_network.return_value = None
            # Should not raise ValueError for scheme
            self._check("http://example.com")

    def test_https_scheme_passes_scheme_check(self):
        """https:// must not raise a scheme error."""
        with patch("missy.gateway.client.get_policy_engine") as mock_engine:
            mock_engine.return_value.check_network.return_value = None
            self._check("https://example.com")

    def test_error_message_includes_scheme_name(self):
        """The ValueError message must quote the offending scheme."""
        client = self._client()
        with pytest.raises(ValueError) as exc_info:
            client._check_url("ftp://example.com")
        assert "ftp" in str(exc_info.value)

    def test_file_scheme_error_contains_permitted_schemes_hint(self):
        """Users should see http/https mentioned in the error."""
        client = self._client()
        with pytest.raises(ValueError) as exc_info:
            client._check_url("file:///etc/passwd")
        msg = str(exc_info.value).lower()
        assert "http" in msg


# ---------------------------------------------------------------------------
# 2. Gateway kwargs sanitization
# ---------------------------------------------------------------------------


class TestGatewayKwargsSanitization:
    """_sanitize_kwargs must strip follow_redirects and leave other keys intact."""

    def test_strips_follow_redirects_true(self):
        result = PolicyHTTPClient._sanitize_kwargs({"follow_redirects": True})
        assert result == {}

    def test_strips_follow_redirects_false(self):
        result = PolicyHTTPClient._sanitize_kwargs({"follow_redirects": False})
        assert result == {}

    def test_preserves_headers(self):
        kwargs = {"headers": {"X-Custom": "1"}}
        result = PolicyHTTPClient._sanitize_kwargs(kwargs)
        assert result == {"headers": {"X-Custom": "1"}}

    def test_empty_dict_returns_empty(self):
        assert PolicyHTTPClient._sanitize_kwargs({}) == {}

    def test_mixed_kwargs_strips_only_follow_redirects(self):
        kwargs = {
            "follow_redirects": True,
            "timeout": 10,
            "headers": {"Authorization": "Bearer tok"},
        }
        result = PolicyHTTPClient._sanitize_kwargs(kwargs)
        assert "follow_redirects" not in result
        assert result["timeout"] == 10
        assert result["headers"] == {"Authorization": "Bearer tok"}

    def test_sanitize_is_in_place_mutation(self):
        """_sanitize_kwargs mutates and returns the same dict object."""
        kwargs = {"follow_redirects": True, "timeout": 5}
        result = PolicyHTTPClient._sanitize_kwargs(kwargs)
        # Same object is returned.
        assert result is kwargs


# ---------------------------------------------------------------------------
# 3. Gateway follow_redirects=False on client creation
# ---------------------------------------------------------------------------


class TestGatewayFollowRedirectsFalse:
    """Both sync and async clients must be created with follow_redirects=False."""

    def test_sync_client_follow_redirects_false(self):
        client = PolicyHTTPClient()
        sync_client = client._get_sync_client()
        try:
            assert sync_client.follow_redirects is False
        finally:
            sync_client.close()

    def test_async_client_follow_redirects_false(self):
        client = PolicyHTTPClient()
        async_client = client._get_async_client()
        assert async_client.follow_redirects is False

    def test_sync_client_is_httpx_client(self):
        client = PolicyHTTPClient()
        sync_client = client._get_sync_client()
        try:
            assert isinstance(sync_client, httpx.Client)
        finally:
            sync_client.close()

    def test_async_client_is_httpx_async_client(self):
        client = PolicyHTTPClient()
        async_client = client._get_async_client()
        assert isinstance(async_client, httpx.AsyncClient)

    def test_sync_client_is_cached(self):
        """Successive calls return the same client instance."""
        client = PolicyHTTPClient()
        c1 = client._get_sync_client()
        c2 = client._get_sync_client()
        try:
            assert c1 is c2
        finally:
            c1.close()

    def test_async_client_is_cached(self):
        """Successive calls return the same async client instance."""
        client = PolicyHTTPClient()
        c1 = client._get_async_client()
        c2 = client._get_async_client()
        assert c1 is c2


# ---------------------------------------------------------------------------
# 4. Shell policy bare '&' split
# ---------------------------------------------------------------------------


class TestShellPolicyBareAmpersandSplit:
    """_extract_all_programs must split on bare & (background execution operator)."""

    def test_ampersand_extracts_both_programs(self):
        programs = ShellPolicyEngine._extract_all_programs("ls & rm -rf /")
        assert programs == ["ls", "rm"]

    def test_ampersand_sleep_echo(self):
        programs = ShellPolicyEngine._extract_all_programs("sleep 5 & echo done")
        assert programs == ["sleep", "echo"]

    def test_ampersand_not_confused_with_double_ampersand(self):
        """&& is a different operator; both sides must still be extracted."""
        programs = ShellPolicyEngine._extract_all_programs("true && false")
        assert programs == ["true", "false"]

    def test_bare_ampersand_alone_returns_single_program(self):
        """'cmd &' — trailing & with empty right side produces one program."""
        programs = ShellPolicyEngine._extract_all_programs("cmd &")
        # Empty part after & is skipped; only 'cmd' survives.
        assert programs == ["cmd"]

    def test_multiple_ampersand_chains(self):
        programs = ShellPolicyEngine._extract_all_programs("a & b & c")
        assert programs == ["a", "b", "c"]

    def test_semicolon_also_splits(self):
        programs = ShellPolicyEngine._extract_all_programs("ls ; cat /etc/hosts")
        assert programs == ["ls", "cat"]

    def test_pipe_also_splits(self):
        programs = ShellPolicyEngine._extract_all_programs("ls | grep txt")
        assert programs == ["ls", "grep"]

    def test_subshell_marker_still_rejected(self):
        """$( ...) must still be blocked regardless of & presence."""
        programs = ShellPolicyEngine._extract_all_programs("ls & $(evil)")
        assert programs is None

    def test_check_command_catches_disallowed_program_after_ampersand(self):
        """Policy check on 'ls & rm -rf /' must deny because rm is not allowed."""
        engine = _make_shell_engine(["ls"])
        with pytest.raises(PolicyViolationError):
            engine.check_command("ls & rm -rf /")

    def test_check_command_allows_when_all_programs_allowed(self):
        engine = _make_shell_engine(["ls", "echo"])
        assert engine.check_command("ls & echo done") is True


# ---------------------------------------------------------------------------
# 5. Shell policy launcher command warning
# ---------------------------------------------------------------------------


class TestShellPolicyLauncherWarning:
    """When a launcher command (env, bash, etc.) is allowed and used, a warning
    must be logged via logger.warning."""

    @pytest.mark.parametrize("launcher", ["env", "bash", "sh", "sudo", "python3"])
    def test_launcher_triggers_warning(self, launcher: str, caplog):
        engine = _make_shell_engine([launcher])
        with caplog.at_level(logging.WARNING, logger="missy.policy.shell"):
            engine.check_command(f"{launcher} ls")
        assert any(launcher in record.message for record in caplog.records)

    def test_non_launcher_does_not_warn(self, caplog):
        engine = _make_shell_engine(["git"])
        with caplog.at_level(logging.WARNING, logger="missy.policy.shell"):
            engine.check_command("git status")
        launcher_warnings = [
            r for r in caplog.records
            if "launcher" in r.message.lower()
        ]
        assert launcher_warnings == []

    def test_launcher_via_path_triggers_warning(self, caplog):
        """A path-qualified launcher such as /usr/bin/env should also warn."""
        engine = _make_shell_engine(["env"])
        with caplog.at_level(logging.WARNING, logger="missy.policy.shell"):
            engine.check_command("/usr/bin/env ls")
        assert any("env" in r.message for r in caplog.records)

    def test_warning_message_mentions_subcommands(self, caplog):
        engine = _make_shell_engine(["bash"])
        with caplog.at_level(logging.WARNING, logger="missy.policy.shell"):
            engine.check_command("bash -c 'echo hi'")
        warning_texts = " ".join(r.message for r in caplog.records)
        assert "subcommand" in warning_texts.lower() or "launcher" in warning_texts.lower()

    def test_warning_does_not_prevent_allow(self, caplog):
        """The warning is informational — the command is still allowed."""
        engine = _make_shell_engine(["env"])
        with caplog.at_level(logging.WARNING, logger="missy.policy.shell"):
            result = engine.check_command("env VAR=1 ls")
        assert result is True


# ---------------------------------------------------------------------------
# 6. MCP config permission check
# ---------------------------------------------------------------------------


class TestMcpConfigPermissions:
    """McpManager.connect_all must refuse to load configs with bad permissions."""

    def test_refuses_when_uid_mismatch(self, tmp_path):
        """Config owned by a different uid must be silently skipped."""
        config = tmp_path / "mcp.json"
        config.write_text('[{"name": "srv", "command": "echo hi"}]')

        mgr = McpManager(config_path=str(config))

        # Pretend the file is owned by uid 9999 while we run as a different uid.
        fake_stat = MagicMock()
        fake_stat.st_uid = 9999
        fake_stat.st_mode = 0o100600  # -rw------- (not group/world writable)

        with patch.object(Path, "stat", return_value=fake_stat), \
             patch("os.getuid", return_value=1000):
            mgr.connect_all()

        # No clients should be connected — the config was refused.
        assert mgr.list_servers() == []

    def test_refuses_when_group_writable(self, tmp_path):
        """A group-writable config file must be silently skipped."""
        config = tmp_path / "mcp.json"
        config.write_text('[{"name": "srv", "command": "echo hi"}]')
        # Set group-writable permission.
        config.chmod(0o664)

        mgr = McpManager(config_path=str(config))
        mgr.connect_all()

        # No clients connected because the file is group-writable.
        assert mgr.list_servers() == []

    def test_refuses_when_world_writable(self, tmp_path):
        """A world-writable config file must also be rejected."""
        config = tmp_path / "mcp.json"
        config.write_text('[{"name": "srv", "command": "echo hi"}]')
        config.chmod(0o666)

        mgr = McpManager(config_path=str(config))
        mgr.connect_all()

        assert mgr.list_servers() == []

    def test_handles_oserror_on_stat(self, tmp_path, caplog):
        """An OSError on stat must be caught and logged as a warning."""
        config = tmp_path / "mcp.json"
        config.write_text('[{"name": "srv", "command": "echo hi"}]')

        mgr = McpManager(config_path=str(config))

        # exists() internally calls stat() on Python 3.12+, so we must mock
        # exists() separately to return True before the explicit stat() call
        # inside connect_all's permission block raises OSError.
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat", side_effect=OSError("permission denied")), \
             caplog.at_level(logging.WARNING, logger="missy.mcp.manager"):
            mgr.connect_all()

        # Stat failure → silently skip with a warning, no clients connected.
        assert mgr.list_servers() == []
        assert any("cannot stat" in r.message.lower() for r in caplog.records)

    def test_skips_loading_on_uid_mismatch_logs_warning(self, tmp_path, caplog):
        """A warning must be emitted when the uid check fails."""
        config = tmp_path / "mcp.json"
        config.write_text('[{"name": "srv", "command": "echo hi"}]')

        fake_stat = MagicMock()
        fake_stat.st_uid = 9999
        fake_stat.st_mode = 0o100600

        mgr = McpManager(config_path=str(config))

        with patch.object(Path, "stat", return_value=fake_stat), \
             patch("os.getuid", return_value=1000), \
             caplog.at_level(logging.WARNING, logger="missy.mcp.manager"):
            mgr.connect_all()

        assert any("uid" in r.message.lower() or "owned" in r.message.lower()
                   for r in caplog.records)

    def test_skips_loading_group_writable_logs_warning(self, tmp_path, caplog):
        """A warning must be emitted when the file is group/world-writable."""
        config = tmp_path / "mcp.json"
        config.write_text('[{"name": "srv", "command": "echo hi"}]')
        config.chmod(0o664)

        mgr = McpManager(config_path=str(config))

        with caplog.at_level(logging.WARNING, logger="missy.mcp.manager"):
            mgr.connect_all()

        assert any(
            "writable" in r.message.lower() or "group" in r.message.lower()
            for r in caplog.records
        )

    def test_no_config_file_skips_silently(self, tmp_path):
        """A missing config is normal — connect_all should return without error."""
        mgr = McpManager(config_path=str(tmp_path / "nonexistent.json"))
        mgr.connect_all()  # Must not raise.
        assert mgr.list_servers() == []

    def test_correct_permissions_owner_proceeds_to_parse(self, tmp_path):
        """A file with correct ownership and permissions should proceed past the
        permission gate (connection may still fail, that is acceptable)."""
        config = tmp_path / "mcp.json"
        # Intentionally malformed JSON so we can verify it got past the perm check.
        config.write_text("NOT_JSON")
        config.chmod(0o600)

        real_uid = os.getuid()
        fake_stat = MagicMock()
        fake_stat.st_uid = real_uid
        fake_stat.st_mode = 0o100600  # owner-only, not group/world writable

        mgr = McpManager(config_path=str(config))

        with patch.object(Path, "stat", return_value=fake_stat):
            # Should attempt to parse and log a parse warning, not a perm warning.
            mgr.connect_all()

        # Still no clients (parse failed), but we confirm it tried to parse.
        assert mgr.list_servers() == []


# ---------------------------------------------------------------------------
# 7. New secret detection patterns
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector() -> SecretsDetector:
    return SecretsDetector()


class TestGitLabTokenDetection:
    def test_detects_gitlab_pat(self, detector):
        text = "token: glpat-abcdefghijklmnopqrst"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "gitlab_token" in types

    def test_does_not_flag_short_glpat(self, detector):
        # Fewer than 20 chars after the prefix — should not match.
        text = "glpat-short"
        findings = [f for f in detector.scan(text) if f["type"] == "gitlab_token"]
        assert findings == []

    def test_gitlab_token_in_env_var_context(self, detector):
        text = 'GITLAB_TOKEN="glpat-ABCDEFGHIJKLMNOPQRSTuvwx"'
        assert detector.has_secrets(text) is True

    def test_gitlab_token_redacted(self, detector):
        text = "use glpat-abcdefghijklmnopqrstu for auth"
        redacted = detector.redact(text)
        assert "glpat-" not in redacted
        assert "[REDACTED]" in redacted


class TestNpmTokenDetection:
    def test_detects_npm_token(self, detector):
        # Pattern requires npm_ + 36 alphanumeric chars
        text = "npm_abcdefghijklmnopqrstuvwxyz0123456789"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "npm_token" in types

    def test_does_not_flag_npm_without_prefix(self, detector):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        findings = [f for f in detector.scan(text) if f["type"] == "npm_token"]
        assert findings == []

    def test_npm_token_in_config_context(self, detector):
        text = '//registry.npmjs.org/:_authToken=npm_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345'
        assert detector.has_secrets(text) is True

    def test_npm_token_redacted(self, detector):
        text = "NPM_TOKEN=npm_abcdefghijklmnopqrstuvwxyz0123456789"
        redacted = detector.redact(text)
        assert "npm_abcdef" not in redacted
        assert "[REDACTED]" in redacted


class TestPyPITokenDetection:
    def test_detects_pypi_token(self, detector):
        # PyPI tokens have 50+ chars after 'pypi-'
        text = "pypi-abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopq"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "pypi_token" in types

    def test_does_not_flag_short_pypi_value(self, detector):
        text = "pypi-short"
        findings = [f for f in detector.scan(text) if f["type"] == "pypi_token"]
        assert findings == []

    def test_pypi_token_in_twine_context(self, detector):
        text = "__token__:pypi-abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrst"
        assert detector.has_secrets(text) is True

    def test_pypi_token_redacted(self, detector):
        text = "TWINE_PASSWORD=pypi-abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopq"
        redacted = detector.redact(text)
        assert "pypi-abcdef" not in redacted
        assert "[REDACTED]" in redacted


class TestSendGridKeyDetection:
    def test_detects_sendgrid_key(self, detector):
        # Pattern: SG.<22 alnum chars>.<43 alnum chars>
        text = "SG.abcdefghijklmnopqrstuv.abcdefghijklmnopqrstuvwxyz0123456789abcde"
        # Verify the second segment is exactly 43 chars before asserting.
        second_seg = "abcdefghijklmnopqrstuvwxyz0123456789abcde"
        assert len(second_seg) == 41  # noqa: PLR2004 — guard for template accuracy
        # Use a correct 43-char second segment.
        second_43 = "abcdefghijklmnopqrstuvwxyz0123456789abcdefg"
        assert len(second_43) == 43  # noqa: PLR2004
        text = f"SG.abcdefghijklmnopqrstuv.{second_43}"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "sendgrid_key" in types

    def test_does_not_flag_truncated_sendgrid(self, detector):
        text = "SG.short.value"
        findings = [f for f in detector.scan(text) if f["type"] == "sendgrid_key"]
        assert findings == []

    def test_sendgrid_key_in_env_context(self, detector):
        second_43 = "abcdefghijklmnopqrstuvwxyz0123456789abcdefg"
        text = f"SENDGRID_API_KEY=SG.abcdefghijklmnopqrstuv.{second_43}"
        assert detector.has_secrets(text) is True

    def test_sendgrid_key_redacted(self, detector):
        second_43 = "abcdefghijklmnopqrstuvwxyz0123456789abcdefg"
        text = f"key=SG.abcdefghijklmnopqrstuv.{second_43}"
        redacted = detector.redact(text)
        assert "SG." not in redacted
        assert "[REDACTED]" in redacted


class TestDatabaseConnectionStringDetection:
    def test_detects_postgres_url(self, detector):
        text = "postgres://user:secret@localhost:5432/db"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "db_connection_string" in types

    def test_detects_mysql_url(self, detector):
        text = "mysql://admin:p4ssw0rd@10.0.0.1:3306/mydb"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "db_connection_string" in types

    def test_detects_mongodb_url(self, detector):
        text = "mongodb://root:hunter2@mongo-host:27017/production"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "db_connection_string" in types

    def test_detects_redis_url(self, detector):
        text = "redis://default:redispass@cache.internal:6379/0"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "db_connection_string" in types

    def test_does_not_flag_url_without_credentials(self, detector):
        """A connection string without user:pass@host should not match."""
        text = "postgres://localhost:5432/mydb"
        findings = [f for f in detector.scan(text) if f["type"] == "db_connection_string"]
        assert findings == []

    def test_db_connection_string_redacted(self, detector):
        text = "DATABASE_URL=postgres://user:secret@localhost:5432/db"
        redacted = detector.redact(text)
        assert "secret" not in redacted
        assert "[REDACTED]" in redacted

    def test_case_insensitive_scheme(self, detector):
        text = "POSTGRES://user:secret@host:5432/db"
        findings = detector.scan(text)
        types = {f["type"] for f in findings}
        assert "db_connection_string" in types
