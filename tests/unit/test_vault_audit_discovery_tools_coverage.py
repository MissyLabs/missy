"""Targeted tests for remaining coverage gaps.


Covers specific uncovered lines identified in the 99% coverage report:
- vault.py lines 25-26 (crypto import fallback)
- audit_logger.py line 207 (blank line skip in security events)
- discovery.py line 125 (non-dict frontmatter)
- self_create_tool.py line 97 (path traversal in delete loop)
- main.py hatching check debug logging
- incus_tools.py error paths
- atspi_tools.py error paths
"""

from __future__ import annotations

import json
import textwrap
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# vault.py: crypto import fallback (lines 25-26)
# ---------------------------------------------------------------------------
class TestVaultCryptoFallback:
    """Test that vault works when cryptography is not installed."""

    def test_vault_raises_when_crypto_unavailable(self, tmp_path):
        """Vault should raise VaultError when crypto package is missing."""
        from missy.security.vault import Vault, VaultError

        with patch("missy.security.vault._CRYPTO_AVAILABLE", False), pytest.raises(VaultError, match="cryptography"):
            Vault(vault_dir=str(tmp_path))

    def test_crypto_available_flag_is_true(self):
        """When cryptography is installed, _CRYPTO_AVAILABLE should be True."""
        from missy.security.vault import _CRYPTO_AVAILABLE

        assert _CRYPTO_AVAILABLE is True


# ---------------------------------------------------------------------------
# audit_logger.py: blank line handling in get_policy_violations (line 207)
# ---------------------------------------------------------------------------
class TestAuditLoggerBlankLines:
    """Test that blank lines in audit log are properly skipped."""

    def test_policy_violations_with_blank_lines(self, tmp_path):
        """Blank lines and whitespace in audit log should be skipped."""
        from missy.observability.audit_logger import AuditLogger

        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        deny_event = {
            "event_type": "network.request",
            "category": "network",
            "result": "deny",
            "timestamp": "2026-03-18T00:00:00Z",
            "detail": {"host": "evil.com"},
        }
        allow_event = {
            "event_type": "network.request",
            "category": "network",
            "result": "allow",
            "timestamp": "2026-03-18T00:00:01Z",
            "detail": {"host": "api.anthropic.com"},
        }

        with open(log_path, "w") as f:
            f.write(json.dumps(deny_event) + "\n")
            f.write("\n")  # blank line
            f.write("   \n")  # whitespace-only line
            f.write(json.dumps(allow_event) + "\n")

        events = logger.get_policy_violations(limit=10)
        assert len(events) == 1
        assert events[0]["result"] == "deny"

    def test_policy_violations_with_invalid_json_lines(self, tmp_path):
        """Invalid JSON lines should be silently skipped."""
        from missy.observability.audit_logger import AuditLogger

        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        deny_event = {
            "event_type": "policy.violation",
            "category": "security",
            "result": "deny",
            "timestamp": "2026-03-18T00:00:00Z",
            "detail": {},
        }

        with open(log_path, "w") as f:
            f.write("{invalid json}\n")
            f.write(json.dumps(deny_event) + "\n")

        events = logger.get_policy_violations(limit=10)
        assert len(events) == 1

    def test_policy_violations_respects_limit(self, tmp_path):
        """Should return at most `limit` violations."""
        from missy.observability.audit_logger import AuditLogger

        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        with open(log_path, "w") as f:
            for i in range(20):
                event = {
                    "event_type": f"test.event.{i}",
                    "result": "deny",
                    "timestamp": f"2026-03-18T00:00:{i:02d}Z",
                }
                f.write(json.dumps(event) + "\n")

        events = logger.get_policy_violations(limit=5)
        assert len(events) == 5


# ---------------------------------------------------------------------------
# discovery.py: non-dict frontmatter (line 125)
# ---------------------------------------------------------------------------
class TestSkillDiscoveryNonDictFrontmatter:
    """Test that non-dict YAML frontmatter raises ValueError."""

    def test_scalar_frontmatter_raises(self, tmp_path):
        """When _parse_yaml returns a non-dict, ValueError should be raised."""
        from missy.skills.discovery import SkillDiscovery

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            textwrap.dedent("""\
            ---
            name: test
            ---
            # A skill
            """)
        )

        discovery = SkillDiscovery()
        # Force _parse_yaml to return a scalar to hit the isinstance check
        with patch.object(
            SkillDiscovery, "_parse_yaml", staticmethod(lambda text: "just a string")
        ), pytest.raises(ValueError, match="not a YAML mapping"):
            discovery.parse_skill_md(str(skill_file))

    def test_list_frontmatter_raises(self, tmp_path):
        """When _parse_yaml returns a list, ValueError should be raised."""
        from missy.skills.discovery import SkillDiscovery

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            textwrap.dedent("""\
            ---
            name: test
            ---
            # A skill
            """)
        )

        discovery = SkillDiscovery()
        with patch.object(
            SkillDiscovery, "_parse_yaml", staticmethod(lambda text: ["item1", "item2"])
        ), pytest.raises(ValueError, match="not a YAML mapping"):
            discovery.parse_skill_md(str(skill_file))


# ---------------------------------------------------------------------------
# self_create_tool.py: delete action paths
# ---------------------------------------------------------------------------
class TestSelfCreateToolDeletePaths:
    """Test SelfCreateTool delete action paths."""

    def test_delete_existing_tool(self, tmp_path):
        """Deleting an existing custom tool should succeed."""
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        # Create a tool file
        (tools_dir / "mytool.sh").write_text("#!/bin/bash\necho hello")
        (tools_dir / "mytool.json").write_text('{"name": "mytool", "description": "test"}')

        tool = SelfCreateTool()
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tools_dir):
            result = tool.execute(action="delete", tool_name="mytool")
            assert result.success is True
            assert not (tools_dir / "mytool.sh").exists()
            assert not (tools_dir / "mytool.json").exists()

    def test_delete_nonexistent_tool(self, tmp_path):
        """Deleting a non-existent tool should report not found."""
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        tool = SelfCreateTool()
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tools_dir):
            result = tool.execute(action="delete", tool_name="nonexistent")
            assert result.success is False or "not found" in (result.output + (result.error or "")).lower()

    def test_delete_invalid_name_rejected(self, tmp_path):
        """Tool names with special chars should be rejected."""
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        tool = SelfCreateTool()
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(action="delete", tool_name="../../../etc/passwd")
            assert result.success is False

    def test_list_empty_directory(self, tmp_path):
        """Listing tools when directory is empty should return empty message."""
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        tool = SelfCreateTool()
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tools_dir):
            result = tool.execute(action="list")
            assert result.success is True

    def test_list_no_directory(self, tmp_path):
        """Listing tools when directory doesn't exist should not crash."""
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        tool = SelfCreateTool()
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path / "nonexist"):
            result = tool.execute(action="list")
            assert result.success is True


# ---------------------------------------------------------------------------
# incus_tools.py: error handling paths
# ---------------------------------------------------------------------------
class TestIncusToolsErrorPaths:
    """Test incus tools graceful error handling."""

    def test_incus_list_binary_not_found(self):
        """IncusListTool should handle missing incus binary."""
        from missy.tools.builtin.incus_tools import IncusListTool

        tool = IncusListTool()
        with patch("subprocess.run", side_effect=FileNotFoundError("incus not found")):
            result = tool.execute()
            assert result.success is False
            assert "not found" in (result.error or "").lower()

    def test_incus_list_command_failure(self):
        """IncusListTool should handle command execution failure."""
        from missy.tools.builtin.incus_tools import IncusListTool

        tool = IncusListTool()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"permission denied"
        mock_result.stdout = b""

        with patch("subprocess.run", return_value=mock_result):
            result = tool.execute()
            assert result.success is False

    def test_incus_list_timeout(self):
        """IncusListTool should handle timeout gracefully."""
        import subprocess

        from missy.tools.builtin.incus_tools import IncusListTool

        tool = IncusListTool()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("incus", 30)):
            result = tool.execute()
            assert result.success is False
            assert "timed out" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# atspi_tools.py: graceful degradation
# ---------------------------------------------------------------------------
class TestAtSpiToolsGracefulDegradation:
    """Test atspi_tools handles missing dependencies."""

    def test_get_tree_handles_import_error(self):
        """AtSpiGetTreeTool should handle missing AT-SPI gracefully."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        tool = AtSpiGetTreeTool()
        with patch(
            "missy.tools.builtin.atspi_tools._get_desktop",
            side_effect=Exception("AT-SPI not available"),
        ):
            result = tool.execute()
            assert result.success is False or "error" in (result.output + (result.error or "")).lower()

    def test_click_handles_missing_element(self):
        """AtSpiClickTool should handle missing elements gracefully."""
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        tool = AtSpiClickTool()
        with patch(
            "missy.tools.builtin.atspi_tools._get_desktop",
            side_effect=Exception("AT-SPI not available"),
        ):
            result = tool.execute(role="push button", name="OK")
            assert result.success is False or "error" in (result.output + (result.error or "")).lower()


# ---------------------------------------------------------------------------
# main.py: hatching check debug logging
# ---------------------------------------------------------------------------
class TestMainHatchingDebugLogging:
    """Test that hatching check failures are logged at debug level."""

    def test_ask_hatching_exception_logged(self, tmp_path):
        """When hatching check raises in 'ask', it should be caught and logged."""
        from click.testing import CliRunner

        from missy.cli.main import cli

        runner = CliRunner()

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "config_version: 2\nproviders:\n  anthropic:\n    name: anthropic\n"
            "    model: claude-sonnet-4-6\n    timeout: 30\n    enabled: true\n"
        )

        with (
            patch(
                "missy.agent.hatching.HatchingManager.needs_hatching",
                side_effect=RuntimeError("hatching broken"),
            ),
            patch("missy.cli.main.logger") as mock_logger,
        ):
            runner.invoke(cli, ["--config", str(cfg_path), "ask", "test"])
            mock_logger.debug.assert_called()


# ---------------------------------------------------------------------------
# Additional edge case: AuditLogger with empty file
# ---------------------------------------------------------------------------
class TestAuditLoggerEdgeCases:
    """Additional edge case tests for audit logger."""

    def test_get_recent_events_empty_file(self, tmp_path):
        """get_recent_events on empty file should return empty list."""
        from missy.observability.audit_logger import AuditLogger

        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("")

        logger = AuditLogger(str(log_path))
        events = logger.get_recent_events(limit=10)
        assert events == []

    def test_get_recent_events_nonexistent_file(self, tmp_path):
        """get_recent_events on missing file should return empty list."""
        from missy.observability.audit_logger import AuditLogger

        logger = AuditLogger(str(tmp_path / "nonexistent.jsonl"))
        events = logger.get_recent_events(limit=10)
        assert events == []

    def test_policy_violations_nonexistent_file(self, tmp_path):
        """get_policy_violations on missing file should return empty list."""
        from missy.observability.audit_logger import AuditLogger

        logger = AuditLogger(str(tmp_path / "nonexistent.jsonl"))
        events = logger.get_policy_violations(limit=10)
        assert events == []
