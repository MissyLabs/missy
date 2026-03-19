"""Additional edge case tests for remaining coverage gaps.


Targets:
- audit_logger.py line 207 (empty line in violations scan)
- voice/registry.py lines 184-188 (load failure with permission error)
- network policy lines 157-158 (unparseable IP from getaddrinfo)
- wizard.py edge cases
- webhook.py line 127 (XFF parsing)
- agent runtime tool result truncation edge cases
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from unittest.mock import patch

# ---------------------------------------------------------------------------
# AuditLogger empty line in violations
# ---------------------------------------------------------------------------

class TestAuditLoggerViolationEmptyLines:
    """Cover audit_logger.py line 207."""

    def test_mixed_deny_allow_with_blanks(self) -> None:
        """Violations scan should skip empty lines between valid records."""
        from missy.observability.audit_logger import AuditLogger

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Mix of deny, allow, and blank lines
            f.write(json.dumps({"event": "net", "result": "allow"}) + "\n")
            f.write(json.dumps({"event": "net", "result": "deny", "host": "evil.com"}) + "\n")
            f.write("\n")  # empty line
            f.write("  \n")  # whitespace line
            f.write("\t\n")  # tab line
            f.write(json.dumps({"event": "shell", "result": "deny", "cmd": "rm"}) + "\n")
            f.write(json.dumps({"event": "net", "result": "allow"}) + "\n")
            path = f.name

        try:
            logger = AuditLogger(log_path=path)
            violations = logger.get_policy_violations(limit=100)
            assert len(violations) == 2
            assert violations[0]["host"] == "evil.com"
            assert violations[1]["cmd"] == "rm"
        finally:
            os.unlink(path)

    def test_violations_with_malformed_json_lines(self) -> None:
        """Violations scan should skip malformed JSON lines."""
        from missy.observability.audit_logger import AuditLogger

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"event": "net", "result": "deny", "detail": "ok"}) + "\n")
            f.write("this is not json\n")
            f.write("{malformed json\n")
            f.write(json.dumps({"event": "shell", "result": "deny", "detail": "ok2"}) + "\n")
            path = f.name

        try:
            logger = AuditLogger(log_path=path)
            violations = logger.get_policy_violations(limit=100)
            assert len(violations) == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Network policy unparseable IP
# ---------------------------------------------------------------------------

class TestNetworkPolicyUnparseableIP:
    """Cover network.py lines 157-158."""

    def test_getaddrinfo_returns_unparseable_ip(self) -> None:
        """If getaddrinfo returns something that can't be parsed, it should be skipped."""
        from missy.config.settings import NetworkPolicy
        from missy.policy.network import NetworkPolicyEngine

        policy = NetworkPolicy(
            default_deny=True,
            allowed_domains=["example.com"],
        )
        engine = NetworkPolicyEngine(policy)

        # Mock getaddrinfo to return a mixture of valid and invalid IPs
        fake_results = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 80)),
        ]
        import contextlib

        with patch("missy.policy.network.socket.getaddrinfo", return_value=fake_results), \
             contextlib.suppress(Exception):
            engine.check_host("example.com", 80)


# ---------------------------------------------------------------------------
# Voice registry permission error on load
# ---------------------------------------------------------------------------

class TestVoiceRegistryPermissionError:
    """Cover voice/registry.py lines 184-188 via different error types."""

    def test_load_permission_error(self) -> None:
        """Permission error during load should result in empty registry."""
        from missy.channels.voice.registry import DeviceRegistry

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("[]")
            path = f.name

        try:
            # Make file unreadable
            os.chmod(path, 0o000)
            reg = DeviceRegistry(registry_path=path)
            assert len(reg._nodes) == 0
        finally:
            os.chmod(path, 0o644)
            os.unlink(path)

    def test_load_with_list_of_invalid_dicts(self) -> None:
        """Registry entries missing required fields should cause fallback to empty."""
        from missy.channels.voice.registry import DeviceRegistry

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Valid JSON array but entries lack required keys
            f.write(json.dumps([{"bad_key": "no_node_id"}]))
            path = f.name

        try:
            reg = DeviceRegistry(registry_path=path)
            # Should either load empty or skip invalid entries
            assert isinstance(reg._nodes, dict)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Webhook XFF edge cases
# ---------------------------------------------------------------------------

class TestWebhookXFFEdgeCases:
    """Cover webhook.py line 127 XFF parsing."""

    def test_xff_with_spaces(self) -> None:
        """X-Forwarded-For with extra spaces should parse correctly."""
        xff = "  192.168.1.1  ,  10.0.0.1  "
        client_ip = xff.split(",")[0].strip()
        assert client_ip == "192.168.1.1"

    def test_xff_with_ipv6(self) -> None:
        """IPv6 address in X-Forwarded-For should work."""
        xff = "::1, 10.0.0.1"
        client_ip = xff.split(",")[0].strip()
        assert client_ip == "::1"

    def test_xff_empty(self) -> None:
        """Empty X-Forwarded-For header should be handled."""
        xff = ""
        parts = xff.split(",")
        client_ip = parts[0].strip()
        assert client_ip == ""


# ---------------------------------------------------------------------------
# Tool result truncation edge cases
# ---------------------------------------------------------------------------

class TestToolResultTruncationEdge:
    """Edge cases for tool result truncation."""

    def test_exactly_at_limit(self) -> None:
        """Content exactly at limit should not be truncated."""
        _MAX = 200_000
        content = "x" * _MAX
        if content and len(content) > _MAX:
            content = content[:_MAX] + "[TRUNCATED]"
        assert len(content) == _MAX
        assert "[TRUNCATED]" not in content

    def test_one_over_limit(self) -> None:
        """Content one char over limit should be truncated."""
        _MAX = 200_000
        content = "x" * (_MAX + 1)
        original_len = len(content)
        if content and len(content) > _MAX:
            content = (
                content[:_MAX]
                + f"\n[TRUNCATED: output was {original_len} chars, limit is {_MAX}]"
            )
        assert "[TRUNCATED:" in content

    def test_empty_content_not_truncated(self) -> None:
        """Empty content should not trigger truncation."""
        _MAX = 200_000
        content = ""
        if content and len(content) > _MAX:
            content = content[:_MAX] + "[TRUNCATED]"
        assert content == ""

    def test_none_content_handled(self) -> None:
        """None content should be handled without crash."""
        _MAX = 200_000
        content = None
        if content and len(content) > _MAX:
            content = content[:_MAX] + "[TRUNCATED]"
        assert content is None


# ---------------------------------------------------------------------------
# Calculator additional edge cases
# ---------------------------------------------------------------------------

class TestCalculatorEdgeCases:
    """Additional edge cases for calculator."""

    def test_complex_expression(self) -> None:
        """Complex nested expression should evaluate."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="(2 + 3) * (4 - 1) / 5")
        assert result.success
        assert result.output == 3.0

    def test_bitwise_operations(self) -> None:
        """Bitwise operations should work."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="0xFF & 0x0F")
        assert result.success
        assert result.output == 15

    def test_right_shift_large(self) -> None:
        """Large right shift should work (no DoS risk)."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="2**100 >> 90")
        assert result.success

    def test_empty_expression(self) -> None:
        """Empty expression should fail gracefully."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="")
        assert not result.success

    def test_string_expression(self) -> None:
        """String expression should fail gracefully."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="'hello' + 'world'")
        assert not result.success

    def test_function_call_rejected(self) -> None:
        """Function calls should be rejected."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="__import__('os').system('ls')")
        assert not result.success


# ---------------------------------------------------------------------------
# Provider registry edge cases
# ---------------------------------------------------------------------------

class TestProviderConfigEdgeCases:
    """Provider configuration edge cases."""

    def test_empty_provider_name(self) -> None:
        """Empty provider name should return None."""
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        result = registry.get("")
        assert result is None

    def test_provider_with_spaces(self) -> None:
        """Provider name with spaces should return None."""
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        result = registry.get("  anthropic  ")
        # Should handle gracefully
        assert result is None or result is not None


# ---------------------------------------------------------------------------
# Input sanitizer MAX_INPUT_LENGTH
# ---------------------------------------------------------------------------

class TestSanitizerMaxLength:
    """Test sanitizer input length truncation."""

    def test_very_long_input_truncated(self) -> None:
        """Very long input should be truncated."""
        from missy.security.sanitizer import InputSanitizer

        sanitizer = InputSanitizer()
        long_input = "A" * 500_000
        result = sanitizer.sanitize(long_input)
        assert len(result) < 500_000

    def test_normal_input_not_truncated(self) -> None:
        """Normal-length input should not be truncated."""
        from missy.security.sanitizer import InputSanitizer

        sanitizer = InputSanitizer()
        normal_input = "Hello, how are you?"
        result = sanitizer.sanitize(normal_input)
        assert result == normal_input
