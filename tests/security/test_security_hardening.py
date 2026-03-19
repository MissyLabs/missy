"""Tests for security hardening: tool output injection, response censoring, MCP safety."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.core.exceptions import PolicyViolationError
from missy.mcp.manager import McpManager
from missy.policy.shell import ShellPolicyEngine


class TestShellPolicyProcessSubstitution:
    """Shell policy must block process substitution markers."""

    def test_reject_input_process_substitution(self):
        """<(...) should be rejected as a subshell marker."""
        policy = MagicMock()
        policy.enabled = True
        policy.allowed_commands = ["diff"]
        engine = ShellPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_command("diff <(cat /etc/shadow) <(echo x)")

    def test_reject_output_process_substitution(self):
        """>(...) should be rejected as a subshell marker."""
        policy = MagicMock()
        policy.enabled = True
        policy.allowed_commands = ["tee"]
        engine = ShellPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_command("echo hello | tee >(cat)")

    def test_reject_heredoc_process_substitution(self):
        """<<(...) should be rejected."""
        policy = MagicMock()
        policy.enabled = True
        policy.allowed_commands = ["cat"]
        engine = ShellPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_command("cat <<(echo bad)")

    def test_still_allows_normal_commands(self):
        """Normal commands without subshell markers should work."""
        policy = MagicMock()
        policy.enabled = True
        policy.allowed_commands = ["ls", "cat"]
        engine = ShellPolicyEngine(policy)
        # Should not raise
        engine.check_command("ls -la")


class TestMcpServerNameValidation:
    """MCP manager must reject server names containing '__'."""

    def test_reject_double_underscore_name(self):
        """Names with __ can cause namespace collision attacks."""
        mgr = McpManager(config_path="/tmp/nonexistent_mcp.json")
        with pytest.raises(ValueError, match="must not contain '__'"):
            mgr.add_server("filesystem__evil", command="echo hi")

    def test_accept_valid_name(self):
        """Valid names without __ should be accepted (connect may fail, that's ok)."""
        mgr = McpManager(config_path="/tmp/nonexistent_mcp.json")
        # connect() will fail since the command doesn't implement MCP, but
        # the name validation should pass — no ValueError about "__"
        try:
            mgr.add_server("valid_name", command="echo hi")
        except ValueError as exc:
            assert "must not contain" not in str(exc)
        except Exception:
            pass  # Any non-ValueError error is fine (connect failure)

    def test_accept_single_underscore(self):
        """Single underscores are fine."""
        mgr = McpManager(config_path="/tmp/nonexistent_mcp.json")
        try:
            mgr.add_server("my_server", command="echo hi")
        except ValueError as exc:
            assert "must not contain" not in str(exc)
        except Exception:
            pass  # Any non-ValueError error is fine (connect failure)


class TestMcpClientTimeout:
    """MCP client RPC must have timeout protection."""

    def test_rpc_has_max_response_bytes(self):
        """McpClient should define a max response size."""
        from missy.mcp.client import McpClient

        assert hasattr(McpClient, "_MAX_RESPONSE_BYTES")
        assert McpClient._MAX_RESPONSE_BYTES == 1024 * 1024

    def test_rpc_timeout_parameter(self):
        """_rpc should accept a timeout parameter."""
        import inspect

        from missy.mcp.client import McpClient

        sig = inspect.signature(McpClient._rpc)
        assert "timeout" in sig.parameters
        assert sig.parameters["timeout"].default == 30.0


class TestToolOutputInjectionScanning:
    """Agent runtime must scan tool output for injection patterns."""

    def test_runtime_has_sanitizer(self):
        """AgentRuntime should create a sanitizer on init."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(provider="anthropic")
        with patch.object(AgentRuntime, "_make_circuit_breaker", return_value=None), \
             patch.object(AgentRuntime, "_make_rate_limiter", return_value=None), \
             patch.object(AgentRuntime, "_make_context_manager", return_value=None), \
             patch.object(AgentRuntime, "_make_memory_store", return_value=None), \
             patch.object(AgentRuntime, "_make_cost_tracker", return_value=None), \
             patch.object(AgentRuntime, "_scan_checkpoints", return_value=[]):
            runtime = AgentRuntime(config)
            assert runtime._sanitizer is not None

    def test_make_sanitizer_returns_sanitizer(self):
        """_make_sanitizer should return an InputSanitizer instance."""
        from missy.agent.runtime import AgentRuntime

        sanitizer = AgentRuntime._make_sanitizer()
        assert sanitizer is not None
        assert hasattr(sanitizer, "check_for_injection")

    def test_make_sanitizer_handles_import_error(self):
        """_make_sanitizer should return None if module unavailable."""
        from missy.agent.runtime import AgentRuntime

        with patch.dict("sys.modules", {"missy.security.sanitizer": None}):
            result = AgentRuntime._make_sanitizer()
            assert result is None


class TestResponseCensoring:
    """Agent runtime must censor secrets from final output."""

    def test_censor_response_import(self):
        """censor_response should be importable from runtime module."""
        from missy.agent import runtime
        assert hasattr(runtime, "censor_response")

    def test_censor_response_redacts_api_key(self):
        """censor_response should redact API keys in output."""
        from missy.security.censor import censor_response

        # Use a key pattern that matches SecretsDetector's openai_key regex
        text = "Here is the key: sk-abc123def456ghi789jkl012mno345pqr678stu901vwx"
        result = censor_response(text)
        assert "[REDACTED]" in result

    def test_censor_response_passes_safe_text(self):
        """censor_response should not modify text without secrets."""
        from missy.security.censor import censor_response

        text = "The answer is 42."
        assert censor_response(text) == text

    def test_censor_response_handles_empty(self):
        """censor_response should handle empty string."""
        from missy.security.censor import censor_response

        assert censor_response("") == ""
        assert censor_response(None) is None


class TestShellPolicyExistingFunctionality:
    """Verify existing shell policy functionality is preserved."""

    def test_command_substitution_still_blocked(self):
        """$(...) should still be blocked."""
        policy = MagicMock()
        policy.enabled = True
        policy.allowed_commands = ["echo"]
        engine = ShellPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_command("echo $(whoami)")

    def test_backtick_still_blocked(self):
        """Backtick substitution should still be blocked."""
        policy = MagicMock()
        policy.enabled = True
        policy.allowed_commands = ["echo"]
        engine = ShellPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_command("echo `whoami`")

    def test_chain_operators_checked(self):
        """Chained commands must all be in the allowlist."""
        policy = MagicMock()
        policy.enabled = True
        policy.allowed_commands = ["ls"]
        engine = ShellPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_command("ls && rm -rf /")
