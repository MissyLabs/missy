"""Session 24: Tests for security fixes from audit.

Tests for:
1. TTS env sanitization (API key filtering)
2. Discord snowflake ID validation (path traversal prevention)
3. Calculator LShift DoS guard
4. X11 shell quoting (shlex.quote vs json.dumps)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# TTS env sanitization
# ---------------------------------------------------------------------------

class TestTTSEnvSanitization:
    """Verify TTS subprocess env does not leak API keys."""

    def test_api_keys_not_in_tts_env(self, monkeypatch) -> None:
        """API keys should be filtered from TTS subprocess environment."""
        from missy.tools.builtin.tts_speak import _ensure_runtime_dir

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "bot-token-secret")
        monkeypatch.setenv("HOME", "/home/test")

        env = _ensure_runtime_dir()
        assert "ANTHROPIC_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env
        assert "DISCORD_BOT_TOKEN" not in env
        assert env.get("HOME") == "/home/test"

    def test_safe_vars_preserved(self, monkeypatch) -> None:
        """Safe variables needed for audio should be preserved."""
        from missy.tools.builtin.tts_speak import _ensure_runtime_dir

        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")

        env = _ensure_runtime_dir()
        assert env.get("PATH") == "/usr/bin"
        assert env.get("LANG") == "en_US.UTF-8"
        assert env.get("DISPLAY") == ":0"
        assert env.get("LD_LIBRARY_PATH") == "/usr/lib"

    def test_piper_env_also_sanitized(self, monkeypatch) -> None:
        """Piper env (which extends _ensure_runtime_dir) should also be safe."""
        from missy.tools.builtin.tts_speak import _piper_env

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("HOME", "/home/test")

        env = _piper_env()
        assert "ANTHROPIC_API_KEY" not in env
        assert env.get("HOME") == "/home/test"


# ---------------------------------------------------------------------------
# Discord snowflake ID validation
# ---------------------------------------------------------------------------

class TestDiscordSnowflakeValidation:
    """Verify snowflake ID validation prevents path traversal."""

    def test_valid_snowflake_accepted(self) -> None:
        """Valid numeric snowflake IDs should pass validation."""
        from missy.channels.discord.rest import _validate_snowflake

        assert _validate_snowflake("123456789012345678") == "123456789012345678"
        assert _validate_snowflake("1") == "1"
        assert _validate_snowflake("99999999999999999999") == "99999999999999999999"

    def test_path_traversal_rejected(self) -> None:
        """Path traversal attempts should be rejected."""
        from missy.channels.discord.rest import _validate_snowflake

        with pytest.raises(ValueError, match="Invalid Discord"):
            _validate_snowflake("../../../users/@me")

    def test_empty_string_rejected(self) -> None:
        """Empty string should be rejected."""
        from missy.channels.discord.rest import _validate_snowflake

        with pytest.raises(ValueError, match="Invalid Discord"):
            _validate_snowflake("")

    def test_alphabetic_id_rejected(self) -> None:
        """Non-numeric IDs should be rejected."""
        from missy.channels.discord.rest import _validate_snowflake

        with pytest.raises(ValueError, match="Invalid Discord"):
            _validate_snowflake("chan-1")

    def test_negative_id_rejected(self) -> None:
        """Negative IDs should be rejected."""
        from missy.channels.discord.rest import _validate_snowflake

        with pytest.raises(ValueError, match="Invalid Discord"):
            _validate_snowflake("-123")

    def test_too_long_id_rejected(self) -> None:
        """IDs longer than 20 digits should be rejected."""
        from missy.channels.discord.rest import _validate_snowflake

        with pytest.raises(ValueError, match="Invalid Discord"):
            _validate_snowflake("1" * 21)

    def test_mixed_content_rejected(self) -> None:
        """IDs with mixed content should be rejected."""
        from missy.channels.discord.rest import _validate_snowflake

        with pytest.raises(ValueError):
            _validate_snowflake("123abc456")

    def test_url_encoded_rejected(self) -> None:
        """URL-encoded path traversal should be rejected."""
        from missy.channels.discord.rest import _validate_snowflake

        with pytest.raises(ValueError):
            _validate_snowflake("..%2F..%2Fusers")


# ---------------------------------------------------------------------------
# Calculator LShift guard
# ---------------------------------------------------------------------------

class TestCalculatorLShiftGuard:
    """Verify calculator prevents memory exhaustion via left shift."""

    def test_normal_shift_allowed(self) -> None:
        """Normal left shifts should work."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="1 << 10")
        assert result.success
        assert result.output == 1024

    def test_large_shift_rejected(self) -> None:
        """Very large left shifts should be rejected."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="1 << 100000")
        assert not result.success
        assert "exceeds" in result.error.lower() or "maximum" in result.error.lower()

    def test_max_boundary_shift(self) -> None:
        """Shift at exactly the limit should work."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="1 << 10000")
        assert result.success

    def test_over_boundary_shift_rejected(self) -> None:
        """Shift just over the limit should be rejected."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="1 << 10001")
        assert not result.success

    def test_right_shift_not_limited(self) -> None:
        """Right shifts should not be restricted (no memory issue)."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="1000000 >> 10")
        assert result.success
        assert result.output == 976

    def test_power_still_guarded(self) -> None:
        """Existing power guard should still work."""
        from missy.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        result = tool.execute(expression="9 ** 9999")
        assert not result.success


# ---------------------------------------------------------------------------
# X11 shell quoting
# ---------------------------------------------------------------------------

class TestX11ShellQuoting:
    """Verify X11 tools use shlex.quote instead of json.dumps."""

    def test_shell_metacharacters_escaped(self) -> None:
        """Shell metacharacters in text should be safely quoted."""
        import shlex
        text = '$(rm -rf /)'
        quoted = shlex.quote(text)
        # shlex.quote wraps in single quotes, which neutralize $() in bash
        assert quoted.startswith("'")
        assert quoted.endswith("'")
        # The entire dangerous string is enclosed in single quotes
        assert quoted == "'$(rm -rf /)'"

    def test_xdotool_type_command_safe(self) -> None:
        """X11TypeTool should use shlex.quote for text parameter."""
        from missy.tools.builtin.x11_tools import X11TypeTool

        tool = X11TypeTool()
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            tool.execute(text='$(dangerous_command)')

        called_cmd = mock_run.call_args[0][0]
        # The command should use single-quote escaping
        assert "$(dangerous_command)" not in called_cmd or "'" in called_cmd

    def test_window_name_safe(self) -> None:
        """Window name should be safely quoted with shlex.quote."""
        from missy.tools.builtin.x11_tools import X11ClickTool

        tool = X11ClickTool()
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            tool.execute(x=100, y=200, window_name='$(whoami)')

        called_cmds = [call[0][0] for call in mock_run.call_args_list]
        # The windowfocus command should have safely quoted the name
        focus_cmd = next((c for c in called_cmds if "windowfocus" in c), "")
        assert "$(whoami)" not in focus_cmd or "'" in focus_cmd
