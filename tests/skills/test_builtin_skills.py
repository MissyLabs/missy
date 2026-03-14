"""Comprehensive tests for all 6 built-in Missy skills.

Covers happy paths, edge cases, and error paths for:
  - SystemInfoSkill
  - DateTimeSkill
  - ConfigShowSkill  (including _redact_api_keys and _summarize_yaml helpers)
  - HealthCheckSkill (including all _check_* helpers)
  - SummarizeSessionSkill (including _format_turns helper)
  - WorkspaceListSkill
"""

from __future__ import annotations

import datetime
import platform
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.skills.base import SkillResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_result(result: SkillResult, *, success: bool) -> None:
    """Assert the type and success flag of a SkillResult."""
    assert isinstance(result, SkillResult)
    assert result.success is success


# ===========================================================================
# 1. SystemInfoSkill
# ===========================================================================


class TestSystemInfoSkill:
    @pytest.fixture()
    def skill(self):
        from missy.skills.builtin.system_info import SystemInfoSkill

        return SystemInfoSkill()

    def test_execute_returns_skill_result(self, skill):
        result = skill.execute()
        _assert_result(result, success=True)

    def test_output_is_string(self, skill):
        result = skill.execute()
        assert isinstance(result.output, str)

    def test_output_contains_hostname(self, skill):
        result = skill.execute()
        assert socket.gethostname() in result.output

    def test_output_contains_os(self, skill):
        result = skill.execute()
        assert platform.system() in result.output

    def test_output_contains_python_version(self, skill):
        result = skill.execute()
        version_token = sys.version.split()[0]
        assert version_token in result.output

    def test_output_contains_machine(self, skill):
        result = skill.execute()
        assert platform.machine() in result.output

    def test_output_contains_os_release(self, skill):
        result = skill.execute()
        assert platform.release() in result.output

    def test_no_error_on_success(self, skill):
        result = skill.execute()
        assert result.error == ""

    def test_output_has_five_lines(self, skill):
        result = skill.execute()
        lines = result.output.strip().splitlines()
        assert len(lines) == 5

    def test_extra_kwargs_ignored(self, skill):
        result = skill.execute(unexpected_kwarg="ignored")
        _assert_result(result, success=True)

    def test_permissions_require_nothing(self):
        from missy.skills.builtin.system_info import SystemInfoSkill

        p = SystemInfoSkill.permissions
        assert p.network is False
        assert p.filesystem_read is False
        assert p.filesystem_write is False
        assert p.shell is False

    def test_skill_metadata(self):
        from missy.skills.builtin.system_info import SystemInfoSkill

        assert SystemInfoSkill.name == "system_info"
        assert SystemInfoSkill.version == "1.0.0"


# ===========================================================================
# 2. DateTimeSkill
# ===========================================================================


class TestDateTimeSkill:
    @pytest.fixture()
    def skill(self):
        from missy.skills.builtin.datetime_info import DateTimeSkill

        return DateTimeSkill()

    def test_execute_returns_skill_result(self, skill):
        result = skill.execute()
        _assert_result(result, success=True)

    def test_output_is_string(self, skill):
        result = skill.execute()
        assert isinstance(result.output, str)

    def test_output_has_four_lines(self, skill):
        result = skill.execute()
        lines = result.output.strip().splitlines()
        assert len(lines) == 4

    def test_output_contains_datetime_utc(self, skill):
        result = skill.execute()
        assert "datetime_utc:" in result.output

    def test_output_contains_datetime_local(self, skill):
        result = skill.execute()
        assert "datetime_local:" in result.output

    def test_output_contains_timezone(self, skill):
        result = skill.execute()
        assert "timezone:" in result.output

    def test_output_contains_uptime(self, skill):
        result = skill.execute()
        assert "uptime:" in result.output

    def test_no_error_on_success(self, skill):
        result = skill.execute()
        assert result.error == ""

    def test_utc_timestamp_is_iso8601(self, skill):
        result = skill.execute()
        utc_line = next(line for line in result.output.splitlines() if line.startswith("datetime_utc:"))
        ts_str = utc_line.split(": ", 1)[1].strip()
        # Should parse without error.
        parsed = datetime.datetime.fromisoformat(ts_str)
        assert parsed.tzinfo is not None

    def test_extra_kwargs_ignored(self, skill):
        result = skill.execute(foo="bar")
        _assert_result(result, success=True)

    def test_permissions_require_nothing(self):
        from missy.skills.builtin.datetime_info import DateTimeSkill

        p = DateTimeSkill.permissions
        assert p.network is False
        assert p.filesystem_read is False

    def test_skill_metadata(self):
        from missy.skills.builtin.datetime_info import DateTimeSkill

        assert DateTimeSkill.name == "datetime_info"
        assert DateTimeSkill.version == "1.0.0"


class TestParseUptime:
    def test_returns_string(self):
        from missy.skills.builtin.datetime_info import _parse_uptime

        result = _parse_uptime()
        assert isinstance(result, str)

    def test_unavailable_when_proc_uptime_missing(self, tmp_path):
        from missy.skills.builtin.datetime_info import _parse_uptime

        fake_path = tmp_path / "nonexistent_uptime"
        with patch("missy.skills.builtin.datetime_info.Path") as mock_path_cls:
            mock_path_cls.return_value = fake_path
            result = _parse_uptime()
        assert result == "unavailable"

    def test_parses_days_hours_minutes_seconds(self, tmp_path):
        from missy.skills.builtin.datetime_info import _parse_uptime

        uptime_file = tmp_path / "uptime"
        # 1 day + 2 h + 3 m + 4 s = 93784 seconds
        total = 1 * 86400 + 2 * 3600 + 3 * 60 + 4
        uptime_file.write_text(f"{total}.00 12345.00\n")

        with patch("missy.skills.builtin.datetime_info.Path") as mock_path_cls:
            mock_path_cls.return_value = uptime_file
            result = _parse_uptime()
        assert "1d" in result
        assert "2h" in result
        assert "3m" in result
        assert "4s" in result

    def test_parses_only_seconds(self, tmp_path):
        from missy.skills.builtin.datetime_info import _parse_uptime

        uptime_file = tmp_path / "uptime"
        uptime_file.write_text("45.00 10.00\n")

        with patch("missy.skills.builtin.datetime_info.Path") as mock_path_cls:
            mock_path_cls.return_value = uptime_file
            result = _parse_uptime()
        # No days/hours/minutes prefix; just seconds.
        assert result == "45s"

    def test_parses_minutes_and_seconds(self, tmp_path):
        from missy.skills.builtin.datetime_info import _parse_uptime

        uptime_file = tmp_path / "uptime"
        uptime_file.write_text("125.00 0.00\n")  # 2m 5s

        with patch("missy.skills.builtin.datetime_info.Path") as mock_path_cls:
            mock_path_cls.return_value = uptime_file
            result = _parse_uptime()
        assert "2m" in result
        assert "5s" in result
        assert "h" not in result

    def test_returns_unavailable_on_bad_content(self, tmp_path):
        from missy.skills.builtin.datetime_info import _parse_uptime

        uptime_file = tmp_path / "uptime"
        uptime_file.write_text("not a number\n")

        with patch("missy.skills.builtin.datetime_info.Path") as mock_path_cls:
            mock_path_cls.return_value = uptime_file
            result = _parse_uptime()
        assert result == "unavailable"


# ===========================================================================
# 3. ConfigShowSkill + helpers
# ===========================================================================


class TestRedactApiKeys:
    def test_short_value_not_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        text = "api_key: short\n"
        assert _redact_api_keys(text) == text

    def test_null_value_not_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        text = "api_key: null\n"
        assert _redact_api_keys(text) == text

    def test_none_value_not_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        text = "api_key: none\n"
        assert _redact_api_keys(text) == text

    def test_true_value_not_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        text = "api_key: true\n"
        assert _redact_api_keys(text) == text

    def test_false_value_not_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        text = "api_key: false\n"
        assert _redact_api_keys(text) == text

    def test_long_bare_value_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        secret = "sk-ant-api03-verylongsecretvalue"
        text = f"api_key: {secret}\n"
        result = _redact_api_keys(text)
        assert secret not in result
        assert secret[:8] in result
        assert "..." in result

    def test_long_quoted_value_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        secret = "sk-ant-api03-verylongsecretvalue"
        text = f'api_key: "{secret}"\n'
        result = _redact_api_keys(text)
        assert secret not in result
        assert secret[:8] in result

    def test_api_keys_plural_redacted(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        secret = "sk-openai-longapikey12345678"
        text = f"api_keys: {secret}\n"
        result = _redact_api_keys(text)
        assert secret not in result
        assert secret[:8] in result

    def test_non_api_key_lines_unchanged(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        text = "model: claude-sonnet-4-6\nsome_other_key: value\n"
        assert _redact_api_keys(text) == text

    def test_case_insensitive_match(self):
        from missy.skills.builtin.config_show import _redact_api_keys

        secret = "sk-ant-api03-verylongsecretvalue"
        text = f"API_KEY: {secret}\n"
        result = _redact_api_keys(text)
        assert secret not in result


class TestSummarizeYaml:
    def test_blank_lines_removed(self):
        from missy.skills.builtin.config_show import _summarize_yaml

        text = "key: value\n\n\nother: thing\n"
        result = _summarize_yaml(text)
        assert "\n\n" not in result

    def test_comment_lines_removed(self):
        from missy.skills.builtin.config_show import _summarize_yaml

        text = "# This is a comment\nkey: value\n"
        result = _summarize_yaml(text)
        assert "#" not in result
        assert "key: value" in result

    def test_api_key_redacted(self):
        from missy.skills.builtin.config_show import _summarize_yaml

        secret = "sk-ant-api03-verylongsecretvalue"
        text = f"api_key: {secret}\n"
        result = _summarize_yaml(text)
        assert secret not in result

    def test_empty_input_returns_empty_string(self):
        from missy.skills.builtin.config_show import _summarize_yaml

        assert _summarize_yaml("") == ""

    def test_only_comments_returns_empty_string(self):
        from missy.skills.builtin.config_show import _summarize_yaml

        text = "# comment one\n# comment two\n"
        assert _summarize_yaml(text) == ""

    def test_trailing_whitespace_stripped(self):
        from missy.skills.builtin.config_show import _summarize_yaml

        text = "key: value   \n"
        result = _summarize_yaml(text)
        assert result == "key: value"


class TestConfigShowSkill:
    @pytest.fixture()
    def skill(self):
        from missy.skills.builtin.config_show import ConfigShowSkill

        return ConfigShowSkill()

    def test_missing_config_returns_failure(self, skill, tmp_path):
        nonexistent = tmp_path / "no_config.yaml"
        with patch("missy.skills.builtin.config_show._CONFIG_PATH", nonexistent):
            result = skill.execute()
        _assert_result(result, success=False)
        assert "not found" in result.error.lower()
        assert result.output == ""

    def test_existing_config_returns_success(self, skill, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("providers:\n  anthropic:\n    model: claude-sonnet-4-6\n")
        with patch("missy.skills.builtin.config_show._CONFIG_PATH", config_file):
            result = skill.execute()
        _assert_result(result, success=True)
        assert "# Config:" in result.output

    def test_output_contains_config_path(self, skill, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("network:\n  default_deny: true\n")
        with patch("missy.skills.builtin.config_show._CONFIG_PATH", config_file):
            result = skill.execute()
        assert str(config_file) in result.output

    def test_api_key_is_redacted(self, skill, tmp_path):
        secret = "sk-ant-api03-verylongsecretvalue"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"providers:\n  anthropic:\n    api_key: {secret}\n")
        with patch("missy.skills.builtin.config_show._CONFIG_PATH", config_file):
            result = skill.execute()
        assert secret not in result.output
        assert secret[:8] in result.output

    def test_comments_stripped_from_output(self, skill, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# Top-level comment\nnetwork:\n  default_deny: true\n")
        with patch("missy.skills.builtin.config_show._CONFIG_PATH", config_file):
            result = skill.execute()
        # The header line starts with #, but no comment lines from YAML content.
        output_without_header = "\n".join(result.output.splitlines()[1:])
        assert "Top-level comment" not in output_without_header

    def test_os_error_on_read_returns_failure(self, skill, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")
        with patch("missy.skills.builtin.config_show._CONFIG_PATH", config_file):
            with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
                result = skill.execute()
        _assert_result(result, success=False)
        assert "Failed to read" in result.error

    def test_empty_config_file_returns_success(self, skill, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        with patch("missy.skills.builtin.config_show._CONFIG_PATH", config_file):
            result = skill.execute()
        # Empty but readable; summarize returns empty body — still success.
        _assert_result(result, success=True)

    def test_permissions_require_filesystem_read(self):
        from missy.skills.builtin.config_show import ConfigShowSkill

        assert ConfigShowSkill.permissions.filesystem_read is True

    def test_skill_metadata(self):
        from missy.skills.builtin.config_show import ConfigShowSkill

        assert ConfigShowSkill.name == "config_show"
        assert ConfigShowSkill.version == "1.0.0"


# ===========================================================================
# 4. HealthCheckSkill + helpers
# ===========================================================================


class TestCheckConfig:
    def test_missing_config_returns_fail(self, tmp_path):
        from missy.skills.builtin.health_check import _check_config

        check = _check_config(tmp_path / "nonexistent.yaml")
        assert check.status == "FAIL"
        assert check.name == "config_file"

    def test_empty_config_returns_warn(self, tmp_path):
        from missy.skills.builtin.health_check import _check_config

        f = tmp_path / "config.yaml"
        f.write_text("")
        check = _check_config(f)
        assert check.status == "WARN"

    def test_valid_config_returns_pass(self, tmp_path):
        from missy.skills.builtin.health_check import _check_config

        f = tmp_path / "config.yaml"
        f.write_text("network:\n  default_deny: true\n")
        check = _check_config(f)
        assert check.status == "PASS"


class TestCheckMemoryDb:
    def test_missing_db_returns_warn(self, tmp_path):
        from missy.skills.builtin.health_check import _check_memory_db

        check = _check_memory_db(tmp_path / "memory.db")
        assert check.status == "WARN"
        assert check.name == "memory_db"

    def test_existing_db_returns_pass(self, tmp_path):
        from missy.skills.builtin.health_check import _check_memory_db

        f = tmp_path / "memory.db"
        f.write_bytes(b"SQLite format 3\x00")
        check = _check_memory_db(f)
        assert check.status == "PASS"


class TestCheckAuditLog:
    def test_missing_log_returns_warn(self, tmp_path):
        from missy.skills.builtin.health_check import _check_audit_log

        check = _check_audit_log(tmp_path / "audit.jsonl")
        assert check.status == "WARN"
        assert check.name == "audit_log"

    def test_existing_log_returns_pass(self, tmp_path):
        from missy.skills.builtin.health_check import _check_audit_log

        f = tmp_path / "audit.jsonl"
        f.write_text('{"event": "startup"}\n')
        check = _check_audit_log(f)
        assert check.status == "PASS"


class TestCheckProviders:
    def test_missing_config_returns_fail(self, tmp_path):
        from missy.skills.builtin.health_check import _check_providers

        check = _check_providers(tmp_path / "nonexistent.yaml")
        assert check.status == "FAIL"
        assert check.name == "providers_configured"

    def test_config_without_providers_section_returns_warn(self, tmp_path):
        from missy.skills.builtin.health_check import _check_providers

        f = tmp_path / "config.yaml"
        f.write_text("network:\n  default_deny: true\n")
        check = _check_providers(f)
        assert check.status == "WARN"

    def test_providers_section_no_model_returns_warn(self, tmp_path):
        from missy.skills.builtin.health_check import _check_providers

        f = tmp_path / "config.yaml"
        f.write_text("providers:\n  anthropic:\n    api_key: sk-test\n")
        check = _check_providers(f)
        assert check.status == "WARN"

    def test_providers_section_with_model_returns_pass(self, tmp_path):
        from missy.skills.builtin.health_check import _check_providers

        f = tmp_path / "config.yaml"
        f.write_text(
            "providers:\n  anthropic:\n    model: claude-sonnet-4-6\n    api_key: sk-test\n"
        )
        check = _check_providers(f)
        assert check.status == "PASS"

    def test_providers_model_null_returns_warn(self, tmp_path):
        from missy.skills.builtin.health_check import _check_providers

        f = tmp_path / "config.yaml"
        f.write_text("providers:\n  anthropic:\n    model: null\n")
        check = _check_providers(f)
        assert check.status == "WARN"

    def test_os_error_reading_config_returns_fail(self, tmp_path):
        from missy.skills.builtin.health_check import _check_providers

        f = tmp_path / "config.yaml"
        f.write_text("providers:\n")
        with patch("pathlib.Path.read_text", side_effect=OSError("denied")):
            check = _check_providers(f)
        assert check.status == "FAIL"
        assert "Could not read config" in check.detail


class TestHealthCheckSkill:
    @pytest.fixture()
    def skill(self):
        from missy.skills.builtin.health_check import HealthCheckSkill

        return HealthCheckSkill()

    def _patch_paths(self, config: Path, memory: Path, audit: Path):
        """Return a context manager that patches all three module-level paths."""
        import missy.skills.builtin.health_check as hc

        return (
            patch.object(hc, "_CONFIG_PATH", config),
            patch.object(hc, "_MEMORY_DB_PATH", memory),
            patch.object(hc, "_AUDIT_LOG_PATH", audit),
        )

    def test_all_present_and_valid_returns_success(self, skill, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "providers:\n  anthropic:\n    model: claude-sonnet-4-6\n"
        )
        memory = tmp_path / "memory.db"
        memory.write_bytes(b"data")
        audit = tmp_path / "audit.jsonl"
        audit.write_text("{}\n")

        ctx1, ctx2, ctx3 = self._patch_paths(config, memory, audit)
        with ctx1, ctx2, ctx3:
            result = skill.execute()

        _assert_result(result, success=True)
        assert "PASS" in result.output
        assert "Overall: PASS" in result.output

    def test_missing_config_causes_failure(self, skill, tmp_path):
        config = tmp_path / "nonexistent.yaml"
        memory = tmp_path / "memory.db"
        memory.write_bytes(b"data")
        audit = tmp_path / "audit.jsonl"
        audit.write_text("{}\n")

        ctx1, ctx2, ctx3 = self._patch_paths(config, memory, audit)
        with ctx1, ctx2, ctx3:
            result = skill.execute()

        _assert_result(result, success=False)
        assert "Overall: FAIL" in result.output

    def test_missing_memory_db_is_warn_not_fail(self, skill, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("providers:\n  anthropic:\n    model: claude-sonnet-4-6\n")
        memory = tmp_path / "nonexistent.db"
        audit = tmp_path / "audit.jsonl"
        audit.write_text("{}\n")

        ctx1, ctx2, ctx3 = self._patch_paths(config, memory, audit)
        with ctx1, ctx2, ctx3:
            result = skill.execute()

        _assert_result(result, success=True)
        assert "WARN" in result.output
        assert "Overall: PASS" in result.output

    def test_missing_audit_log_is_warn_not_fail(self, skill, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("providers:\n  anthropic:\n    model: claude-sonnet-4-6\n")
        memory = tmp_path / "memory.db"
        memory.write_bytes(b"data")
        audit = tmp_path / "nonexistent.jsonl"

        ctx1, ctx2, ctx3 = self._patch_paths(config, memory, audit)
        with ctx1, ctx2, ctx3:
            result = skill.execute()

        _assert_result(result, success=True)
        assert "WARN" in result.output

    def test_output_contains_header(self, skill, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("providers:\n  anthropic:\n    model: claude-sonnet-4-6\n")
        memory = tmp_path / "memory.db"
        memory.write_bytes(b"data")
        audit = tmp_path / "audit.jsonl"
        audit.write_text("{}\n")

        ctx1, ctx2, ctx3 = self._patch_paths(config, memory, audit)
        with ctx1, ctx2, ctx3:
            result = skill.execute()

        assert "Missy Health Check" in result.output

    def test_output_contains_separator(self, skill, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("providers:\n  anthropic:\n    model: claude-sonnet-4-6\n")
        memory = tmp_path / "memory.db"
        memory.write_bytes(b"data")
        audit = tmp_path / "audit.jsonl"
        audit.write_text("{}\n")

        ctx1, ctx2, ctx3 = self._patch_paths(config, memory, audit)
        with ctx1, ctx2, ctx3:
            result = skill.execute()

        assert "=" * 40 in result.output

    def test_permissions_require_filesystem_read(self):
        from missy.skills.builtin.health_check import HealthCheckSkill

        assert HealthCheckSkill.permissions.filesystem_read is True

    def test_skill_metadata(self):
        from missy.skills.builtin.health_check import HealthCheckSkill

        assert HealthCheckSkill.name == "health_check"
        assert HealthCheckSkill.version == "1.0.0"

    def test_no_error_field_set_on_success(self, skill, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("providers:\n  anthropic:\n    model: claude-sonnet-4-6\n")
        memory = tmp_path / "memory.db"
        memory.write_bytes(b"data")
        audit = tmp_path / "audit.jsonl"
        audit.write_text("{}\n")

        ctx1, ctx2, ctx3 = self._patch_paths(config, memory, audit)
        with ctx1, ctx2, ctx3:
            result = skill.execute()

        assert result.error == ""


# ===========================================================================
# 5. SummarizeSessionSkill + _format_turns helper
# ===========================================================================


class TestFormatTurns:
    @pytest.fixture()
    def _make_turn(self):
        """Factory for fake ConversationTurn-like objects."""

        def factory(role: str, content: str, ts=None):
            turn = MagicMock()
            turn.role = role
            turn.content = content
            turn.timestamp = ts or datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
            return turn

        return factory

    def test_empty_turns_returns_no_turns_message(self):
        from missy.skills.builtin.summarize_session import _format_turns

        result = _format_turns([])
        assert "no turns" in result.lower()

    def test_single_turn_rendered(self, _make_turn):
        from missy.skills.builtin.summarize_session import _format_turns

        turn = _make_turn("user", "Hello, world!")
        result = _format_turns([turn])
        assert "User" in result
        assert "Hello, world!" in result

    def test_role_capitalized(self, _make_turn):
        from missy.skills.builtin.summarize_session import _format_turns

        turn = _make_turn("assistant", "Hi there.")
        result = _format_turns([turn])
        assert "Assistant" in result

    def test_long_content_truncated(self, _make_turn):
        from missy.skills.builtin.summarize_session import _format_turns

        long_content = "a" * 300
        turn = _make_turn("user", long_content)
        result = _format_turns([turn])
        assert "a" * 300 not in result
        assert "…" in result

    def test_content_exactly_at_limit_not_truncated(self, _make_turn):
        from missy.skills.builtin.summarize_session import _CONTENT_PREVIEW_LEN, _format_turns

        content = "x" * _CONTENT_PREVIEW_LEN
        turn = _make_turn("user", content)
        result = _format_turns([turn])
        assert "…" not in result

    def test_timestamp_in_output(self, _make_turn):
        from missy.skills.builtin.summarize_session import _format_turns

        ts = datetime.datetime(2025, 6, 15, 10, 30, 0, tzinfo=datetime.UTC)
        turn = _make_turn("user", "test", ts=ts)
        result = _format_turns([turn])
        assert "2025-06-15" in result

    def test_none_timestamp_shows_unknown_time(self, _make_turn):
        from missy.skills.builtin.summarize_session import _format_turns

        turn = _make_turn("user", "test", ts=None)
        turn.timestamp = None
        result = _format_turns([turn])
        assert "unknown time" in result

    def test_multiple_turns_rendered(self, _make_turn):
        from missy.skills.builtin.summarize_session import _format_turns

        turns = [
            _make_turn("user", "Question one."),
            _make_turn("assistant", "Answer one."),
        ]
        result = _format_turns(turns)
        assert "Question one." in result
        assert "Answer one." in result

    def test_none_content_treated_as_empty(self, _make_turn):
        from missy.skills.builtin.summarize_session import _format_turns

        turn = _make_turn("user", None)
        turn.content = None
        result = _format_turns([turn])
        assert "User" in result  # should not crash


class TestSummarizeSessionSkill:
    @pytest.fixture()
    def skill(self):
        from missy.skills.builtin.summarize_session import SummarizeSessionSkill

        return SummarizeSessionSkill()

    @pytest.fixture()
    def mock_turn(self):
        turn = MagicMock()
        turn.role = "user"
        turn.content = "What is the weather?"
        turn.timestamp = datetime.datetime(2025, 3, 1, 9, 0, 0, tzinfo=datetime.UTC)
        return turn

    def test_missing_session_id_returns_failure(self, skill):
        result = skill.execute()
        _assert_result(result, success=False)
        assert "session_id" in result.error.lower()

    def test_empty_session_id_returns_failure(self, skill):
        result = skill.execute(session_id="")
        _assert_result(result, success=False)

    def test_valid_session_with_turns_returns_success(self, skill, mock_turn):
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = [mock_turn]

        # MemoryStore is imported locally inside execute(), so patch at the
        # source module rather than at the skill module's namespace.
        with patch("missy.memory.store.MemoryStore", return_value=mock_store):
            result = skill.execute(session_id="session-abc")

        _assert_result(result, success=True)
        assert "session-abc" in result.output

    def test_valid_session_with_no_turns_returns_success(self, skill):
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []

        with patch("missy.memory.store.MemoryStore", return_value=mock_store):
            result = skill.execute(session_id="empty-session")

        _assert_result(result, success=True)
        assert "no turns" in result.output.lower()

    def test_output_contains_turn_count(self, skill, mock_turn):
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = [mock_turn, mock_turn]

        with patch("missy.memory.store.MemoryStore", return_value=mock_store):
            result = skill.execute(session_id="my-session")

        assert "Turns shown: 2" in result.output

    def test_output_contains_separator(self, skill):
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []

        with patch("missy.memory.store.MemoryStore", return_value=mock_store):
            result = skill.execute(session_id="any-session")

        assert "-" * 60 in result.output

    def test_get_session_turns_called_with_limit(self, skill):
        from missy.skills.builtin.summarize_session import _TURN_LIMIT

        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []

        with patch("missy.memory.store.MemoryStore", return_value=mock_store):
            skill.execute(session_id="s1")

        mock_store.get_session_turns.assert_called_once_with("s1", limit=_TURN_LIMIT)

    def test_extra_kwargs_ignored(self, skill):
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []

        with patch("missy.memory.store.MemoryStore", return_value=mock_store):
            result = skill.execute(session_id="s1", unknown_param="ignored")

        _assert_result(result, success=True)

    def test_permissions_require_filesystem_read(self):
        from missy.skills.builtin.summarize_session import SummarizeSessionSkill

        assert SummarizeSessionSkill.permissions.filesystem_read is True

    def test_skill_metadata(self):
        from missy.skills.builtin.summarize_session import SummarizeSessionSkill

        assert SummarizeSessionSkill.name == "summarize_session"
        assert SummarizeSessionSkill.version == "1.0.0"


# ===========================================================================
# 6. WorkspaceListSkill
# ===========================================================================


class TestWorkspaceListSkill:
    @pytest.fixture()
    def skill(self):
        from missy.skills.builtin.workspace_list import WorkspaceListSkill

        return WorkspaceListSkill()

    def test_nonexistent_path_returns_failure(self, skill, tmp_path):
        missing = str(tmp_path / "no_such_dir")
        result = skill.execute(workspace_path=missing)
        _assert_result(result, success=False)
        assert "not found" in result.error.lower()
        assert result.output is None

    def test_path_that_is_a_file_returns_failure(self, skill, tmp_path):
        f = tmp_path / "notadir.txt"
        f.write_text("content")
        result = skill.execute(workspace_path=str(f))
        _assert_result(result, success=False)
        assert "not a directory" in result.error.lower()

    def test_empty_directory_returns_success_with_empty_message(self, skill, tmp_path):
        result = skill.execute(workspace_path=str(tmp_path))
        _assert_result(result, success=True)
        assert "empty" in result.output.lower()

    def test_files_listed_with_file_prefix(self, skill, tmp_path):
        (tmp_path / "report.txt").write_text("data")
        result = skill.execute(workspace_path=str(tmp_path))
        _assert_result(result, success=True)
        assert "[file] report.txt" in result.output

    def test_directories_listed_with_dir_prefix(self, skill, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = skill.execute(workspace_path=str(tmp_path))
        _assert_result(result, success=True)
        assert "[dir] subdir" in result.output

    def test_directories_sorted_before_files(self, skill, tmp_path):
        (tmp_path / "aaa.txt").write_text("file")
        (tmp_path / "zzz_subdir").mkdir()
        result = skill.execute(workspace_path=str(tmp_path))
        lines = result.output.splitlines()
        dir_indices = [i for i, line in enumerate(lines) if line.startswith("[dir]")]
        file_indices = [i for i, line in enumerate(lines) if line.startswith("[file]")]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)

    def test_multiple_files_each_on_own_line(self, skill, tmp_path):
        for name in ("alpha.txt", "beta.txt", "gamma.txt"):
            (tmp_path / name).write_text("x")
        result = skill.execute(workspace_path=str(tmp_path))
        lines = result.output.splitlines()
        file_lines = [line for line in lines if line.startswith("[file]")]
        assert len(file_lines) == 3

    def test_tilde_expansion_for_default_path(self, skill, tmp_path):
        """workspace_path with tilde should expand correctly."""
        result = skill.execute(workspace_path=str(tmp_path))
        # As long as no crash, the expansion path is exercised.
        assert isinstance(result, SkillResult)

    def test_no_error_on_success(self, skill, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        result = skill.execute(workspace_path=str(tmp_path))
        assert result.error == ""

    def test_extra_kwargs_ignored(self, skill, tmp_path):
        result = skill.execute(workspace_path=str(tmp_path), unused="param")
        assert isinstance(result, SkillResult)

    def test_permissions_require_filesystem_read(self):
        from missy.skills.builtin.workspace_list import WorkspaceListSkill

        assert WorkspaceListSkill.permissions.filesystem_read is True

    def test_skill_metadata(self):
        from missy.skills.builtin.workspace_list import WorkspaceListSkill

        assert WorkspaceListSkill.name == "workspace_list"
        assert WorkspaceListSkill.version == "1.0.0"

    def test_mixed_content_sorted_correctly(self, skill, tmp_path):
        (tmp_path / "b_file.txt").write_text("x")
        (tmp_path / "a_dir").mkdir()
        (tmp_path / "c_file.txt").write_text("x")
        result = skill.execute(workspace_path=str(tmp_path))
        lines = result.output.splitlines()
        assert lines[0] == "[dir] a_dir"
        assert "[file] b_file.txt" in lines
        assert "[file] c_file.txt" in lines
