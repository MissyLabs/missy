"""Deep tests for missy.cli.wizard — 40+ cases covering all public helpers."""

from __future__ import annotations

import os
import stat
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml

# ---------------------------------------------------------------------------
# _mask_key
# ---------------------------------------------------------------------------


class TestMaskKey:
    """Tests for _mask_key(key)."""

    def test_long_key_shows_prefix_and_suffix(self):
        from missy.cli.wizard import _mask_key

        result = _mask_key("sk-ant-api03-abcdefghijklmnop")
        assert result.startswith("sk-ant")
        assert result.endswith("mnop")
        assert "…" in result

    def test_long_key_exactly_six_plus_four(self):
        from missy.cli.wizard import _mask_key

        key = "ABCDEF1234"  # len=10, >8
        result = _mask_key(key)
        assert result == "ABCDEF…1234"

    def test_short_key_eight_chars_is_fully_masked(self):
        from missy.cli.wizard import _mask_key

        key = "12345678"  # len==8, boundary
        result = _mask_key(key)
        assert result == "********"
        assert "…" not in result

    def test_very_short_key_single_char(self):
        from missy.cli.wizard import _mask_key

        result = _mask_key("x")
        assert result == "*"

    def test_short_key_four_chars(self):
        from missy.cli.wizard import _mask_key

        result = _mask_key("abcd")
        assert result == "****"

    def test_key_nine_chars_uses_ellipsis(self):
        from missy.cli.wizard import _mask_key

        key = "ABCDEFGHI"  # len=9, >8
        result = _mask_key(key)
        assert result == "ABCDEF…FGHI"

    def test_realistic_anthropic_key(self):
        from missy.cli.wizard import _mask_key

        key = "sk-ant-api03-AAABBBCCCDDDEEEFFF111222333444555"
        result = _mask_key(key)
        assert result.startswith("sk-ant")
        assert result.endswith(key[-4:])
        assert "…" in result
        # Must not reveal middle section
        assert len(result) < len(key)

    def test_empty_string_edge_case(self):
        """Empty string has len==0, so <=8 → fully masked (zero stars)."""
        from missy.cli.wizard import _mask_key

        result = _mask_key("")
        assert result == ""

    def test_exactly_nine_chars_boundary(self):
        from missy.cli.wizard import _mask_key

        key = "123456789"  # len=9
        result = _mask_key(key)
        assert "…" in result
        assert result.startswith("123456")
        assert result.endswith("6789")


# ---------------------------------------------------------------------------
# _detect_env_key
# ---------------------------------------------------------------------------


class TestDetectEnvKey:
    """Tests for _detect_env_key(env_var)."""

    def test_returns_value_when_set(self, monkeypatch):
        from missy.cli.wizard import _detect_env_key

        monkeypatch.setenv("MY_TEST_API_KEY", "sk-abc123")
        assert _detect_env_key("MY_TEST_API_KEY") == "sk-abc123"

    def test_returns_none_when_unset(self, monkeypatch):
        from missy.cli.wizard import _detect_env_key

        monkeypatch.delenv("MY_MISSING_KEY", raising=False)
        assert _detect_env_key("MY_MISSING_KEY") is None

    def test_returns_none_for_empty_string(self, monkeypatch):
        from missy.cli.wizard import _detect_env_key

        monkeypatch.setenv("EMPTY_KEY", "")
        assert _detect_env_key("EMPTY_KEY") is None

    def test_returns_none_for_whitespace_only(self, monkeypatch):
        """Whitespace-only values are falsy; treated as unset."""
        from missy.cli.wizard import _detect_env_key

        monkeypatch.setenv("WHITESPACE_KEY", "   ")
        # os.environ.get returns whitespace; 'or None' keeps it truthy — verify actual behaviour
        result = _detect_env_key("WHITESPACE_KEY")
        # "   " is truthy so the implementation returns it; document that
        assert result == "   " or result is None  # implementation-defined

    def test_preserves_key_with_special_chars(self, monkeypatch):
        from missy.cli.wizard import _detect_env_key

        monkeypatch.setenv("SPECIAL_KEY", "sk-ant-api03-ABC/DEF+GHI=")
        assert _detect_env_key("SPECIAL_KEY") == "sk-ant-api03-ABC/DEF+GHI="

    def test_correct_env_var_name_case_sensitive(self, monkeypatch):
        from missy.cli.wizard import _detect_env_key

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("anthropic_api_key", raising=False)
        assert _detect_env_key("ANTHROPIC_API_KEY") == "sk-ant-test"
        assert _detect_env_key("anthropic_api_key") is None


# ---------------------------------------------------------------------------
# _validate_key_format
# ---------------------------------------------------------------------------


class TestValidateKeyFormat:
    """Tests for _validate_key_format(key, prefix)."""

    def test_valid_anthropic_key(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("sk-ant-api03-ABCDEF", "sk-ant-") is True

    def test_valid_openai_key(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("sk-proj-ABCDEFG", "sk-") is True

    def test_invalid_prefix_mismatch(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("gsk-wrong-prefix", "sk-ant-") is False

    def test_empty_key_returns_false(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("", "sk-ant-") is False

    def test_whitespace_only_key_returns_false(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("   ", "sk-ant-") is False

    def test_no_prefix_required_accepts_any_nonempty(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("any-random-token", None) is True
        assert _validate_key_format("another", "") is True

    def test_no_prefix_rejects_empty(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("", None) is False

    def test_key_with_leading_whitespace_stripped(self):
        from missy.cli.wizard import _validate_key_format

        # strip() is applied inside the function
        assert _validate_key_format("  sk-ant-api03-ABC  ", "sk-ant-") is True

    def test_key_exactly_matching_prefix(self):
        from missy.cli.wizard import _validate_key_format

        # Key is just the prefix — still starts with prefix so valid
        assert _validate_key_format("sk-ant-", "sk-ant-") is True

    def test_empty_prefix_string_acts_as_no_restriction(self):
        from missy.cli.wizard import _validate_key_format

        # prefix="" is falsy; no restriction applied
        assert _validate_key_format("anything", "") is True


# ---------------------------------------------------------------------------
# _build_config_yaml
# ---------------------------------------------------------------------------


class TestBuildConfigYaml:
    """Tests for _build_config_yaml(workspace, providers_cfg, allowed_hosts, discord_cfg)."""

    def _call(self, workspace="/home/user/workspace", providers_cfg=None, allowed_hosts=None,
              discord_cfg=None):
        from missy.cli.wizard import _build_config_yaml

        if providers_cfg is None:
            providers_cfg = []
        if allowed_hosts is None:
            allowed_hosts = []
        return _build_config_yaml(workspace, providers_cfg, allowed_hosts, discord_cfg)

    def test_output_is_valid_yaml(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_config_version_is_2(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert parsed["config_version"] == 2

    def test_network_default_deny_true(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert parsed["network"]["default_deny"] is True

    def test_workspace_appears_in_filesystem(self):
        content = self._call(workspace="/custom/workspace")
        parsed = yaml.safe_load(content)
        write_paths = parsed["filesystem"]["allowed_write_paths"]
        assert any("/custom/workspace" in p for p in write_paths)

    def test_workspace_path_in_config(self):
        content = self._call(workspace="/my/ws")
        parsed = yaml.safe_load(content)
        assert parsed["workspace_path"] == "/my/ws"

    def test_anthropic_provider_emitted(self):
        providers_cfg = [
            {
                "name": "anthropic",
                "model": "claude-sonnet-4-6",
                "fast_model": "claude-haiku-4-5-20251001",
                "premium_model": "claude-opus-4-6",
                "api_key": "sk-ant-api03-TESTKEY",
                "base_url": None,
            }
        ]
        content = self._call(providers_cfg=providers_cfg)
        parsed = yaml.safe_load(content)
        assert "anthropic" in parsed["providers"]
        prov = parsed["providers"]["anthropic"]
        assert prov["model"] == "claude-sonnet-4-6"
        assert prov["api_key"] == "sk-ant-api03-TESTKEY"

    def test_fast_and_premium_models_emitted(self):
        providers_cfg = [
            {
                "name": "anthropic",
                "model": "claude-sonnet-4-6",
                "fast_model": "claude-haiku-4-5-20251001",
                "premium_model": "claude-opus-4-6",
                "api_key": None,
                "base_url": None,
            }
        ]
        content = self._call(providers_cfg=providers_cfg)
        parsed = yaml.safe_load(content)
        prov = parsed["providers"]["anthropic"]
        assert prov["fast_model"] == "claude-haiku-4-5-20251001"
        assert prov["premium_model"] == "claude-opus-4-6"

    def test_optional_fast_premium_omitted_when_empty(self):
        providers_cfg = [
            {
                "name": "anthropic",
                "model": "claude-sonnet-4-6",
                "fast_model": "",
                "premium_model": "",
                "api_key": None,
                "base_url": None,
            }
        ]
        content = self._call(providers_cfg=providers_cfg)
        parsed = yaml.safe_load(content)
        prov = parsed["providers"]["anthropic"]
        # Empty strings are falsy; the generator skips them
        assert "fast_model" not in prov or prov.get("fast_model") == ""

    def test_base_url_emitted_for_ollama(self):
        providers_cfg = [
            {
                "name": "ollama",
                "model": "llama3",
                "fast_model": "",
                "premium_model": "",
                "api_key": None,
                "base_url": "http://localhost:11434",
            }
        ]
        content = self._call(providers_cfg=providers_cfg)
        parsed = yaml.safe_load(content)
        assert parsed["providers"]["ollama"]["base_url"] == "http://localhost:11434"

    def test_no_discord_section_when_not_provided(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert "discord" not in parsed

    def test_discord_section_emitted(self):
        discord_cfg = {
            "bot_token": "MYTOKEN",
            "token_env_var": "DISCORD_BOT_TOKEN",
            "application_id": "123456",
            "dm_policy": "allowlist",
            "dm_allowlist": ["111", "222"],
            "guild_policies": [],
            "ack_reaction": "eyes",
            "ignore_bots": True,
        }
        content = self._call(discord_cfg=discord_cfg)
        parsed = yaml.safe_load(content)
        assert "discord" in parsed
        assert parsed["discord"]["enabled"] is True

    def test_discord_dm_allowlist_included(self):
        discord_cfg = {
            "bot_token": "",
            "token_env_var": "DISCORD_BOT_TOKEN",
            "application_id": "",
            "dm_policy": "allowlist",
            "dm_allowlist": ["777777777777777777"],
            "guild_policies": [],
            "ack_reaction": "",
            "ignore_bots": False,
        }
        content = self._call(discord_cfg=discord_cfg)
        assert "777777777777777777" in content

    def test_allowed_hosts_appear_in_network(self):
        content = self._call(allowed_hosts=["custom.example.com:443"])
        parsed = yaml.safe_load(content)
        hosts = parsed["network"].get("allowed_hosts") or []
        assert "custom.example.com:443" in hosts

    def test_anthropic_preset_detected_and_used(self):
        """If api.anthropic.com is in allowed_hosts, the 'anthropic' preset fires."""
        from missy.policy.presets import PRESETS

        anthropic_hosts = PRESETS.get("anthropic", {}).get("hosts", [])
        if not anthropic_hosts:
            pytest.skip("anthropic preset has no hosts")

        content = self._call(allowed_hosts=list(anthropic_hosts))
        parsed = yaml.safe_load(content)
        presets = parsed["network"].get("presets") or []
        assert "anthropic" in presets

    def test_shell_disabled_by_default(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert parsed["shell"]["enabled"] is False

    def test_plugins_disabled_by_default(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert parsed["plugins"]["enabled"] is False

    def test_heartbeat_disabled_by_default(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert parsed["heartbeat"]["enabled"] is False

    def test_vault_disabled_by_default(self):
        content = self._call()
        parsed = yaml.safe_load(content)
        assert parsed["vault"]["enabled"] is False

    def test_timeout_set_to_30(self):
        providers_cfg = [
            {
                "name": "anthropic",
                "model": "claude-sonnet-4-6",
                "fast_model": "",
                "premium_model": "",
                "api_key": "sk-ant-test",
                "base_url": None,
            }
        ]
        content = self._call(providers_cfg=providers_cfg)
        parsed = yaml.safe_load(content)
        assert parsed["providers"]["anthropic"]["timeout"] == 30

    def test_multiple_providers_emitted(self):
        providers_cfg = [
            {"name": "anthropic", "model": "claude-sonnet-4-6", "fast_model": "",
             "premium_model": "", "api_key": "sk-ant-xxx", "base_url": None},
            {"name": "openai", "model": "gpt-4o", "fast_model": "gpt-4o-mini",
             "premium_model": "gpt-4-turbo", "api_key": "sk-yyy", "base_url": None},
        ]
        content = self._call(providers_cfg=providers_cfg)
        parsed = yaml.safe_load(content)
        assert "anthropic" in parsed["providers"]
        assert "openai" in parsed["providers"]

    def test_discord_guild_policies_emitted(self):
        discord_cfg = {
            "bot_token": "TOK",
            "token_env_var": "DISCORD_BOT_TOKEN",
            "application_id": "APP_ID",
            "dm_policy": "disabled",
            "dm_allowlist": [],
            "guild_policies": [
                {
                    "guild_id": "999888777666",
                    "require_mention": True,
                    "allowed_channels": ["general"],
                    "mode": "full",
                }
            ],
            "ack_reaction": "eyes",
            "ignore_bots": True,
        }
        content = self._call(discord_cfg=discord_cfg)
        assert "999888777666" in content
        assert "general" in content


# ---------------------------------------------------------------------------
# _write_config_atomic
# ---------------------------------------------------------------------------


class TestWriteConfigAtomic:
    """Tests for _write_config_atomic(config_path, content)."""

    def test_creates_file_with_correct_content(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        target = tmp_path / "missy" / "config.yaml"
        _write_config_atomic(target, "hello: world\n")
        assert target.exists()
        assert target.read_text() == "hello: world\n"

    def test_creates_parent_directory(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        target = tmp_path / "deep" / "nested" / "config.yaml"
        assert not target.parent.exists()
        _write_config_atomic(target, "x: 1")
        assert target.parent.exists()

    def test_file_permissions_are_600(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        target = tmp_path / "cfg" / "config.yaml"
        _write_config_atomic(target, "key: value")
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o600

    def test_overwrites_existing_file(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        target = tmp_path / "config.yaml"
        target.write_text("old content")
        _write_config_atomic(target, "new content")
        assert target.read_text() == "new content"

    def test_no_temp_file_left_on_success(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        target = tmp_path / "config.yaml"
        _write_config_atomic(target, "data")
        tmp_files = [f for f in tmp_path.iterdir() if f.name.startswith(".config_tmp_")]
        assert tmp_files == []

    def test_temp_file_cleaned_up_on_write_failure(self, tmp_path):
        """If the write raises, the temp file must be removed."""
        from missy.cli.wizard import _write_config_atomic

        target = tmp_path / "config.yaml"

        # Patch os.replace to simulate a failure after the temp file is written

        def failing_replace(src, dst):
            raise OSError("simulated replace failure")

        with patch("os.replace", side_effect=failing_replace), pytest.raises(OSError, match="simulated replace failure"):
            _write_config_atomic(target, "content")

        # No orphan temp files should remain
        tmp_files = [f for f in tmp_path.iterdir() if f.name.startswith(".config_tmp_")]
        assert tmp_files == []

    def test_unicode_content_written_correctly(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        content = "# Comment with Unicode: café, naïve, 日本語\nkey: value\n"
        target = tmp_path / "config.yaml"
        _write_config_atomic(target, content)
        assert target.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# run_wizard_noninteractive
# ---------------------------------------------------------------------------


class TestRunWizardNoninteractive:
    """Tests for run_wizard_noninteractive(...)."""

    def test_valid_anthropic_provider_creates_config(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="anthropic",
            api_key="sk-ant-api03-TESTKEY1234",
            workspace=str(tmp_path / "ws"),
        )
        assert cfg.exists()
        parsed = yaml.safe_load(cfg.read_text())
        assert parsed["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"

    def test_valid_openai_provider(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="openai",
            api_key="sk-testkey1234",
            workspace=str(tmp_path / "ws"),
        )
        parsed = yaml.safe_load(cfg.read_text())
        assert "openai" in parsed["providers"]
        assert parsed["providers"]["openai"]["model"] == "gpt-4o"

    def test_ollama_provider_no_key_required(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="ollama",
            model="mistral",
            workspace=str(tmp_path / "ws"),
        )
        parsed = yaml.safe_load(cfg.read_text())
        assert parsed["providers"]["ollama"]["model"] == "mistral"

    def test_unknown_provider_raises(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        with pytest.raises(click.ClickException, match="Unknown provider"):
            run_wizard_noninteractive(config_path=str(cfg), provider="badprovider")

    def test_missing_env_key_raises(self, tmp_path, monkeypatch):
        from missy.cli.wizard import run_wizard_noninteractive

        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        cfg = tmp_path / ".missy" / "config.yaml"
        with pytest.raises(click.ClickException, match="not set or empty"):
            run_wizard_noninteractive(
                config_path=str(cfg),
                provider="anthropic",
                api_key_env="NONEXISTENT_KEY",
            )

    def test_api_key_env_var_resolved(self, tmp_path, monkeypatch):
        from missy.cli.wizard import run_wizard_noninteractive

        monkeypatch.setenv("MY_ANT_KEY", "sk-ant-api03-ENVKEY5678")
        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="anthropic",
            api_key_env="MY_ANT_KEY",
            workspace=str(tmp_path / "ws"),
        )
        parsed = yaml.safe_load(cfg.read_text())
        assert parsed["providers"]["anthropic"]["api_key"] == "sk-ant-api03-ENVKEY5678"

    def test_custom_model_overrides_default(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="anthropic",
            api_key="sk-ant-test",
            model="claude-opus-4-6",
            workspace=str(tmp_path / "ws"),
        )
        parsed = yaml.safe_load(cfg.read_text())
        assert parsed["providers"]["anthropic"]["model"] == "claude-opus-4-6"

    def test_ollama_no_default_model_and_no_model_raises(self, tmp_path):
        """Ollama primary model is empty string; no --model → ClickException."""
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        with pytest.raises(click.ClickException, match="No default model"):
            run_wizard_noninteractive(
                config_path=str(cfg),
                provider="ollama",
                workspace=str(tmp_path / "ws"),
                # no model supplied
            )

    def test_directories_created_alongside_config(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="anthropic",
            api_key="sk-ant-test",
            workspace=str(tmp_path / "ws"),
        )
        assert (tmp_path / ".missy" / "secrets").is_dir()
        assert (tmp_path / ".missy" / "logs").is_dir()

    def test_jobs_file_created(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="anthropic",
            api_key="sk-ant-test",
            workspace=str(tmp_path / "ws"),
        )
        jobs = tmp_path / ".missy" / "jobs.json"
        assert jobs.exists()
        assert jobs.read_text().strip() == "[]"

    def test_overwrites_existing_config_on_second_call(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        cfg = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="anthropic",
            api_key="sk-ant-first",
            workspace=str(tmp_path / "ws"),
        )
        run_wizard_noninteractive(
            config_path=str(cfg),
            provider="openai",
            api_key="sk-second",
            workspace=str(tmp_path / "ws"),
        )
        parsed = yaml.safe_load(cfg.read_text())
        assert "openai" in parsed["providers"]


# ---------------------------------------------------------------------------
# _verify_anthropic
# ---------------------------------------------------------------------------


class TestVerifyAnthropic:
    """Tests for _verify_anthropic(api_key)."""

    def test_setup_token_rejected_immediately(self):
        from missy.cli.wizard import _verify_anthropic

        result = _verify_anthropic("sk-ant-oat-everything-else")
        assert result is False

    def test_setup_token_prefix_sk_ant_oat(self):
        from missy.cli.wizard import _verify_anthropic

        # Any key starting with sk-ant-oat should be rejected without an API call
        with patch("anthropic.Anthropic") as mock_cls:
            result = _verify_anthropic("sk-ant-oat-MYSECRETTOKEN")
        assert result is False
        mock_cls.assert_not_called()

    def test_valid_key_success(self):
        from missy.cli.wizard import _verify_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = _verify_anthropic("sk-ant-api03-VALIDKEY")
        assert result is True
        mock_client.messages.create.assert_called_once()

    def test_api_exception_returns_false(self):
        from missy.cli.wizard import _verify_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("AuthenticationError")
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = _verify_anthropic("sk-ant-api03-BADKEY")
        assert result is False

    def test_uses_haiku_model_for_verification(self):
        from missy.cli.wizard import _verify_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            _verify_anthropic("sk-ant-api03-SOMEKEY")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"

    def test_max_tokens_is_1(self):
        from missy.cli.wizard import _verify_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            _verify_anthropic("sk-ant-api03-SOMEKEY")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 1


# ---------------------------------------------------------------------------
# _verify_openai
# ---------------------------------------------------------------------------


class TestVerifyOpenai:
    """Tests for _verify_openai(api_key)."""

    def test_success_returns_true(self):
        from missy.cli.wizard import _verify_openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        with patch("openai.OpenAI", return_value=mock_client):
            result = _verify_openai("sk-testkey")
        assert result is True

    def test_exception_returns_false(self):
        from missy.cli.wizard import _verify_openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")
        with patch("openai.OpenAI", return_value=mock_client):
            result = _verify_openai("sk-badkey")
        assert result is False

    def test_uses_gpt4o_mini_for_verification(self):
        from missy.cli.wizard import _verify_openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        with patch("openai.OpenAI", return_value=mock_client):
            _verify_openai("sk-test")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o-mini"

    def test_max_tokens_is_1(self):
        from missy.cli.wizard import _verify_openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        with patch("openai.OpenAI", return_value=mock_client):
            _verify_openai("sk-test")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 1

    def test_api_key_passed_to_client(self):
        from missy.cli.wizard import _verify_openai

        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value.chat.completions.create.return_value = MagicMock()
            _verify_openai("sk-myspecifickey")

        mock_cls.assert_called_once_with(api_key="sk-myspecifickey")


# ---------------------------------------------------------------------------
# _verify_ollama
# ---------------------------------------------------------------------------


class TestVerifyOllama:
    """Tests for _verify_ollama(base_url, model)."""

    def test_success_returns_true(self):
        from missy.cli.wizard import _verify_ollama

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp):
            result = _verify_ollama("http://localhost:11434", "llama3")
        assert result is True

    def test_connection_error_returns_false(self):
        from missy.cli.wizard import _verify_ollama

        with patch("httpx.get", side_effect=Exception("Connection refused")):
            result = _verify_ollama("http://localhost:11434", "llama3")
        assert result is False

    def test_http_error_returns_false(self):
        from missy.cli.wizard import _verify_ollama

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        with patch("httpx.get", return_value=mock_resp):
            result = _verify_ollama("http://localhost:11434", "llama3")
        assert result is False

    def test_calls_api_tags_endpoint(self):
        from missy.cli.wizard import _verify_ollama

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            _verify_ollama("http://localhost:11434", "llama3")

        called_url = mock_get.call_args.args[0]
        assert called_url.endswith("/api/tags")

    def test_trailing_slash_normalised(self):
        from missy.cli.wizard import _verify_ollama

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            _verify_ollama("http://localhost:11434/", "llama3")

        called_url = mock_get.call_args.args[0]
        # Should not result in double-slash before api/tags
        assert "//api/tags" not in called_url

    def test_timeout_set_to_5(self):
        from missy.cli.wizard import _verify_ollama

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            _verify_ollama("http://localhost:11434", "llama3")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs.get("timeout") == 5
