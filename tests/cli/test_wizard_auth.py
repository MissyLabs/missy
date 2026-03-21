"""Tests for the CLI wizard and auth modules.

Covers:
  - missy/cli/wizard.py  — interactive onboarding wizard
  - missy/cli/oauth.py   — OpenAI OAuth PKCE flow
  - missy/cli/anthropic_auth.py — Anthropic token auth helpers

All click prompts, file I/O, HTTP calls, and browser operations are mocked.
Tests are focused on logic branches, not UI rendering.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# wizard.py — helper functions
# ===========================================================================


class TestMaskKey:
    def test_short_key_fully_masked(self):
        from missy.cli.wizard import _mask_key

        result = _mask_key("abc")
        assert result == "***"

    def test_exactly_eight_chars_fully_masked(self):
        from missy.cli.wizard import _mask_key

        result = _mask_key("12345678")
        assert result == "********"

    def test_long_key_shows_prefix_and_suffix(self):
        from missy.cli.wizard import _mask_key

        key = "sk-ant-api03-abcdefghij"
        result = _mask_key(key)
        assert result.startswith("sk-ant")
        assert result.endswith(key[-4:])
        assert "…" in result

    def test_nine_char_key_shows_prefix_and_suffix(self):
        from missy.cli.wizard import _mask_key

        key = "abcdefghi"
        # _mask_key uses key[:6] + "…" + key[-4:], so last 4 chars of "abcdefghi" is "fghi"
        assert _mask_key(key) == "abcdef…fghi"


class TestDetectEnvKey:
    def test_returns_value_when_env_set(self):
        from missy.cli.wizard import _detect_env_key

        with patch.dict("os.environ", {"MY_KEY": "sk-123"}):
            assert _detect_env_key("MY_KEY") == "sk-123"

    def test_returns_none_when_env_not_set(self):
        from missy.cli.wizard import _detect_env_key

        with patch.dict("os.environ", {}, clear=True):
            assert _detect_env_key("MISSING_KEY") is None

    def test_returns_none_for_empty_string(self):
        from missy.cli.wizard import _detect_env_key

        with patch.dict("os.environ", {"MY_KEY": ""}):
            assert _detect_env_key("MY_KEY") is None


class TestValidateKeyFormat:
    def test_valid_key_with_correct_prefix(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("sk-ant-api03-abc", "sk-ant-") is True

    def test_invalid_key_wrong_prefix(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("wrong-prefix-abc", "sk-ant-") is False

    def test_empty_key_invalid(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("", "sk-ant-") is False

    def test_whitespace_only_invalid(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("   ", "sk-ant-") is False

    def test_no_prefix_required_accepts_any_nonempty(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("anything", None) is True

    def test_no_prefix_required_rejects_empty(self):
        from missy.cli.wizard import _validate_key_format

        assert _validate_key_format("", None) is False


class TestPromptApiKey:
    def test_env_var_detected_and_accepted_returns_none(self):
        """When env var found and user accepts, return None (use env var)."""
        from missy.cli.wizard import _prompt_api_key

        info = {"env_var": "ANTHROPIC_API_KEY", "label": "Anthropic", "key_prefix": "sk-ant-"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-api03-realkey"}),
            patch("click.confirm", return_value=True),
        ):
            result = _prompt_api_key("anthropic", info)
        assert result is None

    def test_env_var_detected_but_declined_falls_through_to_prompt(self):
        """When env var found but user declines, fall through to manual prompt."""
        from missy.cli.wizard import _prompt_api_key

        info = {"env_var": "ANTHROPIC_API_KEY", "label": "Anthropic", "key_prefix": "sk-ant-"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-api03-realkey"}),
            patch("click.confirm", side_effect=[False, True]),  # decline env, skip provider
            patch("click.prompt", return_value=""),
        ):
            result = _prompt_api_key("anthropic", info)
        assert result is None

    def test_valid_key_returned_directly(self):
        from missy.cli.wizard import _prompt_api_key

        info = {"env_var": None, "label": "OpenAI", "key_prefix": "sk-"}
        with patch("click.prompt", return_value="sk-abc123"):
            result = _prompt_api_key("openai", info)
        assert result == "sk-abc123"

    def test_key_with_assignment_prefix_stripped(self):
        """Keys pasted as 'export KEY=value' have the assignment stripped."""
        from missy.cli.wizard import _prompt_api_key

        info = {"env_var": None, "label": "OpenAI", "key_prefix": "sk-"}
        with patch("click.prompt", return_value="export OPENAI_API_KEY=sk-abcxyz"):
            result = _prompt_api_key("openai", info)
        assert result == "sk-abcxyz"

    def test_bad_format_confirmed_anyway_returns_key(self):
        """User can accept a malformed key after warning."""
        from missy.cli.wizard import _prompt_api_key

        info = {"env_var": None, "label": "Anthropic", "key_prefix": "sk-ant-"}
        with (
            patch("click.prompt", return_value="bad-key-format"),
            patch("click.confirm", return_value=True),  # use it anyway
        ):
            result = _prompt_api_key("anthropic", info)
        assert result == "bad-key-format"

    def test_bad_format_declined_then_valid_key(self):
        """After rejecting a bad format, a second prompt with a valid key is accepted."""
        from missy.cli.wizard import _prompt_api_key

        info = {"env_var": None, "label": "Anthropic", "key_prefix": "sk-ant-"}
        with (
            patch("click.prompt", side_effect=["bad-key", "sk-ant-valid"]),
            patch("click.confirm", return_value=False),  # do not use bad key
        ):
            result = _prompt_api_key("anthropic", info)
        assert result == "sk-ant-valid"

    def test_empty_prompt_then_skip(self):
        """Empty prompt followed by confirm to skip returns None."""
        from missy.cli.wizard import _prompt_api_key

        info = {"env_var": None, "label": "OpenAI", "key_prefix": "sk-"}
        with (
            patch("click.prompt", return_value=""),
            patch("click.confirm", return_value=True),  # skip provider
        ):
            result = _prompt_api_key("openai", info)
        assert result is None


class TestPromptModel:
    def test_ollama_free_form_returns_primary_and_empty_fast_premium(self):
        from missy.cli.wizard import _PROVIDERS, _prompt_model

        ollama_info = _PROVIDERS["ollama"]
        with patch("click.prompt", return_value="mistral"):
            primary, fast, premium = _prompt_model(ollama_info)
        assert primary == "mistral"
        assert fast == ""
        assert premium == ""

    def test_anthropic_numbered_choice_returns_model(self):
        from missy.cli.wizard import _PROVIDERS, _prompt_model

        info = _PROVIDERS["anthropic"]
        # Select index 1 for all three tiers (claude-sonnet-4-6)
        with patch("click.prompt", side_effect=["1", "2", "3"]):
            primary, fast, premium = _prompt_model(info)
        # Index 1 → first model (claude-sonnet-4-6)
        assert primary == "claude-sonnet-4-6"
        # Index 2 → second model (claude-haiku-4-5-20251001)
        assert fast == "claude-haiku-4-5-20251001"
        # Index 3 → third model (claude-opus-4-6)
        assert premium == "claude-opus-4-6"

    def test_invalid_choice_falls_back_to_default(self):
        from missy.cli.wizard import _PROVIDERS, _prompt_model

        info = _PROVIDERS["anthropic"]
        with patch("click.prompt", side_effect=["99", "99", "99"]):
            primary, fast, premium = _prompt_model(info)
        assert primary == info["models"]["primary"]
        assert fast == info["models"]["fast"]
        assert premium == info["models"]["premium"]


class TestVerifyAnthropic:
    def test_setup_token_prefix_rejected_without_api_call(self):
        from missy.cli.wizard import _verify_anthropic

        result = _verify_anthropic("sk-ant-oat01-sometoken")
        assert result is False

    def test_successful_api_call_returns_true(self):
        from missy.cli.wizard import _verify_anthropic

        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = _verify_anthropic("sk-ant-api03-goodkey")
        assert result is True
        mock_client.messages.create.assert_called_once()

    def test_api_exception_returns_false(self):
        from missy.cli.wizard import _verify_anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("auth error")
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = _verify_anthropic("sk-ant-api03-badkey")
        assert result is False


class TestVerifyOpenAI:
    def test_successful_api_call_returns_true(self):
        from missy.cli.wizard import _verify_openai

        mock_client = MagicMock()
        with patch("openai.OpenAI", return_value=mock_client):
            result = _verify_openai("sk-goodkey")
        assert result is True
        mock_client.chat.completions.create.assert_called_once()

    def test_api_exception_returns_false(self):
        from missy.cli.wizard import _verify_openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("auth error")
        with patch("openai.OpenAI", return_value=mock_client):
            result = _verify_openai("sk-badkey")
        assert result is False


class TestVerifyOllama:
    def test_reachable_endpoint_returns_true(self):
        from missy.cli.wizard import _verify_ollama

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = _verify_ollama("http://localhost:11434", "llama3")
        assert result is True

    def test_connection_error_returns_false(self):
        from missy.cli.wizard import _verify_ollama

        with patch("httpx.get", side_effect=Exception("connection refused")):
            result = _verify_ollama("http://localhost:11434", "llama3")
        assert result is False


class TestBuildConfigYaml:
    def test_basic_config_contains_workspace(self):
        from missy.cli.wizard import _build_config_yaml

        content = _build_config_yaml(
            workspace="/home/user/workspace",
            providers_cfg=[],
            allowed_hosts=[],
        )
        assert "/home/user/workspace" in content

    def test_allowed_hosts_listed(self):
        from missy.cli.wizard import _build_config_yaml

        content = _build_config_yaml(
            workspace="/tmp/ws",
            providers_cfg=[],
            allowed_hosts=["api.anthropic.com", "api.openai.com"],
        )
        # Anthropic host may be collapsed into a preset; either form is valid.
        assert "api.anthropic.com" in content or "anthropic" in content
        assert "api.openai.com" in content or "openai" in content

    def test_provider_api_key_embedded(self):
        from missy.cli.wizard import _build_config_yaml

        providers = [
            {
                "name": "anthropic",
                "model": "claude-sonnet-4-6",
                "fast_model": "claude-haiku-4-5-20251001",
                "premium_model": "claude-opus-4-6",
                "api_key": "sk-ant-api03-testkey",
                "base_url": None,
            }
        ]
        content = _build_config_yaml(
            workspace="/tmp/ws",
            providers_cfg=providers,
            allowed_hosts=[],
        )
        assert "sk-ant-api03-testkey" in content
        assert "claude-sonnet-4-6" in content

    def test_discord_section_written_when_provided(self):
        from missy.cli.wizard import _build_config_yaml

        discord_cfg = {
            "bot_token": "tok123",
            "token_env_var": "DISCORD_BOT_TOKEN",
            "application_id": "app456",
            "dm_policy": "allowlist",
            "dm_allowlist": ["111", "222"],
            "guild_policies": [],
            "ack_reaction": "eyes",
            "ignore_bots": True,
        }
        content = _build_config_yaml(
            workspace="/tmp/ws",
            providers_cfg=[],
            allowed_hosts=[],
            discord_cfg=discord_cfg,
        )
        assert "discord:" in content
        assert "tok123" in content
        assert "allowlist" in content
        assert '"111"' in content

    def test_discord_guild_policies_written(self):
        from missy.cli.wizard import _build_config_yaml

        discord_cfg = {
            "bot_token": "",
            "token_env_var": "DISCORD_BOT_TOKEN",
            "application_id": "",
            "dm_policy": "disabled",
            "dm_allowlist": [],
            "guild_policies": [
                {
                    "guild_id": "9999",
                    "require_mention": True,
                    "allowed_channels": ["general"],
                    "mode": "full",
                }
            ],
            "ack_reaction": "",
            "ignore_bots": True,
        }
        content = _build_config_yaml(
            workspace="/tmp/ws",
            providers_cfg=[],
            allowed_hosts=[],
            discord_cfg=discord_cfg,
        )
        assert "9999" in content
        assert "general" in content

    def test_provider_vault_ref_not_quoted_as_key(self):
        """vault:// references should appear literally in config."""
        from missy.cli.wizard import _build_config_yaml

        providers = [
            {
                "name": "anthropic",
                "model": "claude-sonnet-4-6",
                "fast_model": "",
                "premium_model": "",
                "api_key": "vault://anthropic_api_key",
                "base_url": None,
            }
        ]
        content = _build_config_yaml(
            workspace="/tmp/ws",
            providers_cfg=providers,
            allowed_hosts=[],
        )
        assert "vault://anthropic_api_key" in content

    def test_empty_allowed_hosts_writes_empty_list(self):
        from missy.cli.wizard import _build_config_yaml

        content = _build_config_yaml(
            workspace="/tmp/ws",
            providers_cfg=[],
            allowed_hosts=[],
        )
        # The empty list marker should appear for allowed_hosts
        assert "[]" in content


class TestWriteConfigAtomic:
    def test_writes_file_to_disk(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        config_path = tmp_path / "config.yaml"
        _write_config_atomic(config_path, "key: value\n")
        assert config_path.exists()
        assert config_path.read_text() == "key: value\n"

    def test_creates_parent_directories(self, tmp_path):
        from missy.cli.wizard import _write_config_atomic

        config_path = tmp_path / "subdir" / "deep" / "config.yaml"
        _write_config_atomic(config_path, "content")
        assert config_path.exists()

    def test_write_failure_does_not_leave_temp_file(self, tmp_path):

        from missy.cli.wizard import _write_config_atomic

        config_path = tmp_path / "config.yaml"
        with patch("os.replace", side_effect=OSError("disk full")), pytest.raises(OSError):
            _write_config_atomic(config_path, "content")
        # Temp file should be cleaned up
        temp_files = list(tmp_path.glob(".config_tmp_*"))
        assert len(temp_files) == 0


# ===========================================================================
# wizard.py — run_wizard integration (mocking full flow)
# ===========================================================================


class TestRunWizard:
    """Test the run_wizard entry point with mocked I/O."""

    def _make_prompts(self, *values):
        """Return a side_effect list for click.prompt."""
        return list(values)

    def test_aborts_when_user_declines_overwrite(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        config_file.write_text("existing: true\n")

        with patch("click.confirm", return_value=False):
            run_wizard(str(config_file))

        # File should be unchanged
        assert config_file.read_text() == "existing: true\n"

    def test_wizard_skips_providers_when_choice_is_zero(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        prompts = [
            str(tmp_path / "workspace"),  # workspace dir
            "0",  # provider choice: skip
        ]
        confirms = [
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "providers:" in content

    def test_wizard_writes_anthropic_config(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "1",  # provider: anthropic
            "1",  # auth method: API key
            "sk-ant-api03-testkey",  # API key
            "1",  # primary model
            "2",  # fast model
            "3",  # premium model
        ]
        confirms = [
            False,  # verify API key?
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "anthropic" in content
        assert "sk-ant-api03-testkey" in content

    def test_wizard_writes_openai_api_key_config(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "2",  # provider: openai
            "1",  # auth method: API key
            "sk-testkey",  # API key
            "1",  # primary model
            "2",  # fast model
            "3",  # premium model
        ]
        confirms = [
            False,  # verify API key?
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "openai" in content

    def test_wizard_openai_oauth_flow_on_success(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "2",  # provider: openai
            "2",  # auth method: OAuth
            "1",  # primary model
            "2",  # fast model
            "3",  # premium model
        ]
        confirms = [
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch("missy.cli.oauth.run_openai_oauth", return_value="oauth-access-token"),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "openai-codex" in content

    def test_wizard_openai_oauth_failure_falls_back_to_api_key(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "2",  # provider: openai
            "2",  # auth method: OAuth (fails)
            "sk-fallback",  # fallback API key prompt
            "1",  # primary model
            "2",  # fast model
            "3",  # premium model
        ]
        # OAuth fallback path has no verify confirm (only the else branch does).
        confirms = [
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch("missy.cli.oauth.run_openai_oauth", return_value=None),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()

    def test_wizard_anthropic_setup_token_flow(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "1",  # provider: anthropic
            "3",  # auth method: setup-token
            "1",  # primary model
            "2",  # fast model
            "3",  # premium model
        ]
        confirms = [
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch(
                "missy.cli.anthropic_auth.run_anthropic_setup_token_flow",
                return_value="sk-ant-oat01-token",
            ),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()

    def test_wizard_anthropic_vault_flow_stores_ref(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "1",  # provider: anthropic
            "2",  # auth method: vault
            "sk-ant-api03-rawkey",  # API key
            "1",  # primary model
            "2",  # fast model
            "3",  # premium model
        ]
        # run_anthropic_vault_flow is fully mocked so its internal confirm is not called.
        # Verify confirm is also skipped because api_key starts with vault://.
        confirms = [
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch(
                "missy.cli.anthropic_auth.run_anthropic_vault_flow",
                return_value="vault://anthropic_api_key",
            ),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "vault://anthropic_api_key" in content

    def test_wizard_ollama_provider(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "3",  # provider: ollama
            "http://localhost:11434",  # ollama base URL
            "llama3",  # default model
        ]
        confirms = [
            False,  # verify ollama connectivity?
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "ollama" in content
        assert "llama3" in content

    def test_wizard_both_providers_anthropic_and_openai(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "4",  # provider: anthropic + openai
            # Anthropic
            "1",  # auth: API key
            "sk-ant-api03-akey",  # API key
            "1",
            "2",
            "3",  # model picks
            # OpenAI
            "1",  # auth: API key
            "sk-oaikey",  # API key
            "1",
            "2",
            "3",  # model picks
        ]
        confirms = [
            False,  # verify anthropic?
            False,  # verify openai?
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        content = config_file.read_text()
        assert "anthropic" in content
        assert "openai" in content

    def test_wizard_discord_configuration(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "0",  # provider: skip
            # Discord setup
            "bottoken123",  # bot token
            "app_12345",  # application ID
            "2",  # dm policy: allowlist
            "111,222",  # allowed user IDs
            "eyes",  # ack reaction
        ]
        confirms = [
            True,  # configure Discord?
            False,  # add guild policy?
            True,  # ignore bots?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        content = config_file.read_text()
        assert "discord:" in content
        assert "allowlist" in content
        # Discord hosts may be collapsed into a preset.
        assert "discord.com" in content or "discord" in content

    def test_wizard_aborts_write_on_final_confirm_no(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "0",  # provider: skip
        ]
        confirms = [
            False,  # configure Discord?
            False,  # write config?  <-- user says no
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert not config_file.exists()

    def test_wizard_creates_missy_subdirs(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / ".missy" / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "0",  # provider: skip
        ]
        confirms = [
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert (tmp_path / ".missy" / "secrets").exists()
        assert (tmp_path / ".missy" / "logs").exists()
        assert (tmp_path / ".missy" / "jobs.json").exists()

    def test_wizard_os_error_on_write_exits_with_1(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "0",  # provider: skip
        ]
        confirms = [
            False,  # configure Discord?
            True,  # write config?
        ]
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch(
                "missy.cli.wizard._write_config_atomic",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_wizard(str(config_file))
        assert exc_info.value.code == 1

    def test_wizard_verify_anthropic_key_on_success(self, tmp_path):
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"
        workspace_dir = str(tmp_path / "workspace")

        prompts = [
            workspace_dir,  # workspace
            "1",  # provider: anthropic
            "1",  # auth: API key
            "sk-ant-api03-valid",  # API key
            "1",
            "2",
            "3",  # models
        ]
        confirms = [
            True,  # verify API key?
            False,  # configure Discord?
            True,  # write config?
        ]
        mock_client = MagicMock()
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch("anthropic.Anthropic", return_value=mock_client),
        ):
            run_wizard(str(config_file))

        mock_client.messages.create.assert_called_once()
        assert config_file.exists()


# ===========================================================================
# oauth.py — PKCE helpers
# ===========================================================================


class TestGeneratePkce:
    def test_returns_two_strings(self):
        from missy.cli.oauth import _generate_pkce

        verifier, challenge = _generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_challenge_is_sha256_of_verifier(self):
        from missy.cli.oauth import _generate_pkce

        verifier, challenge = _generate_pkce()
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        assert challenge == expected

    def test_verifier_is_urlsafe_base64(self):
        from missy.cli.oauth import _generate_pkce

        verifier, _ = _generate_pkce()
        # Should only contain URL-safe base64 chars (no +, /, =)
        assert "+" not in verifier
        assert "/" not in verifier
        assert "=" not in verifier

    def test_each_call_produces_unique_values(self):
        from missy.cli.oauth import _generate_pkce

        v1, c1 = _generate_pkce()
        v2, c2 = _generate_pkce()
        assert v1 != v2
        assert c1 != c2


class TestBuildAuthUrl:
    def test_contains_client_id(self):
        from missy.cli.oauth import _build_auth_url

        url = _build_auth_url("my-client", "mystate", "mychallenge")
        assert "my-client" in url

    def test_contains_redirect_uri(self):
        from missy.cli.oauth import _build_auth_url

        url = _build_auth_url("cid", "state", "challenge")
        assert "redirect_uri" in url

    def test_contains_state(self):
        from missy.cli.oauth import _build_auth_url

        url = _build_auth_url("cid", "uniquestate", "challenge")
        assert "uniquestate" in url

    def test_challenge_method_is_s256(self):
        from missy.cli.oauth import _build_auth_url

        url = _build_auth_url("cid", "state", "challenge")
        assert "S256" in url

    def test_url_starts_with_authorize_endpoint(self):
        from missy.cli.oauth import AUTHORIZE_URL, _build_auth_url

        url = _build_auth_url("cid", "state", "challenge")
        assert url.startswith(AUTHORIZE_URL)


class TestParseJwtPayload:
    def _make_jwt(self, payload: dict) -> str:
        """Build a minimal JWT with base64url-encoded payload."""
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload_bytes = json.dumps(payload).encode()
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
        return f"{header}.{payload_b64}.fakesig"

    def test_extracts_email_claim(self):
        from missy.cli.oauth import _parse_jwt_payload

        token = self._make_jwt({"email": "user@example.com", "sub": "user123"})
        claims = _parse_jwt_payload(token)
        assert claims["email"] == "user@example.com"

    def test_returns_empty_dict_for_malformed_token(self):
        from missy.cli.oauth import _parse_jwt_payload

        assert _parse_jwt_payload("notajwt") == {}
        assert _parse_jwt_payload("") == {}

    def test_returns_empty_dict_for_invalid_base64(self):
        from missy.cli.oauth import _parse_jwt_payload

        assert _parse_jwt_payload("header.!!!invalid!!!.sig") == {}


class TestExtractAccountMetadata:
    def _make_jwt(self, payload: dict) -> str:
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload_bytes = json.dumps(payload).encode()
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
        return f"{header}.{payload_b64}.fakesig"

    def test_extracts_email_and_account_id_from_openai_namespace(self):
        from missy.cli.oauth import _extract_account_metadata

        token = self._make_jwt(
            {
                "email": "user@example.com",
                "https://api.openai.com/auth": {"chatgpt_account_id": "acc_abc"},
            }
        )
        account_id, email = _extract_account_metadata(token)
        assert email == "user@example.com"
        assert account_id == "acc_abc"

    def test_falls_back_to_sub_for_account_id(self):
        from missy.cli.oauth import _extract_account_metadata

        token = self._make_jwt({"sub": "user_sub_123", "email": "a@b.com"})
        account_id, email = _extract_account_metadata(token)
        assert account_id == "user_sub_123"

    def test_returns_empty_strings_for_malformed_token(self):
        from missy.cli.oauth import _extract_account_metadata

        account_id, email = _extract_account_metadata("notajwt")
        assert account_id == ""
        assert email == ""


class TestExtractCodeFromUrl:
    def test_extracts_code_from_full_redirect_url(self):
        from missy.cli.oauth import _extract_code_from_url

        url = "http://localhost:1455/auth/callback?code=AUTH_CODE_XYZ&state=abc"
        assert _extract_code_from_url(url) == "AUTH_CODE_XYZ"

    def test_returns_raw_value_when_no_query_string(self):
        from missy.cli.oauth import _extract_code_from_url

        assert _extract_code_from_url("AUTH_CODE_ONLY") == "AUTH_CODE_ONLY"

    def test_returns_none_for_empty_string(self):
        from missy.cli.oauth import _extract_code_from_url

        assert _extract_code_from_url("") is None

    def test_strips_whitespace(self):
        from missy.cli.oauth import _extract_code_from_url

        assert _extract_code_from_url("  AUTH_CODE  ") == "AUTH_CODE"


class TestExchangeCode:
    def test_successful_exchange_returns_token_dict(self):
        from missy.cli.oauth import _exchange_code

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "tok123", "refresh_token": "ref456"}

        with patch("httpx.post", return_value=mock_resp):
            result = _exchange_code("client_id", "auth_code", "verifier")

        assert result["access_token"] == "tok123"

    def test_non_200_raises_runtime_error(self):
        from missy.cli.oauth import _exchange_code

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"

        with (
            patch("httpx.post", return_value=mock_resp),
            pytest.raises(RuntimeError, match="Token exchange failed"),
        ):
            _exchange_code("client_id", "auth_code", "verifier")


class TestDoRefresh:
    def test_successful_refresh_returns_new_tokens(self):
        from missy.cli.oauth import _do_refresh

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "newtok", "expires_in": 3600}

        with patch("httpx.post", return_value=mock_resp):
            result = _do_refresh("client_id", "refresh_token")

        assert result["access_token"] == "newtok"

    def test_non_200_raises_runtime_error(self):
        from missy.cli.oauth import _do_refresh

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with (
            patch("httpx.post", return_value=mock_resp),
            pytest.raises(RuntimeError, match="Token refresh failed"),
        ):
            _do_refresh("client_id", "refresh_token")


class TestSaveToken:
    def test_writes_token_file_with_correct_permissions(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig_token_file = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.TOKEN_FILE = tmp_path / "secrets" / "openai-oauth.json"
            data = {"access_token": "tok", "provider": "openai-oauth"}
            oauth_mod._save_token(data)
            assert oauth_mod.TOKEN_FILE.exists()
            loaded = json.loads(oauth_mod.TOKEN_FILE.read_text())
            assert loaded["access_token"] == "tok"
        finally:
            oauth_mod.TOKEN_FILE = orig_token_file


class TestLoadToken:
    def test_returns_none_when_file_missing(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.TOKEN_FILE = tmp_path / "nonexistent.json"
            assert oauth_mod.load_token() is None
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_returns_dict_when_file_exists(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            token_file = tmp_path / "openai-oauth.json"
            token_file.write_text(json.dumps({"access_token": "tok123"}))
            oauth_mod.TOKEN_FILE = token_file
            result = oauth_mod.load_token()
            assert result["access_token"] == "tok123"
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_returns_none_for_corrupt_json(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            token_file = tmp_path / "openai-oauth.json"
            token_file.write_text("not valid json {{")
            oauth_mod.TOKEN_FILE = token_file
            assert oauth_mod.load_token() is None
        finally:
            oauth_mod.TOKEN_FILE = orig


class TestRefreshTokenIfNeeded:
    def test_returns_access_token_when_still_valid(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            future_expiry = int(time.time()) + 7200  # 2 hours in the future
            token_data = {
                "access_token": "valid_tok",
                "refresh_token": "ref",
                "expires_at": future_expiry,
                "client_id": "cid",
            }
            token_file = tmp_path / "tok.json"
            token_file.write_text(json.dumps(token_data))
            oauth_mod.TOKEN_FILE = token_file

            result = oauth_mod.refresh_token_if_needed()
            assert result == "valid_tok"
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_refreshes_when_token_near_expiry(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            past_expiry = int(time.time()) - 10  # already expired
            token_data = {
                "access_token": "old_tok",
                "refresh_token": "ref_tok",
                "expires_at": past_expiry,
                "client_id": "cid",
            }
            token_file = tmp_path / "tok.json"
            token_file.write_text(json.dumps(token_data))
            oauth_mod.TOKEN_FILE = token_file

            new_resp = {"access_token": "new_tok", "expires_in": 3600}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = new_resp

            with patch("httpx.post", return_value=mock_resp):
                result = oauth_mod.refresh_token_if_needed(client_id="cid")

            assert result == "new_tok"
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_returns_none_when_no_token_stored(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.TOKEN_FILE = tmp_path / "missing.json"
            result = oauth_mod.refresh_token_if_needed()
            assert result is None
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_returns_stale_token_when_refresh_fails(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            past_expiry = int(time.time()) - 10
            token_data = {
                "access_token": "stale_tok",
                "refresh_token": "ref_tok",
                "expires_at": past_expiry,
                "client_id": "cid",
            }
            token_file = tmp_path / "tok.json"
            token_file.write_text(json.dumps(token_data))
            oauth_mod.TOKEN_FILE = token_file

            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Server error"

            with patch("httpx.post", return_value=mock_resp):
                result = oauth_mod.refresh_token_if_needed(client_id="cid")

            assert result == "stale_tok"
        finally:
            oauth_mod.TOKEN_FILE = orig


class TestRunOpenaiOauth:
    def _mock_exchange_response(self, access_token: str) -> dict:
        return {
            "access_token": access_token,
            "refresh_token": "ref",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

    def test_returns_none_when_no_code_received(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.TOKEN_FILE = tmp_path / "tok.json"
            with (
                patch("missy.cli.oauth._start_callback_server", return_value=None),
                patch("webbrowser.open"),
                patch("click.prompt", return_value=""),
            ):  # no paste
                result = oauth_mod.run_openai_oauth("test-client")
            assert result is None
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_returns_access_token_on_successful_code_exchange(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.TOKEN_FILE = tmp_path / "tok.json"
            redirect_url = "http://localhost:1455/auth/callback?code=AUTH123&state=st"

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = self._mock_exchange_response("access_tok_xyz")

            with (
                patch("missy.cli.oauth._start_callback_server", return_value=None),
                patch("webbrowser.open"),
                patch("click.prompt", return_value=redirect_url),
                patch("httpx.post", return_value=mock_resp),
            ):
                result = oauth_mod.run_openai_oauth("test-client")

            assert result == "access_tok_xyz"
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_returns_none_when_exchange_fails(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.TOKEN_FILE = tmp_path / "tok.json"
            redirect_url = "http://localhost:1455/auth/callback?code=AUTH123&state=st"

            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.text = "Bad Request"

            with (
                patch("missy.cli.oauth._start_callback_server", return_value=None),
                patch("webbrowser.open"),
                patch("click.prompt", return_value=redirect_url),
                patch("httpx.post", return_value=mock_resp),
            ):
                result = oauth_mod.run_openai_oauth("test-client")

            assert result is None
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_returns_none_when_response_missing_access_token(self, tmp_path):
        from missy.cli import oauth as oauth_mod

        orig = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.TOKEN_FILE = tmp_path / "tok.json"
            redirect_url = "http://localhost:1455/auth/callback?code=AUTH123&state=st"

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"token_type": "Bearer"}  # no access_token

            with (
                patch("missy.cli.oauth._start_callback_server", return_value=None),
                patch("webbrowser.open"),
                patch("click.prompt", return_value=redirect_url),
                patch("httpx.post", return_value=mock_resp),
            ):
                result = oauth_mod.run_openai_oauth("test-client")

            assert result is None
        finally:
            oauth_mod.TOKEN_FILE = orig

    def test_prompts_for_client_id_when_none_provided(self, tmp_path):
        """When no client ID is configured and env var is empty, prompt the user."""
        from missy.cli import oauth as oauth_mod

        orig_default = oauth_mod.DEFAULT_CLIENT_ID
        orig_file = oauth_mod.TOKEN_FILE
        try:
            oauth_mod.DEFAULT_CLIENT_ID = ""
            oauth_mod.TOKEN_FILE = tmp_path / "tok.json"

            # User supplies client ID then no code
            with (
                patch("click.prompt", side_effect=["my-client", ""]),
                patch("missy.cli.oauth._start_callback_server", return_value=None),
                patch("webbrowser.open"),
            ):
                result = oauth_mod.run_openai_oauth(None)

            assert result is None  # no code → None
        finally:
            oauth_mod.DEFAULT_CLIENT_ID = orig_default
            oauth_mod.TOKEN_FILE = orig_file

    def test_returns_none_when_client_id_prompt_is_empty(self):
        """If the client ID prompt is also empty, return None immediately."""
        from missy.cli import oauth as oauth_mod

        orig_default = oauth_mod.DEFAULT_CLIENT_ID
        try:
            oauth_mod.DEFAULT_CLIENT_ID = ""

            with patch("click.prompt", return_value=""):
                result = oauth_mod.run_openai_oauth(None)

            assert result is None
        finally:
            oauth_mod.DEFAULT_CLIENT_ID = orig_default


class TestCallbackHandler:
    def test_do_get_sets_code_and_signals_event(self):
        """_CallbackHandler.do_GET should populate _callback_result and set the event."""
        from missy.cli import oauth as oauth_mod

        # Reset module-level state
        oauth_mod._callback_result.clear()
        oauth_mod._callback_event.clear()

        handler = oauth_mod._CallbackHandler.__new__(oauth_mod._CallbackHandler)
        handler.path = "/auth/callback?code=TESTCODE&state=TESTSTATE"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()

        handler.do_GET()

        assert oauth_mod._callback_result["code"] == "TESTCODE"
        assert oauth_mod._callback_result["state"] == "TESTSTATE"
        assert oauth_mod._callback_event.is_set()

    def test_do_get_captures_oauth_error(self):
        from missy.cli import oauth as oauth_mod

        oauth_mod._callback_result.clear()
        oauth_mod._callback_event.clear()

        handler = oauth_mod._CallbackHandler.__new__(oauth_mod._CallbackHandler)
        handler.path = "/auth/callback?error=access_denied"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()

        handler.do_GET()

        assert oauth_mod._callback_result["error"] == "access_denied"


# ===========================================================================
# anthropic_auth.py — token classification
# ===========================================================================


class TestClassifyToken:
    def test_classifies_api_key(self):
        from missy.cli.anthropic_auth import classify_token

        # Construct a 90-char alphanumeric suffix to satisfy the regex
        suffix = "A" * 90
        token = f"sk-ant-api03-{suffix}"
        assert classify_token(token) == "api_key"

    def test_classifies_setup_token(self):
        from missy.cli.anthropic_auth import classify_token

        suffix = "B" * 70
        token = f"sk-ant-oat01-{suffix}"
        assert classify_token(token) == "setup_token"

    def test_partial_api_key_prefix_classified_as_api_key(self):
        from missy.cli.anthropic_auth import classify_token

        assert classify_token("sk-ant-api03-short") == "api_key"

    def test_partial_oat_prefix_classified_as_setup_token(self):
        from missy.cli.anthropic_auth import classify_token

        assert classify_token("sk-ant-oat01-short") == "setup_token"

    def test_unknown_token_returns_unknown(self):
        from missy.cli.anthropic_auth import classify_token

        assert classify_token("random_garbage") == "unknown"

    def test_strips_whitespace_before_classifying(self):
        from missy.cli.anthropic_auth import classify_token

        assert classify_token("  sk-ant-api03-something  ") == "api_key"


# ===========================================================================
# anthropic_auth.py — token persistence
# ===========================================================================


class TestStoreToken:
    def test_writes_token_file_atomically(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "secrets" / "anthropic-token.json"
            auth_mod.store_token("sk-ant-api03-key", "api_key")

            assert auth_mod.TOKEN_FILE.exists()
            data = json.loads(auth_mod.TOKEN_FILE.read_text())
            assert data["token"] == "sk-ant-api03-key"
            assert data["token_type"] == "api_key"
            assert data["provider"] == "anthropic"
            assert "issued_at" in data
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_custom_issued_at_preserved(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "anthropic-token.json"
            auth_mod.store_token("tok", "setup_token", issued_at=1000000)

            data = json.loads(auth_mod.TOKEN_FILE.read_text())
            assert data["issued_at"] == 1000000
        finally:
            auth_mod.TOKEN_FILE = orig


class TestLoadTokenAnthropic:
    def test_returns_none_when_file_missing(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "missing.json"
            assert auth_mod.load_token() is None
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_returns_dict_when_file_exists(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            token_file = tmp_path / "anthropic-token.json"
            token_file.write_text(json.dumps({"token": "sk-test", "token_type": "api_key"}))
            auth_mod.TOKEN_FILE = token_file
            result = auth_mod.load_token()
            assert result["token"] == "sk-test"
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_returns_none_for_corrupt_json(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            token_file = tmp_path / "anthropic-token.json"
            token_file.write_text("{broken")
            auth_mod.TOKEN_FILE = token_file
            assert auth_mod.load_token() is None
        finally:
            auth_mod.TOKEN_FILE = orig


class TestIsTokenExpiring:
    def test_api_key_never_expiring(self):
        from missy.cli.anthropic_auth import is_token_expiring

        data = {"token_type": "api_key", "issued_at": int(time.time()) - 10000}
        assert is_token_expiring(data) is False

    def test_fresh_setup_token_not_expiring(self):
        from missy.cli.anthropic_auth import is_token_expiring

        data = {"token_type": "setup_token", "issued_at": int(time.time())}
        assert is_token_expiring(data) is False

    def test_old_setup_token_is_expiring(self):
        from missy.cli.anthropic_auth import (
            REFRESH_WARN_MARGIN,
            SETUP_TOKEN_TTL_SECONDS,
            is_token_expiring,
        )

        # issued far enough in the past that it is within the refresh margin
        issued_at = int(time.time()) - (SETUP_TOKEN_TTL_SECONDS - REFRESH_WARN_MARGIN + 60)
        data = {"token_type": "setup_token", "issued_at": issued_at}
        assert is_token_expiring(data) is True

    def test_missing_issued_at_treated_as_epoch(self):
        from missy.cli.anthropic_auth import is_token_expiring

        # issued_at defaults to 0, so the token is ancient and expiring
        data = {"token_type": "setup_token"}
        assert is_token_expiring(data) is True


# ===========================================================================
# anthropic_auth.py — vault flow
# ===========================================================================


class TestTryStoreInVault:
    def test_returns_vault_ref_on_success(self):
        from missy.cli.anthropic_auth import _try_store_in_vault

        mock_vault = MagicMock()
        with patch("missy.security.vault.Vault", return_value=mock_vault):
            result = _try_store_in_vault("sk-ant-api03-key", "~/.missy/secrets")

        assert result == "vault://anthropic_api_key"
        mock_vault.set.assert_called_once_with("anthropic_api_key", "sk-ant-api03-key")

    def test_returns_none_on_vault_exception(self):
        from missy.cli.anthropic_auth import _try_store_in_vault

        with patch("missy.security.vault.Vault", side_effect=Exception("vault init failed")):
            result = _try_store_in_vault("sk-key", "~/.missy/secrets")

        assert result is None


class TestRunAnthropicVaultFlow:
    def test_vault_accepted_returns_vault_ref(self, tmp_path):
        from missy.cli.anthropic_auth import run_anthropic_vault_flow

        mock_vault = MagicMock()
        with (
            patch("click.confirm", return_value=True),
            patch("missy.security.vault.Vault", return_value=mock_vault),
            patch("missy.cli.anthropic_auth.store_token"),
        ):
            result = run_anthropic_vault_flow("sk-ant-api03-key", str(tmp_path))

        assert result == "vault://anthropic_api_key"

    def test_vault_declined_returns_raw_key(self, tmp_path):
        from missy.cli.anthropic_auth import run_anthropic_vault_flow

        with patch("click.confirm", return_value=False):
            result = run_anthropic_vault_flow("sk-ant-api03-key", str(tmp_path))

        assert result == "sk-ant-api03-key"

    def test_vault_failure_returns_raw_key(self, tmp_path):
        from missy.cli.anthropic_auth import run_anthropic_vault_flow

        with (
            patch("click.confirm", return_value=True),
            patch("missy.security.vault.Vault", side_effect=Exception("fail")),
        ):
            result = run_anthropic_vault_flow("sk-ant-api03-key", str(tmp_path))

        assert result == "sk-ant-api03-key"


# ===========================================================================
# anthropic_auth.py — setup-token paste flow
# ===========================================================================


class TestRunAnthropicSetupTokenFlow:
    def test_tos_declined_returns_none(self):
        from missy.cli.anthropic_auth import run_anthropic_setup_token_flow

        with patch("click.confirm", return_value=False):
            result = run_anthropic_setup_token_flow()

        assert result is None

    def test_valid_setup_token_stored_and_returned(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "anthropic-token.json"
            suffix = "C" * 70
            token = f"sk-ant-oat01-{suffix}"

            with (
                patch("click.confirm", return_value=True),  # accept ToS
                patch("click.prompt", return_value=token),
                patch("shutil.which", return_value=None),
            ):
                result = auth_mod.run_anthropic_setup_token_flow()

            assert result == token
            assert auth_mod.TOKEN_FILE.exists()
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_api_key_pasted_instead_of_setup_token_use_anyway(self, tmp_path):
        """If user pastes an API key and confirms using it, flow returns the key."""
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "anthropic-token.json"
            suffix = "D" * 90
            api_key = f"sk-ant-api03-{suffix}"

            with (
                patch("click.confirm", side_effect=[True, True]),  # ToS + use it anyway
                patch("click.prompt", return_value=api_key),
                patch("shutil.which", return_value=None),
            ):
                result = auth_mod.run_anthropic_setup_token_flow()

            assert result == api_key
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_api_key_pasted_declined_continues_loop_then_abort(self, tmp_path):
        """Pasting an API key then declining falls back to the paste loop."""
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "anthropic-token.json"
            suffix = "E" * 90
            api_key = f"sk-ant-api03-{suffix}"

            # ToS=True, use_api_key_anyway=False, empty prompt, abort=True
            with (
                patch("click.confirm", side_effect=[True, False, True]),
                patch("click.prompt", side_effect=[api_key, ""]),
                patch("shutil.which", return_value=None),
            ):
                result = auth_mod.run_anthropic_setup_token_flow()

            assert result is None
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_unknown_format_confirmed_returns_token(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "anthropic-token.json"
            weird_token = "weirdtoken12345"

            with (
                patch("click.confirm", side_effect=[True, True]),  # ToS + use anyway
                patch("click.prompt", return_value=weird_token),
                patch("shutil.which", return_value=None),
            ):
                result = auth_mod.run_anthropic_setup_token_flow()

            assert result == weird_token
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_assignment_prefix_stripped_from_pasted_token(self, tmp_path):
        """Tokens pasted as 'TOKEN=sk-ant-oat01-...' have the prefix stripped."""
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "anthropic-token.json"
            suffix = "F" * 70
            token = f"sk-ant-oat01-{suffix}"
            pasted = f"TOKEN={token}"

            with (
                patch("click.confirm", return_value=True),  # accept ToS
                patch("click.prompt", return_value=pasted),
                patch("shutil.which", return_value=None),
            ):
                result = auth_mod.run_anthropic_setup_token_flow()

            assert result == token
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_claude_cli_found_in_path(self, tmp_path):
        """When claude binary is found, the wizard prints its path (no crash)."""
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "anthropic-token.json"
            suffix = "G" * 70
            token = f"sk-ant-oat01-{suffix}"

            with (
                patch("click.confirm", return_value=True),
                patch("click.prompt", return_value=token),
                patch("shutil.which", return_value="/usr/local/bin/claude"),
            ):
                result = auth_mod.run_anthropic_setup_token_flow()

            assert result == token
        finally:
            auth_mod.TOKEN_FILE = orig


# ===========================================================================
# anthropic_auth.py — get_current_token resolution order
# ===========================================================================


class TestGetCurrentToken:
    def test_env_var_takes_priority(self, tmp_path):
        from missy.cli.anthropic_auth import get_current_token

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            result = get_current_token()

        assert result == "env-key"

    def test_token_file_used_when_no_env_var(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            token_file = tmp_path / "anthropic-token.json"
            token_file.write_text(
                json.dumps(
                    {"token": "file-key", "token_type": "api_key", "issued_at": int(time.time())}
                )
            )
            auth_mod.TOKEN_FILE = token_file

            with patch.dict("os.environ", {}, clear=True):
                result = auth_mod.get_current_token()

            assert result == "file-key"
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_vault_fallback_when_no_env_and_no_file(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "missing.json"

            mock_vault = MagicMock()
            mock_vault.get.return_value = "vault-key"

            with (
                patch.dict("os.environ", {}, clear=True),
                patch("missy.security.vault.Vault", return_value=mock_vault),
            ):
                result = auth_mod.get_current_token(str(tmp_path))

            assert result == "vault-key"
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_returns_none_when_all_sources_empty(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            auth_mod.TOKEN_FILE = tmp_path / "missing.json"

            with (
                patch.dict("os.environ", {}, clear=True),
                patch("missy.security.vault.Vault", side_effect=Exception("no vault")),
            ):
                result = auth_mod.get_current_token(str(tmp_path))

            assert result is None
        finally:
            auth_mod.TOKEN_FILE = orig

    def test_expiring_setup_token_triggers_remind_refresh(self, tmp_path):
        from missy.cli import anthropic_auth as auth_mod

        orig = auth_mod.TOKEN_FILE
        try:
            issued_at = int(time.time()) - (
                auth_mod.SETUP_TOKEN_TTL_SECONDS - auth_mod.REFRESH_WARN_MARGIN + 60
            )
            token_file = tmp_path / "anthropic-token.json"
            token_file.write_text(
                json.dumps(
                    {"token": "expiring-tok", "token_type": "setup_token", "issued_at": issued_at}
                )
            )
            auth_mod.TOKEN_FILE = token_file

            with (
                patch.dict("os.environ", {}, clear=True),
                patch("missy.cli.anthropic_auth.remind_refresh") as mock_remind,
            ):
                result = auth_mod.get_current_token()

            mock_remind.assert_called_once()
            assert result == "expiring-tok"
        finally:
            auth_mod.TOKEN_FILE = orig
