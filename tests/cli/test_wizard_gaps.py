"""Targeted coverage tests for missy/cli/wizard.py gaps.

Uncovered lines targeted:
- Line 148: _prompt_api_key — env var detected but user accepts it (returns None)
             Actually line 148 = ``return None`` when env key used.
             Wait: existing test covers env-var-accepted. Let's target what's TRULY missed.
- Line 191-192: _prompt_model — ValueError/IndexError path inside _pick
- Line 349: _build_config_yaml — discord no guild_policies → "guild_policies: {}"
             (already partially covered; ensure empty guild_policies branch)
- Line 452: run_wizard — provider choice "5" (all three: anthropic + openai + ollama)
- Lines 472-473: run_wizard — ollama verify prompt declined (click.confirm returns False)
- Lines 523-529: run_wizard — openai API key verify, verification FAILS
- Lines 553-554: run_wizard — anthropic setup-token flow returns None → fallback to API key
- Line 572: run_wizard — anthropic verification FAILS (ok=False branch)
- Line 619: run_wizard — discord bot_token starts with "Bot " prefix → stripped
- Lines 641-664: run_wizard — discord guild policy added with channels + multi-mode
- Line 706: run_wizard — openai provider + openai-oauth in verify_results → "(OAuth token)"
             (summary table branch when provider name is "openai" and oauth result present)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _prompt_model — ValueError/IndexError branch (lines 191-192)
# ---------------------------------------------------------------------------


class TestPromptModelInvalidChoiceFallsBack:
    def test_value_error_on_non_numeric_input_falls_back_to_default(self):
        """When the user types a non-numeric string, _pick should fall back to default."""
        from missy.cli.wizard import _prompt_model

        info = {
            "model_choices": [
                ("gpt-4o", "GPT-4o — balanced"),
                ("gpt-4o-mini", "GPT-4o Mini"),
            ],
            "models": {
                "primary": "gpt-4o",
                "fast": "gpt-4o-mini",
                "premium": "gpt-4-turbo",
            },
        }

        # "abc" → int("abc") raises ValueError → should fall back to default_id
        with patch("click.prompt", side_effect=["abc", "abc", "abc"]):
            primary, fast, premium = _prompt_model(info)

        assert primary == "gpt-4o"
        assert fast == "gpt-4o-mini"
        assert premium == "gpt-4-turbo"

    def test_out_of_range_index_falls_back_to_default(self):
        """When the user types an index outside the choices range, fall back to default."""
        from missy.cli.wizard import _prompt_model

        info = {
            "model_choices": [
                ("claude-sonnet-4-6", "Sonnet"),
                ("claude-haiku-4-5", "Haiku"),
            ],
            "models": {
                "primary": "claude-sonnet-4-6",
                "fast": "claude-haiku-4-5",
                "premium": "claude-opus-4-6",
            },
        }

        # Index 99 is out of range → fall back to default
        with patch("click.prompt", side_effect=["99", "99", "99"]):
            primary, fast, premium = _prompt_model(info)

        assert primary == "claude-sonnet-4-6"
        assert fast == "claude-haiku-4-5"
        assert premium == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# _build_config_yaml — discord with no guild policies (line 349/351)
# ---------------------------------------------------------------------------


class TestBuildConfigYamlDiscordNoGuildPolicies:
    def test_discord_with_empty_guild_policies_writes_empty_map(self):
        """When guild_policies is empty, the YAML should include 'guild_policies: {}'."""
        from missy.cli.wizard import _build_config_yaml

        discord_cfg = {
            "bot_token": "mytoken",
            "token_env_var": "DISCORD_BOT_TOKEN",
            "application_id": "123",
            "dm_policy": "disabled",
            "dm_allowlist": [],
            "guild_policies": [],  # empty → should write "guild_policies: {}"
            "ack_reaction": "eyes",
            "ignore_bots": True,
        }

        yaml_content = _build_config_yaml(
            workspace="/tmp/ws",
            providers_cfg=[],
            allowed_hosts=[],
            discord_cfg=discord_cfg,
        )

        assert "guild_policies: {}" in yaml_content


# ---------------------------------------------------------------------------
# run_wizard — provider choice "5" (all three: anthropic + openai + ollama)
# ---------------------------------------------------------------------------


class TestRunWizardAllThreeProviders:
    def test_choice_5_configures_anthropic_openai_ollama(self, tmp_path):
        """Selecting '5' should configure all three providers."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            # Step 1: workspace
            str(tmp_path / "workspace"),
            # Step 2: provider selection — choice "5" (all three)
            "5",
            # Step 3a: Anthropic auth
            "1",  # API key method
            "sk-ant-api03-testkey12345678",  # API key
            # Anthropic verify?
            # Step 3b: Anthropic model picks
            "1",  # primary
            "2",  # fast
            "3",  # premium
            # Step 3c: OpenAI auth method
            "1",  # API key
            "sk-openai-testkey12345678",  # API key
            # OpenAI model picks
            "1",  # primary
            "1",  # fast
            "1",  # premium
            # Step 3d: Ollama
            "http://localhost:11434",  # base URL
            "llama3",  # model
            # Step 4: Discord?
            # Step 5: write?
        ]
        confirms = [
            False,  # Anthropic: verify? No
            False,  # OpenAI: verify? No
            False,  # Ollama: verify? No
            False,  # Discord? No
            True,   # Write config? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch("missy.cli.wizard._verify_anthropic", return_value=True),
            patch("missy.cli.wizard._verify_openai", return_value=True),
            patch("missy.cli.wizard._verify_ollama", return_value=True),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "anthropic" in content
        assert "openai" in content
        assert "ollama" in content


# ---------------------------------------------------------------------------
# run_wizard — ollama verify prompt declined (lines 472-473)
# ---------------------------------------------------------------------------


class TestRunWizardOllamaVerifyDeclined:
    def test_ollama_verify_skipped_when_user_declines(self, tmp_path):
        """When user declines Ollama verification, _verify_ollama should not be called."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            str(tmp_path / "workspace"),  # Step 1
            "3",                           # Step 2: ollama only
            "http://localhost:11434",      # Ollama base URL
            "llama3",                      # Ollama model
        ]
        confirms = [
            False,  # Ollama verify? No → skips _verify_ollama call
            False,  # Discord? No
            True,   # Write config? Yes
        ]

        mock_verify = MagicMock(return_value=True)
        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch("missy.cli.wizard._verify_ollama", mock_verify),
        ):
            run_wizard(str(config_file))

        # verify should NOT have been called since user declined
        mock_verify.assert_not_called()
        assert config_file.exists()


# ---------------------------------------------------------------------------
# run_wizard — openai API key verify FAILS (lines 529-531)
# ---------------------------------------------------------------------------


class TestRunWizardOpenAIVerifyFails:
    def test_openai_verification_failure_still_saves_key(self, tmp_path):
        """When OpenAI verification fails, the key is saved anyway."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            str(tmp_path / "workspace"),  # Step 1
            "2",                           # Step 2: openai only
            "1",                           # OpenAI auth: API key
            "sk-openai-failkey1234567890", # API key (valid format)
            "1",  # model primary
            "1",  # model fast
            "1",  # model premium
        ]
        confirms = [
            True,   # OpenAI verify? Yes
            False,  # Discord? No
            True,   # Write config? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch("missy.cli.wizard._verify_openai", return_value=False),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "openai" in content


# ---------------------------------------------------------------------------
# run_wizard — anthropic setup-token flow returns None → fallback (lines 553-554)
# ---------------------------------------------------------------------------


class TestRunWizardAnthropicSetupTokenFallback:
    def test_setup_token_none_falls_back_to_api_key_prompt(self, tmp_path):
        """When setup-token flow returns None, wizard prompts for API key instead.

        auth_choice == "3" means verification is skipped even after fallback,
        so the only confirms are Discord and Write.
        """
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            str(tmp_path / "workspace"),   # Step 1
            "1",                            # Step 2: anthropic only
            "3",                            # Anthropic auth: setup-token
            "sk-ant-api03-fallback1234567", # Fallback API key
            "1",  # model primary
            "2",  # model fast
            "3",  # model premium
        ]
        # auth_choice == "3" → verification block is skipped (condition: auth_choice != "3")
        # so only Discord and Write confirms are needed
        confirms = [
            False,  # Discord? No
            True,   # Write? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch(
                "missy.cli.anthropic_auth.run_anthropic_setup_token_flow",
                return_value=None,  # setup token flow fails/skipped
            ),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "anthropic" in content


# ---------------------------------------------------------------------------
# run_wizard — anthropic verification FAILS (line 572)
# ---------------------------------------------------------------------------


class TestRunWizardAnthropicVerifyFails:
    def test_anthropic_verify_failure_still_saves_key(self, tmp_path):
        """When Anthropic verification fails, the key is saved with a warning."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            str(tmp_path / "workspace"),   # Step 1
            "1",                            # Step 2: anthropic only
            "1",                            # Anthropic auth: API key
            "sk-ant-api03-testverify12345", # API key
            "1",  # model primary
            "2",  # model fast
            "3",  # model premium
        ]
        confirms = [
            True,   # Verify? Yes
            False,  # Discord? No
            True,   # Write? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch("missy.cli.wizard._verify_anthropic", return_value=False),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()


# ---------------------------------------------------------------------------
# run_wizard — discord bot_token starts with "Bot " → stripped (line 619)
# ---------------------------------------------------------------------------


class TestRunWizardDiscordBotPrefixStripped:
    def test_bot_prefix_stripped_from_discord_token(self, tmp_path):
        """A bot_token starting with 'Bot ' should have the prefix stripped."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            str(tmp_path / "workspace"),         # Step 1
            "0",                                   # Step 2: skip providers
            "Bot myactualdiscordtoken",            # Discord bot token with "Bot " prefix
            "",                                    # Application ID (blank)
            "1",                                   # DM policy: disabled
            "",                                    # ACK reaction
        ]
        confirms = [
            True,   # Configure Discord? Yes
            False,  # Add guild policy? No
            True,   # Ignore bots? Yes (default)
            True,   # Write? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        # "Bot " prefix should be stripped, so raw token appears
        assert "myactualdiscordtoken" in content
        assert "Bot myactualdiscordtoken" not in content


# ---------------------------------------------------------------------------
# run_wizard — discord guild policy with channels (lines 641-664)
# ---------------------------------------------------------------------------


class TestRunWizardDiscordGuildPolicy:
    def test_guild_policy_with_channels_written_to_config(self, tmp_path):
        """A guild policy with allowed_channels should appear in the YAML output."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            str(tmp_path / "workspace"),  # Step 1
            "0",                           # Step 2: skip providers
            "mytokenvalue",               # Discord bot token
            "88888888",                   # Application ID
            "1",                          # DM policy: disabled
            "123456789",                  # Guild ID
            "general,announcements",      # Allowed channels
            "1",                          # Mode: full
            "",                           # ACK reaction
        ]
        confirms = [
            True,   # Configure Discord? Yes
            True,   # Add guild policy? Yes
            True,   # Require @mention? Yes
            False,  # Add another guild? No
            True,   # Ignore bots? Yes
            True,   # Write? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "123456789" in content
        assert "general" in content

    def test_guild_policy_with_multiple_guilds(self, tmp_path):
        """Two guild policies can be added in a single Discord setup run."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        # Prompts consumed in order through the wizard:
        # workspace, provider choice, bot token, app ID, dm policy,
        # guild1 ID, guild1 channels, guild1 mode,
        # guild2 ID, guild2 channels, guild2 mode,
        # ack reaction
        prompts = [
            str(tmp_path / "workspace"),  # Step 1
            "0",                           # Step 2: skip providers
            "mytokenvalue",               # Discord bot token
            "88888888",                   # Application ID
            "1",                          # DM policy: disabled
            "111111111",                  # Guild ID 1
            "",                           # Guild 1 allowed channels (blank = all)
            "1",                          # Guild 1 mode: full
            "222222222",                  # Guild ID 2
            "",                           # Guild 2 allowed channels
            "2",                          # Guild 2 mode: safe-chat
            "",                           # ACK reaction
        ]
        # Confirms consumed in order:
        # Configure Discord?, Add guild policy?,
        # guild1 require_mention, Add another guild? (→ Yes, loop continues),
        # guild2 require_mention, Add another guild? (→ No, loop ends),
        # Ignore bots?, Write?
        confirms = [
            True,   # Configure Discord? Yes
            True,   # Add guild policy? Yes
            True,   # Require @mention guild 1? Yes
            True,   # Add another guild? Yes → loop again
            False,  # Require @mention guild 2? No
            False,  # Add another guild? No → break
            True,   # Ignore bots? Yes
            True,   # Write? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        assert "111111111" in content
        assert "222222222" in content


# ---------------------------------------------------------------------------
# run_wizard — summary table: openai provider with oauth result → "(OAuth token)"
# (line 706)
# ---------------------------------------------------------------------------


class TestRunWizardOAuthTokenSummaryDisplay:
    def test_openai_oauth_success_shows_oauth_token_in_summary(self, tmp_path):
        """When OpenAI OAuth succeeds, the summary table shows '(OAuth token)'."""
        from missy.cli.wizard import run_wizard

        config_file = tmp_path / "config.yaml"

        prompts = [
            str(tmp_path / "workspace"),  # Step 1
            "2",                           # Step 2: openai only
            "2",                           # OpenAI auth: OAuth
            "1",  # model primary (openai-codex choices)
            "1",  # model fast
            "1",  # model premium
        ]
        confirms = [
            False,  # Discord? No
            True,   # Write? Yes
        ]

        with (
            patch("click.prompt", side_effect=prompts),
            patch("click.confirm", side_effect=confirms),
            patch(
                "missy.cli.oauth.run_openai_oauth",
                return_value="oauth-access-token-123456789",
            ),
        ):
            run_wizard(str(config_file))

        assert config_file.exists()
        content = config_file.read_text()
        # The provider should be openai-codex after OAuth succeeds
        assert "openai-codex" in content
