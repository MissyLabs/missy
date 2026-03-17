"""Tests for non-interactive wizard setup (Feature 2)."""

from __future__ import annotations

import click
import pytest
import yaml


class TestNonInteractiveWizard:
    """Tests for run_wizard_noninteractive."""

    def test_noninteractive_creates_valid_config(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        config_file = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(config_file),
            provider="anthropic",
            api_key="sk-ant-test-key-12345",
            workspace=str(tmp_path / "workspace"),
        )

        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text())
        assert "providers" in data
        assert "anthropic" in data["providers"]
        assert data["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"

    def test_noninteractive_requires_provider(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        config_file = tmp_path / ".missy" / "config.yaml"
        with pytest.raises(click.ClickException, match="Unknown provider"):
            run_wizard_noninteractive(
                config_path=str(config_file),
                provider="nonexistent",
            )

    def test_noninteractive_api_key_env(self, tmp_path, monkeypatch):
        from missy.cli.wizard import run_wizard_noninteractive

        monkeypatch.setenv("TEST_API_KEY", "sk-ant-from-env-67890")
        config_file = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(config_file),
            provider="anthropic",
            api_key_env="TEST_API_KEY",
            workspace=str(tmp_path / "workspace"),
        )

        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text())
        assert data["providers"]["anthropic"]["api_key"] == "sk-ant-from-env-67890"

    def test_noninteractive_api_key_env_missing(self, tmp_path, monkeypatch):
        from missy.cli.wizard import run_wizard_noninteractive

        monkeypatch.delenv("MISSING_KEY", raising=False)
        config_file = tmp_path / ".missy" / "config.yaml"
        with pytest.raises(click.ClickException, match="not set or empty"):
            run_wizard_noninteractive(
                config_path=str(config_file),
                provider="anthropic",
                api_key_env="MISSING_KEY",
            )

    def test_noninteractive_default_model(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        config_file = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(config_file),
            provider="anthropic",
            api_key="sk-ant-test-12345",
            workspace=str(tmp_path / "workspace"),
        )

        data = yaml.safe_load(config_file.read_text())
        # Should default to anthropic's primary model
        assert data["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"

    def test_noninteractive_custom_model(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        config_file = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(config_file),
            provider="anthropic",
            api_key="sk-ant-test-12345",
            model="claude-opus-4-6",
            workspace=str(tmp_path / "workspace"),
        )

        data = yaml.safe_load(config_file.read_text())
        assert data["providers"]["anthropic"]["model"] == "claude-opus-4-6"

    def test_noninteractive_ollama_no_key(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        config_file = tmp_path / ".missy" / "config.yaml"
        run_wizard_noninteractive(
            config_path=str(config_file),
            provider="ollama",
            model="llama3",
            workspace=str(tmp_path / "workspace"),
        )

        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text())
        assert data["providers"]["ollama"]["model"] == "llama3"
