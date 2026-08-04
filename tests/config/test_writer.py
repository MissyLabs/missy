"""Tests for missy.config.writer: targeted, backed-up config.yaml field writes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from missy.config.writer import (
    ConfigWriteError,
    set_account_weights,
    set_default_provider,
    set_provider_weight,
)

_BASE_CONFIG = """
providers:
  openai:
    name: openai
    model: gpt-5.5
  anthropic:
    name: anthropic
    model: claude-sonnet-4-6
"""


def _write_config(tmp_path: Path, content: str = _BASE_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestSetDefaultProvider:
    def test_persists_default_provider(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_default_provider(str(path), "openai")
        data = yaml.safe_load(path.read_text())
        assert data["default_provider"] == "openai"

    def test_preserves_existing_fields(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_default_provider(str(path), "anthropic")
        data = yaml.safe_load(path.read_text())
        assert data["providers"]["openai"]["model"] == "gpt-5.5"
        assert data["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"

    def test_creates_backup(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_default_provider(str(path), "openai")
        backup_dir = tmp_path / "config.d"
        assert backup_dir.exists()
        assert list(backup_dir.glob("config.yaml.*"))

    def test_no_op_when_already_set(self, tmp_path: Path):
        path = _write_config(tmp_path, _BASE_CONFIG + "\ndefault_provider: openai\n")
        set_default_provider(str(path), "openai")
        backup_dir = tmp_path / "config.d"
        # Nothing changed -- no backup should have been created.
        assert not backup_dir.exists()

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ConfigWriteError):
            set_default_provider(str(tmp_path / "nope.yaml"), "openai")

    def test_file_permissions_are_owner_only(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_default_provider(str(path), "openai")
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestSetProviderWeight:
    def test_persists_weight(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_provider_weight(str(path), "openai", 2.5)
        data = yaml.safe_load(path.read_text())
        assert data["providers"]["openai"]["weight"] == 2.5

    def test_unconfigured_provider_raises(self, tmp_path: Path):
        path = _write_config(tmp_path)
        with pytest.raises(ConfigWriteError, match="not configured"):
            set_provider_weight(str(path), "ollama", 1.0)

    def test_negative_weight_raises(self, tmp_path: Path):
        path = _write_config(tmp_path)
        with pytest.raises(ConfigWriteError, match="weight"):
            set_provider_weight(str(path), "openai", -1.0)

    def test_zero_weight_is_valid(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_provider_weight(str(path), "openai", 0.0)
        data = yaml.safe_load(path.read_text())
        assert data["providers"]["openai"]["weight"] == 0.0

    def test_no_op_when_already_set(self, tmp_path: Path):
        path = _write_config(
            tmp_path, _BASE_CONFIG.replace("model: gpt-5.5", "model: gpt-5.5\n    weight: 3.0")
        )
        set_provider_weight(str(path), "openai", 3.0)
        backup_dir = tmp_path / "config.d"
        assert not backup_dir.exists()


class TestSetAccountWeights:
    def test_persists_account_weights(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_account_weights(str(path), "openai", [3.0, 1.0])
        data = yaml.safe_load(path.read_text())
        assert data["providers"]["openai"]["account_weights"] == [3.0, 1.0]

    def test_unconfigured_provider_raises(self, tmp_path: Path):
        path = _write_config(tmp_path)
        with pytest.raises(ConfigWriteError, match="not configured"):
            set_account_weights(str(path), "ollama", [1.0])

    def test_non_positive_weight_raises(self, tmp_path: Path):
        path = _write_config(tmp_path)
        with pytest.raises(ConfigWriteError):
            set_account_weights(str(path), "openai", [0.0, 1.0])

    def test_empty_list_resets_to_equal_weighting(self, tmp_path: Path):
        path = _write_config(
            tmp_path,
            _BASE_CONFIG.replace(
                "model: gpt-5.5", "model: gpt-5.5\n    account_weights: [2.0, 1.0]"
            ),
        )
        set_account_weights(str(path), "openai", [])
        data = yaml.safe_load(path.read_text())
        assert data["providers"]["openai"]["account_weights"] == []
