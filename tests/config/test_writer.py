"""Tests for missy.config.writer: targeted, backed-up config.yaml field writes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from missy.config.writer import (
    ConfigWriteError,
    set_account_weights,
    set_agent_field,
    set_default_provider,
    set_provider_field,
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


class TestSetContextWorkerFields:
    def test_persists_context_worker_provider_and_model(self, tmp_path: Path):
        path = _write_config(tmp_path)

        assert (
            set_provider_field(str(path), "anthropic", "context_worker_provider", "ollama")
            == "ollama"
        )
        assert (
            set_provider_field(str(path), "anthropic", "context_worker_model", "qwen3:8b")
            == "qwen3:8b"
        )

        data = yaml.safe_load(path.read_text())
        assert data["providers"]["anthropic"]["context_worker_provider"] == "ollama"
        assert data["providers"]["anthropic"]["context_worker_model"] == "qwen3:8b"


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


class TestProviderControlCenterWrites:
    def test_persists_extended_provider_tuning(self, tmp_path: Path):
        path = _write_config(tmp_path)

        assert set_provider_field(str(path), "openai", "max_wait_seconds", "12.5") == 12.5
        assert set_provider_field(str(path), "openai", "circuit_breaker_threshold", "3") == 3
        assert (
            set_provider_field(str(path), "openai", "key_rotation_strategy", "round_robin")
            == "round_robin"
        )

        provider = yaml.safe_load(path.read_text())["providers"]["openai"]
        assert provider["max_wait_seconds"] == 12.5
        assert provider["circuit_breaker_threshold"] == 3
        assert provider["key_rotation_strategy"] == "round_robin"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("temperature", 1.2),
            ("max_iterations", 16),
            ("max_sub_agents", 12),
            ("max_concurrent_agents", 4),
            ("max_sub_agent_depth", 3),
            ("max_spend_usd", 5.5),
            ("global_budget_period", "daily"),
        ],
    )
    def test_persists_agent_orchestration_fields(self, tmp_path: Path, field: str, value):
        path = _write_config(tmp_path)

        assert set_agent_field(str(path), field, value) == value
        assert yaml.safe_load(path.read_text())[field] == value

    def test_rejects_concurrency_above_agent_limit(self, tmp_path: Path):
        path = _write_config(tmp_path)
        set_agent_field(str(path), "max_sub_agents", 4)

        with pytest.raises(ConfigWriteError, match="cannot exceed"):
            set_agent_field(str(path), "max_concurrent_agents", 5)

    def test_rejects_unallowlisted_agent_field(self, tmp_path: Path):
        path = _write_config(tmp_path)

        with pytest.raises(ConfigWriteError, match="not editable"):
            set_agent_field(str(path), "api_key", "secret")

    @pytest.mark.parametrize(
        ("field", "value"),
        [("temperature", "nan"), ("max_spend_usd", "inf")],
    )
    def test_rejects_non_finite_agent_values(self, tmp_path: Path, field: str, value: str):
        path = _write_config(tmp_path)

        with pytest.raises(ConfigWriteError):
            set_agent_field(str(path), field, value)
