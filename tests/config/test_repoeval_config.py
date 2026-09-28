"""RepoEval Foundry configuration is strictly opt-in and syntactically scoped."""

from __future__ import annotations

import pytest

from missy.config.settings import RepoevalFoundryConfig, get_default_config, load_config
from missy.core.exceptions import ConfigurationError
from missy.tools.builtin import register_builtin_tools
from missy.tools.registry import ToolRegistry


def _load(tmp_path, section: str):
    path = tmp_path / "config.yaml"
    path.write_text("providers: {}\nrepoeval_foundry:\n" + section, encoding="utf-8")
    return load_config(str(path)).repoeval_foundry


def _registered(config: RepoevalFoundryConfig) -> bool:
    registry = ToolRegistry()
    register_builtin_tools(registry, repoeval_foundry_config=config)
    return registry.get("repoeval_foundry") is not None


def test_default_disabled_and_no_endpoint(tmp_path):
    default = get_default_config().repoeval_foundry
    assert default == RepoevalFoundryConfig()
    assert not _registered(default)
    assert not _registered(_load(tmp_path, "  enabled: false\n"))


def test_operator_assertion_and_incomplete_opt_in_are_configurable(tmp_path):
    config = _load(
        tmp_path,
        "  enabled: true\n"
        "  base_url: https://foundry.example.test\n"
        "  project_id: pinned-project\n"
        "  allowed_hosts: [foundry.example.test]\n",
    )
    assert config.enabled is True
    assert config.api_available is False
    assert not _registered(config)


def test_explicit_authenticated_config_is_parsed_without_filesystem_access(tmp_path):
    token_file = tmp_path / "foundry-token"
    token_file.write_text("temporary-test-token\n", encoding="ascii")
    token_file.chmod(0o600)
    config = _load(
        tmp_path,
        "  enabled: true\n"
        "  api_available: true\n"
        "  base_url: https://foundry.example.test:443\n"
        "  project_id: pinned-project\n"
        "  allowed_hosts: [foundry.example.test]\n"
        f"  token_file: {token_file}\n",
    )
    assert config.enabled is True
    assert config.api_available is True
    assert config.token_file == str(token_file)
    assert _registered(config)
    token_file.chmod(0o644)
    assert not _registered(config)
    token_file.unlink()
    assert not _registered(config)


@pytest.mark.parametrize(
    "section",
    [
        "  enabled: true\n  api_available: true\n",
        "  enabled: true\n  api_available: true\n  token_file: /run/secrets/foundry\n",
        "  enabled: true\n  api_available: true\n  base_url: https://foundry.example.test\n  project_id: project\n  allowed_hosts: [other.test]\n",
        "  enabled: true\n  api_available: true\n  base_url: https://user:pass@foundry.example.test\n  project_id: project\n  allowed_hosts: [foundry.example.test]\n",
        "  enabled: true\n  api_available: true\n  base_url: http://foundry.example.test\n  project_id: project\n  allowed_hosts: [foundry.example.test]\n",
        "  enabled: true\n  api_available: true\n  base_url: https://foundry.example.test:0\n  project_id: project\n  allowed_hosts: [foundry.example.test]\n  token_file: /run/secrets/foundry\n",
        "  enabled: true\n  api_available: true\n  base_url: https://bad_host.test\n  project_id: project\n  allowed_hosts: [bad_host.test]\n  token_file: /run/secrets/foundry\n",
        "  enabled: true\n  api_available: true\n  base_url: https://foundry.example.test/v1\n  project_id: project\n  allowed_hosts: [foundry.example.test]\n  token_file: /run/secrets/foundry\n",
        "  enabled: true\n  api_available: true\n  base_url: https://foundry.example.test\\@evil.test\n  project_id: project\n  allowed_hosts: [foundry.example.test]\n  token_file: /run/secrets/foundry\n",
        "  enabled: 'maybe'\n",
        "  enabled: 1\n",
        "  api_available: 'false'\n",
        "  enabld: true\n",
        "  enabled: true\n  api_available: true\n  base_url: https://foundry.example.test\n  project_id: bad/project\n  allowed_hosts: [foundry.example.test]\n",
        "  token_file: relative/token\n",
        "  token_file: ~/secret\n",
        "  token_file: /run//secret\n",
        "  token_file: /run/../secret\n",
        "  token_file: /run/./secret\n",
        "  token_file: //run/secret\n",
        "  token_file: '/run/se\\cret'\n",
    ],
)
def test_invalid_opt_in_rejected(tmp_path, section):
    with pytest.raises(ConfigurationError):
        _load(tmp_path, section)
