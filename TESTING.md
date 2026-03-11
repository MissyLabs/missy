# Testing

Missy has a comprehensive test suite covering policy enforcement, provider
abstraction, configuration parsing, scheduling, plugins, and more.

---

## Test Suite Layout

```
tests/
  agent/              AgentRuntime tests
    test_runtime.py
  channels/           Channel abstraction tests
    test_cli_channel.py
  cli/                CLI command tests
    test_main.py
  config/             Configuration loading and validation tests
    test_settings.py
  core/               Session management tests
    test_session.py
  integration/        End-to-end policy enforcement tests
    test_policy_enforcement.py
  memory/             Memory store tests
    test_store.py
  observability/      Audit logger tests
    test_audit_logger.py
  plugins/            Plugin system tests
    test_base.py
    test_loader.py
  policy/             Policy engine tests
    test_engine.py
    test_filesystem.py
    test_network.py
    test_shell.py
  providers/          Provider abstraction tests
    test_anthropic.py
    test_base.py
    test_ollama.py
    test_openai.py
    test_registry.py
  scheduler/          Scheduler tests
    test_jobs.py
    test_manager.py
    test_parser.py
  security/           Input sanitization and secrets detection tests
    test_sanitizer.py
    test_secrets.py
  skills/             Skill system tests
    test_base.py
    test_registry.py
  tools/              Tool system tests
    test_calculator.py
    test_registry.py
  unit/               Unit tests (Discord, gateway)
    test_discord_channel.py
    test_discord_config.py
    test_gateway_client.py
```

---

## Running Tests

### Run all tests

```bash
python3 -m pytest tests/ -v
```

### Run a single test file

```bash
python3 -m pytest tests/policy/test_network.py -v
```

### Run tests by name pattern

```bash
pytest -k "test_check_host_allowed"
```

This runs any test whose name contains the given substring.  The `-k` flag
supports basic boolean expressions:

```bash
pytest -k "test_network and not test_deny"
```

### Run a specific test directory

```bash
python3 -m pytest tests/policy/ -v
python3 -m pytest tests/providers/ -v
```

---

## Code Coverage

### Generate a coverage report

```bash
pytest tests/ --cov=missy --cov-report=html
```

This produces an HTML report in `htmlcov/`.  Open `htmlcov/index.html` in a
browser to view line-by-line coverage.

### Terminal coverage summary

```bash
pytest tests/ --cov=missy --cov-report=term-missing
```

### Current coverage target

The project targets **85% code coverage** (currently at 86% with 740 tests).

---

## What the Key Test Categories Cover

### Policy Enforcement (`tests/policy/`)

- **test_engine.py** -- Tests the `PolicyEngine` facade: verifies that
  `check_network()`, `check_read()`, `check_write()`, and `check_shell()`
  delegate correctly and raise `PolicyViolationError` on denied operations.

- **test_network.py** -- Tests the `NetworkPolicyEngine`: host allowlist
  matching, CIDR range evaluation, domain suffix matching, default-deny
  behaviour, empty host rejection.

- **test_filesystem.py** -- Tests the `FilesystemPolicyEngine`: path prefix
  matching for read and write operations, tilde expansion, denial of paths
  outside allowed trees.

- **test_shell.py** -- Tests the `ShellPolicyEngine`: enabled/disabled state,
  command allowlist matching, denial of unlisted commands.

### Provider Abstraction (`tests/providers/`)

- **test_base.py** -- Tests the `Message`, `CompletionResponse`, and
  `BaseProvider` interface contracts.

- **test_anthropic.py** -- Tests the `AnthropicProvider`: system message
  extraction, API call construction, error handling (timeout, auth, API
  errors), availability checking, audit event emission.

- **test_openai.py** -- Tests the `OpenAIProvider`: message formatting, SDK
  client construction with `base_url` override, error handling.

- **test_ollama.py** -- Tests the `OllamaProvider`: HTTP request construction
  via `PolicyHTTPClient`, response parsing from Ollama's `/api/chat` format,
  availability checking via `/api/tags`.

- **test_registry.py** -- Tests the `ProviderRegistry`: `from_config()`
  construction, get/list/get_available queries, unknown provider handling.

### Configuration (`tests/config/`)

- **test_settings.py** -- Tests YAML parsing, default values, error handling
  for missing/invalid files, provider config parsing with API key resolution.

### Discord Access Control (`tests/unit/`, `tests/channels/`)

- **test_discord_config.py** -- Tests Discord configuration parsing: account
  config, guild policies, DM policy enum values, token resolution.

- **test_discord_channel.py** -- Tests Discord channel behaviour: message
  routing, guild policy enforcement, bot-ignore logic, mention requirements.

### Integration Tests (`tests/integration/`)

- **test_policy_enforcement.py** -- End-to-end tests that verify the full
  pipeline enforces policy: network requests through `PolicyHTTPClient` are
  denied for unlisted hosts, filesystem access outside allowed paths is
  rejected, shell commands not on the allowlist are blocked.

### Other Categories

- **Scheduler** (`tests/scheduler/`) -- Schedule string parsing, job
  serialisation/deserialisation, manager lifecycle.
- **Plugins** (`tests/plugins/`) -- Plugin loading with policy gates,
  execution with audit events, denial of unlisted plugins.
- **Skills** (`tests/skills/`) -- Skill registration, execution, error
  handling, audit event emission.
- **Memory** (`tests/memory/`) -- Turn persistence, session queries, file
  I/O edge cases.
- **Security** (`tests/security/`) -- Input sanitization patterns, prompt
  injection detection, secrets detection and redaction.
- **CLI** (`tests/cli/`) -- Click command invocation, output formatting,
  error rendering.

---

## How to Write New Tests

### Conventions

1. Place test files in the appropriate subdirectory under `tests/`.
2. Name test files with the `test_` prefix (e.g. `test_my_feature.py`).
3. Name test functions with the `test_` prefix.
4. Use `pytest` fixtures for shared setup.
5. Use `unittest.mock` for mocking external dependencies (API calls, file
   I/O, etc.).

### Example Test

```python
"""Tests for the schedule parser."""

import pytest
from missy.scheduler.parser import parse_schedule


def test_parse_interval_minutes():
    result = parse_schedule("every 5 minutes")
    assert result == {"trigger": "interval", "minutes": 5}


def test_parse_daily():
    result = parse_schedule("daily at 09:00")
    assert result == {"trigger": "cron", "hour": 9, "minute": 0}


def test_parse_invalid_raises():
    with pytest.raises(ValueError, match="Unrecognised schedule string"):
        parse_schedule("sometime next week")
```

### Testing Policy Enforcement

When writing tests that involve the policy engine, initialise it with a
known configuration:

```python
from missy.config.settings import MissyConfig, NetworkPolicy, get_default_config
from missy.policy.engine import init_policy_engine

def test_network_deny():
    cfg = get_default_config()  # Secure defaults: deny everything
    init_policy_engine(cfg)

    from missy.policy.engine import get_policy_engine
    engine = get_policy_engine()

    with pytest.raises(PolicyViolationError):
        engine.check_network("evil.example.com")
```

### Mocking Providers

Provider tests should mock the SDK to avoid real API calls:

```python
from unittest.mock import MagicMock, patch
from missy.providers.anthropic_provider import AnthropicProvider
from missy.config.settings import ProviderConfig

def test_anthropic_complete():
    config = ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022",
                            api_key="test-key")
    provider = AnthropicProvider(config)

    with patch("missy.providers.anthropic_provider._anthropic_sdk") as mock_sdk:
        mock_client = MagicMock()
        mock_sdk.Anthropic.return_value = mock_client
        # ... set up mock response ...
        result = provider.complete([Message(role="user", content="Hello")])
        assert result.provider == "anthropic"
```

### Cleaning Up Singletons

Many subsystems use module-level singletons.  Tests that initialise these
should either:

1. Re-initialise at the start of each test (the `init_*` functions replace
   the existing singleton).
2. Use `autouse` fixtures to reset state between tests.
3. Clear the event bus with `event_bus.clear()` to prevent cross-test
   pollution of audit events.
