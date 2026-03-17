# Providers

Missy supports multiple AI providers through a pluggable abstraction layer.
This document covers the provider model, configuration for each built-in
provider, API key management, and fallback behaviour.

---

## Provider Abstraction Model

All providers implement the `BaseProvider` abstract base class defined in
`missy/providers/base.py`.  The interface consists of two methods:

```python
class BaseProvider(ABC):
    name: str  # e.g. "anthropic", "openai", "ollama"

    @abstractmethod
    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        """Run a completion against the provider."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is ready to accept requests."""
        ...
```

### Message

A provider-agnostic conversation turn:

| Field | Type | Description |
|---|---|---|
| `role` | str | One of `"user"`, `"assistant"`, or `"system"` |
| `content` | str | The text content of the message |

### CompletionResponse

The canonical return type from any provider:

| Field | Type | Description |
|---|---|---|
| `content` | str | The generated text from the model |
| `model` | str | The exact model identifier used |
| `provider` | str | The provider name (e.g. `"anthropic"`) |
| `usage` | dict | Token counts: `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `raw` | dict | The raw deserialized response payload from the provider API |

---

## Built-in Providers

### Anthropic

**Implementation**: `missy/providers/anthropic_provider.py`

Uses the official `anthropic` Python SDK to call the Messages API.  The SDK is
imported lazily -- if the package is not installed, `is_available()` returns
`False` without raising.

**Default model**: `claude-sonnet-4-6`

**Required environment variable**: `ANTHROPIC_API_KEY`

**System message handling**: The Anthropic Messages API requires system
messages to be passed as a separate `system` parameter rather than in the
messages list.  The provider extracts any `"system"` role message and forwards
it correctly.

**Supported kwargs** (passed to `complete()`):

- `temperature` (float) -- sampling temperature
- `max_tokens` (int) -- maximum completion tokens (default: 4096)
- `model` (str) -- override the configured model

**Configuration**:

```yaml
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Network policy**: Add `api.anthropic.com` to `network.allowed_hosts`.

---

### OpenAI

**Implementation**: `missy/providers/openai_provider.py`

Uses the official `openai` Python SDK to call the Chat Completions API.  The
SDK is imported lazily.  The `base_url` parameter allows targeting any
OpenAI-compatible endpoint (Groq, Together AI, local vLLM, etc.).

**Default model**: `gpt-4o`

**Required environment variable**: `OPENAI_API_KEY`

**System message handling**: System, user, and assistant messages are
forwarded as-is to the Chat Completions API.

**Supported kwargs**:

- `temperature` (float) -- sampling temperature
- `max_tokens` (int) -- maximum completion tokens
- `model` (str) -- override the configured model

**Configuration**:

```yaml
providers:
  openai:
    name: openai
    model: "gpt-4o"
    timeout: 30
```

```bash
export OPENAI_API_KEY="sk-..."
```

**Network policy**: Add `api.openai.com` to `network.allowed_hosts`.

**Using with OpenAI-compatible services**: Set `base_url` to the alternative
endpoint and add that host to `network.allowed_hosts`:

```yaml
providers:
  groq:
    name: openai              # Use the OpenAI provider implementation
    model: "llama-3.1-70b"
    base_url: "https://api.groq.com/openai/v1"
    timeout: 30
```

---

### Ollama

**Implementation**: `missy/providers/ollama_provider.py`

Communicates with the Ollama local-inference server via its REST API.  Unlike
the Anthropic and OpenAI providers, Ollama does **not** use a third-party SDK.
Instead, it routes all HTTP traffic through `PolicyHTTPClient`
(`missy/gateway/client.py`), which means every request is subject to the active
network policy automatically.

**Default model**: `llama3.2`

**Default base URL**: `http://localhost:11434`

**Required environment variable**: None (Ollama does not require an API key).

**Availability check**: `is_available()` sends `GET /api/tags` to the
configured base URL.  Returns `True` if the server responds with HTTP 200.

**API endpoint**: `POST /api/chat` with `stream: false`.

**Supported kwargs**:

- `model` (str) -- override the configured model
- `temperature` (float) -- forwarded inside the `options` payload

**Configuration**:

```yaml
providers:
  ollama:
    name: ollama
    model: "llama3.2"
    base_url: "http://localhost:11434"
    timeout: 60
```

**Network policy**: Add `localhost:11434` (or the custom host) to
`network.allowed_hosts`.

**Local setup**:

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Ollama runs on `localhost:11434` by default.
4. Add `localhost:11434` to `network.allowed_hosts` in your config.

---

## Provider Registry

The `ProviderRegistry` (`missy/providers/registry.py`) is the single point of
truth for which provider instances are active.  It is initialised by
`init_registry(cfg)` during startup and accessed via `get_registry()`.

The registry is built from `config.providers` by the `from_config()` class
method:

1. For each key in `config.providers`, look up the `name` field.
2. Map the name to a known provider class (`anthropic`, `openai`, `ollama`).
3. Construct the provider instance with the `ProviderConfig`.
4. Register it under the config key.

Unknown provider names are skipped with a warning.  Provider construction
failures are logged and skipped.

---

## Per-provider Enable/Disable

Each provider config supports an `enabled` field (default: `true`).  When set
to `false`, the provider is still constructed and registered, but reports
itself as unavailable.  This allows temporarily disabling a provider without
removing its configuration:

```yaml
providers:
  openai:
    name: openai
    model: "gpt-4o"
    enabled: false        # Provider is loaded but will not be used
```

---

## Provider Fallback Behaviour

When the `AgentRuntime` resolves a provider, it follows this logic:

1. Look up the configured provider by name in the registry.
2. If found and `is_available()` returns `True`, use it.
3. If not available, log a warning and query `registry.get_available()`.
4. If any other provider is available, use the first one (fallback).
5. If no providers are available at all, raise `ProviderError`.

To disable fallback and ensure only a specific provider is used, configure
only that provider in your YAML file.

---

## API Key Resolution

For each provider, the API key is resolved at configuration parse time in
`_parse_providers()` (`missy/config/settings.py`):

1. Check the `api_key` field in the YAML provider block.
2. If not set, check the environment variable `<KEY>_API_KEY`, where `<KEY>`
   is the uppercased config mapping key (e.g. a provider keyed as `anthropic`
   checks `ANTHROPIC_API_KEY`).
3. If neither is set, `api_key` remains `None` and the provider's
   `is_available()` returns `False`.

**Important**: Never put API keys in the configuration file.  Use environment
variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

---

## Audit Events

Every provider completion emits a `provider_invoke` audit event with category
`"provider"` and result `"allow"` (on success) or `"error"` (on failure).  The
event detail includes the provider name, model, and a human-readable message.

These events can be reviewed with:

```bash
missy audit recent --category provider
```
