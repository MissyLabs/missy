# Provider Abstraction Layer

The provider layer abstracts AI model access behind a common interface so
that the agent runtime can switch providers without changing any calling
code. The layer consists of base types, concrete implementations, and a
singleton registry.

**Source files:**

- `missy/providers/base.py` -- abstract base and interchange types
- `missy/providers/anthropic_provider.py` -- Anthropic Claude
- `missy/providers/openai_provider.py` -- OpenAI
- `missy/providers/ollama_provider.py` -- Ollama (local)
- `missy/providers/registry.py` -- singleton registry and factory

---

## BaseProvider Interface

```python
class BaseProvider(ABC):
    name: str  # e.g. "anthropic", "openai", "ollama"

    @abstractmethod
    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        """Run a completion against the provider.

        Args:
            messages: Ordered list of conversation turns.
            **kwargs: Provider-specific overrides:
                - session_id (str): Session identifier (popped, used for audit only)
                - task_id (str): Task identifier (popped, used for audit only)
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum completion tokens
                - model (str): Model override

        Returns:
            CompletionResponse with the model's reply.

        Raises:
            ProviderError: On any provider-side failure.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can service requests.

        Implementations check that required credentials are present
        and (optionally) that the upstream service is reachable.
        """
        ...
```

---

## Message Type

```python
@dataclass
class Message:
    role: str      # "user", "assistant", or "system"
    content: str   # The text content of the message
```

`Message` is the canonical interchange format for conversation turns. All
provider implementations accept and produce `Message` objects. The `role`
field uses the OpenAI convention (`"system"`, `"user"`, `"assistant"`);
providers that use a different wire format (e.g. Anthropic's separate
`system` parameter) translate internally.

---

## CompletionResponse Type

```python
@dataclass
class CompletionResponse:
    content: str     # Generated text from the model
    model: str       # Exact model identifier used
    provider: str    # Provider name (e.g. "anthropic")
    usage: dict      # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    raw: dict        # Raw deserialized API response
```

`CompletionResponse` standardises the response shape regardless of which
provider produced it. The `raw` field preserves the full provider-specific
payload for debugging or accessing provider-specific fields.

---

## Anthropic Provider Implementation

**Class:** `AnthropicProvider` in `missy/providers/anthropic_provider.py`

### Construction

```python
provider = AnthropicProvider(ProviderConfig(
    name="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
    timeout=30,
))
```

The `anthropic` SDK is imported lazily at module load time. If the import
fails, the module sets `_ANTHROPIC_AVAILABLE = False` and `is_available()`
returns `False`.

### `is_available()`

Returns `True` when both conditions are met:
1. The `anthropic` SDK is importable
2. An API key is present (either from `ProviderConfig.api_key` or the
   `ANTHROPIC_API_KEY` environment variable)

### `complete(messages, **kwargs)`

1. Pops `session_id`, `task_id`, and `model` from kwargs
2. Separates system messages from the message list (Anthropic's API
   requires the system prompt as a separate `system` parameter)
3. Constructs the API call with `max_tokens` (default 4096), optional
   `temperature`, and any remaining kwargs
4. Creates an `anthropic.Anthropic` client with the configured API key
   and timeout
5. Calls `client.messages.create(**call_kwargs)`
6. On success, extracts `content[0].text`, builds the usage dict from
   `input_tokens` and `output_tokens`, and emits a `provider_invoke`
   audit event with result `"allow"`
7. On failure, catches `APITimeoutError`, `AuthenticationError`,
   `APIError`, and generic `Exception`, emits an error audit event, and
   raises `ProviderError`

### Audit Events

```json
{
  "event_type": "provider_invoke",
  "category": "provider",
  "result": "allow" | "error",
  "detail": {
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "message": "completion successful"
  }
}
```

---

## ProviderRegistry

**Class:** `ProviderRegistry` in `missy/providers/registry.py`

### Registration and Lookup

```python
registry = ProviderRegistry()
registry.register("anthropic", provider_instance)

provider = registry.get("anthropic")       # -> BaseProvider | None
names = registry.list_providers()           # -> sorted list of names
available = registry.get_available()        # -> list of available providers
```

### `get_available()`

Iterates all registered providers and returns those where `is_available()`
returns `True`, in insertion order. Exceptions from `is_available()` are
caught, logged, and the provider is treated as unavailable.

### `from_config(config)` Factory

```python
registry = ProviderRegistry.from_config(config)
```

Iterates `config.providers` (a `dict[str, ProviderConfig]`) and for each
entry:

1. Checks if `provider_config.enabled` is `True` (skips disabled)
2. Resolves the provider name to a concrete class via the `_PROVIDER_CLASSES`
   mapping:
   - `"anthropic"` -> `AnthropicProvider`
   - `"openai"` -> `OpenAIProvider`
   - `"ollama"` -> `OllamaProvider`
3. Constructs the provider instance with `provider_cls(provider_config)`
4. Registers it under the config key name
5. Unknown provider names are logged and skipped
6. Construction failures are logged and skipped

### Singleton Management

```python
from missy.providers.registry import init_registry, get_registry

init_registry(config)          # Build from config and install
registry = get_registry()      # Retrieve (thread-safe, raises RuntimeError if not init'd)
```

Both functions use a `threading.Lock` for atomic replacement/retrieval.

---

## Provider Fallback in AgentRuntime

`AgentRuntime._get_provider()` implements the fallback logic:

```
1. Look up configured provider name in registry
2. If found AND is_available() -> use it
3. If found but NOT available -> log warning, fall through
4. Call registry.get_available() -> list of available providers
5. If non-empty -> use first available as fallback
6. If empty -> raise ProviderError("No providers available...")
```

This means the system degrades gracefully: if the preferred provider's API
key is missing or the service is down, the agent automatically falls back
to the next available provider.

---

## How Providers Use PolicyHTTPClient

Currently, providers do **not** use `PolicyHTTPClient` directly. The
Anthropic provider creates its own `anthropic.Anthropic` client internally.
OpenAI and Ollama providers follow the same pattern with their respective
SDKs.

Network policy enforcement for provider API endpoints is achieved by
requiring the operator to add the provider's API hostname to the network
policy allow-lists. For example:

```yaml
network:
  default_deny: true
  allowed_domains:
    - "api.anthropic.com"
    - "api.openai.com"
  provider_allowed_hosts:
    - "localhost:11434"   # Ollama
```

---

## ProviderConfig Reference

```python
@dataclass
class ProviderConfig:
    name: str                    # e.g. "anthropic"
    model: str                   # e.g. "claude-3-5-sonnet-20241022"
    api_key: Optional[str]       # Falls back to {NAME}_API_KEY env var
    base_url: Optional[str]      # Override for self-hosted endpoints
    timeout: int = 30            # Request timeout in seconds
    enabled: bool = True         # Set False to skip during registry build
```

API keys are resolved at config-load time from either the YAML field or the
`{PROVIDER_KEY}_API_KEY` environment variable (e.g. `ANTHROPIC_API_KEY`).

---

## Adding a New Provider

1. **Create the module** at `missy/providers/my_provider.py`:

   ```python
   from missy.providers.base import BaseProvider, Message, CompletionResponse
   from missy.config.settings import ProviderConfig
   from missy.core.exceptions import ProviderError

   class MyProvider(BaseProvider):
       name = "myprovider"

       def __init__(self, config: ProviderConfig) -> None:
           self._api_key = config.api_key
           self._model = config.model
           self._timeout = config.timeout

       def is_available(self) -> bool:
           return bool(self._api_key)

       def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
           session_id = kwargs.pop("session_id", "")
           task_id = kwargs.pop("task_id", "")
           # ... call your API ...
           return CompletionResponse(
               content=...,
               model=self._model,
               provider=self.name,
               usage={...},
               raw={...},
           )
   ```

2. **Register the class** in `missy/providers/registry.py`:

   ```python
   from .my_provider import MyProvider

   _PROVIDER_CLASSES["myprovider"] = MyProvider
   ```

3. **Add config** in the YAML:

   ```yaml
   providers:
     myprovider:
       name: myprovider
       model: my-model-v1
       api_key: null  # or set MYPROVIDER_API_KEY env var
       timeout: 30
   ```

4. **Emit audit events** from your `complete()` method following the
   pattern in `AnthropicProvider._emit_event()`.
