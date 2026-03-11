# Agent Execution Loop

This document describes how `AgentRuntime.run()` works, step by step, from
the moment user input arrives until a response string is returned to the
caller.

**Source file:** `missy/agent/runtime.py`

---

## Overview

`AgentRuntime` is a synchronous, single-turn orchestrator. Each call to
`run(user_input, session_id=None)` performs one complete request/response
cycle through these stages:

1. Session resolution
2. Audit event emission (start)
3. Provider resolution (with fallback)
4. Message list construction
5. Provider completion call
6. Audit event emission (complete or error)
7. Return the content string

The runtime does **not** currently perform input sanitisation or secrets
detection inline -- those utilities exist in `missy.security.sanitizer` and
`missy.security.secrets` and are intended to be wired in by the channel or
CLI layer before calling `run()`. The runtime also does not interact with
`MemoryStore` directly; memory persistence is the caller's responsibility.

---

## Step-by-Step Walkthrough

### 1. Session Resolution (`_resolve_session`)

```python
session = self._resolve_session(session_id)
sid = str(session.id)
task_id = str(self._session_mgr.generate_task_id())
```

The runtime owns a `SessionManager` instance (one per `AgentRuntime`).
`_resolve_session` checks for an existing thread-local session via
`SessionManager.get_current_session()`. If one exists, it is reused. If
not, a new `Session` is created with `SessionManager.create_session()`.
When the caller supplies a `session_id` string, it is stored in the new
session's `metadata["caller_session_id"]` for audit correlation.

A fresh `task_id` (UUID) is generated for every `run()` invocation,
regardless of whether the session was new or reused.

### 2. Start Audit Event

```python
self._emit_event(
    session_id=sid,
    task_id=task_id,
    event_type="agent.run.start",
    result="allow",
    detail={"user_input_length": len(user_input)},
)
```

An `agent.run.start` event is published to the global `event_bus` with
category `"provider"`. The detail includes only the input length, never the
input text itself (to avoid leaking sensitive content into the audit trail).

### 3. Provider Resolution (`_get_provider`)

```python
provider = self._get_provider()
```

Resolution follows this logic:

1. Look up `self.config.provider` (default `"anthropic"`) in the
   `ProviderRegistry` singleton via `get_registry().get(name)`.
2. If the provider is found and `provider.is_available()` returns `True`,
   use it.
3. If the provider is found but **not** available, log a warning and
   fall through.
4. Call `registry.get_available()`, which iterates all registered providers
   and returns those where `is_available()` is `True`, in registration
   order.
5. If any are available, use the first one as a fallback.
6. If none are available, raise `ProviderError`.

On `ProviderError`, an `agent.run.error` event is emitted with
`detail.stage = "provider_resolution"` and the exception is re-raised.

### 4. Message Construction (`_build_messages`)

```python
messages = self._build_messages(user_input)
```

This constructs a simple two-element list:

```python
[
    Message(role="system", content=self.config.system_prompt),
    Message(role="user", content=user_input),
]
```

The system prompt defaults to `"You are Missy, a helpful local assistant."`.
It can be overridden via `AgentConfig.system_prompt`.

### 5. Provider Completion Call

```python
complete_kwargs = {
    "session_id": sid,
    "task_id": task_id,
    "temperature": self.config.temperature,
}
if self.config.model:
    complete_kwargs["model"] = self.config.model

response: CompletionResponse = provider.complete(messages, **complete_kwargs)
```

The provider-specific `complete()` method receives the message list plus
keyword arguments for session/task tracking and model parameters. Inside
the Anthropic provider, `session_id` and `task_id` are popped from kwargs
and used solely for audit events -- they are not sent to the API.

The provider is responsible for:
- Extracting system messages (Anthropic requires them as a separate parameter)
- Constructing the API call
- Emitting a `provider_invoke` audit event
- Translating errors into `ProviderError`

### 6. Completion Audit Event

On success:

```python
self._emit_event(
    event_type="agent.run.complete",
    result="allow",
    detail={
        "provider": response.provider,
        "model": response.model,
        "usage": response.usage,
    },
)
```

On error (either `ProviderError` or unexpected `Exception`):

```python
self._emit_event(
    event_type="agent.run.error",
    result="error",
    detail={"error": str(exc), "stage": "completion", "provider": provider.name},
)
```

Unexpected exceptions are wrapped in `ProviderError` before re-raising.

### 7. Return Path

```python
return response.content
```

The plain string content from `CompletionResponse.content` is returned to
the caller. The `CompletionResponse` object itself (which contains model
name, usage stats, and the raw provider payload) is not exposed.

---

## Input Sanitisation and Secrets Detection

While `AgentRuntime.run()` does not call these directly, the framework
provides two security modules intended to be applied by the calling layer
before invoking the agent:

**`missy.security.sanitizer.InputSanitizer`:**
- Truncates input beyond 10,000 characters
- Scans for 13 prompt-injection patterns (e.g. "ignore previous
  instructions", `<|im_start|>`, `[INST]`)
- Logs a warning on match but still returns the text (caller decides)

**`missy.security.secrets.SecretsDetector`:**
- Scans for 9 secret patterns (API keys, AWS keys, private keys, GitHub
  tokens, passwords, JWTs, Stripe keys, Slack tokens)
- `redact()` replaces matches with `[REDACTED]`
- `has_secrets()` provides a short-circuit boolean check

---

## Memory Store Interaction

`MemoryStore` (`missy/memory/store.py`) is not called by `AgentRuntime`
directly. It is a separate persistence layer that callers (CLI, channels)
use to:

- `add_turn(session_id, role, content, provider)` -- persist each
  user/assistant turn
- `get_session_turns(session_id, limit)` -- retrieve recent history to
  build multi-turn context
- `clear_session(session_id)` -- wipe a conversation

---

## ASCII Sequence Diagram

```
  Caller          AgentRuntime        SessionManager     ProviderRegistry     Provider       EventBus
    |                  |                   |                    |                 |               |
    | run(input, sid)  |                   |                    |                 |               |
    |----------------->|                   |                    |                 |               |
    |                  | get_current()     |                    |                 |               |
    |                  |------------------>|                    |                 |               |
    |                  |   Session|None    |                    |                 |               |
    |                  |<------------------|                    |                 |               |
    |                  |                   |                    |                 |               |
    |                  | [if None]         |                    |                 |               |
    |                  | create_session()  |                    |                 |               |
    |                  |------------------>|                    |                 |               |
    |                  |   Session         |                    |                 |               |
    |                  |<------------------|                    |                 |               |
    |                  |                   |                    |                 |               |
    |                  | publish(agent.run.start)               |                 |               |
    |                  |---------------------------------------------------------------->------->|
    |                  |                   |                    |                 |               |
    |                  | get(provider_name)|                    |                 |               |
    |                  |-------------------------------->----->|                  |               |
    |                  |   BaseProvider    |                    |                 |               |
    |                  |<--------------------------------<-----|                  |               |
    |                  |                   |                    |                 |               |
    |                  | is_available()?   |                    |                 |               |
    |                  |------------------------------------------------>------->|               |
    |                  |   True/False      |                    |                 |               |
    |                  |<------------------------------------------------<-------|               |
    |                  |                   |                    |                 |               |
    |                  | [if unavailable: get_available() -> fallback]            |               |
    |                  |                   |                    |                 |               |
    |                  | build messages    |                    |                 |               |
    |                  |---+               |                    |                 |               |
    |                  |   | [system, user]|                    |                 |               |
    |                  |<--+               |                    |                 |               |
    |                  |                   |                    |                 |               |
    |                  | complete(messages, **kwargs)           |                 |               |
    |                  |------------------------------------------------>------->|               |
    |                  |                   |                    |                 | API call      |
    |                  |                   |                    |                 |---+           |
    |                  |                   |                    |                 |   |           |
    |                  |                   |                    |                 |<--+           |
    |                  |                   |                    |                 |               |
    |                  |                   |                    |                 | publish(      |
    |                  |                   |                    |                 |  provider_    |
    |                  |                   |                    |                 |  invoke)      |
    |                  |                   |                    |                 |------>------->|
    |                  |                   |                    |                 |               |
    |                  |   CompletionResponse                   |                |               |
    |                  |<------------------------------------------------<-------|               |
    |                  |                   |                    |                 |               |
    |                  | publish(agent.run.complete)            |                 |               |
    |                  |---------------------------------------------------------------->------->|
    |                  |                   |                    |                 |               |
    |  response.content|                   |                    |                 |               |
    |<-----------------|                   |                    |                 |               |
```

---

## AgentConfig Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"anthropic"` | Registry key of the provider to use. |
| `model` | `Optional[str]` | `None` | Model override forwarded to the provider. When `None`, the provider's own default is used. |
| `system_prompt` | `str` | `"You are Missy, a helpful local assistant."` | System-level instruction prepended to every conversation. |
| `max_iterations` | `int` | `10` | Maximum provider calls per `run()` invocation (reserved for future multi-step agentic use). |
| `temperature` | `float` | `0.7` | Sampling temperature forwarded to the provider. |

---

## Error Handling Summary

| Stage | Exception | Audit Event Type | Re-raised? |
|-------|-----------|-----------------|------------|
| Provider resolution | `ProviderError` | `agent.run.error` | Yes |
| Completion | `ProviderError` | `agent.run.error` | Yes |
| Completion | Any other `Exception` | `agent.run.error` | Yes (wrapped in `ProviderError`) |
| Audit emission | Any `Exception` | (none -- logged) | No (swallowed) |
