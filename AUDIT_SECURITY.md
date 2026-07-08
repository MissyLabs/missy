# AUDIT_SECURITY

- Timestamp: 2026-07-08 13:56:58 EDT

## Expected common security and operations docs

- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md
- present: docs/providers.md

## OpenAI Provider Security Notes

- Credentials remain loaded from provider config or `OPENAI_API_KEY`; the new
  Responses path does not log API keys or request headers.
- The Responses path reuses the existing SDK client construction, including
  policy-aware HTTP client wiring where available.
- `base_url` providers are deliberately kept on Chat Completions compatibility
  so Missy does not assume third-party support for native OpenAI Responses
  semantics.
- Tool-call transcripts are not routed to Responses yet, avoiding unsafe loss
  of required provider output items during replay.
- Vision inputs still pass through prior safe-image filtering before either
  Responses or Chat request construction.
- Provider invoke/error events remain in place; retry, fallback, and detailed
  cost audit events are still pending.

## Residual Risks

- Native Responses tool/function calling needs a dedicated transcript model
  before enabling.
- Responses streaming needs event-level parsing and malformed-event handling.
- OpenAI-native structured output needs schema validation and fallback tests.
