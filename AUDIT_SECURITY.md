# AUDIT_SECURITY

- Timestamp: 2026-07-08 14:14:39 EDT

## Expected common security and operations docs

- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and OpenAI provider scan

- OpenAI credentials still come from provider config or `OPENAI_API_KEY`; this
  session did not add any logging or persistence of API keys.
- Native Responses streaming reuses the existing SDK client construction, so
  provider egress remains routed through the policy-aware HTTP client when it
  can be built.
- Responses stream failed/error events are converted into `ProviderError`
  instead of being silently ignored.
- Unsafe image URL filtering and transcript repair audit events remain in the
  OpenAI adapter boundary.
- Existing unrelated `LOOP_INSTRUCTIONS.md` working-tree modification remains
  unmodified.

## Remaining Security Work

- Add explicit retry/rate-limit/fallback/cost audit events.
- Add OpenAI doctor probes for credential presence, exact endpoint policy,
  model-list access, and redaction.
- Add native structured-output validation denial events when OpenAI rejects or
  cannot satisfy the requested schema.
