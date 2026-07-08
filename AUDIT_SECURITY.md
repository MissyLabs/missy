# AUDIT_SECURITY

- Timestamp: 2026-07-08 14:34:10 EDT

## Expected common security and operations docs

- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security Notes From This Session

- No secret-loading behavior was changed.
- OpenAI API keys still come from provider config or `OPENAI_API_KEY`; the
  changed code does not log request payloads or credentials.
- OpenAI SDK construction still attempts to use Missy's policy-aware HTTP
  client, preserving the network-policy egress boundary.
- Structured-output schema payloads contain Pydantic JSON Schema metadata, not
  credentials or protected local configuration.
- The agent-layer structured-output runner remains provider-neutral; OpenAI
  request-shape specifics are isolated to `missy/providers/openai_provider.py`.

## Remaining Security Work

- Add OpenAI diagnostics/doctor commands that prove credential presence,
  redaction, model-list access, network-policy posture, and structured-output
  capability without leaking secrets.
- Extend provider audit events for retry, rate-limit cooldown, fallback,
  usage/cost recording, and provider-side validation denials.
- Add native Responses tool/function calling only after replayable Responses
  output items can be stored without confusing provider-neutral transcripts.
