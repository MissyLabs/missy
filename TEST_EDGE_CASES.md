# TEST_EDGE_CASES

- Timestamp: 2026-07-08 15:03:20 EDT

Current edge-case focus:

- OpenAI API keys are loaded securely and redacted everywhere.
- OpenAI diagnostics report credential source but never credential values.
- OpenAI diagnostics redact secret-bearing custom `base_url` components and
  expose only endpoint host/posture.
- OpenAI endpoint access goes through network policy; diagnostics inspect local
  allowlists without DNS or live API calls.
- Streaming handles partial chunks, repeated full content, and malformed deltas.
- Tool/function calling schemas normalize correctly for OpenAI.
- Structured output and JSON failures are handled safely.
- Retries, timeouts, rate limits, and cancellations are deterministic.
- Provider usage, token, and cost accounting is correct.
- Mocked OpenAI success and failure responses are covered.
- Future overhaul compatibility for Anthropic, Ollama, tools, Discord, and Web
  TUI remains preserved by provider-neutral hooks.
