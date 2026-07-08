# TEST_EDGE_CASES

- Timestamp: 2026-07-08 14:34:10 EDT

Current edge-case focus:

- OpenAI API keys are loaded securely and redacted everywhere.
- OpenAI endpoint access goes through network policy.
- Streaming handles partial chunks, repeated full content, and malformed
  deltas.
- Tool/function calling schemas normalize correctly for OpenAI.
- Structured output and JSON failures are handled safely.
- OpenAI-native structured output maps to Responses `text.format` when using
  the native Responses path.
- OpenAI-native structured output maps to Chat Completions `response_format`
  when using `base_url` or compatibility paths.
- The structured-output runner remains provider-neutral and only passes native
  schema kwargs through providers that explicitly implement the hook.
- Retries, timeouts, rate limits, and cancellations are deterministic.
- Provider usage, token, and cost accounting is correct.
- Mocked OpenAI success and failure responses are covered.
- Provider diagnostics do not leak tokens or unsafe config.
- Future overhaul compatibility for Anthropic, Ollama, tools, Discord, and Web
  TUI remains preserved.
