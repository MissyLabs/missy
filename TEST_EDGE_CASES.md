# TEST_EDGE_CASES

- Timestamp: 2026-07-08 14:14:39 EDT

Current edge-case focus:

- OpenAI API keys are loaded securely and redacted everywhere
- OpenAI endpoint access goes through network policy
- native OpenAI Responses streams emit deltas without duplicating final/full
  text snapshots
- Responses stream failure/error events become deterministic `ProviderError`
  failures
- Chat Completions streaming fallback remains available for `base_url`
  OpenAI-compatible providers
- streaming handles partial chunks, repeated full content, and malformed deltas
- tool/function calling schemas normalize correctly for OpenAI
- structured output and JSON failures are handled safely
- retries, timeouts, rate limits, and cancellations are deterministic
- provider usage, token, and cost accounting is correct
- mocked OpenAI success and failure responses are covered
- provider diagnostics do not leak tokens or unsafe config
- future overhaul compatibility for Anthropic, Ollama, tools, Discord, and Web TUI
