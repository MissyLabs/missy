# TEST_EDGE_CASES

- Timestamp: 2026-07-08 15:10:45

Current edge-case focus:
- OpenAI API keys are loaded securely and redacted everywhere
- OpenAI endpoint access goes through network policy
- streaming handles partial chunks, repeated full content, and malformed deltas
- tool/function calling schemas normalize correctly for OpenAI
- structured output and JSON failures are handled safely
- retries, timeouts, rate limits, and cancellations are deterministic
- provider usage, token, and cost accounting is correct
- mocked OpenAI success and failure responses are covered
- provider diagnostics do not leak tokens or unsafe config
- future overhaul compatibility for Anthropic, Ollama, tools, Discord, and Web TUI
