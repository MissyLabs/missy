# TEST_EDGE_CASES

- Timestamp: 2026-07-08 13:56:58 EDT

Current edge-case focus:

- Native OpenAI clients should use Responses for compatible plain text/vision
  requests.
- OpenAI-compatible `base_url` providers must remain on Chat Completions.
- Tool-result/tool-call transcripts must not be routed to Responses until
  replayable Responses output-item preservation exists.
- Responses `output_text` may be empty; structured output parts must be parsed.
- Responses usage counters must map to Missy's canonical token fields.
- OpenAI API keys are loaded securely and redacted everywhere.
- OpenAI endpoint access goes through network policy.
- Streaming handles partial chunks, repeated full content, malformed deltas,
  and future Responses events.
- Tool/function calling schemas normalize correctly for OpenAI.
- Structured output and JSON failures are handled safely.
- Retries, timeouts, rate limits, and cancellations are deterministic.
- Provider diagnostics do not leak tokens or unsafe config.
- Future overhaul compatibility for Anthropic, Ollama, tools, Discord, and Web TUI.
