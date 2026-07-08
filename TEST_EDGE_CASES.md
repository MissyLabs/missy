# TEST_EDGE_CASES

- Timestamp: 2026-07-08 13:29:18 EDT

Current OpenAI provider edge-case focus:

- OpenAI user content lists preserve safe text and image blocks.
- Unsafe OpenAI image URL schemes are stripped before provider invocation.
- Assistant tool calls with missing IDs or names are removed.
- Duplicate tool-call IDs are removed before request submission.
- Tool-result messages without a pending assistant tool call are dropped.
- Transcript repair emits structured audit events with correlation IDs.
- Malformed provider responses still produce safe `CompletionResponse` values
  or `ProviderError` failures.
- Streaming must reconcile delta chunks, full content, tool-call deltas,
  partial streams, and provider-side validation errors.
- Responses API migration must not leak OpenAI-only message assumptions into
  Anthropic, Ollama, Codex, or ACPX providers.
