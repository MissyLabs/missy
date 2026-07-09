# TEST_EDGE_CASES

- Timestamp: 2026-07-09 15:25:53 EDT

Current edge-case focus:

- frequent-request detection without noisy false positives
- generated tool candidates remain disabled until approved
- enabled candidates still remain unloaded unless
  `tool_intelligence.candidate_runtime.enabled` is explicitly set
- candidate runtime loading rejects missing implementation metadata
- candidate runtime loading rejects invalid schemas, unsafe names, unknown
  permission keys, missing provenance, missing delegated targets, name
  conflicts, self-delegation, and provider-disabled candidates
- generated tool permissions are least-privilege
- tool metadata includes provenance, schema, tests, versions, ownership,
  lifecycle state, provider flags, and implementation binding metadata
- benchmark runs compare OpenAI, Ollama, Anthropic, and mock/local providers
  where available
- provider-specific enablement respects benchmark thresholds
- unsafe or flaky tools are flagged or disabled per provider
- tool schema incompatibilities are detected and reported
- benchmark failures do not enable or load tools accidentally
- runtime candidate loading does not execute generated code
- future overhaul compatibility for Discord, scheduling, and provider routing
- vision capture retry tests tolerate exhausted mocked monotonic clocks without
  leaking `StopIteration`
