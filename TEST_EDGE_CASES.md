# TEST_EDGE_CASES

- Timestamp: 2026-07-09 13:52:21 EDT

Current edge-case focus:

- frequent-request detection without noisy false positives
- generated tool candidates remain disabled until reviewed, benchmarked,
  approved, and enabled
- candidate lifecycle transitions reject skipped benchmark/approval gates
- disabled candidates cannot be resurrected in place
- deprecated candidates can be restored or disabled with auditability
- generated tool permissions are least-privilege
- tool metadata includes provenance, schema, tests, versions, and lifecycle
  state
- benchmark runs compare OpenAI, Ollama, Anthropic, and mock/local providers
  where available
- provider-specific enablement respects benchmark thresholds
- unsafe or flaky tools are flagged or disabled per provider
- tool schema incompatibilities are detected and reported
- benchmark failures do not enable tools accidentally
- future overhaul compatibility for Discord, scheduling, and provider routing
