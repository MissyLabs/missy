# TEST_EDGE_CASES

- Timestamp: 2026-07-09 14:47 EDT

Current edge-case focus:
- frequent-request detection without noisy false positives
- generated tool candidates remain disabled until approved
- generated tool permissions are least-privilege
- tool metadata includes provenance, schema, tests, versions, and lifecycle state
- benchmark runs compare OpenAI, Ollama, Anthropic, and mock/local providers where available
- provider-specific enablement respects benchmark thresholds
- unsafe or flaky tools are flagged or disabled per provider
- tool schema incompatibilities are detected and reported
- benchmark failures do not enable tools accidentally
- Web/API candidate controls require typed confirmations
- Web/API candidate denial requires an explicit review reason
- Web/API candidate actions cannot skip `CandidateStore` lifecycle gates
- future overhaul compatibility for Discord, scheduling, and provider routing
