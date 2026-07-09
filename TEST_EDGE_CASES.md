# TEST_EDGE_CASES

- Timestamp: 2026-07-09 14:12:59 EDT

Current edge-case focus:

- frequent-request detection without noisy false positives
- generated tool candidates remain disabled until approved
- generated tool permissions are least-privilege
- tool metadata includes provenance, schema, tests, versions, and lifecycle state
- benchmark runs compare OpenAI, Ollama, Anthropic, and mock/local providers where available
- provider-specific enablement respects benchmark thresholds
- benchmark-to-candidate reconciliation does not approve or enable tools
- benchmark import handles missing candidates and missing benchmark evidence cleanly
- insufficient benchmark samples do not mark a provider enabled
- low safety or schema-adherence scores keep provider flags disabled
- unsafe or flaky tools are flagged or disabled per provider
- tool schema incompatibilities are detected and reported
- benchmark failures do not enable tools accidentally
- future overhaul compatibility for Discord, scheduling, and provider routing
