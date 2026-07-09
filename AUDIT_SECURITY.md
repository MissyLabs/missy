# AUDIT_SECURITY

- Timestamp: 2026-07-09 13:52:21 EDT

## Security Posture For This Session

- Store-level lifecycle enforcement now protects every caller of
  `CandidateStore.transition()`, including CLI, runtime automation, and
  future Web/API controls.
- Candidate enablement cannot jump directly from `proposed` to `enabled`.
- Candidate approval cannot jump directly from `proposed` to `approved`;
  benchmark evidence must first move the record to `benchmarked`.
- Disabled candidates are terminal and cannot be resurrected in place. A new
  candidate/version should be created instead, preserving denial history.
- Invalid lifecycle requests emit `tool.candidate.transition_denied` audit
  events with `result="deny"` and include previous/requested states, actor,
  and notes.
- Existing generated-tool safety remains in place: automatic generation is
  opt-in, generated candidates start as `proposed`, shell permission remains
  denied unless explicitly configured, and runtime tool execution policy is
  still independent from candidate storage.

## Reviewed Threats

- Prompt injection attempting to force tool creation: still gated by
  `tool_intelligence.candidate_generation.enabled`.
- Malicious generated tool requesting immediate enablement: rejected by the
  candidate lifecycle gate.
- Benchmark bypass: direct approval from proposed state is rejected.
- Unsafe rollback/re-enable: deprecated candidates can be restored, but
  disabled candidates cannot.
- Audit evasion: denied transitions are audited before raising.
- Provider-specific unsafe enablement: unchanged; `ToolProviderGate` and
  operator overrides remain separate and audited.

## Residual Risks

- Benchmark results are not yet automatically reconciled into candidates, so
  operators still need manual or future CLI/API glue to move candidates to
  `benchmarked`.
- Enabled candidates still need a controlled runtime loading path with schema,
  provenance, permissions, and rollback checks.
- Web/API candidate lifecycle controls are not yet implemented.
