# Build Status

Last updated: 2026-07-08 14:34:10 EDT

## Current State

Primary focus remains the OpenAI provider overhaul. The provider layer now has
native OpenAI Responses execution for compatible plain text, vision, streaming,
and structured-output requests. Chat Completions remains the compatibility path
for OpenAI-compatible `base_url` providers and transcripts containing tool
results or assistant tool-call history.

This session added OpenAI-native structured output support:

- Added `BaseProvider.structured_output_kwargs(schema)` as a provider-neutral
  optional hook for native schema enforcement.
- Updated `StructuredOutputRunner` to request native provider kwargs while
  preserving Missy's generic Pydantic prompt, validation, and retry loop.
- Implemented OpenAI JSON Schema request formatting for both native Responses
  (`text.format`) and Chat Completions compatibility (`response_format`).
- Sanitized OpenAI schema names to API-safe values and preserved strict mode
  and schema descriptions.
- Added focused tests for the runner hook and OpenAI Responses/Chat structured
  output request shapes.
- Updated provider documentation and provider-abstraction implementation docs.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Provider interface compliance | improved | OpenAI still returns canonical `CompletionResponse`; provider-neutral structured-output hook added to `BaseProvider`. |
| Secure credential loading | in place | API key comes from config/env; changed path does not log secrets. |
| Network policy integration | in place | SDK client is built with policy-aware HTTP where available. |
| Model selection | in place | `auto` resolves through model listing with preferred current chat models and fallback. |
| Responses API path | improved | Native OpenAI text/vision completions, compatible streams, and structured outputs can use Responses. |
| Chat compatibility | preserved | `base_url` providers, tool transcripts, and Chat structured outputs continue through Chat Completions. |
| Streaming reconciliation | improved | Responses stream deltas and final/full-content snapshots are reconciled. |
| Tool schema normalization | in place | OpenAI tool schemas delegate to `schema_adapter.normalize_for_provider()`. |
| Tool transcript repair | improved | Invalid/duplicate assistant tool calls and orphan tool results are dropped before request. |
| Vision input support | improved | Safe image blocks are preserved and converted for Responses when eligible. |
| Structured output | improved | OpenAI-native JSON Schema enforcement is requested where available; generic Pydantic validation remains the final contract. |
| Auditability | partial | Provider invoke/error and transcript repair events exist; retry/fallback/cost events remain. |
| Tests | improved | Focused provider/agent suites and full repository suite pass. |

## Current Architecture State

- OpenAI-specific message normalization, transcript repair, Responses routing,
  stream-event reconciliation, and JSON Schema request formatting remain
  contained in `missy/providers/openai_provider.py`.
- The structured-output runner stays provider-neutral by calling an optional
  provider hook and still validating returned content against the Pydantic
  schema.
- The Responses path remains conservative: it is used only for native OpenAI
  clients without `base_url` and without tool transcript state.
- Chat Completions remains the compatibility path for OpenAI-like providers
  and current tool-calling behavior.
- Existing unrelated `LOOP_INSTRUCTIONS.md` modification remains in the working
  tree and was not touched.

## Tests

- `python3 -m ruff format missy/providers/base.py missy/agent/structured_output.py missy/providers/openai_provider.py tests/agent/test_structured_output.py tests/providers/test_openai_provider.py`: passed; 5 files left unchanged.
- `python3 -m pytest tests/agent/test_structured_output.py tests/providers/test_openai_provider.py -q`: 101 passed.
- `python3 -m pytest tests/providers -q`: 843 passed.
- `python3 -m pytest tests/agent -q`: 4109 passed, 4 skipped.
- `python3 -m ruff format --check .`: 731 files already formatted.
- `python3 -m ruff check .`: passed.
- `python3 -m pytest -q`: 20487 passed, 6 skipped, 3 warnings in 397.50s.

## Remaining Work

1. Add native Responses tool/function calling once Missy's tool-loop transcript
   model can preserve Responses output items safely.
2. Add provider-native tool-call stream delta reconciliation and final response
   validation for streamed tool workflows.
3. Add embeddings support if vector-memory workflows need an external OpenAI
   embedding backend.
4. Add provider diagnostics/doctor checks for OpenAI credentials, model list,
   network policy, rate-limit posture, redaction, and structured-output support.
5. Extend audit events for retry, rate-limit cooldown, usage/cost recording,
   fallback, and provider-side validation denials.

## Blockers

- No code blocker for the next OpenAI provider slice.

## Next Actions

Begin OpenAI provider diagnostics/doctor coverage or design the Responses
tool/function-call transcript model needed for native Responses tool calling.
