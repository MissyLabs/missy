# Build Status

Last updated: 2026-07-08 14:14:39 EDT

## Current State

Primary focus remains the OpenAI provider overhaul. The provider layer now has
native OpenAI Responses execution for compatible plain text/vision completion
requests and native Responses streaming for compatible stream requests. Chat
Completions remains the compatibility path for OpenAI-compatible `base_url`
providers and transcripts containing tool-result turns or assistant tool-call
history.

This session added the Responses streaming slice:

- Added conservative stream routing that uses `client.responses.stream` only
  for native OpenAI clients with compatible transcripts.
- Preserved Chat Completions streaming fallback for `base_url` endpoints and
  unsupported SDK/client shapes.
- Converted streaming requests through the same Responses input/instructions
  payload builder used by non-streaming completions.
- Reconciled `response.output_text.delta` events with final/full-text
  Responses events so repeated snapshots do not duplicate emitted text.
- Converted Responses stream `response.failed` and `error` events into
  `ProviderError` with safe provider messages.
- Expanded focused OpenAI provider tests for Responses stream routing,
  full-content reconciliation, failed events, and Chat fallback.
- Updated OpenAI provider documentation for native Responses streaming.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Provider interface compliance | improved | OpenAI still returns canonical `CompletionResponse` and `stream()` yields canonical text chunks. |
| Secure credential loading | in place | API key comes from config/env; changed path does not log secrets. |
| Network policy integration | in place | SDK client is built with policy-aware HTTP where available. |
| Model selection | in place | `auto` resolves through model listing with preferred current chat models and fallback. |
| Responses API path | improved | Native OpenAI text/vision completions and compatible streams can use Responses. |
| Chat compatibility | preserved | `base_url` providers and tool transcripts continue to use Chat Completions. |
| Streaming reconciliation | improved | Responses stream deltas and final/full-content snapshots are reconciled. |
| Tool schema normalization | in place | OpenAI tool schemas delegate to `schema_adapter.normalize_for_provider()`. |
| Tool transcript repair | improved | Invalid/duplicate assistant tool calls and orphan tool results are dropped before request. |
| Vision input support | improved | Safe image blocks are preserved and converted for Responses when eligible. |
| Auditability | partial | Provider invoke/error and transcript repair events exist; retry/fallback/cost events remain. |
| Tests | improved | Provider and full repository suites pass after the streaming slice. |

## Current Architecture State

- OpenAI-specific message normalization, transcript repair, Responses routing,
  and stream-event reconciliation remain contained in
  `missy/providers/openai_provider.py`.
- The Responses path is intentionally conservative: it is used only for native
  OpenAI clients without `base_url` and without tool transcript state.
- Chat Completions remains the compatibility path for OpenAI-like providers
  and current tool-calling behavior.
- Shared provider abstractions remain provider-neutral.
- Existing unrelated `LOOP_INSTRUCTIONS.md` modification remains in the working
  tree and was not touched.

## Tests

- `python3 -m ruff format missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed; 1 file reformatted.
- `python3 -m ruff check missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed.
- `python3 -m pytest tests/providers/test_openai_provider.py -q`: 33 passed.
- `python3 -m pytest tests/providers -q`: 841 passed.
- `python3 -m ruff format --check .`: 731 files already formatted.
- `python3 -m ruff check .`: passed.
- `python3 -m pytest -q`: 20484 passed, 6 skipped, 3 warnings in 397.19s.

## Remaining Work

1. Add native Responses tool/function calling once Missy's tool-loop transcript
   model can preserve Responses output items safely.
2. Add provider-native tool-call stream delta reconciliation and final response
   validation for streamed tool workflows.
3. Add structured output support that can use OpenAI-native response formats
   when available and fall back to Missy's generic validator otherwise.
4. Add embeddings support if vector-memory workflows need an external OpenAI
   embedding backend.
5. Add provider diagnostics/doctor checks for OpenAI credentials, model list,
   network policy, rate-limit posture, and redaction.
6. Extend audit events for retry, rate-limit cooldown, usage/cost recording,
   fallback, and provider-side validation denials.

## Blockers

- No code blocker for the next OpenAI provider slice.
