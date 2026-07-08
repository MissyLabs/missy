# Build Status

Last updated: 2026-07-08 13:56:58 EDT

## Current State

Primary focus remains the OpenAI provider overhaul. The provider layer now has
a first-class native OpenAI Responses execution path for compatible plain
text/vision requests while retaining Chat Completions for OpenAI-compatible
`base_url` providers and tool-call transcripts.

This session added the Responses adapter slice:

- Added internal OpenAI request routing that prefers `client.responses.create`
  only for native OpenAI clients with compatible transcripts.
- Preserved Chat Completions fallback for `base_url` endpoints, tool-result
  turns, and assistant tool-call history.
- Converted normalized OpenAI text/image content into Responses `input_text`
  and `input_image` parts, with system prompts mapped to `instructions`.
- Added Responses output extraction from `output_text` and structured
  `response.output` content parts.
- Mapped Responses usage counters back into Missy's canonical
  `prompt_tokens`, `completion_tokens`, and `total_tokens` shape.
- Added focused tests covering Responses routing, output extraction, vision
  conversion, and Chat fallback.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Provider interface compliance | improved | OpenAI still returns canonical `CompletionResponse` for both Responses and Chat paths. |
| Secure credential loading | in place | API key comes from config/env; changed path does not log secrets. |
| Network policy integration | in place | SDK client is built with policy-aware HTTP where available. |
| Model selection | in place | `auto` resolves through model listing with preferred current chat models and fallback. |
| Responses API path | started | Native OpenAI text/vision requests can use Responses; tool calls remain on Chat. |
| Chat compatibility | preserved | `base_url` providers continue to use Chat Completions. |
| Tool schema normalization | in place | OpenAI tool schemas delegate to `schema_adapter.normalize_for_provider()`. |
| Tool transcript repair | improved | Invalid/duplicate assistant tool calls and orphan tool results are dropped before request. |
| Vision input support | improved | Safe image content blocks are preserved and converted for Responses when eligible. |
| Auditability | partial | Provider invoke/error and transcript repair events exist; retry/fallback/cost events remain. |
| Tests | improved | Provider tests now cover native Responses routing and fallback behavior. |

## Current Architecture State

- OpenAI-specific message normalization and transcript repair remain contained
  in `missy/providers/openai_provider.py`.
- The Responses path is intentionally conservative: it is used only for native
  OpenAI clients without `base_url` and without tool transcript state.
- Chat Completions remains the compatibility path for OpenAI-like providers
  and current tool-calling behavior.
- Shared provider abstractions remain provider-neutral.
- Existing unrelated `LOOP_INSTRUCTIONS.md` modification remains in the working
  tree and was not touched.

## Tests

- `python3 -m ruff format missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed.
- `python3 -m ruff check missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed.
- `python3 -m pytest tests/providers/test_openai_provider.py -q`: 29 passed.
- `python3 -m pytest tests/providers -q`: 837 passed.
- `python3 -m ruff format --check .`: 731 files already formatted.
- `python3 -m ruff check .`: passed.
- `python3 -m pytest -q`: 20480 passed, 6 skipped, 3 warnings in 401.95s.

## Remaining Work

1. Add native Responses tool/function calling once Missy's tool-loop transcript
   model can preserve Responses output items safely.
2. Expand OpenAI streaming reconciliation to cover Responses stream events,
   provider-native tool-call deltas, and final response validation.
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
