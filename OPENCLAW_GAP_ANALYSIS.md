# OpenClaw Gap Analysis

Last updated: 2026-07-08 14:34:10 EDT

## Current Focus

The active primary focus is the OpenAI provider overhaul. OpenClaw and Odin
remain references for provider-turn validation, raw stream diagnostics,
operator ergonomics, safe controls, and auditability; Missy implementation
remains clean-room and Python-native.

## OpenAI Provider Status

| Capability | Status | Notes |
|---|---|---|
| Provider interface compliance | improved | OpenAI keeps returning Missy's canonical `CompletionResponse`; `BaseProvider` now has an optional structured-output hook. |
| Secure key handling | in place | Config/env loading remains; changed path does not log secrets. |
| Network policy integration | in place | SDK client attempts policy-aware HTTP wiring. |
| Model listing/selection | in place | `auto` uses model listing with current preferred chat-model fallback. |
| Chat/text generation | improved | Native OpenAI text/vision can use Responses; compatibility path remains Chat. |
| Responses API path | improved | Plain native OpenAI requests, compatible streams, and structured outputs route to Responses when safe. |
| Tool schema normalization | live | Uses provider schema adapter; native Responses tool path remains future work. |
| Tool transcript repair | improved | Invalid/duplicate/orphaned tool turns are removed before SDK calls. |
| Vision input support | improved | Safe image blocks are preserved and converted to Responses `input_image` when eligible. |
| Streaming reconciliation | improved | Responses stream text deltas and final/full snapshots reconcile without duplicate output. |
| Structured output | improved | OpenAI-native JSON Schema request formatting is implemented for Responses and Chat compatibility paths. |
| Embeddings | not_started | Needed only if external vector workflows require OpenAI embeddings. |
| Diagnostics/doctor | partial | General diagnostics exist; OpenAI-specific probes remain. |
| Audit events | improved | Provider invoke/error and transcript repair are covered; retry/fallback/cost events remain. |

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | tested | Runtime support remains in place; provider stream chunk quality improved for OpenAI Responses. |
| A2 | Layered tool policy pipeline | hardened | Provider/tool policy surfaces remain active. |
| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Repeated failing tool calls are surfaced to the model. |
| A4 | Compaction retry coordination | not_started | Manager-level retry coordination remains future work. |
| A5 | Auth profile cooldown + fallback | not_started | Preserve pinned provider behavior. |
| A6 | Per-provider tool schema normalization | live | OpenAI delegates to schema adapter; structured output now uses provider-native request-shape normalization. |
| A7 | Block-reply chunking with flush points | not_started | Channel delivery remains future work. |
| A8 | Per-channel identity cascade | not_started | |
| A9 | Before/after hook system | not_started | |
| A10 | Sub-agent depth + child caps | not_started | |
| A11 | Raw-stream JSONL diagnostics | partial | OpenAI Responses stream events are parsed; raw JSONL diagnostics remain future work. |
| A12 | Transcript dual-repair | improved | OpenAI repairs invalid/orphaned tool turns before SDK calls. |
| A13 | Context-window guard | not_started | |

## Recommended Next Slice

Add OpenAI provider diagnostics/doctor checks or design the Responses
tool-call transcript model needed for native Responses function calling.
