# OpenClaw Gap Analysis

Last updated: 2026-07-08

## Current Focus

The active primary focus is the Web TUI and operator console overhaul. OpenClaw and Odin are reference points for operator ergonomics, diagnostics, auditability, safe controls, run visibility, and control-plane clarity; Missy implementation remains clean-room and Python-native.

## Web TUI / Operator Experience Status

| Capability | Status | Notes |
|---|---|---|
| Secure local Web UI entrypoint | started | `/login` and `/` implemented with cookie sessions and CSRF. |
| Explicit authentication/session handling | started | API key login, HttpOnly cookie, in-memory expiry, logout revocation. |
| Polished dashboard | started | Runtime, providers, tools, sessions, and security posture are shown. |
| Session/run viewer | not_started | Needs streaming output, tool calls, errors, costs, routing, fallback, resume context. |
| Audit log browser | not_started | Needs API and UI filters plus redaction guarantees. |
| Diagnostics/doctor views | not_started | Needs Discord, providers, scheduler, tools, memory, gateway, policy, network posture. |
| Safe operator controls | not_started | Must be policy-gated, default-deny, audited, and confirmation guarded. |
| Responsive/accessibility coverage | partial | CSS is responsive; browser/visual tests still needed. |
| Backend Web TUI security | started | Auth, CSRF, rate limit, and hardened headers in place. |

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | tested | Runtime support remains in place. |
| A2 | Layered tool policy pipeline | hardened | Policy surfaces include current security updates. |
| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Repeated identical failing tool calls are fingerprinted and surfaced to the model. |
| A4 | Compaction retry coordination | not_started | Manager-level retry coordination remains future work. |
| A5 | Auth profile cooldown + fallback | not_started | Must preserve user-pinned profile behavior. |
| A6 | Per-provider tool schema normalization | live | Provider schema methods delegate to `normalize_for_provider()` with fallbacks. |
| A7 | Block-reply chunking with flush points | not_started | Channel delivery remains future work. |
| A8 | Per-channel identity cascade | not_started | |
| A9 | Before/after hook system | not_started | |
| A10 | Sub-agent depth + child caps | not_started | |
| A11 | Raw-stream JSONL diagnostics | not_started | Relevant to Web TUI run viewer. |
| A12 | Transcript dual-repair | not_started | |
| A13 | Context-window guard | not_started | |

## Recommended Next Slice

Refactor the Web TUI/session helpers out of `missy/api/server.py`, then add structured audit events and a first audit log browser API/view with filtering and redaction.
