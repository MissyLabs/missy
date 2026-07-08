# OpenClaw Gap Analysis

Last updated: 2026-07-08 10:57:27 EDT

## Current Focus

The active primary focus is the Web TUI and operator console overhaul. OpenClaw
and Odin remain references for operator ergonomics, diagnostics, auditability,
safe controls, run visibility, and control-plane clarity; Missy implementation
remains clean-room and Python-native.

## Web TUI / Operator Experience Status

| Capability | Status | Notes |
|---|---|---|
| Secure local Web UI entrypoint | started | `/login` and `/` implemented with cookie sessions and CSRF. |
| Explicit authentication/session handling | improved | Browser session storage is extracted into `missy/api/web_sessions.py`. |
| Polished dashboard | started | Runtime, providers, tools, sessions, diagnostics, controls, security posture, and audit trail are shown. |
| Session/run viewer | not_started | Needs streaming output, tool calls, errors, costs, routing, fallback, resume context. |
| Audit log browser | improved | `/api/v1/audit` supports filters, facets, file/memory sources, redaction, IDs, totals, offsets, and `has_more`; UI has filters, pagination, and details. |
| Diagnostics/doctor views | improved | `/api/v1/diagnostics` covers Web, providers, tools, memory, policy, gateway, Discord, scheduler, runtime, and remediation hints. |
| Safe operator controls | improved | Provider default switching plus scheduler pause/resume are confirmed, validated, CSRF-protected for browser sessions, and audited. |
| Responsive/accessibility coverage | partial | CSS is responsive; renderer tests cover hooks and escaping; browser/visual tests still needed. |
| Backend Web TUI security | improved | Auth, CSRF, rate limit, hardened headers, audit events, redaction, XSS-resistant dashboard rendering, and redacted audit search are in place. |
| Renderer/assets architecture | improved | Login, message, console shell, CSS, and JavaScript live in `missy/api/web_console.py`; server routing is thinner. |

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | tested | Runtime support remains in place. |
| A2 | Layered tool policy pipeline | hardened | Policy surfaces include current security updates and are reflected in diagnostics. |
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

Add the next safe controls slice for tools or channels with explicit policy
gates, confirmation text, denial audit events, and focused API/UI tests.
