# Runtime/provider hardening

Code and isolated fixture evidence only, no live deployment or credentials used.

Implemented: database-global fenced concurrency; bounded shutdown cancellation and
drain; durable separately mode-bound exact `run_tools` through ordinary policy,
approvals and journals; separate immutable `no_tools` Runtime; recursive security
projections at every provider request (including direct Store admission/replay),
assistant output censor; raw tool evidence stored separately and atomically from
bounded safe model receipts. Exact tools never call a provider. Original request,
checkpoint and raw source evidence remain immutable and scope-isolated.

Providers add explicit injectable credential references, multi-account round-robin
and cooldown, and native OpenAI Responses custom functions/NoTools. No hosted tool
execution is accepted. This is the safe API option for compatible Codex models,
not conversion of an existing CLI OAuth subscription. Codex CLI remains blocked.
Native optional SSE/NDJSON assembly uses the policy HTTP streaming gateway,
requires complete terminal/usage/finish evidence, and never dispatches partial
tool deltas. Runtime returns durable completion, not speculative partial tokens.

Action-only fabrication/promise guards use relevant current-turn successful tool
receipts. Shared persisted retries consume the normal step/token/cost/time budget.
Exhaustion preserves the answer with notice. These conservative heuristics are
not byte-identical legacy regex parity and are not a proof of factual truth.

Explicit OAuth refresh profiles use policy HTTP and operator reference names.
The selected private encrypted vault saves rotated refresh/access tokens before
inference. Nonblocking vault locks serialize refresh across processes. No login
flow, token discovery, legacy OAuth scraping or native CLI tool bypass exists.
Externally provisioned access_token_env has operator-managed lifetime and is saved
with its refresh generation on first use. A 401 expires only the rejected vault
generation, survives restart and does not silently replay a possibly billed
completion. The next explicitly requested completion may refresh the generation.
Refresh network uncertainty or
rotated-token persistence failure requires operator credential recovery.

Database-global lifetime cumulative spend/token caps reserve complete per-turn
ceilings before work, retain interrupted/uncertain reservations, release unused
known completed usage, and pin policy against silent disabling in another Runtime.
Unknown billing blocks further admissions under global limits. These are not
daily windows or proof a provider cannot exceed a declared bound. Overages remain
recorded and stop effects. Direct Provider consumers outside Runtime require their
own reservation path. Nested orchestration needs sufficient configured concurrency,
not an unbounded exemption. Account cooldown itself is in-process, not durable.

Operator APIs: display-name metadata rename; logical history cutoff clear;
idle/nonterminal/uncertain guarded session_cleanup with dry-run; exact scope and
argument-hash effect reconciliation with evidence and no replay; explicit billing
reconciliation that abandons unknown completion instead of regenerating it.

Focused four-crate tests and strict clippy passed after OAuth/global/operator additions:
`cargo test -p missy-runtime -p missy-providers -p missy-store -p
missy-provider-process --all-targets --locked --offline` with 2 jobs. Tests include
exact dispatch/restart/mode binding, capacity race/expiry, security projection,
guard exhaustion, account rotation, Responses tool security, and all four native
HTTP streaming adapters, including completed tool-call argument assembly.
OAuth TCP fixtures prove independent simultaneous OS processes serialize one
refresh, persist refresh rotation across process restarts, and durably invalidate
401 without inference replay. Additional fixtures cover cancellation, bounded
refresh timeout, blocked policy, private-vault lock contention and sanitized errors.
The ignored OAuth child fixture is launched explicitly by its parent test; it is
not an untested skip. Store fixtures cover legacy nullable/new empty session names,
expired unknown reservations, shared parent/child budget and policy downgrade denial.

Native operator CLI: `providers list`, `providers check NAME`, `providers keypool
NAME`, `providers oauth-refresh NAME`. List/check/keypool are offline and show
reference names/availability, never credential values. OAuth check does not require
bootstrap environment values after tokens have been vaulted and does not claim to
authenticate or inspect the vault. OAuth refresh explicitly contacts the selected
configured issuer, persists its result without inference, and handles shutdown
signals through cancellation. No login/discovery or profile credential import is
offered. Provider failover currently follows configured order/circuit state, not
weighted/tier routing. Native API model IDs are configured rather than discovered.

Whole-workspace tests and CLI integration validation remain the final owners' gate.
No live provider, desktop, service or ~/.missy state was touched.
