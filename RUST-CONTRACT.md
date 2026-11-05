# Rust rewrite working contracts

Workspace rooted here, all new implementation in `crates/`. Python remains untouched as a parity reference and rollback path, not a runtime dependency. Do not deploy or touch any live `.missy`.

Shared DTO/config/traits are in `missy-core`. Root workspace dependency versions are authoritative. No secrets in config or diagnostics: environment variable references only. All tests isolated with tempdirs and mock servers.

## Planned public interfaces

- `missy_policy::{Policy, PolicyHttpClient}`: `Policy::new(PolicyConfig) -> Result<Self>`; read/write/shell checks and bounded, DNS-pinned, no-proxy, no-redirect HTTP via `PolicyHttpClient`. Coordinate exact async HTTP interface directly with providers/tools owners and report parent.
- `missy_store::Store`: `open(path)`, scope-isolated durable sessions/turns, leases/fencing, message history, tool execution journal, scoped memory/FTS, audit and checkpoints. Store owner publishes exact APIs immediately to runtime/gateway owners.
- `missy_providers::ProviderPool`: concrete `Provider` implementation with sequential failover and per-provider circuit breaker, configured from `Vec<ProviderConfig>` and `PolicyConfig` (or `PolicyHttpClient`), no hidden bypass routes. Publish exact constructors.
- `missy_tools::BuiltinTools`: concrete `ToolExecutor`, configured `PolicyConfig`, `RuntimeConfig`, optionally store. Scoped memory tools and pinned opt-in MCP. Publish constructor.
- `missy_runtime::Runtime`: construct with shared Store, Arc<dyn Provider>, Arc<dyn ToolExecutor>, system prompt, RuntimeConfig. `run(scope, prompt, request_id)`, `resume(scope, turn_id)`, cancellation and status API. Exact return DTO published by owner.
- `missy_gateway`: HTTP API + browser console + Discord + durable scheduler wired to runtime. No production startup in tests.

All teams report interface changes immediately. Security claims require adversarial tests. Unsupported legacy functions fail explicitly; no pretend compatibility. No automatic replay of possibly executed side effects after crashes.
