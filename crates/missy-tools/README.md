# Rust tool boundary

`BuiltinTools::new(PolicyConfig, RuntimeConfig, Arc<Store>) -> Result<Self>`
implements `missy_core::ToolExecutor`. Optionally await
`.with_mcp(Vec<McpConfig>) -> Result<Self>` **before** exposing the executor.

Builtins: `file_read`, `file_write`, `list_files`, `shell_exec`, `web_fetch`,
`memory_put`, `memory_get`, `memory_search`, `memory_delete`. Argument structs deny
unknown fields. No caller can select a memory scope, provenance, trust, or author;
authenticated `ToolContext` determines all of these. Memory and every tool result
remain untrusted data. Runtime must still scan/censor before feeding results back
to a provider. Size rejection after a side effect does not mean the effect failed.

File operations use the policy crate's descriptor-relative no-symlink operations.
HTTP uses only its policy client. Shell uses absolute executable + exact argv,
empty environment, `/` working directory, no shell string evaluation, bounded
aggregate stdout/stderr, and a deadline. Operators can explicitly approve a shell
executable plus exact script argv; this is a trusted capability, not an implicit
interpreter allowlist. Arbitrary approved executables are **not sandboxed** and can
escape process-group containment with setsid, write outside file policy, or use
their own network. Do not approve untrusted programs as commands.

An owned worker thread performs nonblocking process I/O. On cancellation, timeout,
output overflow, parse failure, completed calls, or dropping the execution future,
the worker kills its owned process group and waits its immediate child. Dropping
the async guard synchronously joins the worker before returning. Grandchildren
are killed within that group, but not waitable children of this process; reaping
those is their parent's or init's responsibility. A program escaping the process
group is outside this guarantee. No global process kill or subreaper mutation.

## MCP

Both `policy.mcp_enabled` and an exact operator-approved argv in
`policy.allowed_commands` are mandatory. Enabling MCP alone grants nothing.
`shell_enabled` is independent. Only explicitly configured `McpConfig.env` entries
are passed. The caller's environment is cleared. Each discovery/call owns a fresh
process, eliminating shared multiplexed sessions or stale mutable caches.

MCP protocol `2024-11-05` uses newline-delimited JSON-RPC on stdio. Initialize,
initialized notification, and tools/list are performed. Notifications, mismatched
IDs, error responses, unsupported protocol/schema, pagination, and unexpected
response fields fail closed. The pin is SHA256 of the complete tools/list result
with recursively sorted object keys and preserved array order. Exported
`manifest_digest(&serde_json::Value)` computes it. A trusted operator must obtain
and review the manifest independently, then configure the pin.

Discovery verifies the pin. Every call repeats initialization and verifies list
twice on the same connection immediately before tools/call, then rechecks after
the result. Changed manifests fail. All MCP tools are effect-capable regardless
of untrusted read-only annotations. A failure after dispatch is uncertain, not a
safe retry. MCP does not provide an atomic "call if manifest hash still equals"
primitive, so no client can prove a malicious server internally used the
advertised implementation. Pinning verifies advertised manifests, not executable
integrity or an adversarial server's internal state. Approved servers remain a
trust boundary. This is not a claim of impossibly atomic server-side TOCTOU safety.

Supported MCP argument schema keywords: `type`, `properties`, `required`,
`additionalProperties:false`, `items`, `enum`, `description`, `title`, `default`.
Object arguments reject unknown fields recursively. Unsupported constraints or
combinators fail discovery rather than being silently ignored. Server-provided
schemas and annotations cannot turn off effect journaling.

## Explicit exclusions

No arbitrary Python plugin compatibility, eval-based calculator, desktop/camera,
voice, browser automation, legacy dynamic tool loading, or implicit global memory.
Unsupported tool names fail explicitly. The Python fixture in isolated tests is
only a local protocol adversary, never a production plugin runtime.
