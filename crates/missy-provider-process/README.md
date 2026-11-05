# Native CLI provider boundary

This crate is an optional **chat-only** `missy_core::Provider`. It starts an
explicitly approved external executable directly, not a Python bridge, ACP
bridge, shell command, OAuth scraper, or token importer.

## Public API

- `CliProvider::new(CliProviderConfig) -> anyhow::Result<CliProvider>`
- `create_provider(CliProviderConfig) -> Result<Arc<dyn Provider>>`
- `CliProviderKind::{ClaudeCli, CodexCli}`, serialized as `claude_cli`, `codex_cli`
- `CliProvider::manual_login_argv(kind, absolute_executable) -> Vec<String>`
- `CliProvider::probe_auth(config, cancellation).await -> Result<AuthStatus>`
- `AuthStatus::{Authenticated, NotAuthenticated}`

Integrators must explicitly select this factory instead of the HTTP provider
factory and use a **NoTools runtime**. `complete` rejects all tool definitions,
tool-call history, tool-role messages, unsupported roles, misplaced system
messages, and conversations not starting/ending with a user message. No native
`ToolCall` support is claimed. Operator-controlled prompts are passed on stdin
as JSON conversation data; leading system messages go into a private prompt
file. This does not make language-model role interpretation a security boundary.

Configuration is serde `deny_unknown_fields`, with no arbitrary args, environment,
shell, working directory, command hooks, fallback binary discovery, or imported
credentials. Required fields:

```json
{
  "name": "claude-chat",
  "kind": "claude_cli",
  "executable": "/absolute/operator-approved/claude",
  "auth_home": "/absolute/operator-provisioned/private-home",
  "model": "an-exact-model-id",
  "approved_executable": true,
  "timeout_seconds": 120,
  "max_stdout_bytes": 1048576,
  "max_stderr_bytes": 65536,
  "max_input_bytes": 1048576,
  "input_microusd_per_million": null,
  "output_microusd_per_million": null
}
```

Paths are canonicalized; executable and auth home must be root/current-user
owned and not group/world writable. Executable ancestors are checked too,
allowing root-owned sticky temporary directory parents. A trusted executable
may itself be a script; the adapter never invokes an interpreter implicitly.
Use a self-contained binary or wrapper with absolute interpreter/dependency
paths: no inherited `PATH` is supplied. Trust approval is not a binary signature
or protection against an operator replacing a binary after construction.

## Tool ownership

Claude runs with fixed `--tools ""`, disallowed `mcp__*`, strict empty MCP config,
no setting sources, slash commands disabled, `dontAsk`, hooks disabled,
one turn, no session persistence, and an explicit replacement system prompt.
The parser requires an init event advertising zero native tools/MCP servers,
an assistant-only text response from the exact configured model, and a matching
successful terminal result. Tool-use blocks, tool calls, permission denials,
role changes, fallback models, unexpected protocol events and malformed JSON
fail closed. Thinking blocks are discarded, never presented as response text.

**Codex construction is deliberately disabled.** Local `codex exec --help`
inspection showed `--sandbox read-only`, `--ignore-user-config` and
`--ignore-rules`, but no verified complete no-tools protocol. Read-only sandbox
does not prevent model-directed shell/read/web/MCP operations. Advertising an
operational Codex adapter on that basis would bypass Missy policy. The serde
kind, manual login argv and non-inference status probe are available for future
integration, but there is no pretend tool-safe completion implementation.

## Credentials and process lifecycle

An operator must manually provision the CLI's own login in the selected
`auth_home`. Login argv is returned as **data**, not executed. Status probes run
only `claude auth status` or `codex login status`, bounded to 20 seconds, and
return only an auth boolean enum. They never run inference or return raw account
metadata/credentials. Missy does not read auth files or invoke OAuth/login.

Child environment is cleared and rebuilt from HOME, explicit CLAUDE_CONFIG_DIR
and CODEX_HOME, LANG, NO_COLOR and DISABLE_AUTOUPDATER only. All inherited API
keys, endpoint overrides, proxies, loader/interpreter injection variables and
agent nesting flags are absent. The approved CLI necessarily accesses its own
operator-provisioned credentials and network; this is not an OS containment
sandbox or HTTP origin-policy wrapper. Never configure an untrusted executable.

Each call gets an empty mode-0700 temporary working directory, removed on drop.
Stdout and stderr are independently byte bounded. Stderr is counted and
discarded rather than retained or printed. All transport/protocol errors are
generic. Stdin/out/err use nonblocking pipes with bounded scheduling slices.
Timeout, cancellation, output overflow, normal exit and dropping/aborting a
completion future kill the dedicated child process group and synchronously wait
for the immediate child. This covers ordinary CLI descendants, not malicious
children that deliberately escape the group with `setsid`. The trust model does
not treat a hostile executable as sandboxed. Linux fixture tests inspect process
state; grandchildren killed by SIGKILL can temporarily remain zombies pending
their actual parent's/system init reaping, but cannot continue execution.

## Usage, pricing and acceptance limits

Input/output counters come only from terminal structured usage, including cache
input in total input tokens. Reported nonnegative finite `total_cost_usd` is
converted conservatively to micro-USD. Without that, explicit configured input
and output rates are used only when counters exist and cache usage is zero.
Cache rates are not invented. Missing usage or prices means `cost_known=false`,
never a claim of a free call. Unknown counters are represented by zero because
the shared `Usage` type has no optional counters; this is not an estimate.

Tests use only approved temporary shell fixtures, not actual provider inference,
OAuth, live accounts or credentials. They verify protocol/flags, usage, finish
reasons, unknown costs, role/tool rejection, secret-safe failures, env/cwd,
explicit setup/status, cancellation, task-abort and timeout process cleanup.
No live OAuth acceptance or cross-version CLI compatibility is claimed. An
unexpected CLI version/protocol fails closed rather than silently weakening the
tool restrictions. The adapters are Unix-only; process cleanup tests use Linux.
