# missy-plugins

Safe native replacement for dynamic Python `self_create_tool` behavior: a data-only
declarative registry plus approved operator-supplied ELF binaries. The model does
not install arbitrary code, compile Rust, evaluate Python, invoke a shell, grant
itself policy capabilities, or load changed source into a running runtime.

## Control plane and integration

When plugins are enabled, `PluginTools` advertises the model-callable
`self_create_tool` tool with a typed `{ "manifest": PluginManifest }` argument.
It is side-effecting and requires the same durable exact-call `PluginApproval`
gate as execution. Authenticated scope, cancellation, closed fields, schema,
capability policy and a 64KiB argument limit are checked before persistence.
Its only effect is creating a pending scope-bound proposal. The response contains
the exact review hash and explicitly reports `approved=false`, `enabled=false`
and `requires_new_runtime=true`. The model cannot approve itself, add allowed IDs,
change policy, write executable source, compile code, execute the proposal or
change the current tool catalog/fingerprint. Proposal creation is available with
an empty execution-ID allowlist, so an operator need not pre-activate a nonexistent
plugin merely to permit proposing it. Duplicate IDs are never overwritten.

`ToolRegistry::open(absolute_private_directory)` creates a current-user-owned 0700
directory under an existing parent. Records are private, size-bounded JSON files.
`propose(&Scope, PluginManifest)` validates the manifest and returns a proposal
containing the exact SHA256 to review. `review(scope,id,hash,true)` is an operator
API, not a model-callable approval tool. `enable(scope,id,hash,&PluginConfig)` also
requires matching configured scope and an explicitly allowed ID. Rejection clears
activation. Existing IDs cannot be silently overwritten by new proposals.

`PluginTools::new(registry,config,Arc<Policy>,Arc<dyn PluginApproval>)` snapshots
enabled records once. Runtime construction fails closed for missing or unreviewed
configured IDs, wrong executable hashes, or unavailable confinement. Activation
only affects the next runtime construction, never the current tool catalog.
The configuration fingerprint includes policy, registry proposal hashes/manifests,
scope, limits, runtime library roots and approval-gate fingerprint. The integrating
runtime must retain original tool call IDs in its normal durable side-effect
journal and apply its exact-call approval guard. This crate does not replace that
journal. Each `plugin__ID` definition is forcibly required to be `side_effect=true`.

The required `PluginApproval` trait deliberately has no allow-all implementation.
Embed it in the existing durable human approval boundary; never implement approval
by trusting model arguments or the registry's proposal approval alone. Preflight
and execution both check scope, exact expanded argv policy, argument shape and
approval. Registry scope is an authorization scope, not an execution principal:
children still run as the service OS user inside confinement.

## Manifest

The manifest consists of `id`, typed `definition` (`ToolDefinition`), `command`
(`executable`, lowercase `sha256`, `args`) and `capabilities.read_roots`.
Its name must be `plugin__{id}`. Parameter schemas must be closed root objects
with only string/number/integer scalar properties. Required fields, enums,
length/numeric bounds are supported. All references, URI schema vocabularies,
nested objects, arrays, regexes and unsupported keywords are rejected. An argv
placeholder occupies an entire entry, such as `{text}`; it cannot select the
executable or produce additional argv entries. Substitutions are literal strings,
never shell syntax. Every expanded vector must already appear in the owned
policy's `allowed_commands` and `shell_enabled` must be true. A template does not
expand the host's command authority.

Only operator-provided native 64-bit ELF is executed. Shebangs and known shell,
interpreter, compiler and loader executable names are rejected. Name checks are
defense in depth, not a malware classifier: exact binary-hash human review remains
essential, and renamed interpreters are not made trustworthy by their filename.
The binary is read, hash-checked, copied to a sealed memfd and that exact sealed
image is executed, eliminating executable pathname replacement between hash and
exec. The hash is checked at construction and again at each dispatch.

## Confinement

Linux Landlock ABI >=3 is mandatory; no unsandboxed fallback. Child-only Landlock
permits only manifest read roots, explicit operator `runtime_read_roots`, and
write-only access to a fresh private temporary directory. Manifest read roots
must fall within policy read roots. Dynamic ELF libraries/loaders require explicit
bounded real directory roots, for example `/usr/lib/x86_64-linux-gnu` on the test
host. `/`, dot paths and symlinked roots are rejected. There are no implicit `/etc`,
home, PATH, credential, or library grants. Environment is empty, stdin null,
non-stdio FDs close on exec, core dumps disabled, and cwd is the temporary directory.

A seccomp filter denies socket creation and network syscalls for TCP, UDP and Unix
sockets, plus io_uring, tracing, common namespace/privilege escapes, group/session
escape and cross-process signals. Process groups are killed and the leader reaped
on success, timeout, output overflow, cancellation, error and future drop. PID
reservation via `waitid(WNOWAIT)` avoids killing a reused PID. Output is jointly
bounded and always labeled untrusted. Time is at most 60 seconds and output at
most 1 MiB. This is not a VM or complete hostile-native-code boundary: kernel
vulnerabilities, resource exhaustion, same-UID/root configuration tampering,
concurrent administrator mount/path replacement and malicious approved native
program logic remain outside the guarantee. Only install operator-reviewed code.

## Benchmark

The separate benchmark API uses existing configured providers only, no tool
definitions, bounded literal prompts and at most three calls. It records actual
provider-reported usage/cost and elapsed wall time, not invented GPU throughput.
Unknown billing or uncertain calls fail closed in its durable receipt ledger.
See the public benchmark request/config/result types and the embedding CLI.

## Verification and deliberate limits

`cargo test -p missy-plugins --offline` runs local mocks plus native `/usr/bin/printf`
and a disposable C fixture. Tests verify exact-hash/scope approval, remote-ref
rejection, closed schemas, inert shell-looking strings, exact argv policy,
no PATH/HOME inheritance, filesystem/network denial, hash drift and timeout.
The fixture compiler is used only by tests, never plugin execution.

No source-generation/autocompilation service, WASM runtime, dynamic Python,
in-process native library loading or automatic external MCP installation is
provided. Existing approved external MCP registration remains a separate facility.
If an outer workflow writes a proposed Rust source artifact, it must use existing
policy-write roots and durable approval, and it must not silently compile or run
that artifact. This crate's functional installation path is reviewed native ELF.
