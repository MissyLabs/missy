# missy-sandbox

Opt-in Linux Landlock **child-only containment** and a hardened Docker **command
builder**. Builder does not execute Docker, pull images or claim live isolation.
Nothing restricts the calling thread, parent service, test runner or desktop.

## Native API

`SandboxedCommand::new(PolicyConfig, SandboxConfig)` validates policy/config and
queries ABI without restricting the parent. `check_command(argv)` is preflight.
`execute(argv, &ToolContext).await` validates exact argv with missy-policy;
`execute_mcp` uses the MCP gate (one-shot, not persistent MCP transport).

SandboxConfig serde fields: read_roots, write_roots, deny_tcp (default true),
timeout_ms (default 30000), max_output_bytes (default 65536). Unknown fields denied.
Roots must fit corresponding policy allowlists and be existing absolute dirs, no
dot components or symlinks in any component. Merged-/usr needs real paths, not
/lib symlinks. Executable, interpreter, loader and libraries need explicit read
roots. Write roots do not imply read/execute. No implicit home/tmp/proc/env grants.

Before fork: O_PATH|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC root descriptors open one
component at a time, ruleset built without restricting parent. **Only dedicated
exec child** runs private pre_exec setrlimit(core=0), PR_SET_NO_NEW_PRIVS,
landlock_restrict_self, close_range(CLOEXEC). No allocation, formatting, locking or
path opens in pre_exec. Failure aborts exec, no unsandboxed fallback. Rust exec-error
pipe works before exec; every fd above stderr closes on successful exec. Stdin
/dev/null, environment cleared, cwd `/`.

Raw syscalls implement Linux documented UAPI, packed 12-byte path-beneath and
ABI-dependent ruleset size. ABI>=3 required for REFER/TRUNCATE; deny_tcp requires
ABI>=4. ABI>=5 also denies device ioctls. Later ABIs retain ABI-5 protection, not
newer optional scopes/rights. Unsupported kernel/LSM/syscall/close_range fails closed.

**TCP denial is NOT complete network isolation.** Blocks bind/connect including
loopback, not UDP, Unix-domain sockets, raw sockets or all socket calls. false
explicitly allows TCP. Never label this network_disabled. Full network isolation
needs a separately managed container/namespace boundary.

Combined raw stdout/stderr byte cap, concurrent nonblocking drain, timeout and
cancellation. Overflow/cancel/timeout/drop kills exact owned process group then
synchronously reaps direct child. Normal exit also kills leftover group members.
waitid(WNOWAIT) reserves leader PID until kill to avoid reused-PID kills. Exec/drop
briefly block normally; uninterruptible kernel I/O can delay drop indefinitely.
Approved commands must not setsid/setpgid to escape group cleanup.

Result JSON contains trust=untrusted, actual backend/ABI/TCP-denial metadata and
status/success/stdout/stderr. Raw cap excludes JSON escaping/invalid-UTF8 replacement
expansion; caller must separately bound serialized envelope.

## Container builder

`build_container_command(policy,sandbox,container,approved_runtime,argv,mcp)` requires
exact operator-approved absolute runtime path, prebuilt immutable image
repository@sha256:<64 lowercase hex>, nonroot uid:gid, bounded pids/memory/CPU.
Payload argv separately validated with missy-policy. No shell interpolation.

Plan includes pull=never, network=none, cap-drop=ALL, no-new-privileges, read-only,
ipc=none, log-driver=none, resource limits and explicit entrypoint. Only explicit
sandbox roots bind-mounted. Read mounts require recursive readonly; unsupported
runtime must error not drop flags. Write mounts necessarily readable too. Mounts
overlapping proc/sys/dev/run/var-run/Docker/containerd state or exposing socket files
rejected, bounded inspection. Unique missy-sandbox-UUID name, exact cleanup argv
`rm --force -- <name>`. Consumer owns bounded runtime I/O and exact-name cleanup on
drop/cancel. Builder creates no containers. Tests inspect argv, not live isolation.

Daemon opens mount paths after planning; snapshot inspection cannot eliminate
TOCTOU or safely walk a hostile mutable tree. Roots/topology must be administrator
controlled, stable until setup. Runtime approval is trusted operator input, never
tool arguments. No Docker socket mounts. No broad cleanup commands.

## Threat limits

Landlock is defense in depth, not VM/seccomp/cgroup/UID boundary. Does not hide
process metadata, prevent same-UID signals/ptrace, stop group escapes, limit native
memory/CPU/process count or remove root capabilities. Run dedicated unprivileged
UID. **Root execution is not safe untrusted-code hosting.** Commands remain trusted.
Administrators, same-UID attackers modifying approved roots/code/mounts, kernel or
daemon compromise and image-baked secrets are outside boundary. Parent filesystem
policy remains userspace, not Landlock.

## Verification

`cargo test -p missy-sandbox --offline -- --nocapture` exercises real child allowed
read/write, forbidden read/write/truncate, symlink escape, loopback TCP connect/bind
denial, cleared env, inherited non-CLOEXEC fd closure, unchanged parent authority,
exact argv/MCP gates, symlink-root rejection, overflow, timeout/cancel/abort reap.
Python3 is test probe dependency only. Unsupported ABI/probe prints SKIP explicitly;
feature rejection unit tests still run. No desktop/service changes, containers or
external network calls.
