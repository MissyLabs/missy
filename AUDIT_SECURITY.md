# Security Audit

**Project:** Missy AI Agent Framework
**Audit Date:** 2026-03-11
**Auditor:** Internal
**Scope:** Policy enforcement, input sanitisation, secrets detection, audit
logging, plugin isolation

---

## Executive Summary

Missy is designed with a secure-by-default posture.  Every capability —
network access, filesystem access, shell execution, plugin loading — is
disabled or denied at construction time and must be explicitly unlocked via
configuration.  The audit found strong enforcement of these defaults, a
complete audit trail, and effective credential detection.  The main areas
requiring ongoing attention are the inherent limitations of heuristic prompt-
injection detection and the scheduler's reliance on process isolation rather
than sandboxing for job execution.

**Overall Risk Rating: LOW** (with the caveats documented per area below)

---

## 1. Default Security Posture

### Analysis

All four policy dataclasses (`NetworkPolicy`, `FilesystemPolicy`,
`ShellPolicy`, `PluginPolicy`) default to maximum restriction:

```python
NetworkPolicy(default_deny=True, allowed_cidrs=[], allowed_domains=[], allowed_hosts=[])
FilesystemPolicy(allowed_read_paths=[], allowed_write_paths=[])
ShellPolicy(enabled=False, allowed_commands=[])
PluginPolicy(enabled=False, allowed_plugins=[])
```

`get_default_config()` is the canonical safe baseline and returns this
configuration.  Nothing is permitted without an explicit, named grant.

### What Is Secure

- Zero-trust defaults require opt-in for every capability.
- Configuration is loaded from YAML; no environment variable overrides bypass
  policy at runtime.
- Changing policy requires updating the YAML file and restarting; there is no
  runtime API to relax policy mid-session.

### What to Watch

- Operators must review every allow-list entry added to their YAML file.
  Overly broad entries (`"*"` in domains, `"0.0.0.0/0"` in CIDRs) would
  negate the default-deny benefit.
- There is no schema validation layer that rejects obviously dangerous entries
  such as `0.0.0.0/0`; that responsibility lies with the operator.

**Risk Rating: LOW**

---

## 2. Network Policy Enforcement

### Analysis

`NetworkPolicyEngine` implements a six-step decision algorithm:

1. If `default_deny=False`, allow everything immediately.
2. If the host is a bare IP address, evaluate CIDR allow-lists only (no DNS
   lookup for IPs).
3. Exact hostname match against `allowed_hosts`.
4. Wildcard/suffix match against `allowed_domains`.
5. DNS resolution of the hostname followed by CIDR re-evaluation.
6. Deny.

Each step short-circuits on the first match, so the order is deterministic and
cannot be bypassed by crafting a hostname that matches a later step after
failing an earlier one.

### What Is Secure

- Bare IP addresses bypass DNS and go directly to CIDR comparison, preventing
  DNS rebinding as a bypass vector.
- All hostnames are lowercased before comparison; case-sensitivity bypass is
  not possible.
- IPv6 bracket notation (`[::1]`) is stripped before evaluation.
- Mixed IPv4/IPv6 comparisons are caught as `TypeError` and skipped safely
  rather than raising.
- Every evaluation — allow or deny — emits an `AuditEvent`, producing a
  complete forensic trail.

### What to Watch

- The DNS fallback in step 5 introduces network latency.  A slow or
  unresponsive DNS server will delay the deny decision.  Consider setting a
  short timeout on `socket.getaddrinfo` if the agent runs in latency-sensitive
  contexts.
- DNS caching is delegated to the operating system.  DNS TTLs are honoured by
  the OS resolver; Missy does not implement its own TTL check.
- `default_deny=False` mode disables all network checks.  This mode should only
  be used in fully trusted development environments.

**Risk Rating: LOW**

---

## 3. Filesystem Sandboxing

### Analysis

`FilesystemPolicyEngine` resolves all paths with `Path.resolve(strict=False)`
before comparison.  This means:

- `..` components are collapsed before any comparison, preventing path
  traversal via `workspace/../../etc/passwd`.
- Symlink targets are resolved where the symlink itself exists, preventing
  symlink-in-workspace attacks that point outside the allowed directory.
- Non-existent files (write targets about to be created) are resolved as far
  as their existing parent allows, then the final component is appended.

The `is_relative_to` method (Python 3.9+, used here on Python 3.11+) returns
`True` only when one path is strictly nested inside another, with no substring
false-positives.

### What Is Secure

- Symlink resolution prevents a common class of container escape analogues.
- Path traversal attempts collapse to the real target before evaluation.
- Read and write permissions are fully independent lists.
- Empty allow-lists deny all access without special-case logic.

### What to Watch

- `strict=False` resolves symlinks only as far as they exist.  If a symlink
  target is itself a symlink, the chain is followed.  An adversary who can
  create new symlinks inside the workspace after policy evaluation but before
  the actual file operation could still redirect writes.  This is a TOCTOU
  window inherent in any filesystem policy enforcement layer; mitigate by
  running the agent as a low-privilege user with minimal filesystem write
  permissions at the OS level.
- There is no file-type or file-size enforcement; the policy engine checks
  path containment only.  Operators who need to restrict by extension or size
  must add that logic above the policy layer.

**Risk Rating: LOW** (TOCTOU window is low-severity in the agent's typical
threat model)

---

## 4. Shell Policy

### Analysis

`ShellPolicyEngine` applies a two-stage gate:

1. Global `enabled` flag — if `False`, every command is denied regardless of
   the allow-list.
2. Allow-list check — the program name (first token from `shlex.split`) is
   compared by basename against `allowed_commands`.

`shlex.split` uses POSIX rules.  Unmatched quotes return `None` from the
extraction helper, which maps to a deny rather than a crash.

Basename comparison is exact: `"git"` does not match `"gitk"`.  Path-qualified
programs like `/usr/bin/git` are normalised to their basename before matching,
so `"git"` allows both `git status` and `/usr/bin/git status`.

### What Is Secure

- Shell execution is off by default; no accidental enablement.
- The allow-list check is on the extracted program basename, not the full
  command string, preventing bypass via `git; malicious-cmd`.
- Malformed commands (unmatched quotes) are denied rather than falling through.
- Empty and whitespace-only commands are denied.

### What to Watch

- The policy engine checks the declared program name only.  It does not prevent
  an allowed program from performing dangerous actions (e.g. `git` can run
  hooks that execute arbitrary code).  Operators should consider what each
  allowed command can do transitively.
- Shell metacharacters (`;`, `&&`, `|`, `$()`) in the command string are not
  individually stripped.  The engine trusts that the caller constructs commands
  from safe inputs.  If user-supplied data is interpolated into a command string
  before calling `check_command`, the caller must sanitise that data first.
- `enabled=True` with an empty `allowed_commands` list is safe (denies all) but
  confusing.  Document the intended state clearly.

**Risk Rating: LOW** (with the caveat that transitive program behaviour is not
audited)

---

## 5. Plugin Security Model

### Analysis

Plugins require two explicit grants before loading:

1. `plugins.enabled = True` in configuration.
2. The plugin's `name` must appear in `plugins.allowed_plugins`.

After passing both gates, the plugin's `initialize()` method is called.  A
return value of `False` or an exception during initialisation is treated as a
load failure; the plugin is not registered and `enabled` remains `False`.

Every load and execute attempt emits an `AuditEvent` regardless of outcome.

### What Is Secure

- Double gate prevents accidental plugin loading.
- Plugins that fail initialisation cannot be executed.
- `PluginPermissions` provides a manifest for security review, listing the
  network, filesystem, and shell permissions a plugin claims to need.
- Audit events cover every lifecycle transition: load allow, load deny, execute
  start, execute allow, execute error.

### What to Watch

- `PluginPermissions` is a declarative manifest only.  Missy does not enforce
  that a plugin's `execute()` method actually restricts itself to the declared
  permissions.  A plugin declaring `network=False` is still free to call
  `urllib.request.urlopen` unless its code is independently audited.  Operators
  should review plugin source code before adding names to `allowed_plugins`.
- The `allowed_plugins` list contains plugin name strings.  There is no
  cryptographic signing of plugin packages; supply-chain security is outside
  the current scope.
- Plugin execution errors propagate to the caller.  A buggy plugin can raise
  arbitrary exceptions that the agent runtime must handle gracefully.

**Risk Rating: MEDIUM** (the manifest is advisory, not enforced at runtime)

---

## 6. Input Sanitisation and Injection Prevention

### Analysis

`InputSanitizer` applies two defences:

1. **Truncation** — input longer than 10,000 characters is truncated and
   suffixed with `[truncated]` before being forwarded.
2. **Pattern matching** — thirteen heuristic regular expressions are checked
   case-insensitively.  A match logs a warning; the input is still returned
   (possibly truncated) for the caller to decide how to handle.

The sanitiser does not silently drop or transform injection attempts; it reports
them.  This is intentional: the framework does not know whether a flagged
pattern is genuinely adversarial or a legitimate user discussing prompt
injection.

### What Is Secure

- Truncation prevents memory and token-limit exhaustion attacks.
- Pattern coverage includes all common prompt injection formulations observed in
  published adversarial datasets.
- Warning logs provide a detection signal for monitoring systems.

### What to Watch

- Heuristic detection can be evaded by adversaries who are aware of the
  patterns (e.g. inserting zero-width characters, using homoglyphs, or
  splitting keywords across message boundaries).  This layer is a
  speed-bump, not a guarantee.
- The sanitiser does not modify or remove detected patterns.  Callers that
  forward flagged input to an LLM must implement their own policy for how to
  handle suspicious content.
- There is no test for Unicode normalisation attacks (e.g. `ｉｇｎｏｒｅ`
  instead of `ignore`).  Consider adding normalisation before pattern matching.

**Risk Rating: MEDIUM** (heuristic bypass is a known limitation of all
pattern-based injection detection)

---

## 7. Secrets Detection

### Analysis

`SecretsDetector` maintains nine compiled regular expression patterns covering:
API keys, AWS access keys, PEM private keys, GitHub personal access tokens,
passwords, generic tokens/secrets, Stripe keys, Slack tokens, and JWTs.

The `redact()` method replaces matches right-to-left so that earlier match
offsets remain valid after each substitution.

### What Is Secure

- Patterns are compiled once at construction time for performance.
- Right-to-left redaction is correct for overlapping matches.
- `has_secrets()` short-circuits at the first match for fast pre-checks.
- Multiple secrets in the same string are all detected and redacted.

### What to Watch

- The pattern set is not exhaustive.  New credential formats (e.g. a new
  provider's API key scheme) must be added manually.
- The `password` and `token` patterns require a label prefix followed by the
  value.  Bare passwords without a label (e.g. a raw password string) are not
  detected.
- There is no integration with the audit logger that would emit an event when
  secrets are detected.  Add a hook if you need alerting on credential exposure
  attempts.

**Risk Rating: LOW**

---

## 8. Audit Logging Completeness

### Analysis

The in-process event bus (`event_bus`) receives `AuditEvent` objects for every
policy decision.  `AuditLogger` writes these events to a structured log file.
Each event carries:

- `timestamp` (UTC, timezone-aware)
- `session_id` and `task_id` for request correlation
- `event_type` (e.g. `network_check`, `filesystem_write`, `shell_check`,
  `plugin.load`)
- `category` (e.g. `"network"`, `"filesystem"`, `"shell"`, `"plugin"`)
- `result` (`"allow"` or `"deny"`)
- `policy_rule` (the specific rule that produced the result, or `None`)
- `detail` (structured dict with host, path, command, or plugin name)

### What Is Secure

- Every allow and deny decision is logged — no silent outcomes.
- Events are emitted before the exception is raised on deny, so the audit trail
  is complete even when the caller catches the exception.
- `session_id` and `task_id` allow reconstruction of a full request lifecycle
  across multiple policy checks.

### What to Watch

- The event bus is in-process only.  If the process crashes mid-operation, the
  most recent events may not have been flushed to disk.  Consider writing events
  synchronously or using a durable queue for high-assurance environments.
- Audit log files are not rotated or encrypted by default.  Add log rotation
  and access controls appropriate to your deployment.
- There is no alerting integration; detecting anomalous patterns requires
  external log analysis tooling.

**Risk Rating: LOW**

---

## 9. Summary of Findings

| Area | Status | Risk |
|------|--------|------|
| Default security posture | All capabilities off by default | LOW |
| Network enforcement | CIDR + domain + DNS chain; audit trail complete | LOW |
| Filesystem sandboxing | Symlink-resolved path containment | LOW |
| Shell policy | Disabled by default; basename allow-list | LOW |
| Plugin security | Double-gate (enabled + allowlist); advisory manifest | MEDIUM |
| Input sanitisation | Heuristic injection detection; truncation | MEDIUM |
| Secrets detection | Nine-pattern credential scanner with redaction | LOW |
| Audit logging | Complete event trail; in-process bus | LOW |

### Recommendations

1. Add OS-level privilege separation (run the agent as a dedicated low-privilege
   user) to complement filesystem path policy.
2. Audit plugin source code before adding names to `allowed_plugins`; the
   manifest is advisory.
3. Consider Unicode normalisation before prompt-injection pattern matching.
4. Restrict CIDR and domain allow-lists to the minimum required for each
   deployment.  Document and review every allow-list entry.
5. Add log rotation and access controls to the audit log file.
6. Emit an `AuditEvent` when `SecretsDetector` finds a match, to enable
   alerting on credential exposure attempts.
