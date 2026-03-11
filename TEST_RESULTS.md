# Test Results

## Overview

| Metric | Value |
|--------|-------|
| Total tests | 740 |
| Passed | 740 |
| Failed | 0 |
| Errors | 0 |
| Coverage | 86% |
| Required coverage threshold | 85% |

Run command: `python3 -m pytest tests/ -v`

---

## Test Categories

### Unit Tests

Direct, isolated tests for individual classes and functions. External I/O (HTTP
clients, DNS, file handles) is replaced with controlled fakes or mocks so that
every unit test is deterministic and fast.

**Modules covered:**

- `missy.config.settings` — config parsing, default values, YAML loading
- `missy.core.session` — session lifecycle and thread management
- `missy.core.events` — audit event construction and event bus filtering
- `missy.core.exceptions` — exception hierarchy and attribute contract
- `missy.gateway.client` — policy-gated HTTP client (sync and async)
- `missy.providers.*` — Anthropic, OpenAI, and Ollama provider adapters
- `missy.tools.builtin.calculator` — safe expression evaluator
- `missy.tools.registry` — tool registration and dispatch
- `missy.skills.base` / `missy.skills.registry` — skill API and registry
- `missy.channels.cli_channel` — CLI input/output channel
- `missy.agent.runtime` — agent run loop, provider resolution
- `missy.cli.main` — CLI entry-points
- `missy.security.sanitizer` — prompt injection detection and truncation
- `missy.security.secrets` — credential pattern scanning and redaction
- `missy.observability.audit_logger` — structured audit log writing
- `missy.memory.store` — in-memory key/value store
- `missy.scheduler.parser` — cron/interval expression parsing
- `missy.scheduler.jobs` — job data structures
- `missy.scheduler.manager` — job scheduling and execution

### Policy Tests

Focused tests for the three policy engine classes and the central facade.
These verify every branch of the allow/deny decision trees individually.

**Files:**

- `tests/policy/test_network.py` (38 tests) — CIDR matching, domain wildcard,
  exact host, DNS fallback, audit event fields, edge cases
- `tests/policy/test_filesystem.py` (28 tests) — path containment, symlink
  resolution, read/write separation, trailing slash normalisation
- `tests/policy/test_shell.py` (26 tests) — disabled state, allow-list prefix
  matching, path-qualified programs, malformed commands
- `tests/policy/test_engine.py` (18 tests) — facade delegation, singleton
  lifecycle, thread safety of init/get

### Integration Tests

End-to-end tests that instantiate real engines with real policy objects and
verify that enforcement decisions are correct without any mocking of the
engines themselves.

**File:** `tests/integration/test_policy_enforcement.py` (72 tests)

| Class | Tests | What it proves |
|-------|-------|----------------|
| `TestNetworkPolicyEnforcement` | 23 | Domain blocking, CIDR gating, wildcard domains, audit events |
| `TestFilesystemPolicyEnforcement` | 17 | Workspace sandboxing, /etc/passwd blocking, traversal prevention |
| `TestShellPolicyEnforcement` | 16 | Default-disabled shell, allow-list enforcement, basename matching |
| `TestPluginPolicyEnforcement` | 11 | Plugin gating, deny events, load/execute lifecycle |
| `TestPolicyEngineFacade` | 7 | Facade delegation, singleton install/retrieve |

### Security Tests

Tests that verify the input-sanitisation and secrets-detection layers work
correctly against adversarial inputs.

**Files:**

- `tests/security/test_sanitizer.py` — injection pattern matching, truncation,
  combined `sanitize()` behaviour
- `tests/security/test_secrets.py` — API key patterns, AWS keys, GitHub tokens,
  JWT detection, redaction correctness

### Provider Tests

Verify that each AI provider adapter correctly constructs requests, handles
errors, and emits audit events.

**Files:** `tests/providers/test_anthropic.py`, `test_openai.py`,
`test_ollama.py`, `test_base.py`, `test_registry.py`

### Plugin Tests

Verify plugin loading, policy enforcement at load time and execute time, the
singleton lifecycle, and audit event emission for every outcome.

**Files:** `tests/plugins/test_loader.py`, `tests/plugins/test_base.py`

### Scheduler Tests

Verify that cron and interval expressions parse correctly and that the job
manager schedules, executes, and cancels jobs as configured.

**Files:** `tests/scheduler/test_parser.py`, `test_jobs.py`, `test_manager.py`

---

## Coverage by Module (Selected)

| Module | Coverage |
|--------|----------|
| `missy/policy/engine.py` | 100% |
| `missy/policy/shell.py` | 100% |
| `missy/policy/network.py` | 98% |
| `missy/policy/filesystem.py` | 95% |
| `missy/security/sanitizer.py` | 100% |
| `missy/security/secrets.py` | 100% |
| `missy/tools/builtin/calculator.py` | 100% |
| `missy/providers/base.py` | 100% |
| `missy/config/settings.py` | 97% |
| `missy/core/events.py` | 95% |
| **Total** | **86%** |

---

## Key Test Areas

1. **Default-deny posture** — Every policy class defaults to the most
   restrictive configuration.  Tests verify that a freshly constructed engine
   with no allow-list entries denies everything.

2. **CIDR boundary enforcement** — Tests verify that the last host in a block
   is allowed, the first address outside the block is denied, and mixed
   IPv4/IPv6 comparisons do not crash.

3. **Wildcard domain semantics** — `*.example.com` allows the root domain,
   every subdomain, and deep multi-level subdomains, but never a different
   root.

4. **Path containment** — Symlink-resolved paths are compared against
   allow-lists using `Path.is_relative_to()`.  Tests exercise path traversal
   attempts (`../../etc/passwd`), sibling directories, and the empty allow-list
   case.

5. **Shell basename matching** — `/usr/bin/git status` matches entry `"git"`;
   `gitk` does not match `"git"`.  Unmatched quotes are denied gracefully.

6. **Plugin double-gate** — Both `plugins.enabled` (global toggle) and
   `allowed_plugins` (per-name allow-list) must pass before a plugin loads.

7. **Audit event completeness** — Every allow and deny action produces exactly
   one structured `AuditEvent` with correct `result`, `category`, `policy_rule`,
   `session_id`, `task_id`, and `detail` fields.

8. **Secrets detection** — Nine distinct credential patterns are tested
   individually and in combination, including redaction correctness.

9. **Prompt injection** — Thirteen heuristic patterns are tested, covering
   "ignore previous instructions", system prompt injection, model-control
   tokens, and override attempts.

10. **Calculator safety** — The built-in calculator's expression evaluator
    blocks `__import__`, `exec`, `eval`, `open`, and exponent DoS attempts
    while permitting standard arithmetic and bitwise operations.

---

## Running Tests

```bash
# Full suite
python3 -m pytest tests/ -v

# With coverage
python3 -m pytest tests/ --cov=missy --cov-report=term-missing

# Integration tests only
python3 -m pytest tests/integration/ -v

# Policy tests only
python3 -m pytest tests/policy/ -v

# Security tests only
python3 -m pytest tests/security/ -v

# Fast run (no coverage)
python3 -m pytest tests/ -q
```
