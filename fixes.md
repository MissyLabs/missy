# Missy — Fixes, Gaps & Feature Backlog

Scope: static review of `missy/` on branch `codex_loop_04_27_2026_b` (HEAD `3a7d03b`).
Focus areas requested: implementation gaps, data-consistency issues, scheduling gaps,
rate-limit risk to external APIs, security issues, plus new features.

Each item lists **file/line**, **problem**, **impact**, and a **suggested fix** so it can
be actioned directly. Severity: 🔴 high, 🟠 medium, 🟡 low.

---

## 1. Rate-limit risk to external APIs

### 1.1 🔴 Token bucket is never pre-charged — TPM limiting is effectively disabled
- **Where:** `missy/providers/anthropic_provider.py:142` and `:266`; `missy/providers/base.py:227-238`.
- **Problem:** `complete()` / `complete_with_tools()` call `self._acquire_rate_limit()`
  with **no `estimated_tokens` argument**, so `RateLimiter.acquire(tokens=0)` only spends
  the *request* bucket. The token bucket is decremented only afterwards via
  `record_usage()`. A burst of large-context requests can all pass `acquire()`
  simultaneously because the TPM budget is never checked *before* dispatch.
- **Impact:** TPM rate limiting does not actually throttle outbound calls — Missy can
  blow past provider tokens-per-minute limits and trigger 429s (or spend) under load.
- **Fix:** Estimate prompt tokens before the call (sum of message lengths / 4, or a real
  tokenizer) and pass to `_acquire_rate_limit(estimated_tokens=...)`. Then use
  `record_usage()` to reconcile estimate vs. actual (it already subtracts the delta).
  Do the same in `openai_provider.py` and `ollama_provider.py`.

### 1.2 🟠 RateLimiter is hard-coded; config RPM/TPM values are ignored
- **Where:** `missy/providers/registry.py:236` — `instance.rate_limiter = RateLimiter()`.
- **Problem:** Every provider gets the default `RateLimiter(60 rpm, 100_000 tpm)`. There is
  no `requests_per_minute` / `tokens_per_minute` field in `ProviderConfig`
  (`missy/config/settings.py` has no such keys), so operators cannot tune limits to match
  their actual plan tier. A user on a low tier can't lower it; a user on a high tier is
  needlessly throttled.
- **Impact:** Wrong limits for the account → either 429 storms or artificial slowdowns.
- **Fix:** Add `requests_per_minute: int = 60` and `tokens_per_minute: int = 100_000` to
  `ProviderConfig`, and construct `RateLimiter(rpm, tpm)` from those in `from_config`.

### 1.3 🟠 `on_rate_limit_response()` blocks the shared limiter thread with `time.sleep`
- **Where:** `missy/providers/rate_limiter.py:149-164`.
- **Problem:** On a 429, `on_rate_limit_response()` calls `time.sleep(min(retry_after, max_wait))`.
  The RateLimiter is shared across threads (async runtime, scheduler, Discord). A synchronous
  sleep inside this method blocks the calling thread and, if called from the async event
  loop path, stalls unrelated coroutines.
- **Impact:** One provider's 429 can freeze the whole agent for up to `max_wait` (30s).
- **Fix:** Drain the buckets (already done) but do **not** sleep inside the limiter. Let the
  next `acquire()` compute the wait naturally, or push the retry-after as a "not before"
  timestamp the buckets honor. In async paths use `asyncio.sleep`.

### 1.4 🟡 `retry-after` header parse can raise and mask the real error
- **Where:** `anthropic_provider.py:158-159`, `openai_provider.py:141`.
- **Problem:** `float(headers.get("retry-after", 5))` will raise `ValueError` if the header
  is a non-numeric HTTP-date (RFC 7231 permits an HTTP-date form). That exception replaces
  the clean `ProviderError` with an unrelated traceback.
- **Fix:** Wrap the parse in try/except; fall back to a default (e.g. 5s) on any parse error,
  and support the HTTP-date form.

---

## 2. Scheduling gaps

### 2.1 🔴 Retries are silently dropped when they land outside `active_hours`
- **Where:** `missy/scheduler/manager.py:332-355` (`_run_job` active-hours gate) combined with
  the retry path at `:410-446`.
- **Problem:** A failed job schedules a one-shot retry via `add_job(trigger="date")`. When that
  retry fires, `_run_job` first checks `job.should_run_now()`; if the backoff pushed the retry
  outside the active-hours window, the method `return`s immediately **without rescheduling**.
  The retry is lost and `consecutive_failures` is never advanced or cleared.
- **Impact:** Jobs with `active_hours` set can permanently stop retrying after a single failure
  near the window boundary — a silent reliability hole.
- **Fix:** In the active-hours skip branch, if the job is a retry (or has pending failures),
  reschedule to the next in-window time instead of dropping it.

### 2.2 🟠 `retry_on` categories are declared but never enforced
- **Where:** `missy/scheduler/jobs.py:108-122` (`should_retry` ignores `error`), field defined at `:57`.
- **Problem:** `ScheduledJob.retry_on` defaults to `["network", "provider_error"]` and is
  documented as "error category tags that trigger a retry," but `should_retry()` retries on
  **every** error type up to `max_attempts`. Non-retryable errors (e.g. auth failure, invalid
  prompt) are retried needlessly.
- **Impact:** Wasted API spend and delay retrying deterministic failures that will never succeed.
- **Fix:** Match the exception string/category against `retry_on` in `should_retry()`, or drop
  the field to avoid the misleading contract.

### 2.3 🟠 `scheduling.max_jobs` is defined in config but never enforced
- **Where:** `missy/config/settings.py:193` (documented), `add_job` in `manager.py:103-196`
  performs **no count check**.
- **Problem:** The config schema advertises `max_jobs` (0 = unlimited) but `SchedulerManager`
  never consults it, so the cap has no effect.
- **Impact:** Unbounded job creation (e.g. via a compromised or buggy caller) → resource
  exhaustion; the documented safety limit is a no-op.
- **Fix:** In `add_job`, if `max_jobs > 0` and `len(self._jobs) >= max_jobs`, raise
  `SchedulerError`. Thread the config value into `SchedulerManager.__init__`.

### 2.4 🟡 Retry-job IDs are keyed by failure count and can collide / leak
- **Where:** `manager.py:417` — `retry_job_id = f"{job_id}_retry_{failures}"`.
- **Problem:** On success the retry APScheduler jobs are not removed, and re-failing at the same
  `failures` index (after a manual reset) reuses the same ID. Also these `_retry_N` IDs never map back
  into `self._jobs`, so `_run_job` on a retry that references a *removed* base job just logs and
  exits — but the orphaned APScheduler entries accumulate.
- **Fix:** Remove completed/última retry jobs from APScheduler after they fire; consider a monotonic
  suffix (timestamp) instead of the failure index.

### 2.5 🟡 `next_run` is only refreshed on success, never after a skip or failure
- **Where:** `manager.py:488-491`.
- **Problem:** `job.next_run` is updated only in the success branch. After a skipped (active-hours)
  or failed run the persisted `next_run` is stale, so `missy schedule list` shows wrong data.
- **Fix:** Update `next_run` from `self._scheduler.get_job(...)` in a `finally` block.

---

## 3. Data-consistency issues

### 3.1 🟠 API session registry and memory-store turn counts drift
- **Where:** `missy/api/server.py:142-182` (in-memory `_SessionRegistry`) vs.
  `_handle_list_sessions:588-597` (overlays DB counts).
- **Problem:** Sessions live only in process memory and are lost on restart, while conversation
  turns persist in SQLite. After a restart, `/chat` with a previously valid `session_id` returns
  404 ("Session not found") even though the history exists in the DB. Turn counts are reconciled
  only in `list`, not in `get`/`history`.
- **Impact:** Confusing client behavior; history is orphaned relative to the session lifecycle.
- **Fix:** Back the session registry with the memory store's `sessions` table (there's already a
  `register_session`/`list_sessions` API), or on 404 fall back to the DB session if turns exist.

### 3.2 🟠 `CostTracker` in-memory totals diverge from persisted per-call cost after eviction
- **Where:** `missy/agent/cost_tracker.py:174-177` and runtime persistence at `runtime.py:1935-1946`.
- **Problem:** `CostTracker` evicts old `UsageRecord`s past `_MAX_RECORDS` (totals stay correct in
  memory), but `call_count` is derived from `len(self._records)` (`:253-256`), so after eviction the
  reported call count is wrong/underreported. The persisted `record_cost` rows in the store and the
  live tracker can't be reconciled.
- **Impact:** `missy cost` under-reports call count on long sessions; budget summary `call_count`
  is misleading.
- **Fix:** Track `self._call_count` as a separate integer counter incremented on every `record()`,
  independent of the retained-records list.

### 3.3 🟠 `_save_jobs` swallows write failures — silent persistence loss
- **Where:** `manager.py:524-554`.
- **Problem:** The whole atomic-write is wrapped in `try/except Exception` that only logs. If the
  disk is full or the directory is unwritable, the in-memory job state advances (run counts,
  failure counters) but is never persisted, and the caller never learns. Combined with the retry
  logic, this can double-run jobs after a crash.
- **Fix:** Re-raise (or surface via an audit event + return status) on persistence failure for
  operations initiated by the user (`add`/`remove`/`pause`/`resume`); keep the swallow only for the
  background `_run_job` bookkeeping path.

### 3.4 🟡 FTS `search()` wraps the whole query in quotes, disabling documented operators
- **Where:** `missy/memory/sqlite_store.py:508-511`.
- **Problem:** The docstring promises FTS5 boolean/prefix operators (`"python AND async"`), but the
  sanitizer wraps the entire input in double quotes, turning it into a single literal phrase. So
  `AND`, `OR`, `*` prefix, and column filters no longer work.
- **Impact:** Documented search behavior is broken; API `/memory/search` results are phrase-only.
- **Fix:** Either update the docs to say phrase-only, or implement a proper FTS token escaper that
  preserves operators while neutralizing injection.

---

## 4. Security issues

### 4.1 🔴 Sandbox fails **open** to unsandboxed shell when Docker is unavailable
- **Where:** `missy/security/sandbox.py:350-363` (`get_sandbox`) → `FallbackSandbox.execute`
  (`:242-329`) runs `subprocess.run(command, shell=True, executable="/bin/bash")`.
- **Problem:** When `sandbox.enabled: true` but Docker isn't accessible, `get_sandbox` logs a
  warning and returns `FallbackSandbox`, which executes the command **on the host with no
  isolation** (just a scrubbed env). An operator who enabled sandboxing for safety silently loses
  it. `ShellExecTool.__init__` has the same fall-through (`shell_exec.py:98-111`).
- **Impact:** False sense of containment. A model-driven or injected command runs directly on the
  host despite `enabled: true`.
- **Fix:** Add a `require_isolation`/`strict` flag. When sandbox is enabled and Docker is
  unavailable, **refuse to execute** (return a failed `ToolResult`) rather than silently falling
  back. At minimum, make the fallback opt-in.

### 4.2 🔴 `FallbackSandbox` ignores network isolation, memory/CPU limits, and read-only root
- **Where:** `missy/security/sandbox.py:242-329`.
- **Problem:** The `SandboxConfig` promises `network_disabled`, `memory_limit`, `cpu_limit`,
  `read_only_root`. The fallback honors **none** of them — it only sanitizes env and caps timeout
  to 300s. So a config that looks locked-down provides almost none of its guarantees off-Docker.
- **Impact:** Commands have full network + full filesystem write access under a config that claims
  otherwise.
- **Fix:** Document the fallback's limits loudly, gate it behind explicit opt-in (see 4.1), and
  where possible apply `resource.setrlimit` (address space, CPU) and a Landlock/`unshare` network
  namespace in the fallback path (Missy already has `security/landlock.py`).

### 4.3 🟠 REST-policy L7 rules are only enforced inside `PolicyHTTPClient`
- **Where:** `missy/gateway/client.py:355-357` calls `_check_rest_policy` only when a `method` is
  passed. Confirm all outbound paths route through the gateway.
- **Problem:** Provider SDKs (Anthropic/OpenAI) create their **own** `httpx`/`requests` clients
  (`anthropic_provider.py:84`), bypassing `PolicyHTTPClient` entirely. So network policy + REST
  policy + response-size caps + audit events do **not** apply to the largest volume of outbound
  traffic. The CLAUDE.md claim that the gateway is "the single enforcement point for ALL outbound
  HTTP" is not true for provider calls.
- **Impact:** Egress controls and audit are blind to provider traffic; a compromised base_url or
  proxy setting can exfiltrate without policy review.
- **Fix:** Pass a policy-aware `http_client` into the provider SDKs (both Anthropic and OpenAI SDKs
  accept a custom `http_client=`), or wrap them so requests transit the gateway.

### 4.4 🟠 Scheduled-job tasks are sanitized but still executed even when injection is detected
- **Where:** `manager.py:379-394`.
- **Problem:** `_run_job` runs `InputSanitizer.check_for_injection(job.task)` and, on a hit, only
  **logs a warning** — then proceeds to run the task. A tampered `jobs.json` (or a job added via a
  future API) with injected instructions executes anyway.
- **Impact:** Detection without enforcement; defense-in-depth is only cosmetic here.
- **Fix:** On injection detection, emit a `security` audit event and either skip the run or require
  an approval gate (`ApprovalGate`) before executing.

### 4.5 🟠 Vault store contents are not integrity-checked against key-file swap
- **Where:** `missy/security/vault.py:55-95`.
- **Problem:** The code guards the *key file* against symlink/hard-link/permission attacks (good),
  but if `vault.enc` is replaced with ciphertext encrypted under a **different** key that an
  attacker also writes to `vault.key` (e.g. after deleting both), `_load_or_create_key` will happily
  mint/accept it. There's no per-install binding (e.g. AAD tying ciphertext to a machine/user id).
- **Impact:** Limited (attacker with write access to `~/.missy/secrets` already has a lot), but the
  AEAD `associated_data` parameter is unused (`_encrypt`/`_decrypt` pass `None`).
- **Fix:** Pass stable AAD (e.g. `b"missy-vault-v1:" + uid`) to ChaCha20Poly1305 so cross-context
  ciphertext substitution fails authentication.

### 4.6 🟡 API server binds a background thread with no request timeout / slowloris protection
- **Where:** `missy/api/server.py:807-813` — plain `HTTPServer` + `BaseHTTPRequestHandler`.
- **Problem:** `HTTPServer` has no socket read timeout, so a client that opens a connection and
  sends bytes slowly ties up the single-threaded server (it's not even `ThreadingHTTPServer`),
  blocking all API clients.
- **Impact:** Trivial local DoS of the API; also a slow `Content-Length` body stalls
  `_read_body`'s `rfile.read(length)`.
- **Fix:** Set `handler.timeout`, use `ThreadingHTTPServer`, and read the body with a deadline.

### 4.7 🟡 API `/chat` mutates shared runtime provider globally per-request
- **Where:** `api/server.py:518-522` — `runtime.switch_provider(provider_override)`.
- **Problem:** `switch_provider` changes the runtime's active provider process-wide. Two concurrent
  `/chat` requests with different `provider` values race, and one request's override leaks into the
  other's run.
- **Fix:** Pass the provider per-run (as a `run()` argument) rather than mutating shared state, or
  serialize/scope provider selection per session.

---

## 5. Implementation gaps / correctness

### 5.1 🟠 `cleanup_memory` constructs a throwaway `MemoryStore()` with default path
- **Where:** `manager.py:302-326`.
- **Problem:** It imports `missy.memory.store.MemoryStore` and instantiates a fresh one rather than
  using the runtime's configured store. If the DB path is customized, cleanup targets the wrong
  (default) database and silently deletes nothing / the wrong data.
- **Fix:** Inject the actual store instance into `SchedulerManager`.

### 5.2 🟠 Response-size cap in gateway buffers the whole body before checking (chunked path)
- **Where:** `missy/gateway/client.py:465-476`.
- **Problem:** When there's no `Content-Length`, the fallback does `len(response.content)`, which
  forces httpx to buffer the **entire** body into memory first — defeating the purpose of the
  50 MB guard against memory exhaustion.
- **Fix:** Use streaming (`client.stream(...)`) and abort once bytes exceed the cap, or cap via
  httpx transport limits.

### 5.3 🟡 `ModelRouter.score_complexity` treats history length as a raw turn count
- **Where:** `missy/providers/registry.py:257-280`.
- **Problem:** `history_length > 10` forces "premium" regardless of prompt simplicity, so any long
  chat silently escalates to the most expensive tier — a cost surprise not reflected in budget
  planning.
- **Fix:** Make thresholds configurable and factor in token estimate, not turn count alone.

### 5.4 🟡 `_extract_all_programs` blanket-rejects `<<` but the tool advertises redirection
- **Where:** `missy/policy/shell.py:135` (`_SUBSHELL_MARKERS` includes `"<<"`) vs.
  `shell_exec.py` description that promotes pipes/redirection.
- **Problem:** Any command containing `<<` (including the substring in some legitimate args) is
  rejected. This is intentional for heredocs but the matcher is a plain substring test, so
  `echo "a<<b"` is also refused. Minor UX/correctness mismatch.
- **Fix:** Tokenize before rejecting, or scope the check to shell-operator positions.

---

## 6. New features

### 6.1 🟢 Feature: Provider-tier & rate-limit configuration surface (`providers.*.limits`)
- **Motivation:** Ties together items 1.1–1.3. Today limits are hard-coded and TPM is unenforced.
- **Proposal:** Add a `limits` block to `ProviderConfig`:
  ```yaml
  providers:
    anthropic:
      limits:
        requests_per_minute: 50
        tokens_per_minute: 40000
        max_wait_seconds: 30
  ```
  Wire into `ProviderRegistry.from_config` to build a correctly-sized `RateLimiter`, add
  `missy providers limits` CLI to display current buckets (`request_capacity`/`token_capacity`),
  and estimate prompt tokens before `acquire()` so TPM is actually honored. Emit an audit event
  when a request blocks on the limiter so operators can see throttling.
- **Value:** Turns the existing but inert rate limiter into a real cost/throughput control.

### 6.2 🟢 Feature: Egress audit report (`missy audit egress`) + universal gateway routing
- **Motivation:** Item 4.3 — provider SDKs bypass the gateway, so there's no single egress ledger.
- **Proposal:**
  1. Inject a policy-aware `httpx` client into every provider SDK so all outbound HTTP flows
     through `PolicyHTTPClient` (network + REST policy + size cap + `network_request` audit events).
  2. Add `missy audit egress [--since] [--host]` that aggregates `network_request` audit events
     into a per-host / per-method summary (counts, bytes, denied vs allowed), reading from
     `~/.missy/audit.jsonl`.
- **Value:** Delivers on the "single enforcement point for ALL outbound HTTP" promise and gives
  operators a real egress dashboard for a security-first product.

### 6.3 🟢 (Optional) Feature: Persistent, restart-safe API sessions
- **Motivation:** Item 3.1 — API sessions are memory-only and orphan their DB history on restart.
- **Proposal:** Back `_SessionRegistry` with the memory store's `sessions` table (load on startup,
  write-through on create/touch/delete). Add a `last_provider` column so `/chat` can resume with the
  correct provider after a restart.
- **Value:** Makes the Agent-as-a-Service API durable and consistent with persisted history.

---

## Suggested priority order for Opus

1. **4.1 / 4.2** sandbox fail-open (security, high) — refuse-on-no-Docker + document fallback.
2. **1.1 / 1.2** TPM not enforced + limits unconfigurable (rate-limit, high).
3. **2.1** retries dropped outside active-hours (scheduling reliability, high).
4. **4.3** provider traffic bypasses gateway (security/audit, high) → enables 6.2.
5. **3.1 / 3.3** session drift + silent job-persistence loss (data consistency).
6. Remaining 🟠/🟡 items and features 6.1–6.3.

*Note:* several fixes have existing tests under `tests/` (policy, scheduler, providers,
memory). Update/extend those suites alongside each change; coverage threshold is 90%.
