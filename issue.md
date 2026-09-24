# Incident: background SleeptimeWorkers exhausted Claude/ACPX usage

## Summary

On September 17, 2026, Missy consumed the operator's Claude/ACPX allowance
without a corresponding user request. The consumption came from background
memory summarization, not an unknown external caller.

Multiple `SleeptimeWorker` threads repeatedly called the ACPX provider against
the same shared conversation store. Successful calls immediately after Claude
quota resets consumed the renewed allowance; the workers then continued making
thousands of rejected calls while the account was rate-limited.

The bot was stopped at approximately 08:16 EDT on September 17. No further
Missy ACPX audit events or ACPX sandbox session files were created after the
shutdown. Restarting the bot without a fix may recreate the problem.

## Severity

High. The defect causes autonomous provider usage, rapidly exhausts a paid or
subscription-backed quota, bypasses the expected per-runtime provider choice,
and creates a large retry storm with incomplete cost accounting.

## Confirmed impact

From 00:00 through the 08:16 EDT shutdown on September 17:

- 1,314 successful ACPX/Claude calls were recorded.
- 24,624 additional ACPX calls failed after Claude reported that the usage
  limit had been reached.
- All calls had empty user session and task scope, consistent with
  `SleeptimeWorker._llm_summarize()` calling the provider directly rather than
  a user-triggered agent run.
- Successful calls appeared in two bursts immediately after quota resets:
  - Approximately 653 calls from 01:50 through 02:07 EDT.
  - Approximately 661 calls from 06:50 through 07:07 EDT.
- Claude session records contained usage metadata for 1,268 of the successful
  calls:
  - 3,757 uncached input tokens
  - 19,379 output tokens
  - 126,034 cache-creation input tokens
  - 12,735,252 cache-read input tokens
  - Approximately 12.88 million recorded token operations in total
- The other 46 successful calls did not contain usage metadata, so the token
  total is a lower bound. Claude may weight cached tokens differently for
  subscription usage.
- 25,938 Claude session files were created that day under Missy's isolated
  ACPX sandbox. None were created after the bot stopped.

## Timeline

All times below are America/New_York.

- 00:53: A separate scheduled OpenAI Codex job ran. This affected Codex usage,
  not Claude usage.
- 01:50: Claude allowance reset; background ACPX calls began succeeding.
- 02:07: The renewed allowance was exhausted and calls returned to failures.
- 04:53: The scheduled OpenAI Codex job ran again.
- 06:50: Claude allowance reset again; another background success burst began.
- 07:07: The allowance was exhausted again.
- 08:15: The last ACPX attempt and sandbox session file were recorded.
- 08:16: `missy-gateway.service` stopped successfully.

## Root cause

Several defects compound into the incident.

### 1. Every runtime automatically starts a background worker

`AgentRuntime` constructs and starts a `SleeptimeWorker` for every runtime,
including short-lived or task-specific runtimes:

- `missy/agent/runtime.py:822`
- `missy/agent/runtime.py:5163`

There is no ownership model ensuring only one memory worker exists for a
shared memory store or gateway process.

### 2. Scheduled executions leak their runtime and worker

Every scheduler execution creates a new `AgentRuntime`:

- `missy/scheduler/manager.py:604`
- `missy/scheduler/manager.py:612`

The scheduler does not call `agent.shutdown()` on either success or failure.
The runtime's daemon `SleeptimeWorker` therefore remains alive after the job
finishes. The enabled four-hour job had reached 90 completed runs, allowing
workers to accumulate across the gateway's lifetime.

The approximately 40–44 successful calls per minute during reset windows is
consistent with dozens of leaked workers operating concurrently.

### 3. Background summarization ignores the runtime provider

`SleeptimeWorker._llm_summarize()` sorts all registered provider names and
selects the first provider that reports itself available:

- `missy/agent/sleeptime.py:646`
- `missy/agent/sleeptime.py:663`
- `missy/agent/sleeptime.py:686`

Because provider names are sorted alphabetically and ACPX was enabled, `acpx`
was selected even though the default and scheduled-job provider was
`openai-codex`.

This makes background-provider selection implicit and unrelated to the
operator's requested provider for the runtime.

### 4. Workers independently process the same shared sessions

Each leaked worker queries the same SQLite memory store and independently
identifies sessions requiring summarization. There is no cross-worker claim,
lease, or single-flight mechanism preventing duplicate processing of the same
session batch.

### 5. Provider failures do not stop the processing cycle

When an ACPX call fails with a quota-limit response,
`_llm_summarize()` returns `None`. The worker then continues through other
sessions and later cycles. There is no provider-wide cooldown, circuit breaker,
or "abort this cycle after quota exhaustion" behavior.

This produced 24,624 rejected attempts in approximately eight hours.

### 6. Plain ACPX completion bypasses the configured rate limiter

`AcpxProvider.complete_with_tools()` acquires the shared provider rate limiter,
but the plain `AcpxProvider.complete()` path used by SleeptimeWorker does not.
Consequently, concurrent workers can invoke ACPX without the configured RPM or
TPM constraint.

### 7. Background calls bypass normal cost accounting

SleeptimeWorker calls `provider.complete()` directly instead of going through
`AgentRuntime`'s normal cost-recording path. ACPX also returns zero token usage
in its `CompletionResponse` even when Claude's local session record contains
usage. The SQLite cost table therefore did not record these Claude calls or
their actual usage.

## Other possible sources investigated

The host audit found no evidence that another automated service caused this
morning's Claude consumption:

- No cron entry, systemd timer, container, or other ACPX/Anthropic process was
  found.
- One standalone `claude resume --allow-dangerously-skip-permissions` process
  has existed since September 1. It was idle during the investigation, showed
  negligible CPU/I/O change, and its session data contained no new API turn on
  September 17.
- All new Claude session files were under Missy's ACPX sandbox.
- ACPX activity stopped exactly with `missy-gateway.service`.

The scheduled "EveMarket Goblin Radar actual work" job did run at 00:53 and
04:53 EDT and made 23 OpenAI Codex completions. Those calls are separate from
the Claude incident but demonstrate that scheduled background work was active.

## Required remediation

1. Wrap scheduler-created runtimes in `try/finally` and always call
   `agent.shutdown()`.
2. Do not start SleeptimeWorker automatically for ephemeral, delegated,
   scheduler, benchmark, or other task-specific runtimes.
3. Enforce one SleeptimeWorker owner per process/shared memory store, or add a
   database-backed lease that prevents duplicate workers.
4. Make background summarization opt-in and explicitly configure its provider.
   Never choose the first provider alphabetically.
5. Default background summarization to a local or explicitly budgeted provider,
   or use deterministic keyword summarization when no worker provider is
   configured.
6. Abort the remaining summarization cycle after quota/rate-limit/auth failures
   and apply exponential backoff with a provider-wide circuit breaker.
7. Apply the ACPX rate limiter to plain `complete()` calls as well as
   `complete_with_tools()`.
8. Route background completions through normal session budget and audit
   accounting.
9. Parse ACPX/Claude usage metadata so successful calls do not report zero
   tokens.
10. Add an operator-visible kill switch for SleeptimeWorker and expose worker
    count, provider, recent calls, failures, and next retry in diagnostics.

## Acceptance criteria

- A scheduled job can run repeatedly without increasing the number of live
  `missy-sleeptime` threads.
- Scheduler runtimes call `shutdown()` after both success and failure.
- At most one worker processes a given memory store/session batch at a time.
- Background processing makes zero provider calls unless explicitly enabled
  with a configured provider.
- A quota-limit response causes no further calls until a bounded backoff has
  expired.
- Plain ACPX completion respects RPM and TPM limits.
- Background ACPX calls have non-empty audit scope and appear in cost/usage
  reports.
- Tests reproduce the historical conditions with dozens of scheduler runs and
  prove there is no worker/thread leak or duplicate provider fan-out.
- Stopping the gateway leaves no SleeptimeWorker or ACPX child process alive.

## Current containment

`missy-gateway.service` was stopped and verified inactive. No ACPX audit event
or ACPX sandbox session file appeared after shutdown. Keep the gateway stopped
until the worker lifecycle, provider selection, and failure-backoff defects are
fixed or background LLM summarization is explicitly disabled.
