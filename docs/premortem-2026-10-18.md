# Missy premortem: 2026-10-18

**Premise:** it is April 2027 and Missy has caused a serious incident or been
switched off. This document works backward from that outcome to the causes
already present in the code and in this deployment. Each item cites evidence
from `master` at `513a0cc` and the live install, gives early warning signs, and
proposes the smallest mitigation. Nothing was changed while writing it.

**Method:** four parallel reviews (security boundary, agent loop and providers,
state and persistence, operability and maintainability), followed by direct
verification of every item marked ✅. Items marked ◐ are supported by the code
but were not reproduced end to end. The security review was cut short by an
automated filter, so its section rests only on facts verified directly, and it
deliberately describes risks rather than attack steps.

## Summary

| # | Failure | Likelihood | Impact | Status |
|---|---|---|---|---|
| 1 | Unrestricted host shell for anyone in the allowed Discord channels | High | Critical | ✅ config |
| 2 | Checkpoint database already corrupt, and nothing alarms | Happening | High | ✅ live |
| 3 | Anti-hedging guard can repeat actions after "ok thanks" (introduced today) | High | Critical | ✅ |
| 4 | Promise/fabrication guards replace ordinary answers with a canned refusal | High | High | ✅ |
| 5 | Default provider reports zero cost, so budget caps cannot trip | High | High | ✅ |
| 6 | Discord can die permanently while the process looks healthy | Medium | Critical | ✅ |
| 7 | Optional subsystems fail at startup silently | Happened | High | ✅ |
| 8 | Production runs from the development working tree | High | High | ✅ |
| 9 | Test runs write into the live `~/.missy` | High | High | ◐ |
| 10 | Worst-case provider calls per turn are effectively unbounded | Medium | High | ◐ |
| 11 | Device policy changes from the CLI are lost | Medium | High | ◐ |
| 12 | Audit window shrinks to weeks of heartbeat noise | High | Medium | ◐ |
| 13 | Maintainability: god-modules, drifted CLAUDE.md, brittle tests, unpinned deps | High | High | ✅ |

Items 2 and 3 are worth acting on now; the rest are ranked for planning.

---

## 1. Unrestricted host shell for anyone in the allowed Discord channels ✅

**Story.** Several opt-ins that are each reasonable on their own combine into
one chain. Any member of the five allowed channels (and any other bot) can ask
Missy to run commands, and nothing between the message and the host narrows it.

**Evidence (verified against `~/.missy/config.yaml` and the host):**
- `shell.enabled: true`, `shell.unrestricted: true`, no `allowed_commands`:
  every non-empty command is allowed without allow-list matching.
- Both guild policies (5 channels total) have no `allowed_users`,
  `allowed_role_ids`, or `allowed_roles`; `ignore_bots: false`.
- `desktop.allow_software_install` and `auto_approve_software_install` are both
  `true`; `discord.auto_approve_uploads: true`.
- `network.default_deny: false`.
- The `missy` user is in the `docker`, `incus-admin`, and `admin` groups, so
  host shell access is effectively root-equivalent.

**Likelihood / impact.** High / critical: the only remaining controls are the
model's own judgment and the prompt-injection scanner.

**Early warning.** `shell_exec` calls from authors who are not the operator;
`install_software_confirmed` audit events; uploads of unexpected paths.

**Mitigation.** Add `allowed_users` (operator IDs) or `allowed_role_ids` to
both guild policies and set `ignore_bots: true` outside validation runs. Replace
`shell.unrestricted` with an allow-list, or gate shell behind the approval gate
for non-operator authors. Turn `network.default_deny` back on.

## 2. Checkpoint database already corrupt, and nothing alarms ✅ (live)

**Story.** `~/.missy/checkpoints.db` became corrupt around 2026-09-28. Every run
since has silently proceeded without checkpoints, `missy recover` has nothing to
resume, and the daily retention job reports success while failing.

**Evidence.**
- Read-only `PRAGMA quick_check`: `Tree 3 page 48742: btreeInitPage() returns
  error code 11`. The file is 245 MB with 52,644 of 59,878 pages free (88%
  dead); nothing ever runs `VACUUM`.
- Audit log: `maintenance.retention` records `"checkpoints": "error: database
  disk image is malformed"` daily, yet publishes `result: "allow"`
  (`missy/scheduler/maintenance.py:84-90`, `:190`).
- `missy/agent/runtime.py:1852-1863` logs checkpoint init failure at DEBUG.
- `missy doctor` checks only that the file exists (`missy/cli/main.py:4471-4475`).
- Likely trigger ◐: `missy/agent/checkpoint.py:161-163` opens and closes the
  live database file on every run (`CheckpointManager` is built per run). On
  Linux, closing any descriptor releases every POSIX lock the process holds on
  that file, including SQLite's, so concurrent connections in the same process
  lose their locks. `missy/memory/sqlite_store.py:297-299` uses the same
  pattern on `memory.db` (still `ok` today).

**Mitigation.** Only create the file when it does not exist; build
`CheckpointManager` once per runtime; make retention publish `result="error"`
when any step fails; run `PRAGMA quick_check` in `missy doctor` and the
watchdog for every SQLite store; quarantine and recreate a malformed
`checkpoints.db`. Back up `memory.db` now.

## 3. Anti-hedging guard can repeat actions after "ok thanks" ✅ (introduced today)

**Story.** `_AFFIRMATION_REQUEST` (added in `8b9e007`) treats "ok thanks" and
"sounds good, thank you" as approval of a pending action. If Missy closes with
"Let me know if you'd like me to do anything else", the guard injects a retry
telling her to "call the appropriate tool now", which can repeat the previous
turn's side effect.

**Evidence.** `detect_hedging("You're welcome! Let me know if you'd like me to
do anything else.", "ok thanks", [])` returns `True`. Separately (pre-existing),
it also returns `True` for "This will permanently delete 412 files. Do you want
me to proceed?" after "delete everything in ~/workspace/old", and for "Should I
also restart nginx after?" after "check the nginx config". The retry path emits
no audit event.

**Mitigation.** Exclude thanks/closing phrases from the affirmation pattern and
apply it only when the previous assistant turn proposed a specific action. Never
treat a confirmation question before a destructive action as hedging. Emit an
audit event on every hedging retry.

## 4. Promise/fabrication guards replace ordinary answers ✅

**Story.** Every Discord turn runs through `_tool_loop`. The promise and
fabrication detectors run on every final answer regardless of whether an action
was requested. Ordinary phrasing triggers a retry, and a second match replaces
the answer with `_UNVERIFIED_COMPLETION_RESPONSE`, which is then saved to the
session history.

**Evidence.** Verified `True` with no tools used: "Sure! I'll explain how DNS
works…", "I'm assuming you want metric units…" (promise); "I reviewed the code
you pasted…", "I verified the math: 17*3 = 51" (fabrication). Patterns at
`missy/agent/response_guards.py:228-233` and `:43-80`; overwrites at
`missy/agent/runtime.py:3369` and `:3437`. Today's "Running tally" fix narrowed
only the leading-gerund pattern.

**Mitigation.** Run both guards only when the request is an action request (as
`detect_hedging` already checks). On exhaustion, keep the model's answer with a
short caveat rather than substituting canned text.

## 5. Default provider reports zero cost ✅

**Story.** `CodexProvider` returns zero token usage on every response, so cost
tracking records $0 and neither `max_spend_usd` nor `GlobalBudget` can trip for
the default provider. The token-per-minute limiter sees nothing. The only real
limit is the upstream subscription quota, which ran out on 2026-10-18.

**Evidence.** `missy/providers/codex_provider.py:636` and `:833`:
`usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}`. The
live config sets no spend cap and `max_iterations: 25`.

**Mitigation.** Read usage from the stream's completion event (not yet checked
whether it is present) or estimate from characters, and add a per-turn cap on
provider calls that works regardless of pricing data.

## 6. Discord can die permanently while the process looks healthy ✅

**Story.** A non-reconnectable close (rotated token, disallowed intents) stops
the Discord gateway, but the process keeps running, so systemd never restarts it
and the primary channel stays dark.

**Evidence.** `missy/channels/discord/gateway.py:365-372` sets
`self._running = False` and breaks. `channel.py:255` starts the gateway task with
no done-callback, and nothing in the supervisor loop checks it. The unit has
`Restart=on-failure` and no `WatchdogSec`. Diagnostics check configuration, not
connection state.

**Mitigation.** Add a done-callback on the gateway task that exits non-zero (or
marks the channel failed) on a fatal close; expose connection status in
`/diagnostics` and `missy gateway status`.

## 7. Optional subsystems fail at startup silently ✅ (happened today)

**Story.** A hardening change alters a subsystem's preconditions; on restart it
raises, `gateway_start` logs a warning and continues, and systemd reports
`active`. Today the voice channel refused to start (plaintext non-loopback bind)
from 08:32 until a manual restart at 09:22.

**Evidence.** Proactive, scheduler, heartbeat, voice, screencast, web console,
and Landlock startup each sit in their own `except Exception` in
`missy/cli/main.py` (88 such handlers in that file). The watchdog checks only
provider registry, memory store, and MCP servers. `missy gateway status`
re-reads config; it does not report whether subsystems are running.

**Mitigation.** Record each subsystem's start result in a registry surfaced by
`/diagnostics` and `missy gateway status`; register started channels with the
watchdog; offer a strict mode where an explicitly enabled channel failing to
start is fatal.

## 8. Production runs from the development working tree ✅

**Story.** The gateway imports `/home/missy/missy/missy` (editable install), so
whatever branch is checked out becomes production on the next restart.
Separately, the documented ComfyUI topology is not real.

**Evidence.** `import missy` resolves to the working tree. The running gateway
has no `MISSY_*` environment variables, so `video_generate` never targets the
documented `10.0.0.230` RTX 5080 and always uses the local 3070.

**Mitigation.** Deploy from a tagged install in a separate venv or worktree; put
environment in an `EnvironmentFile=` for the unit; have `missy doctor` print the
code path, git SHA, and ComfyUI target.

## 9. Test runs write into the live `~/.missy` ◐

**Story.** Tests that build real stores run on the same machine as the gateway
and open production databases and logs, giving a second process for item 2's
lock problem and polluting forensic records.

**Evidence.** `~/.missy/audit.log` (the default path, distinct from the
configured `audit.jsonl`) contains test fixtures such as `/tmp/does-not-exist.jpg`;
`~/.missy/missy.log` contains `/tmp/pytest-of-missy/...` paths. There is no
root `tests/conftest.py` isolating `HOME`.

**Mitigation.** Add an autouse fixture in `tests/conftest.py` that points `HOME`
(or a `MISSY_HOME`) at `tmp_path`.

## 10. Worst-case provider calls per turn are effectively unbounded ◐

**Story.** Guard retries continue the loop. Counting 16 retry paths that consume
iterations, an 18-iteration grace budget, and a finalization call, one top-level
run can make up to about 44 provider calls at `max_iterations: 25`. Each
`delegate_task` can start up to 10 sub-runs, nested two deep, each with its own
loop. There is no wall-clock deadline in `_tool_loop`.

**Evidence.** `missy/agent/runtime.py:2038-2048` (grace budget), `:3611`
(finalization), `missy/agent/sub_agent.py` limits. Independent per-guard
counters also mean a correct second attempt can trip a different guard; nine
sites overwrite `final_text`, and later guards can run after an overwrite.

**Mitigation.** One shared guard-retry budget per turn (for example 3), a
per-turn wall-clock deadline, a cap on sub-runs per top-level task, and return
immediately once `final_text` has been overwritten.

## 11. Device policy changes from the CLI are lost ◐

**Story.** `missy devices policy <id> --mode muted` writes `devices.json`, but
the gateway's in-memory registry never reloads it, and the node's next heartbeat
saves the stale copy back over the file.

**Evidence.** `missy/channels/voice/registry.py:291-314` and `:511-546` (every
heartbeat path saves); the server's per-message policy re-fetch
(`server.py:473`) is an in-memory lookup. CLAUDE.md's claim that a CLI mute
applies on the next message holds only within one process.

**Mitigation.** Use the `jobs.json` approach: `flock` plus reload-on-mtime-change
before every read and save, or route device changes through the gateway API.

## 12. Audit window shrinks to weeks of heartbeat noise ◐

**Story.** Rotation is bounded (5 × 50 MB), but most volume is routine
heartbeats, so security events age out after a few weeks.

**Evidence.** In the active `audit.jsonl` (about 31 MB), `watchdog.health_check`
accounts for about 13 MB and Discord heartbeats about 12 MB. The oldest retained
rotation dates from 2026-09-15.

**Mitigation.** Audit watchdog and heartbeat events only on state change, or
sample them; optionally route security events to a separate log.

## 13. Maintainability ✅

- **God-modules.** `missy/cli/main.py` is 8,445 lines and `missy/agent/runtime.py`
  6,737; `gateway_start` spans about 1,000 lines. No test starts the gateway end
  to end, and startup ordering is tested by searching source text. Of the last
  101 commits, 47 begin with "fix". *Mitigation:* extract a gateway supervisor
  with an ordered `(name, start, stop, required)` subsystem list.
- **CLAUDE.md drift.** 86 KB of accreted history with stale claims ("250+
  injection patterns"; the real count is 101) and the false ComfyUI claim.
  *Mitigation:* reduce it to an architecture map, move rationale to `docs/adr/`.
- **Brittle tests.** 279 real-clock `time.sleep` calls, exact pattern-count
  assertions, and `filterwarnings` that hide never-awaited coroutines; CI runs
  with `-x` and no `pytest-timeout`. Two load-dependent flakes were seen today.
  *Mitigation:* injectable clocks, `>=` count assertions, `pytest-timeout`.
- **Unpinned dependencies.** Core dependencies are floor-only with no lockfile,
  and CI installs an unpinned `ruff`. *Mitigation:* a lockfile used by both CI
  and deployment; pin `ruff`.

## Checked and found solid

- Discord gateway reconnect uses bounded exponential backoff with jitter; the
  REST governor and shutdown path behave as documented.
- Audit rotation is bounded, writes are serialized under `_chain_lock`, and a
  scan of all active lines found no hash-chain breaks. `jobs.json`,
  `patches.json`, `trust.json`, and `playbook.json` use `flock` plus atomic
  replace.
- Context fitting never drops the current request; tool exceptions stay inside
  `ToolResult`; the budget is checked before every iteration and fallback.
- `memory.db` passes `quick_check`.

## Not verified

The definitive trigger of the checkpoint corruption; whether the Codex stream
exposes token usage; OBS/VTube/MCP restart behavior; screencast lifecycle; and
how often the identity, capability-denial, and security-refusal guards produce
false positives.
