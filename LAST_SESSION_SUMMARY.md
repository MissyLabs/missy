# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed

1. Preserved and hardened the existing `voice_commands.py` fix, adding
   regression tests that caught and fixed a real trailing-comma leak bug.
2. **FX-A** (first slice): forced the acpx delegate provider through
   Missy's structured tool protocol — zero native tools, fail-closed
   permissions, isolated sandbox cwd, config-can't-override sanitization,
   a versioned delegation envelope, and defensive stripping of leaked
   transcript markers.
3. **FX-B** (root cause fixed): `AgentRuntime._make_memory_store()` was
   using the wrong memory backend (JSON file instead of SQLite),
   explaining "only 3 rows despite 937 agent.run.start events." Fixed
   the store class, the `_save_turn()` call convention, made the user
   turn persist before the provider call, and found/fixed the identical
   bug independently in `VisionMemoryBridge`.
4. **FX-E + SR-1.2/1.3 (critical)**: found and fixed a live,
   unauthenticated code-self-approval vulnerability, not just the
   narrower "don't suggest bypasses" symptom FX-E describes. Missy's own
   default system prompt instructed the agent to approve and apply its
   own code changes via `code_evolve`, and `CodeEvolutionManager.approve()`/
   `apply()`/`rollback()` perform zero authentication — they trusted any
   caller. A second, independent instance of the same flaw existed in
   Discord's ✅/❌ reaction-based evolution approval handler, which let
   *any* Discord user approve a code change with no admin/owner check at
   all. Fixed both: the agent-facing `code_evolve` tool no longer exposes
   approve/apply/rollback (refused before `CodeEvolutionManager` is even
   constructed); the default system prompt no longer instructs
   self-approval and gained an explicit "never bypass a gate" rule;
   Discord's approve-reaction path now refuses and emits a deny audit
   event instead of calling `mgr.approve()`. The legitimate path (`missy
   evolve approve/apply/rollback` CLI, requiring a real terminal session
   on the host) is untouched.

## Verification

```text
python3 -m pytest tests/tools/ tests/agent/ tests/channels/ tests/security/ -q
9530 passed, 6 skipped
```

```text
python3 -m pytest tests/ -q -o faulthandler_timeout=120 \
  --deselect tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl \
  --deselect tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped \
  --deselect tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl
20686 passed, 13 skipped, 3 deselected in 447.60s (0:07:27)
```

Full detail in `BUILD_STATUS.md` and `TEST_RESULTS.md`.

## Remains

- FX-A bullet 6: live end-to-end proof across tool categories (not yet
  attempted).
- FX-B loose ends (see prior session summary, preserved in git history).
- SR-1.2/1.3 is **not fully solved** — this session closed the two live
  bypass paths (agent tool, Discord reactions) but
  `CodeEvolutionManager.approve()`/`apply()`/`rollback()` still perform
  zero authentication themselves. A genuinely "unforgeable,
  proposal-bound, expiring approval artifact" and disposable-sandbox
  validation before promotion are still missing. The CLI's terminal
  session is the only real trust boundary today.
- FX-C, FX-D (remaining, beyond what the FX-A envelope covers), FX-F,
  FX-G not yet started.
- SR-1.1, SR-1.4 through SR-4.8 not yet started.
- Full 89-case tool-specific validation backlog not yet re-run.
- Pre-existing vision `CameraDiscovery` cache-TTL flake (3 tests,
  tracked separately, unrelated to this session).

## First Next Step

Given the severity of what FX-E investigation turned up, do a focused
sweep of the remaining SR-1.x findings for the same class of bug
(unauthenticated privileged action reachable from the agent's own tool
loop or from an unauthenticated channel like Discord) before moving on
to FX-C/F/G — SR-1.5 (Incus policy composition), SR-1.6 (browser network
enforcement), SR-1.12 (Discord pairing self-approval — the pairing
handler's "admin only — simplified" comment in `channel.py` is worth
checking immediately, it may have the identical flaw), and SR-1.13
(uniform Discord ingress authorization) are the most likely candidates
based on this session's findings.
