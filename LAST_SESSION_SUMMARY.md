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
5. **SR-1.12 (critical)**: auditing for the same bug class found an even
   more directly exploitable instance in Discord's DM pairing flow —
   `!pair accept <target_id>` was processed with zero check on who sent
   it, letting any unpaired stranger self-approve their own pairing with
   two DM messages. Closed: in-band accept/deny commands are now
   unconditionally refused and audited; `accept_pair()`/`deny_pair()`
   remain the only legitimate path (not yet wired to an authenticated
   operator surface — tracked as follow-up, task #12).
6. **FX-D**: added an explicit structural boundary marker to the
   flattened acpx prompt (previously only described in prose) so the
   delegate cannot mistake prior history/tool-results for something it
   should keep responding to. Made `complete()`/`complete_with_tools()`
   fail closed (raise `ProviderError`) rather than silently returning an
   empty "successful" response when a leaked transcript marker strips
   away the entire delegate output. ~20 new regression tests covering
   boundary tracking, quoted transcript safety, long history, malicious
   history instructions, and full DISC-CMD-006 + report-followup
   end-to-end reproductions.

## Verification

```text
python3 -m pytest tests/providers/test_acpx_provider.py -q
136 passed
```

```text
python3 -m pytest tests/ -q -o faulthandler_timeout=120 \
  --deselect tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl \
  --deselect tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped \
  --deselect tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl
20703 passed, 13 skipped, 3 deselected in 456.40s (0:07:36)
```

Full detail in `BUILD_STATUS.md`, `AUDIT_SECURITY.md`, and
`TEST_RESULTS.md`.

## Remains

- FX-A bullet 6: live end-to-end proof across tool categories (not yet
  attempted).
- FX-B loose ends (see prior session summary, preserved in git history).
- SR-1.2/1.3 and SR-1.12 are **not fully solved** — both fixes close live
  bypass paths but neither is the complete remediation the security
  review asks for. `CodeEvolutionManager.approve()`/`apply()`/
  `rollback()` still perform zero authentication themselves (CLI
  terminal session is the only real trust boundary). Discord pairing has
  no working approval path at all right now (task #12).
- FX-C, FX-F, FX-G not yet started.
- SR-1.1, SR-1.4 through SR-1.11, SR-1.13, SR-2.x through SR-4.x not yet
  started.
- Full 89-case tool-specific validation backlog not yet re-run.
- Pre-existing vision `CameraDiscovery` cache-TTL flake (3 tests,
  tracked separately, unrelated to this session).

## First Next Step

Continue the SR-1.x sweep for the same bug class (unauthenticated
privileged action reachable from the agent's own tool loop or an
unauthenticated channel) — SR-1.5 (Incus policy composition), SR-1.6
(browser network enforcement), and SR-1.13 (uniform Discord ingress
authorization) are the most likely remaining candidates given this
session's findings (two independent instances of the identical pattern
already found and fixed). Alternatively, wire the Discord pairing
approval endpoint (task #12) to restore working pairing functionality,
or move to FX-C (grounding factual state claims in fresh tool evidence)
per the prompt's stated dependency order.
