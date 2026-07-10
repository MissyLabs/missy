# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed

1. Preserved and hardened the existing `voice_commands.py` fix, adding
   regression tests that caught and fixed a real trailing-comma leak bug
   in channel-name parsing.
2. **FX-A** (first slice): forced the acpx delegate provider through
   Missy's structured tool protocol — zero native tools
   (`--allowed-tools ""`, verified against the actual pinned
   `acpx@0.3.1` source), fail-closed permissions
   (`--non-interactive-permissions deny`), a fail-closed `is_available()`
   health check, an isolated sandbox cwd, config-can't-override
   sanitization of security flags, a versioned delegation envelope, and
   defensive stripping of leaked transcript markers (fixes the exact
   `DISC-CMD-006` failure shape).
3. **FX-B** (root cause fixed): `AgentRuntime._make_memory_store()` was
   constructing the JSON-file-backed `MemoryStore` (`~/.missy/memory.json`)
   instead of the SQLite backend (`~/.missy/memory.db`) that every other
   production consumer (memory tools, compaction, large-content
   retrieval, hatching, vision memory, this validation harness) already
   assumes. This is the direct explanation for "only 3 rows in
   memory.db despite 937 agent.run.start events." Fixed to use
   `SQLiteMemoryStore`, fixed `_save_turn()`'s call convention (object,
   not kwargs), made the user turn persist *before* the provider call
   (so a crashing delegate no longer erases the incoming request),
   upgraded silent-swallow logging to a warning + new
   `memory.persist_failed` audit event, and found/fixed the identical
   bug independently in `VisionMemoryBridge.store_observation()` (vision
   observations were never actually being persisted either). Added a
   real integration test (`tests/integration/test_discord_memory_persistence.py`)
   driving `AgentRuntime.run()` the same way Discord's channel handler
   does, against a real on-disk SQLite store — no memory-layer mocking.

## Verification

```text
python3 -m pytest tests/integration/test_discord_memory_persistence.py tests/agent/test_coverage_gaps.py tests/vision/ -q
3041 passed, 3 failed (pre-existing, unrelated), 0 new failures
```

```text
python3 -m pytest tests/ -q -o faulthandler_timeout=120 \
  --deselect tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl \
  --deselect tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped \
  --deselect tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl
20701 passed, 13 skipped, 3 deselected in 452.13s (0:07:32)
```

Full detail (root-cause writeups, all evidence) in `BUILD_STATUS.md` and
`TEST_RESULTS.md`.

## Remains

- FX-A bullet 6: live end-to-end proof across tool categories (not yet
  attempted — needs a real or scripted acpx delegate invocation).
- FX-B loose ends: `run_stream()`'s CLI-only streaming path still saves
  both turns post-completion rather than user-first; redaction/secret
  scrubbing before persistence not yet covered (overlaps SR-1.10).
  `MEM-001`, `MEM-004`, `SEC-PI-004`, `XT-006` still need live harness
  re-validation.
- FX-C through FX-G not yet started.
- SR-1.1 through SR-4.8 security-review remediation not yet started —
  note that SR-3.1 ("one production memory backend") and part of SR-3.2/
  SR-3.3 are now substantially addressed by the FX-B fix; still needs
  explicit re-verification and write-up against the security review's
  own finding IDs.
- Full 89-case tool-specific validation backlog not yet re-run.
- Pre-existing vision `CameraDiscovery` cache-TTL flake (3 tests,
  unrelated to this session, tracked separately).

## First Next Step

Two good options, pick based on what's available: (a) if a safe way to
exercise a real/scripted acpx delegate call exists, prioritize proving
FX-A bullet 6 end-to-end since it unblocks the largest failure cluster;
otherwise (b) continue with FX-D/FX-E (independent-grading enforcement
and gated-tool-bypass refusal) since they're implementation-only (no
live delegate needed) and partially already started via the FX-A
envelope work. After either, map SR-3.1/3.2/3.3 findings in
`~/Missy-security-review.md` explicitly onto the FX-B fix just landed
and update `AUDIT_SECURITY.md`.
