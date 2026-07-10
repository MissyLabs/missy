# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed

- Preserved and hardened the existing `missy/channels/discord/voice_commands.py`
  fix (already committed as `voice_fix` on master before this session).
  Added 9 regression tests directly against `parse_voice_intent()`. Found
  and fixed a real bug the new tests exposed: a trailing comma attached
  directly to a channel name (no preceding whitespace) leaked into the
  parsed channel name. Fixed `_TRAILING_CLAUSE_RE` and target stripping
  in `_normalise_channel_target()`.
- Implemented the first concrete slice of **FX-A** (dominant root cause
  behind ~30 of 43 failing validation cases) in
  `missy/providers/acpx_provider.py`:
  - Every acpx invocation now forces `--allowed-tools ""` and
    `--non-interactive-permissions deny`, verified against the actual
    pinned `acpx@0.3.1` source (not just `--help` text) to confirm the
    empty-string form really means zero native tools for the Claude Code
    delegate adapter.
  - `_sanitize_extra_flags()` strips security-critical flags (including
    `--cwd`) from operator-configured `base_url` so they cannot be
    reintroduced via a mutable local config file.
  - `is_available()` now fails closed if the installed acpx version
    doesn't document the required flags in `--help`.
  - Added `_isolated_cwd()` (`~/.missy/acpx_sandbox`, mode 0700) so the
    delegate never defaults into Missy's actual repository.
  - Added a versioned delegation envelope
    (`_render_delegation_envelope`) replacing the bare `[System]:` text
    boundary, explicitly stating the delegate is Missy's planning
    component with no native tools and must not fabricate additional
    conversation turns or a self-authored scorecard (overlaps FX-D).
  - Added `_strip_leaked_transcript_markers()` defensive scrub, applied
    before tool-call parsing, with a regression test reproducing the
    exact `DISC-CMD-006` failure shape.
  - Confirmed tool-call execution already routes through
    `AgentRuntime._tool_loop()` → `ToolRegistry.execute()` — no change
    needed there.
  - Live-verified against the real installed `acpx` binary (health
    check + argv construction with a hostile `base_url`), not just
    mocks.

## Verification

```text
python3 -m pytest tests/channels/discord/test_voice_commands.py tests/providers/test_acpx_provider.py -q
157 passed
```

```text
python3 -m ruff check missy/providers/acpx_provider.py tests/providers/test_acpx_provider.py missy/channels/discord/voice_commands.py tests/channels/discord/test_voice_commands.py
All checks passed!
python3 -m ruff format --check <same files>
4 files already formatted
```

```text
python3 -m pytest tests/providers/ tests/agent/ -q
4995 passed, 4 skipped in 78.27s
```

```text
python3 -m pytest -q -o faulthandler_timeout=120   # full suite
20692 passed, 3 failed, 13 skipped in 490.29s (0:08:10)
```

The 3 failures are a pre-existing `CameraDiscovery` cache-TTL bug in
`missy/vision/discovery.py`, confirmed present on master before this
session's changes (via `git stash`) and unrelated to acpx/voice work.
Tracked as a follow-up task, not fixed this session.

Live smoke test against the real `acpx@0.3.1` binary (no LLM calls —
`--version`/`--help` only):

```text
$ python3 -c "... AcpxProvider(...).is_available() ..."
is_available(): True
isolated cwd: /home/missy/.missy/acpx_sandbox (mode 0700)
```

## Remains

- FX-A bullet 6: representative end-to-end proof (real acpx delegate
  invocation, not mocks) across filesystem, shell, browser, X11, AT-SPI,
  vision, audio, Discord upload, memory, `self_create_tool`, and
  `code_evolve` categories. This requires either a live Claude Code
  delegate call through acpx or a scripted fake ACP agent — not yet
  attempted.
- FX-B through FX-G not yet started (see `BUILD_STATUS.md` for detail
  and priority order).
- SR-1.1 through SR-4.8 security-review remediation not yet started.
- Full 89-case tool-specific validation backlog not yet re-run.
- Pre-existing vision `CameraDiscovery` cache-TTL flake (3 tests).

## First Next Step

Wire real Discord conversation-turn persistence to `SQLiteMemoryStore`
(FX-B) — the harness observed only 3 memory rows across 937
`agent.run.start` events, meaning almost no production Discord traffic
was ever durably remembered. Trace every runtime/Discord call path to
`add_turn()`, add an integration test through the real Discord handler,
and rerun `MEM-001`, `MEM-004`, `SEC-PI-004`, `XT-006`. Alternatively,
if a live/sandboxed way to exercise a real acpx delegate call becomes
available, prioritize proving FX-A bullet 6 first since it unblocks the
largest cluster of the 43 failing cases.
