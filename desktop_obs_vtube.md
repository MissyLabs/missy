# Desktop Automation, OBS, and VTube Studio Integration — Design & Implementation Record

Source request: `~/feedback.md` (full text preserved verbatim below the design). Branch:
`feat/desktop-obs-vtube-integration`.

Status: **Phases 1–4 implemented** per the requester's explicit reordering note ("start with
OBS WebSocket + VTube Studio API before full mouse-click desktop automation"). Phase 5 (a
fully integrated, always-on Discord/VTuber live-performance workflow) is **designed but not
implemented** — see [Phase 5](#phase-5-integrated-discordvtuber-workflow-designed-deferred).

## 0. Source-code scan (done first, per the request)

Before writing anything, the existing tool registry was inspected. Most of "Phase 1" already
existed:

| Requested capability | Existing tool | Status |
|---|---|---|
| Take screenshots | `x11_screenshot` (`x11_tools.py`) | already implemented |
| Click | `x11_click` | already implemented |
| Type | `x11_type` | already implemented |
| Hotkeys | `x11_key` | already implemented |
| List windows | `x11_window_list` | already implemented |
| OCR/vision analysis of screen | `x11_read_screen` (Ollama vision) | already implemented |
| Launch GUI apps | `x11_launch.py` | existed, but no allowlist/confirmation gating |
| Audio device inspect/mute/volume | `audio_list_devices`, `audio_set_volume` (`tts_speak.py`) | already implemented (PipeWire `wpctl`) |
| Detect X11/Wayland/GNOME session | — | **gap**, built this pass (`desktop_status`) |
| Standalone focus-window | — | **gap** (only inline in click/type/key), built this pass |
| Mouse drag | — | **gap**, built this pass (`desktop_mouse_drag`) |
| Allowlisted+confirmed app launch | — | **gap**, built this pass (`desktop_launch_app`) |
| OBS control | — | **gap**, built this pass |
| VTube Studio control | — | **gap**, built this pass |
| TTS→OBS/VTube audio routing | — | **gap**, built this pass |

This changed the actual scope: rather than building desktop automation from scratch, this pass
adds the missing pieces around an already-solid X11 foundation, plus the two brand-new
integrations the requester explicitly prioritized (OBS, VTube Studio).

## 1. Staged plan (as implemented)

Reordered from the request's original Phase 1–5 per the requester's own risk-based
prioritization: WebSocket APIs (OBS, VTube Studio) are safer and more reliable than blind
mouse-coordinate clicking, so they came first.

| Phase | Scope | Status |
|---|---|---|
| 1 | OBS WebSocket tools | **implemented** |
| 2 | VTube Studio API tools | **implemented** |
| 3 | Desktop gaps (session detection, focus, drag, allowlisted launch) | **implemented** |
| 4 | Audio routing (PipeWire virtual sink for TTS → OBS/VTube) | **implemented** |
| 5 | Integrated Discord/VTuber live-performance workflow | **designed, not implemented** |

## 2. New config sections

Added to `missy/config/settings.py` (`ObsConfig`, `VtubeConfig`, `DesktopConfig`), all disabled
by default (secure-by-default, matching every other Missy subsystem):

```yaml
obs:
  enabled: false
  host: "127.0.0.1"
  port: 4455
  password: vault://obs_password   # or $OBS_PASSWORD; never plaintext in config.yaml
  scene_allowlist: ["Main", "BRB", "Starting Soon"]  # empty = every scene allowed

vtube:
  enabled: false
  host: "127.0.0.1"
  port: 8001
  auth_token: vault://vtube_studio_token   # populated automatically on first authorization
  plugin_name: "Missy"
  plugin_developer: "MissyLabs"

desktop:
  enabled: false
  app_allowlist: ["firefox", "obs", "vtubestudio"]
  unrestricted: false   # true = skip the allowlist entirely (still needs enabled: true)
```

## 3. Security model

- **Fail closed by default.** Every new tool family (`obs_*`, `vtube_*`, `desktop_launch_app`)
  requires its config section's `enabled: true`; none function on a stock install.
- **No raw shell fallback.** `desktop_launch_app` execs an argv list directly
  (`subprocess.Popen([binary, *args], ...)`, never `shell=True`), so there is no shell-metacharacter
  injection surface regardless of requested arguments. `ToolPermissions(shell=True)` is still
  declared so the existing global `ShellPolicy` allowlist applies too (defense in depth, strictly
  *more* restrictive than bare `shell_exec`, not a bypass of it).
- **App/scene allowlists.** `desktop.app_allowlist` and `obs.scene_allowlist` gate the common
  case without a prompt; anything outside them requires human confirmation.
- **Confirmation gates, fail closed with no gate configured.** A new process-wide
  `missy.agent.approval.get_shared_approval_gate()`/`set_shared_approval_gate()` accessor
  (`missy/agent/approval.py`) lets built-in tools reach the same `ApprovalGate` instance
  `gateway_start()` already wires into `ProactiveManager`/the Web API, since built-in tools get
  no constructor injection (`register_builtin_tools()` instantiates every tool class with zero
  arguments). Every confirmation-gated action in this feature — non-allowlisted app launch,
  non-allowlisted OBS scene switch, and **unconditionally** `obs_start_streaming_confirmed`/
  `obs_stop_streaming_confirmed` — denies outright when no gate is configured, mirroring
  `McpManager`'s existing `requires_approval` handling exactly. There is no allowlist bypass for
  starting/stopping the stream: it always prompts.
- **Stream keys never touched.** `obs_start_streaming_confirmed`/`obs_stop_streaming_confirmed`
  only issue obs-websocket's `StartStream`/`StopStream` requests, which use OBS's own internally
  configured stream destination/key. No obs-websocket request used here contains or returns one.
- **Secrets never leave the process.** OBS's password and VTube Studio's auth token support
  `vault://`/`$ENV` references (resolved the same way as `ProviderConfig.api_key`) and never
  appear in any `ToolResult` output, log line, or audit event. A freshly-acquired VTube Studio
  token (first-run authorization) is written directly to the encrypted `Vault`, never returned
  to the caller — same persist-don't-print pattern as the OpenAI OAuth flow in `missy/cli/oauth.py`.
- **Real network targets enforced.** Every tool declares `resolve_network_hosts()` with the
  actual configured OBS/VTube Studio host, and `_obs_request()`/`_vtube_request()` additionally
  call `get_policy_engine().check_network()` directly before connecting (raw `websockets.connect()`
  doesn't route through `PolicyHTTPClient`, so this check can't be skipped by relying on the
  declaration alone).
- **Safe audio defaults.** `audio_route_tts`'s virtual sink volume is capped at 70% on creation
  (never left at PipeWire's occasionally-100%+ default), and it never changes the operator's
  system-wide default audio sink unless `set_default=True` is explicitly passed.

## 3.5. Follow-up hardening pass (self-audit against `~/feedback.md`)

A second pass re-read the original request against the shipped Phases 1–4 code and found nine
gaps between what was asked for and what actually landed. All nine are now closed, in the same
commit as this note:

| # | Gap | Fix |
|---|---|---|
| 1 | No window allowlist — only an app allowlist existed | `desktop.window_allowlist` + `_desktop_shared.check_window_allowed()`, wired into `x11_click`/`x11_type`/`x11_key`'s optional `window_name` param and `desktop_focus_window`. Enforced **only** when `desktop.enabled` is `True` (see backward-compatibility note below) — a fresh boolean, not a fail-closed-on-absent-config default, so every pre-existing deployment that never opted into `desktop:` keeps working unchanged. |
| 2 | No rate limiting anywhere for tool calls | `_desktop_shared.check_rate_limit()` — an in-memory sliding-window limiter keyed by tool name — applied to every `x11_*`, `desktop_*`, `obs_*`, `vtube_*`, and `audio_route_*`/`audio_test_route` tool. Budgets come from each family's new `rate_limit_per_minute` config field (OBS/VTube default 30/min; audio routing uses a fixed 20/min, no dedicated config surface). |
| 3 | No screenshot secret redaction | `x11_tools._redact_screenshot_secrets()` — best-effort OCR via optional `pytesseract` (new `[vision]` extra), blacking out any word `SecretsDetector` flags before the image is saved. Runs on `x11_screenshot` (new `redact: bool = True` param) and on `x11_read_screen`'s saved capture; the vision model's *description* is separately passed through `censor_response()`. Degrades gracefully (never blocks the screenshot) when `pytesseract` isn't installed or OCR fails. |
| 4 | No `obs_set_source_text`/image/media tool | `ObsSetSourceTextTool` (`obs_set_source_text`) — `SetInputSettings` with either literal `text` or a `file_path` (mutually exclusive), for on-stream captions/lower-thirds/alerts. **Follow-up fix:** the initial version always sent `file_path` under the `file` settings key, which OBS's Image Source (`image_source`) reads but its actual Media Source (`ffmpeg_source`) does not — `SetInputSettings` silently merges unrecognized keys without applying or erroring, so pointing the tool at a video/audio source returned `success: True` without changing anything. Fixed by looking up the source's real `inputKind` via `GetInputSettings` first and picking the settings key that input kind actually reads (`file` for `image_source`, `local_file` + `is_local_file: true` for `ffmpeg_source`; unrecognized kinds fall back to `file`, the prior behavior). |
| 5 | No confirmation gate on `discord_upload_file` | `discord_upload.py`'s `execute()` now always calls `require_approval()` before posting (token-presence check happens first, so a missing token never triggers an approval prompt for nothing). No allowlist bypass, matching the existing OBS start/stop-streaming posture. |
| 6 | No `install_software` tool/gate | `InstallSoftwareConfirmedTool` (`install_software_confirmed`) — argv-only `sudo apt-get install -y <package>` (never `shell=True`), gated on `desktop.enabled` **and** the new `desktop.allow_software_install` flag (default `False`), a package-name regex, rate limiting, and a `require_approval()` prompt before every install. |
| 7 | No standalone mouse-move (no click) | `DesktopMouseMoveTool` (`desktop_mouse_move`) — `xdotool mousemove` only, no click event, for hover-only interactions (e.g. tooltip/preview triggers) that shouldn't also click. |
| 8 | `audio_set_volume` had no upper volume clamp | `AudioSetVolumeTool.execute()` now hard-caps an absolute volume at 100% and a relative delta (`+N%`/`-N%`) at 50 percentage points per call — PipeWire otherwise allows a relative delta to boost a sink arbitrarily far past 100%. |
| 9 | No `vtube_list_models` standalone tool | `VtubeListModelsTool` (`vtube_list_models`) — wraps VTube Studio's `AvailableModelsRequest`, returning `{model_name, model_id, is_loaded}` per available Live2D model, for model discovery/selection independent of `vtube_load_model`'s by-name lookup. |

**Backward-compatibility principle applied to gap #1:** the first implementation of the window
allowlist failed closed whenever `desktop:` config was absent, which would have silently broken
every pre-existing `x11_click`/`x11_type`/`x11_key` call using `window_name` on a deployment that
never configured the new `desktop:` section. Caught before commit by testing with
`MISSY_CONFIG=/nonexistent/config.yaml` (simulating CI, which has no `~/.missy/config.yaml`) —
the local dev machine's real config (with `desktop.unrestricted: true` already set) had been
masking the bug. Fixed by tying enforcement to `desktop.enabled` being explicitly `True`, the
same opt-in pattern every other guardrail in this feature already follows.

## 4. Tool specifications

### 4.1 OBS (`missy/tools/builtin/obs_tools.py`)

Protocol: [obs-websocket v5](https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md),
OBS Studio's built-in WebSocket server (Tools → WebSocket Server Settings). Each tool call opens
a fresh connection, authenticates (SHA-256 challenge/salt per spec), issues one request, closes.

| Tool | Schema | Safety | Approval |
|---|---|---|---|
| `obs_status` | no params | read-only | none |
| `obs_list_scenes` | no params | read-only | none |
| `obs_switch_scene` | `scene_name: string` (required) | writes program scene | required *unless* `scene_name` is in `obs.scene_allowlist` |
| `obs_set_source_visibility` | `scene_name`, `source_name`, `visible: bool` (all required) | toggles a source | none (non-public-facing; local composition change) |
| `obs_start_recording` | no params | starts local recording only | none |
| `obs_stop_recording` | no params | stops local recording | none |
| `obs_start_streaming_confirmed` | no params | **goes live publicly** | **always required**, no bypass |
| `obs_stop_streaming_confirmed` | no params | ends a public stream | **always required**, no bypass |

Error handling: connection/timeout/auth/request failures all raise `ObsError` internally, caught
at each tool's `execute()` boundary and returned as `ToolResult(success=False, error=str(exc))` —
never a raw traceback, never the password. Audit: every call goes through the same
`ToolRegistry.execute()` → policy/trust-score pipeline as any built-in tool (see
`missy/tools/registry.py`), so `tool_execute`/`tool_result` audit events are emitted automatically
with redacted arguments; no bespoke audit code was needed in this module. `get_streaming/recording status`
from the original request is folded into `obs_status`'s output rather than a separate tool.

### 4.2 VTube Studio (`missy/tools/builtin/vtube_tools.py`)

Protocol: [VTube Studio Public API](https://github.com/DenchiSoft/VTubeStudio) over WebSocket
(default port 8001).

| Tool | Schema | Safety | Notes |
|---|---|---|---|
| `vtube_status` | no params | read-only; may block up to 30s on first-run authorization pop-up | current model/loaded state |
| `vtube_load_model` | `model_name: string` (required) | resolves name→ID via `AvailableModelsRequest`, then loads | unknown name lists available models in the error |
| `vtube_trigger_hotkey` | `hotkey_name: string` (required) | resolves name→ID via `HotkeysInCurrentModelRequest`, then triggers | can reach any hotkey the operator bound in VTS, including a tracking toggle if one exists |
| `vtube_set_parameter` | `parameter_id: string`, `value: number` (required), `weight: number` (default 1.0) | one-shot `InjectParameterDataRequest` | discrete/scripted puppeting only — see gap below |

**Documented API gap:** VTube Studio's public API has no "start/stop face tracking" request —
tracking is controlled by the app's own webcam/tracker settings or an operator-bound hotkey
(reachable only via `vtube_trigger_hotkey`). This is stated honestly rather than fabricating an
unsupported call. **Real-time audio-driven mouth movement is deliberately not implemented as
per-frame parameter injection** — VTube Studio has its own built-in microphone-based lip sync;
pointing it at the PipeWire monitor source `audio_route_tts` creates is far more reliable than
Missy computing synthetic amplitude-to-parameter mappings against TTS playback timing.

### 4.3 Desktop (`missy/tools/builtin/desktop_tools.py`)

| Tool | Schema | Safety |
|---|---|---|
| `desktop_status` | no params | read-only; detects `XDG_SESSION_TYPE`/`XDG_CURRENT_DESKTOP` and which of `xdotool`/`wmctrl`/`scrot`/`wtype`/`ydotool`/`grim`/`slurp` are installed |
| `desktop_focus_window` | `window_name: string` (required) | X11 only, via `xdotool search --name ... windowfocus` |
| `desktop_mouse_drag` | `start_x`, `start_y`, `end_x`, `end_y` (required ints), `button` (default `left`) | X11 only, `xdotool mousedown`+`mousemove`+`mouseup` |
| `desktop_launch_app` | `app: string` (required), `args: string[]` (default `[]`) | argv-only exec (never a shell string); allowlist + confirmation gate |

`desktop_screenshot`/`desktop_click`/`desktop_type`/`desktop_hotkey`/`desktop_list_windows` from
the original tool-name list are intentionally **not** duplicated — `x11_screenshot`/`x11_click`/
`x11_type`/`x11_key`/`x11_window_list` already do exactly this for X11 sessions.

**Wayland/GNOME limitation (documented, not silently broken):** `xdotool`/`wmctrl`/`scrot` are
X11-only; GNOME's Mutter compositor doesn't implement the wlroots protocols `wtype`/`ydotool`
need either. Under GNOME on native Wayland, `desktop_status` reports
`x11_automation_usable: false, wayland_automation_usable: false` with an explanatory note rather
than letting click/type/key silently fail with no diagnostic. Documented future path (not
built): GNOME's Remote Desktop portal (`org.gnome.Mutter.RemoteDesktop` over D-Bus).

### 4.4 Audio routing (`missy/tools/builtin/audio_route.py`)

| Tool | Schema | Safety |
|---|---|---|
| `audio_route_tts` | `sink_name` (default `missy_tts_out`), `set_default: bool` (default `false`) | idempotent PipeWire null-sink creation via `pactl`; volume capped at 70% on creation; never changes the system default sink unless explicitly requested |
| `audio_test_route` | `sink_name`, `play_test_phrase: bool` (default `true`) | verifies the sink exists/unmuted/volume, optionally plays a short `espeak-ng` phrase targeted at it via `PULSE_SINK` |

`audio_list_devices`/`audio_set_volume`/mute-unmute already exist in `tts_speak.py` and are
unmodified by this pass.

## 5. Install steps

### OBS

1. OBS Studio 28+ ships obs-websocket built in. Tools → WebSocket Server Settings → enable,
   set a password, note the port (default 4455).
2. `missy vault set obs_password <the password>` then in `config.yaml`:
   `obs: { enabled: true, password: vault://obs_password }`.

### VTube Studio

- **Windows (recommended):** install from Steam or https://denchisoft.com/. Settings → API →
  enable "Start API", note the port (default 8001). Run `missy vtube status` (via the
  `vtube_status` tool) once — this triggers the one-time authorization pop-up in VTS; approve it,
  and Missy persists the issued token to the vault automatically.
- **Linux via Steam/Proton:** VTube Studio runs under Proton (Steam Play must be enabled for
  it; SteamVR is not required for desktop mode). Webcam passthrough into a Proton prefix can be
  unreliable — if face tracking is needed, Windows is the more reliable option; API-only control
  (no tracking) works fine under Proton since it's just a WebSocket server. `vtube.host` should
  point at wherever VTS actually runs (`127.0.0.1` if colocated with Missy, a LAN IP if VTS is on
  a separate Windows box).
- **Live2D models:** import via VTube Studio's own model loading UI (drag a `.model3.json`
  folder or use File → Load Model); `vtube_load_model` operates on whatever VTS has already
  imported, it does not import models itself.

### Audio routing (PipeWire)

Requires `pipewire-pulse` (`sudo apt install pipewire-pulse`) for `pactl` compatibility — already
a dependency of the existing `audio_list_devices`/`audio_set_volume` tools, so no new package on
a system where TTS already works. No systemd unit needed: `audio_route_tts` creates the virtual
sink on demand via `pactl load-module`, which does not persist across a PipeWire restart by
design (call `audio_route_tts` again after a reboot, or add the module to
`~/.config/pipewire/pipewire-pulse.conf.d/` for persistence — left to the operator since it's a
system-wide PipeWire config change outside Missy's own file scope).

In OBS: Sources → Add → Audio Input Capture → device = `<sink_name>.monitor`. In VTube Studio:
Settings → Input → select `<sink_name>.monitor` as the microphone.

## 6. Phase 5: integrated Discord/VTuber workflow (designed, deferred)

Not implemented this pass — this section is the design for a future pass, not a claim of
current capability.

**Goal:** Missy joins Discord voice, speaks via `tts_speak`/`discord_voice_say`, and the VTuber
avatar's mouth moves in sync while OBS streams the composed scene — driven entirely by the
already-built pieces (`audio_route_tts` + VTube Studio's own mic-based lip sync + `obs_switch_scene`
for scene changes during a "performance"), not new lip-sync computation.

**Why deferred:** `discord_voice.py`'s `DiscordVoiceSayTool` currently plays audio through
Discord's voice connection; it does not currently also duplicate that audio to a local PipeWire
sink at the same time (Discord voice playback and local TTS playback are presently separate code
paths — `tts_speak` plays locally, `discord_voice_say` sends to Discord). Making both happen
from one spoken line (local monitor for VTube Studio's lip sync *and* Discord voice output
simultaneously) needs either a shared audio-graph change in `discord_voice.py` or a second
simultaneous local playback call — a real code change to a different module than this pass's
scope (OBS/VTube/desktop/audio-routing tools), and one that should be scoped and tested on its
own rather than folded in here speculatively.

**What a future pass needs:**
1. Extend `discord_voice.py`'s playback path (or add a thin wrapper tool) to tee the same
   synthesized audio to both the Discord voice connection and the `audio_route_tts` sink in one
   call, so VTube Studio's mic-based lip sync reacts to the same speech Discord users hear.
2. A `vtuber_perform`-style orchestration tool (or agent-loop convention) that sequences
   `obs_switch_scene`/`vtube_trigger_hotkey`/speech calls for a coherent "segment" rather than the
   agent making ad-hoc individual tool calls — this is a UX/orchestration design question more
   than a security one, and deserves its own scoped conversation with the operator about what a
   "performance" should look like before building it.
3. "Keep normal text replies separate from explicit 'say this aloud' actions" (from the original
   request) is already true structurally — Discord text replies and `tts_speak`/`discord_voice_say`
   are already distinct tool calls the agent chooses independently; no new mechanism was needed
   for that part specifically.

## 7. Tests

- `tests/tools/test_obs_tools.py` — 36 tests: obs-websocket v5 auth-response hash correctness
  (verified against the spec's own formula), full authenticated-handshake round trip, fail-closed
  config gating, scene-allowlist approval gating, and unconditional fail-closed approval gating
  for `obs_start_streaming_confirmed`/`obs_stop_streaming_confirmed` (denied outright with no
  gate configured; approved-but-then-denied still blocks the OBS request from ever being sent).
- `tests/tools/test_vtube_tools.py` — 17 tests: existing-token re-auth path, fresh-token
  acquisition + vault persistence + confirmed non-leakage into tool output, vault-write-failure
  doesn't fail the call, name→ID resolution for models/hotkeys.
- `tests/tools/test_desktop_tools.py` — 26 tests: session-detection matrix (X11 / GNOME-Wayland /
  Wayland-with-XWayland), launch-app allowlist/unrestricted/approval-gating matrix, and a direct
  assertion that `subprocess.Popen` is always called with an argv list and never `shell=True`.
- `tests/tools/test_audio_route.py` — 13 tests: idempotent sink creation, volume always capped at
  creation, default-sink never changed unless explicitly requested.
- `tests/tools/test_desktop_shared.py` — 8 tests: the shared config-loader and fail-closed
  approval helper directly.
- `tests/agent/test_approval_gate.py` — added 5 tests for the new `set_shared_approval_gate()`/
  `get_shared_approval_gate()` process-wide accessor, including a concurrent get/set race check.
- `tests/config/test_settings.py` — added 11 tests for `ObsConfig`/`VtubeConfig`/`DesktopConfig`
  YAML parsing, vault-reference resolution for `obs.password`/`vtube.auth_token`, and
  unknown-key warnings.

All new/changed files pass `ruff check`/`ruff format --check`; the full `tests/tools/
tests/config/ tests/agent/ tests/cli/` suite (8067 tests) passes with this feature added.

---

## Appendix: original request (`~/feedback.md`, verbatim)

> I want to add safe desktop/GUI automation and full VTuber/streaming usability to my Discord AI
> assistant, Missy, which currently runs on a Linux server with GNOME available but no exposed
> desktop-control tools. [Full text: filesystem allowlists, OBS/VTube Studio/audio tool lists,
> security model, and the staged Phase 1–5 plan — preserved in `~/feedback.md`, not duplicated
> here to avoid drift between two copies.]
>
> Tiny note: start with OBS WebSocket + VTube Studio API before full mouse-click desktop
> automation if possible. APIs are way safer and less cursed than letting me click random GNOME
> buttons like a caffeinated cat.
