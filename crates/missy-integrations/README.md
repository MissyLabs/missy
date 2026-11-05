# Native integration adapters

This crate contains native HTTP adapters for Nomad, Incus, and ComfyUI, and native
OBS WebSocket v5 / VTube Studio Public API WebSocket adapters. It implements
`missy_core::ToolExecutor`; the embedding application composes it with BuiltinTools.
No Python process, arbitrary HTTP tool, command runner, application launch, local
host probing, authorization popup, camera access or desktop session access exists.

## Constructor contract

`IntegrationTools::new(http: PolicyHttpClient, config: IntegrationConfig) -> Result<Self>`

`IntegrationConfig` has optional `nomad: NomadConfig`, `incus: IncusConfig`,
`comfyui: ComfyConfig`, `obs: ObsConfig`, `vtube: VtubeConfig` members. Each has
`connection: ConnectionConfig` with explicit `enabled`, `endpoint`, `token_env`
(environment variable NAME only), `timeout_seconds` (1..120), `max_output_bytes`
(128..262144), and `allow_mutations`. Default is disabled, blank endpoint, no
credential, 10 seconds, 32000 bytes, and no mutations. Unknown config keys fail.
No connection is made during construction. Credentials resolve once at construction;
use environment secret injection before starting the process, never model arguments.
There is no Debug/Serialize implementation for the credential-holding transport.

HTTP endpoints must be bare `http(s)://host:port` origins; WS endpoints bare
`ws(s)://host:port` origins. No userinfo, paths, fragments or query credentials.
`PolicyHttpClient` still requires the exact HTTP equivalent origin for WS, and
private destinations require policy opt-in. All DNS, redirects, address pinning,
TLS and network authorization remain the gateway's responsibility. The adapters
never construct another HTTP client or raw production socket. TLS certificate
validation is not disabled. Incus must use an explicitly configured bearer-auth
proxy/API endpoint; native mTLS client certificate provisioning is not implemented.

## Approval boundary

All `_mutate` tools and `comfyui_submit` set `side_effect=true`. They require
operator `allow_mutations=true` AND exact resource/action allowlists AND a trusted
`ApprovalGate`. The default gate denies every mutation. Install a host gate using
`with_approval(Arc<dyn ApprovalGate>)`; `approve(&ToolContext, &ToolCall)` is async.
The gate verifies exact scope, turn, call ID, tool name and arguments, not an
untrusted `confirm` flag. The host runtime journals effects and must not retry
uncertain results after cancellation, timeouts or lost responses.

When the host already has an outer `GuardedTools` durable approval wrapper, a
private always-true marker gate is permissible **only** if every effectful integration
tool is mandatory in that outer wrapper's sensitive tool set and there is no other
dispatch path to the inner executor. Validate this composition before exposing
tools. The marker is privileged application wiring, never a config or model API.
Do not consume the same approval twice. Callers without that mandatory wrapper
must supply a real per-call human approval verifier, or leave the deny gate intact.

Configuration fingerprints hash the full nonsecret operator config plus outbound
HTTP policy. They change when endpoints, routes' resource allowlists, workflows,
deployments, limits, or policy change. Credential bytes are not fingerprint inputs.
`ToolExecutor::preflight` performs pure strict argument and exact resource/action
validation and returns typed `ToolDispatchRefused` on known pre-dispatch refusals.
`execute` repeats preflight, so direct callers cannot skip those checks. Approval
is not consumed during preflight; the installed approval gate runs at dispatch.

## Capability matrix

* `nomad_read`: `list`, `status {job}`. One exact configured namespace; results
  project only summary fields and filter exact job IDs. No task Env/Vault/templates.
* `nomad_mutate`: `deploy {job}` selects a complete operator-authored Job object
  whose ID and namespace match config; model-supplied commands/driver/mount fields
  cannot enter it. `stop {job}` uses DELETE with `purge=false`. No host default,
  discovery, exec, scale, arbitrary API route, purge or generated job payload.
* `incus_read`: allowed instance `list` and `status {instance}` within one exact
  configured project; no instance config/expanded secrets.
* `incus_mutate`: exact instance and action from configured lifecycle set
  (start/stop/restart/freeze/unfreeze), non-forced, 30-second server stop timeout.
  Returns async operation metadata only. Never follows untrusted operation URLs.
  No delete, exec, file, devices, profiles, mounts, launch or arbitrary API calls.
* `comfyui_submit`: `workflow` selects an operator-authored API graph; optional
  `prompt` can replace only `CLIPTextEncode.inputs.text` at one configured node.
  No arbitrary graph input. Node classes require explicit allowlist, graph links
  must resolve, cycles and malformed nodes fail, nodes/bytes bound. Operators must
  trust all allowed Comfy node implementations and the exact static graph, including
  local model/path inputs. Class allowlisting is not a sandbox for custom nodes.
* `comfyui_history`: only IDs submitted in the same full Scope, no arbitrary ID
  enumeration. Returns structured `ArtifactMetadata` for images/videos/audio/gifs.
  Filename/subfolder components reject absolute paths, dot traversal, escapes,
  backslashes, percent encodings and URL schemes. Only output/temp storage allowed.
  No download route, local artifact file claim, file writes or path dereference.
  Tracking is in-memory, capacity bounded (1..10000); restart loses ownership.
  Ambiguous submission consumes a reservation and is never replayed automatically.
* `obs_read`: `status`, `list_scenes`; filtered scene metadata only. V5 Hello,
  challenge/SHA256/base64 auth, Identify, negotiated RPC v1, exact request correlation,
  strict success response checks. No event subscription or unsolicited effects.
* `obs_mutate`: explicit action set and exact scenes/scene:item-id resource set.
  Switch scene, visibility, recording/stream start/stop are separate actions. No
  arbitrary requestType, source settings, plugin calls, output file path or text input.
* `vtube_read`: allowed current model, models and hotkeys. Requires pre-provisioned
  token; never token issuance or persistence/popups. Checks API name/version,
  request ID and response type before accepting data.
* `vtube_mutate`: exact models/hotkeys or exact parameters with finite per-parameter
  bounds. No arbitrary messageType, item loads, file paths, microphone or audio.

Results carry `trust: untrusted`; upstream messages never become instructions.
Credentials are redacted from output values and keys across all configured adapters;
errors do not include upstream bodies, auth comments, credentials or raw URLs.
Protocol JSON, request args, control frames, message/output sizes and total operation
deadlines are bounded. Cancellation may leave an external mutation in an uncertain
state; these adapters do not claim a failed network read rolled back the effect.

## Verification and limitations

`cargo test -p missy-integrations --offline` runs actual HTTP and WS connections to
in-process ephemeral loopback mocks only, with explicit policy opt-in. No production
app, desktop session, hardware, GPU generation or real media was exercised. Native
FFmpeg file conversion, TTS synthesis/playback, device/volume operations and video
workflow generation are deliberately not implemented here. Existing Python behavior
is not a parity claim for those functions; network integrations were prioritized.

Verified locally: 19 mock tests pass; `cargo clippy -p missy-integrations --all-targets
--offline -- -D warnings` passes, and `cargo fmt -p missy-integrations` was applied.
The unit-variant unknown-field regression is specifically tested: Serde's internally
tagged unit variants ignored extra fields despite deny_unknown_fields, so the adapter
now validates its exact advertised object schema independently before deserialization.
