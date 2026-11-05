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

All `_mutate` tools, `incus_operation`, `comfyui_submit`, and `comfyui_artifact`
set `side_effect=true`. They require
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
  arbitrary API route, purge or caller-generated job payload. Node discovery and
  reviewed scaling are provided by the extensions below.
* `incus_read`: allowed instance `list` and `status {instance}` within one exact
  configured project; no instance config/expanded secrets.
* `incus_mutate`: exact instance and action from configured lifecycle set
  (start/stop/restart/freeze/unfreeze), non-forced, 30-second server stop timeout.
  Returns async operation metadata only. Never follows untrusted operation URLs.
  Broader operations use the independently pinned `incus_operation` grants below.
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
  History itself does not download files or dereference server-provided URLs.
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
FFmpeg file conversion, TTS synthesis/playback, physical desktop device operations and video
workflow generation are deliberately not implemented here. Existing Python behavior
is not a parity claim for those functions; network integrations were prioritized.

Verified locally: 29 tests pass (original 19, extended HTTP/guard mocks 9, MP4 bounds 1);
`cargo +1.99.1 clippy -p missy-integrations --all-targets --no-deps -- -D warnings`
passes, and rustfmt was applied to the owned crate sources and tests.
The unit-variant unknown-field regression is specifically tested: Serde's internally
tagged unit variants ignored extra fields despite deny_unknown_fields, so the adapter
now validates its exact advertised object schema independently before deserialization.

## Extended typed infrastructure and artifact capabilities

The legacy narrow matrix above is extended by these native adapters. All operations
still use PolicyHttpClient, exact origins, per-task cancellation/deadlines and the
same strict outer approval boundary. There is no direct API passthrough.

* `nomad_read`: additionally `nodes`, `node {node}`, `placement {job}`,
  `allocations {job}`, `results {job}`. Node discovery filters an exact configured
  `nodes` ID set. Placement calls the actual server-side `/v1/job/{job}/plan` with
  the pinned Job, rather than fabricating local host-placement scores. Allocation
  results are exact job/namespace task states and events, not stdout downloads.
* `nomad_mutate`: additionally `plan {job}` and `scale {grant}`. Each `scales`
  entry pins job, group, and count (maximum 10000). Group must exist in a pinned
  deployment. Model callers cannot change the count or workload. Offload/benchmark
  workloads can be reviewed complete Job definitions selected through deploy.
  Runtime scheduler persistence/reconciliation belongs to the embedding runtime.
* `incus_discover`: `list {kind}`, `inspect {resource}`, `snapshots {instance}`.
  `resources` grants are typed exact instance/image/network/storage pool/custom
  volume/profile/project resources. Keys: `instance/name`, `image/fingerprint`,
  `network/name`, `storage_pool/name`, `storage_volume/pool/name`, `profile/name`,
  `project/name`. Lists filter grants and project summary fields. Inspect deliberately
  exposes configuration of the explicitly allowed resource. Project resources must
  match the configured project; instance resources must also be allowed instances.
* `incus_operation {grant}`: selects a strict tagged operator-authored `operations`
  entry, never a caller method/path/body. Create/update/delete support instances,
  networks, storage pools/custom volumes, profiles and the configured project;
  images support update/delete. Definitions reject unknown top-level fields and
  name mismatches. Instance creation requires explicit container/VM type and a
  full locally available allowlisted image fingerprint, no remote source URL.
  This creates an instance; starting it is a separate approved lifecycle call.
  Exact pinned noninteractive exec argv uses an absolute executable, no websocket
  stdin, bounded operation wait, and same-instance validated UUID exec-log paths.
  Results may still be running and must not be replayed. Other operation variants:
  exact guest text file write (content pinned), snapshot create/restore/delete,
  device set/delete, and same-project copy/move between two allowed instance names.
  Operator-pinned config/devices/profiles are privileged capabilities, not a sandbox.
  Device edits read and patch the local devices map, so concurrent external changes
  can race the read/modify/write. No ETag conflict protection is claimed.
* `incus_file_read {grant}`: exact guest instance/path from `file_reads` only.
  Guest paths must be absolute without dot components. This remote guest HTTP
  capability grants no host filesystem access. Files are bounded UTF-8 text.
* `comfyui_artifact {prompt_id,node,kind,output_index,output_path}`: requires owned
  completed history and exact-call human approval. Fixed `/view` uses the complete
  server descriptor's filename, subfolder and storage type, with no guessed defaults.
  `max_artifact_bytes` defaults to 0 (disabled); configure 1..4194304. Larger artifacts
  are explicitly denied, not truncated. Policy write roots/max_file_bytes must also
  allow the target. PNG/JPEG/WebP/GIF decode under 8192-pixel/64-MiB allocation bounds.
  MP4 receives bounded top-level ISO-BMFF structural validation, **not codec/frame
  decoding**, accurately labeled in the result. Other media is denied. Writes use
  descriptor-relative atomic mode-0600 Policy replacement; bytes, SHA256 and media
  facts are returned only after successful write. No hidden server URL or host path
  permission is inherited from history. Tracking remains in-memory and scope-bound.
* `comfyui_submit` additionally accepts optional named `conditioning`. The operator
  `conditioning` map binds an exact Policy-readable host path to one approved
  workflow's `LoadImage.inputs.image` node. Images are decoded with bounds,
  re-encoded to PNG, uploaded as bounded multipart through PolicyHttpClient to the
  fixed `/upload/image` route, and only the validated exact returned filename is
  substituted into that node. No model-provided file paths, image URLs, upload
  destinations, arbitrary graph changes, or opaque source bytes enter this path.
  The existing submit approval covers both upload and generation; an interrupted
  upload/submission can leave an unused remote input file and must not be replayed.

Extended verification uses real ephemeral loopback HTTP mocks. Tests exercise typed
routes, exact bodies, no-network denials, guest raw text wire I/O, Nomad plan/scale,
actual PNG decoding and artifact persistence, metadata forwarding, invalid media,
size bounds, scope isolation and symlink write denial. No live service was contacted.

Outer approval coverage is automatic in the parent CLI: it collects every enabled
integration definition with `side_effect=true`, requires durable approval config,
and refuses startup unless the entire mutation set occurs in `sensitive_tool_names`.
Thus `incus_operation` and `comfyui_artifact` are mandatory without a separate static
name list. Regression tests compose the actual governance `GuardedTools`, durable
SQLite approval registry and the integration executor with the privileged inner
marker. Both tools deny before network/write without exact approval, reject changed
grant/scope/output path bindings, execute the approved effect once, and reject replay
after consumption. The inner marker is not itself operator consent.
