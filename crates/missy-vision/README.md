# Native explicit-file vision

`VisionTools::new(PolicyConfig, VisionConfig) -> anyhow::Result<VisionTools>`
implements `missy_core::ToolExecutor`. `VisionConfig` is serde-defaulted,
unknown-field-denying, and **disabled by default**. `VisionKind` serializes as
`openai`, `anthropic`, or `ollama`.

## Tools and approval boundary

* `vision_analyze`: exactly one of `imagepath` (absolute string) or `imagepaths`
  (1–4 absolute strings), and a nonempty bounded `question`.
* `vision_scene`: persistent scope-bound session actions create/add_observation/
  update_state/summarize/close. SQLite observations are append-only untrusted
  caller assertions; state is never automatically inferred from provider output.
  Without action, accepts one absolute `imagepath` and integer `max_objects` (1–100).
  Prompts for `{"description":"...","objects":[{"label":"...","confidence":0.5}]}`.
  Rejects extra keys, malformed JSON, markdown fences, oversized strings, invalid
  confidences and too many objects. Invalid output returns `scene: null`; never
  invents or repairs entities. An empty object list is valid.

* `vision_monitor`: bounded interval polling of one explicit image file, normalized
  pixel-change threshold and cooldown; calls the billable backend only on change.
  Never silently opens camera/display. See the complete contracts and fixture
  evidence in [`media-production.md`](../../docs/rust/media-production.md).

These tools have **`side_effect: true`**, because state is persisted or images leave the host and provider
calls may incur charges. The embedding caller **must enforce operator approval**
before invoking them. `ToolContext` cannot attest approval, so this crate cannot
enforce the outer approval mechanism itself. Do not expose the executor directly
to unapproved callers. Image contents and model answers are untrusted, not new
instructions; no provider tool calls or model-selected file reads are executed.

No camera, hardware, desktop, URL image fetching, shell, Python, subprocess or
automatic path discovery exists. Only caller-explicit image files are read through
the existing descriptor-anchored `Policy::read_file` gateway. Its allowlisted roots,
symlink/hardlink rules and byte limit remain authoritative.

## Provider setup

OpenAI `base_url` normally ends in `/v1`; requests go to `/chat/completions` below
it with real `image_url` **data:image/png;base64** multimodal blocks. Anthropic's
base normally ends in `/v1`; `/messages` receives `source` image blocks. Ollama
normally uses its origin as the base; `/api/chat` receives `images[]` in its user
message and `options.num_predict`. No tool argument selects the model, endpoint,
credential or price.

Set an exact allowed HTTP(S) origin in `PolicyConfig.allowed_origins` (including
nondefault ports). All network traffic uses `PolicyHttpClient`: origin and DNS/IP
policy, DNS snapshot pinning, denied redirects, response cap and timeout remain in
force. HTTPS is required except explicitly permitted loopback with private-network
policy enabled. A configured remote Ollama is not assumed to be free.

`api_key_env`, if provided, must name a `MISSY_` uppercase environment variable.
OpenAI/Anthropic require this explicit binding. The credential is snapshotted at
construction and hashed into the configuration fingerprint without disclosure.
Backend error bodies are never surfaced; model text containing the credential is
rejected. A response must identify the exact configured `model`, have an assistant
role and normal completion, and carry no provider tool/function call. Configure
the exact returned model ID, not a provider alias that silently resolves elsewhere.

## Image safety

PNG/JPEG/WebP magic is checked, dimensions are parsed before decode, and dimension,
pixel, decoder-allocation and source-byte caps apply. A fresh RGBA pixel buffer is
re-encoded as PNG, discarding source EXIF/XMP/comments/trailing arbitrary bytes.
The sanitized PNG is bounded again. Combined serialized requests cannot exceed the
policy HTTP gateway's 4 MiB cap. Unsupported bytes never become text attachments.
Default caps: 2 MiB/image, 8 million pixels, 8192 pixels/axis, four images,
4096 question bytes, 1024 output tokens, 64 KiB returned JSON and 60 seconds.
This limits decoder work, not an OS-level decoder sandbox. Cancellation may leave
an already-running bounded blocking image decoder finishing locally; it cannot
dispatch a later HTTP request after the calling future has been dropped.

## Independent durable billing ledger

Set an absolute `ledger_path` in an existing private, effective-UID-owned **0700**
directory. The crate creates a **0600** SQLite ledger and refuses symlink/hardlink
or nonprivate state. Never place it in image read roots or delete it to retry a
failed request. SQLite immediate transactions serialize cross-process reservations.

Remote providers require both `input_microusd_per_million` and
`output_microusd_per_million`, or explicit `allow_unpriced_remote: true`. Omitted
prices are treated as zero only for loopback Ollama. Per-call and cumulative caps
are `max_call_cost_microusd` and `max_total_cost_microusd`. Up front we reserve
configured output-token ceiling plus operator `image_input_tokens_estimate` per
image, prompt-byte estimate and framing allowance, priced with upward rounding.
Unknown price reserves the whole per-call cap and will require reconciliation.

**These are estimates, not provider-enforced prepaid financial hard caps.** Image
tokenization varies, configured prices may be stale, provider usage can be wrong,
and a failed/cancelled call can still be billed. Actual returned token usage is
priced locally. Missing/unknown usage, unsupported Anthropic cached-token billing,
model/role mismatch, costs exceeding reservations, cancellation, timeout, malformed
response or HTTP failure leave the ledger pending/blocked. Further vision requests
are denied across restarts until an operator reconciles actual billing. No automatic
retry, release or replay is performed. The ledger's `vision_calls` table records
reservation ID, reserved micro-USD, actual micro-USD (when known), and state. An
operator must verify provider billing before editing a pending record or marking
it settled; changing configuration is not reconciliation.

Returned `usage` includes token counts, configured-price cost and `cost_known`;
`accounting` explicitly says `runtime_usage_includes_this_call: false`. This ledger
is separate from the main Runtime's provider budget, and the crate does not claim
that budget accounts for these external calls. Uncertain billing releases no
analysis/scene. The current ToolExecutor contract can block subsequent **vision**
calls, not force the whole runtime turn to terminate or disable unrelated tools.
Embedding runtimes requiring that stronger guarantee must add a terminal-error
contract and runtime integration rather than infer it from this executor.

## Verification

`cargo test -p missy-vision` exercises actual loopback HTTP for all three multimodal
payload formats, strict scene schema, native format sanitization, policy-denied
files and arguments, cancellation/timeouts, redirect/private-DNS denial, strict
provider responses, uncertain billing and durable reservation behavior. Tests use
synthetic images and fixed mock provider replies. They establish wire payload and
validation behavior, **not actual visual LLM inference or recognition accuracy**.
