# Native provider routing and operational admission

Native routing replaces the legacy ModelRouter fast/premium selection and worker
provider controls with **explicit operator profiles**, not extra model grants.
The native runner never asks a model to classify a request or choose its own
provider. No accounts, credential stores, or running processes are modified by
configuration loading or route selection.

## Defaults and deterministic order

Omitting `routing` retains ordered failover in configuration order. Profile
priority defaults to zero; lower numbers run first, equal priorities keep their
original order. `mode: weighted` uses deterministic smooth weighted round-robin
within the best eligible priority group. The chosen primary moves to the front;
remaining eligible profiles retain priority/configuration order for failover.
Weights do not grant eligibility. No eligible profile means a local error, never
an implicit premium or unrelated provider fallback. Every eligible profile is
attempted at most once per completion, subject to circuit and admission state.

Example native YAML fragments (combine with the documented native config):

```yaml
routing:
  mode: weighted
  default_tier: fast
  task_tiers:
    worker: fast
    review: premium
  rules:
    - min_input_chars: 20000
      min_tools: 0
      tier: premium
  max_wait_ms: 500
providers:
  - name: local-fast
    kind: ollama
    base_url: http://127.0.0.1:11434
    model: operator-selected-model
    input_microusd_per_million: 0
    output_microusd_per_million: 0
    routing:
      priority: 0
      weight: 3
      tiers: [fast]
      tasks: []
      requests_per_minute: 120
      tokens_per_minute: 200000
      max_concurrent: 4
      circuit_failure_threshold: 1
      circuit_cooldown_ms: 500
```

Each provider is a full native profile with an explicit model, API policy origin,
credential **reference**, and prices. To use a provider's fast/premium models,
create two named profiles rather than inferring hidden model aliases. Worker
profiles use `tasks: [worker]` and caller-selected task metadata. Empty `tasks` or
`tiers` means unrestricted along that axis, not a capability grant. Restricted
tasks require explicit matching task metadata. A profile restricted to a tier
requires that tier to be selected. Labels are nonempty ASCII alphanumerics,
underscore or hyphen, at most 64 bytes.

Precedence is explicit caller tier, task-to-tier mapping, first matching rule,
then default tier. Rules match immutable **user-message** character count and
operator-supplied tool definition count only. Assistant responses and tool
results cannot change routing. The caller must not derive task/tier metadata from
model-produced tool arguments. For stronger isolation, omit rules and supply an
operator-fixed tier. `ProviderPool::complete_for(task, tier, messages, tools,
cancellation)` is the trusted worker/task entry point; ordinary `Provider`
completion uses no task/tier override. CLI runner multi-profile pools consume
`Config.routing`; dedicated single-profile probes and explicitly selected
isolated worker pools need not inherit global tier filters.

## Limiter, reservations, cancellation and circuits

Admission is process-local, synchronized per pool and provider, and bounded:

* `requests_per_minute` and `tokens_per_minute` are optional positive limits;
  omit them for unlimited. Zero is rejected, never interpreted silently.
* Fixed 60-second windows begin on first admission. This is a conservative local
  guard, not a promise of exact upstream vendor rolling-window quotas.
* Token reservations use serialized request bytes plus maximum output tokens,
  multiplied by the number of possible account attempts. RPM likewise reserves
  every possible account attempt. This deliberately over-reserves when accounts
  are cooling down or reject a request; account rotation remains unchanged.
* Reservations are not refunded after errors, cancellation, or short responses.
  The provider may already have received and billed the request. Concurrency is
  released by an RAII guard, including future cancellation/drop.
* `max_concurrent` defaults to 16 and is bounded to 1..1024. `max_wait_ms` defaults
  to zero (immediate local failover), at most 60000, and is shared across the
  entire failover chain, not multiplied by provider count. Waiting is cancellable
  and contains no mutex held across an await. Cancellation never dispatches a
  fallback request. Reservations larger than a whole window fail locally.
* Circuit failure threshold defaults to 1 (1..32), initial cooldown to 500ms
  (1..30000ms), doubling on consecutive failures up to 30 seconds. A success resets
  the provider circuit. Existing per-account rejection cooldown is separate.

All HTTP remains `PolicyHttpClient` mediated. Provider costs, circuit state and
rotation are not rewritten by generated text. A failed remote attempt makes the
successful fallback's total cost unknown; a locally skipped limiter does not.
Unknown pricing remains unknown even when only one eligible fallback remains.
`require_known_prices` still rejects *any* profile without explicit input/output
prices, including profiles restricted to other tiers. It cannot accidentally
admit a future costly fallback. Runtime spend reservation/unknown-cost stopping
semantics are unchanged. Native config and pool fingerprints include routing,
limiter settings and credential reference names, never resolved values.

## Operator list and safe switch

`missy-rs --config /private/config.yaml provider list` is offline and redacts
credential values, reference names, OAuth vault paths and base URLs. It includes
route weights, tiers, tasks and limits. `provider check`/`keypool` retain their
explicit reference-availability inspection, never print values.

`provider switch PROFILE` prints a restart-only change plan. Add `--confirm` to
atomically replace the owned private native configuration with the selected
profile first and `mode: ordered`. Its priority becomes the existing minimum;
other profiles, prices, policy, eligibility and account references are retained.
Tier/task filters still apply, so a restricted profile is not made universally
eligible. This is not a runtime hot swap and never signals an admitted Runtime.
It requires a controlled restart before new settings take effect. This is an
explicit named CLI command, not a model-exposed tool.

The write path refuses symlinks, nonprivate/multilink files, unsafe parent
directories and concurrent config operations. It uses a private same-directory
temporary file, full candidate validation, before-write content comparison,
atomic rename and directory fsync. Existing config rollback and switch share the
native config mutation lock. No account data, OAuth flows or process deployments
are part of a switch.

## Focused verification

`cargo test -p missy-providers -j 2` covers stable weighted distribution, task and
tier isolation, cancellation/backpressure, capacity fallback without false
unknown billing, last eligible unknown-priced fallback, fingerprints, unchanged
HTTP adapters/streaming/account rotation and OAuth fixture behavior.
`cargo test -p missy-cli --bin missy-rs provider_ops::tests -j 2` covers offline
redacted profile projection, no-write plans, atomic switch, retained references,
private file modes and symlink/nonprivate-file refusal. All fixtures use
disposable local endpoints/configuration, no real provider accounts.
