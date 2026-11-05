# Offline legacy configuration conversion

The converter in `crates/missy-migrate/src/config.rs` accepts an explicit offline
exported YAML via `MigrationInput.config_path`. It never discovers `~/.missy`,
reads `config.d`, resolves vault references, interpolates environment variables,
launches integrations or changes accounts. Merge includes and sanitize secrets
offline first. This is source conversion, not deployment or gateway activation.

Sources: `missy/config/settings.py` (`MissyConfig`, policies, `ProviderConfig`),
`missy/channels/discord/config.py`, `examples/config.yaml`. The production test
fixture follows these actual schemas, not an invented eight-key subset.
Destination schema: `missy_core::Config`.

## Deliverables and disabled templates

Confirmed applicable imports publish validated native `config.yaml`, metadata-only
**`config-review.json`**, `plan.json`, `operator-resolution.json` and the normal
integrity manifest. Verification covers the review file. Applicable conversion
does **not** mean production-ready equivalence. Resolve required items before
admitting replacement work.

Review templates have `enabled: false`, `reviewed: false`,
`required_before_activation: true`, a fixed section/native surface and concrete
instructions. They include **no source values**. They are not executable extension
manifests and are never wired into `Config.extensions`. Flipping their enabled
field cannot activate anything. Separately create schema-valid native extension
configs after review.

## Conversion matrix

| Legacy input | Native candidate and review |
|---|---|
| `config_version` | Validate nonnegative stamp; native version remains 1. |
| `max_iterations` | `runtime.max_steps`, 1..1000, absent legacy default 10. |
| Explicit normalized `max_turn_seconds`, `tool_timeout_seconds`, `max_tokens`, `max_guard_retries` | Bounded runtime fields. These AgentConfig overrides appear in operator YAML; older MissyConfig loaders ignored some. No claim of old-loader support. |
| `max_spend_usd`, `global_max_spend_usd` | USD to microdollars, rounding **down**. Smallest positive cap becomes a stricter cross-session **lifetime** ceiling, not a per-turn replacement. Native per-turn default narrowed if needed. Positive sub-microdollar rejects rather than becomes unlimited. Zero does not remove native safety defaults. |
| `global_budget_period` | Validate total/daily/monthly. No automatic reset imported. Lifetime enforcement is stricter. Required resolution reconciles historical spent/reserved costs and reduces remaining ceiling before activation; new DB must not reset consumed budget. Missing provider pricing fails closed under cost ceilings. |
| Explicit offline `system_prompt` | Bounded nonempty operator prompt retained. No persona/history automatically promoted. Prompt boundary/lineage review required. |
| `temperature` | Validate 0..2, disabled review. Native provider schema lacks an equivalent. |
| `providers` | Preserve YAML priority and supported selected default. OpenAI/Anthropic/Ollama kind, model, safe URL, timeout and environment-only key reference. Disabled entries omitted without resolving/validating inactive options. |
| OAuth/CLI/unknown providers | Disabled setup template. Unsupported/disabled/absent selected default **disables whole candidate pool**, never silently changes accounts. |
| Provider RPM/TPM, breaker/wait, weights, model tiers, worker routing | Disabled review. No alternate model/account/worker egress invented. YAML priority is not claimed to reproduce weighted balancing. |
| Raw/vault keys and multi-key rotation | Removed/redacted, blocking for active supported providers. Use environment references and review account/rotation semantics. |
| `discord` | Channel **disabled** until review. One account/one enabled guild with numeric snowflake channels/users, full mode, no role restrictions can populate inert candidate allowlists. Environment token name/tighter concurrency retained. Never union guild restrictions into cross-products. |
| Discord roles/names/modes/multi-account/multi-guild/DM pairing | No flattened grants. DMs stay disabled. Owner approvals, bot exceptions, user rates, uploads require native review. No live ID resolution. |
| Legacy host/domain/CIDR/category/preset/REST network rules with default-deny | Review+omit grants. Exact origins remain empty, private access false. No schemes/ports inferred from URLs. |
| Explicit offline `allowed_origins`, `allow_private_network` | Strict exact-origin adapter, not inferred permission. Private flag false without origins. Mixing legacy rules with native origins rejects because dropping category/REST restrictions could widen grants. Default-allow rejects. |
| Absolute filesystem roots | Retain reviewed roots without dot traversal. Home-relative and `.missy` roots omitted with review, never expanded/opened or exposed to agent. Protected write exclusions reject rather than disappear. |
| Enabled shell | Only explicit normalized exact argv with absolute executable. Unrestricted/program-name/environment inheritance grants reject. No inferred commands. |
| Disabled shell | Inactive command/env/unrestricted options do not block and never grant execution. |
| Python plugins | Disabled review only. No imports/execution or implicit native tools/MCP. |
| Nondefault `tools`/`agents` restrictions | Blocking. Dropping deny/allow/profile/provider restrictions could widen active native tools. Default full/empty policy accepted. |

Other recognized root fields emit disabled specific templates: `workspace_path`,
`audit_log_path`, `scheduling`, `nomad`, `heartbeat`, `observability`, `vault`,
`proactive`, `sleeptime`, `features`, `retention`, `sandbox`, `container`, `vision`,
`repoeval_foundry`, `tool_intelligence`, `obs`, `vtube`, `desktop`, delegation
count/concurrency/depth and `landlock_enabled`. Extension paths stay unset.
Workspace is not a filesystem grant, telemetry URL is not network authority,
enabled camera is not consent, retention is not deletion permission. No kernel
Landlock enforcement is falsely claimed to be installed.
If Landlock or an enabled sandbox/container was required, candidate filesystem
roots and shell grants are cleared until equivalent native enforcement is
verified. A disabled review template is not a substitute for an isolation boundary.

## Redaction and failure contract

Safe complete denial/omission is a review. Unsupported optional operations are
disabled entirely, not left running after losing restrictions. Unsupported
constraints on active native tools/network/filesystem are errors. Invalid
supported types/bounds, secret fields, and unknown root/policy/provider/active
Discord keys reject. Unknown key names are hashed, values excluded.
Opaque disabled extension sections are never loaded or copied; review their
actual schemas from the private offline export before recreating them.

Known raw credential values are scanned recursively, including unknown secret
keys and disabled sections. Duplication into a supported output field/map key
suppresses the **whole candidate**. Environment-reference identifiers are not
plaintext credentials. Arbitrary secrets disguised as prose cannot be reliably
detected: sanitize the offline copy. Reports may include private operator prompt
or identity metadata; publishing them is a separate operator decision.

## Validation

```text
CARGO_BUILD_JOBS=2 cargo test -p missy-migrate --test config_production --test migration --test production
```

Tests cover full production root/provider/Discord schemas, the real repository
example, bounded runtime/prompt/budgets, emitted native round-trip, apply+manifest
verification, inactive options, denied grants, unsafe restrictions, unknown/secret
redaction and unsupported-default account safety. Fixtures use private temp
exports only. No machine/service/account access is needed.
