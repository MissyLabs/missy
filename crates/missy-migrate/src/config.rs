use super::*;
use missy_core::ProviderConfig;
use serde_json::Map;
use url::Url;

fn object<'a>(v: &'a Value, p: &mut MigrationPlan, loc: &str) -> Option<&'a Map<String, Value>> {
    if let Some(o) = v.as_object() {
        Some(o)
    } else {
        p.error(
            "invalid_type",
            loc,
            "expected an object; value not included",
        );
        None
    }
}
fn unknown(o: &Map<String, Value>, known: &[&str], p: &mut MigrationPlan, loc: &str) {
    for key in o.keys().filter(|k| !known.contains(&k.as_str())) {
        p.error("unsupported_key", &format!("{loc}.unknown_sha256_{}", digest(key.as_bytes())), "unsupported configuration key; not silently dropped (key name hashed to avoid disclosing credentials)");
    }
}
fn boolean(
    o: &Map<String, Value>,
    key: &str,
    default: bool,
    p: &mut MigrationPlan,
    loc: &str,
) -> bool {
    match o.get(key) {
        None => default,
        Some(Value::Bool(v)) => *v,
        _ => {
            p.error("invalid_type", loc, "boolean required");
            default
        }
    }
}
fn texts(o: &Map<String, Value>, key: &str, p: &mut MigrationPlan, loc: &str) -> Vec<String> {
    match o.get(key) {
        None => vec![],
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => {
            a.iter().map(|v| v.as_str().unwrap().into()).collect()
        }
        _ => {
            p.error("invalid_type", loc, "array of strings required");
            vec![]
        }
    }
}
fn environment(v: &str) -> bool {
    !v.is_empty()
        && v.len() <= 128
        && v.bytes()
            .enumerate()
            .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
}
fn identifier(v: &str) -> bool {
    !v.is_empty()
        && v.len() <= 256
        && v.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:/".contains(&b))
}
fn provider_url(v: &str) -> bool {
    !v.contains('*')
        && Url::parse(v).is_ok_and(|u| {
            ["https", "http"].contains(&u.scheme())
                && u.host_str().is_some()
                && u.username().is_empty()
                && u.password().is_none()
                && u.query().is_none()
                && u.fragment().is_none()
        })
}
fn exact_origin(v: &str) -> bool {
    provider_url(v) && !v.contains('*') && Url::parse(v).is_ok_and(|u| u.path() == "/")
}

// Detect credential values even when an offline config copies one into another
// supported field. Do not let a model/name/URL echo a removed API key in a report.
fn raw_credentials(v: &Value, found: &mut Vec<String>, sensitive: bool) {
    match v {
        Value::Object(o) => {
            for (k, v) in o {
                let k = k.to_ascii_lowercase();
                raw_credentials(
                    v,
                    found,
                    sensitive
                        || ((!k.ends_with("_env")
                            && !k.ends_with("_env_var")
                            && !k.ends_with("_envs"))
                            && (k.contains("secret")
                                || k.contains("password")
                                || k.contains("token")
                                || k.contains("credential")
                                || k == "authorization"))
                        || k == "api_key"
                        || k == "api_keys",
                );
            }
        }
        Value::Array(a) => {
            for v in a {
                raw_credentials(v, found, sensitive);
            }
        }
        Value::String(s)
            if sensitive && !s.is_empty() && !(s.starts_with("${") && s.ends_with('}')) =>
        {
            found.push(s.clone())
        }
        _ => {}
    }
}

fn contains_credential(v: &Value, credentials: &[String]) -> bool {
    match v {
        Value::String(s) => credentials.iter().any(|credential| s.contains(credential)),
        Value::Array(a) => a.iter().any(|v| contains_credential(v, credentials)),
        Value::Object(o) => o.iter().any(|(k, v)| {
            credentials.iter().any(|c| k.contains(c)) || contains_credential(v, credentials)
        }),
        _ => false,
    }
}

// Metadata-only templates, never runtime-loaded manifests. No source values.
fn template(p: &mut MigrationPlan, section: &str, native: &str, instruction: &str) {
    if p.config_templates
        .iter()
        .any(|v| v["legacy_section"] == section && v["instruction"] == instruction)
    {
        return;
    }
    p.config_templates
        .push(json!({"legacy_section":section,"native_surface":native,
        "enabled":false,"reviewed":false,"required_before_activation":true,
        "source_values_included":false,"instruction":instruction}));
    p.review("disabled_extension_template", section, instruction);
    p.resolutions.push(Resolution::new(
        "review_config_conversion",
        section,
        instruction,
        true,
    ));
}

fn bounded(
    root: &Map<String, Value>,
    key: &str,
    min: u64,
    max: u64,
    p: &mut MigrationPlan,
) -> Option<u64> {
    let value = root.get(key)?;
    if let Some(n) = value.as_u64().filter(|n| *n >= min && *n <= max) {
        Some(n)
    } else {
        p.error(
            "invalid_bounds",
            &format!("config.{key}"),
            "integer outside supported bounds; value suppressed",
        );
        None
    }
}

fn money(root: &Map<String, Value>, key: &str, p: &mut MigrationPlan) -> Option<u64> {
    let value = root.get(key)?;
    if let Some(v) = value
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= 1_000_000_000.0)
    {
        // Round DOWN so fractional microdollars never increase a ceiling.
        let n = (v * 1_000_000.0).floor() as u64;
        if v > 0.0 && n == 0 {
            p.error(
                "budget_below_precision",
                &format!("config.{key}"),
                "positive budget below one microdollar cannot become unlimited",
            );
            return None;
        }
        Some(n)
    } else {
        p.error(
            "invalid_budget",
            &format!("config.{key}"),
            "finite nonnegative USD budget required; value suppressed",
        );
        None
    }
}

fn main_knobs(root: &Map<String, Value>, cfg: &mut Config, p: &mut MigrationPlan) {
    // Explicit normalized AgentConfig overrides, including older loader omissions.
    for (key, target, min, max) in [
        (
            "max_turn_seconds",
            &mut cfg.runtime.max_turn_seconds,
            1,
            86400,
        ),
        (
            "tool_timeout_seconds",
            &mut cfg.runtime.tool_timeout_seconds,
            1,
            86400,
        ),
        ("max_tokens", &mut cfg.runtime.max_tokens, 1, 100_000_000),
    ] {
        if let Some(n) = bounded(root, key, min, max, p) {
            *target = n;
        }
    }
    if let Some(n) = bounded(root, "max_guard_retries", 0, 10, p) {
        cfg.runtime.max_guard_retries = n as u32;
    }
    // A session ceiling is NOT per-turn. Use a stricter aggregate lifetime cap.
    let session = money(root, "max_spend_usd", p);
    let global = money(root, "global_max_spend_usd", p);
    if let Some(n) = [session, global]
        .into_iter()
        .flatten()
        .filter(|n| *n > 0)
        .min()
    {
        cfg.runtime.global_max_cost_microusd = n;
        cfg.runtime.max_cost_microusd = cfg.runtime.max_cost_microusd.min(n);
        p.review("budget_narrowed", "config.budgets", "Session/period ceilings enforced as a stricter cross-session lifetime ceiling. Historical spend/reservations not imported; reconcile ledgers before admitting work. Missing native provider pricing fails closed.");
        p.resolutions.push(Resolution::new("reconcile_budget_ledger", "budgets", "Reconcile historical spent/reserved costs before activation and reduce the remaining native lifetime ceiling. Do not reset a consumed legacy budget.", true));
    }
    if let Some(v) = root.get("global_budget_period") {
        if !v
            .as_str()
            .is_some_and(|s| ["total", "daily", "monthly"].contains(&s))
        {
            p.error(
                "invalid_budget_period",
                "config.global_budget_period",
                "expected total, daily or monthly",
            );
        } else if v.as_str() != Some("total") {
            template(
                p,
                "global_budget_period",
                "runtime.global_max_cost_microusd",
                "Periodic reset not imported. Lifetime ceiling is stricter; review reset semantics and ledger before changing it.",
            );
        }
    }
    if let Some(v) = root.get("system_prompt") {
        if let Some(s) = v
            .as_str()
            .filter(|s| !s.trim().is_empty() && s.len() <= 100_000)
        {
            cfg.system_prompt = s.into();
            p.review("operator_prompt", "config.system_prompt", "Explicit offline operator prompt retained. Not persona/history promotion; review boundaries before activation.");
        } else {
            p.error(
                "invalid_prompt",
                "config.system_prompt",
                "bounded nonempty operator prompt required",
            );
        }
    }
    if let Some(v) = root.get("temperature") {
        if !v.as_f64().is_some_and(|n| (0.0..=2.0).contains(&n)) {
            p.error(
                "invalid_temperature",
                "config.temperature",
                "temperature must be a finite number in 0..2",
            );
        }
        template(
            p,
            "temperature",
            "providers",
            "Native provider sampling has no equivalent temperature field. Review sampling behavior; no unsupported setting emitted.",
        );
    }
    for (section, native, instruction) in [
        (
            "workspace_path",
            "policy.read_roots/write_roots",
            "Workspace path is not a grant. Keep explicit reviewed filesystem roots; no home expansion or path creation.",
        ),
        (
            "audit_log_path",
            "state audit store",
            "Native audit uses its own ledger. Keep original logs separately; never splice chains or reuse the legacy path.",
        ),
        (
            "scheduling",
            "extensions.automation_config",
            "Recreate scheduler policy and paused jobs with scopes, timezone, concurrency, active hours and misfire review. No jobs enabled.",
        ),
        (
            "nomad",
            "extensions.workloads_config",
            "Rebuild Nomad namespace/pool/image/resource restrictions and reconcile external effects. No workload submission enabled.",
        ),
        (
            "heartbeat",
            "extensions.automation_config",
            "Review heartbeat interval, active hours, workspace and authenticated scope; no heartbeat registered.",
        ),
        (
            "observability",
            "extensions.telemetry_config",
            "Review log retention and exact telemetry egress origins; no exporter enabled.",
        ),
        (
            "vault",
            "native vault",
            "Create a new native vault and reauthenticate. No legacy secret path, key or vault authority copied.",
        ),
        (
            "proactive",
            "extensions.triggers_config",
            "Review trigger conditions, confirmation, scope, cooldown and effect receipts; every trigger remains disabled.",
        ),
        (
            "sleeptime",
            "extensions.automation_config",
            "Review background provider, scope, budget and schedule; no background run enabled.",
        ),
        (
            "features",
            "extensions retrieval/evolution",
            "Review graph/semantic/routing/condenser/proposal semantics; no optional feature enabled.",
        ),
        (
            "retention",
            "extensions.operations_config",
            "Translate retention for each native store; no cleanup job or deletion grant enabled.",
        ),
        (
            "sandbox",
            "extensions.sandbox_config",
            "Rebuild equivalent sandbox restrictions and validate enforcement before activation. No process configuration imported.",
        ),
        (
            "container",
            "extensions.sandbox_config",
            "Review container image pins, mounts, network, resources and enforcement; no container launched.",
        ),
        (
            "vision",
            "extensions.vision_config",
            "Review camera device, consent, limits and memory ownership; no camera or vision process enabled.",
        ),
        (
            "repoeval_foundry",
            "extensions.specialty_config",
            "Review authenticated endpoint, project pin and exact origins; no Foundry integration enabled.",
        ),
        (
            "tool_intelligence",
            "extensions.evolution_config",
            "Review candidate proposal and benchmark gating; no candidate generation, promotion or gating enabled.",
        ),
        (
            "obs",
            "extensions.integrations_config",
            "Review OBS endpoint, environment-only secret, scene allowlist and approval gates; no broadcast enabled.",
        ),
        (
            "vtube",
            "extensions.integrations_config",
            "Review VTube endpoint, plugin identity, environment-only secret and rate limits; no integration enabled.",
        ),
        (
            "desktop",
            "extensions.desktop_config",
            "Review app/window allowlists, installation gates and consent; desktop remains disabled.",
        ),
    ] {
        if root.contains_key(section) {
            template(p, section, native, instruction);
        }
    }
    for (key, max) in [
        ("max_sub_agents", 1000),
        ("max_concurrent_agents", 1000),
        ("max_sub_agent_depth", 100),
    ] {
        if root.contains_key(key) {
            bounded(root, key, 0, max, p);
        }
    }
    if [
        "max_sub_agents",
        "max_concurrent_agents",
        "max_sub_agent_depth",
    ]
    .iter()
    .any(|k| root.contains_key(*k))
    {
        template(
            p,
            "delegation_limits",
            "extensions.subagents_config",
            "Recreate count, concurrency and depth limits. Delegation stays disabled; depth zero never becomes a grant.",
        );
    }
    if root.contains_key("landlock_enabled") {
        boolean(
            root,
            "landlock_enabled",
            false,
            p,
            "config.landlock_enabled",
        );
        template(
            p,
            "landlock_enabled",
            "extensions.sandbox_config",
            "No process-wide Landlock rule installed. Rebuild equivalent kernel restrictions and validate before activation.",
        );
    }
    // Dropping tool restrictions widens authority, unlike disabling an extension.
    for key in ["tools", "agents"] {
        if let Some(v) = root.get(key) {
            let safe = v.as_object().is_some_and(|o| {
                o.is_empty()
                    || (key == "tools"
                        && o.iter().all(|(k, v)| {
                            (k == "profile" && (v.is_null() || v.as_str() == Some("full")))
                                || (["allow", "deny", "also_allow", "disabled_tools"]
                                    .contains(&k.as_str())
                                    && v.as_array().is_some_and(Vec::is_empty))
                                || (["by_provider", "by_model", "groups"].contains(&k.as_str())
                                    && v.as_object().is_some_and(Map::is_empty))
                        }))
            });
            if !safe {
                p.error("tool_restrictions_require_port", &format!("config.{key}"), "Non-default tool restrictions cannot be dropped or broadened; port equivalent execute-time policy first");
            }
        }
    }
}

fn snowflake(s: &str) -> bool {
    (15..=20).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit())
}

fn discord(root: &Map<String, Value>, cfg: &mut Config, p: &mut MigrationPlan) {
    let Some(value) = root.get("discord").filter(|v| !v.is_null()) else {
        return;
    };
    let Some(d) = object(value, p, "config.discord") else {
        return;
    };
    unknown(
        d,
        &[
            "enabled",
            "accounts",
            "auto_approve_uploads",
            "max_concurrent_conversations",
        ],
        p,
        "config.discord",
    );
    let enabled = boolean(d, "enabled", false, p, "config.discord.enabled");
    template(
        p,
        "discord",
        "discord + extensions.discord_tools_config",
        "Discord remains disabled pending review of account, per-user rate limits, bot-author rules, owner approvals and upload confirmations. Candidate snowflake allowlists are inert until explicitly enabled; channel-name/role/mode/pairing policies are never flattened into grants.",
    );
    if let Some(n) = bounded(d, "max_concurrent_conversations", 1, 16, p) {
        cfg.runtime.max_concurrent_turns = cfg.runtime.max_concurrent_turns.min(n as usize);
    }
    // Disabled channel options are deliberately not resolved or revalidated.
    if !enabled {
        return;
    }
    let Some(accounts) = d.get("accounts").and_then(Value::as_array) else {
        p.error(
            "invalid_type",
            "config.discord.accounts",
            "array of account objects required",
        );
        return;
    };
    if accounts.len() != 1 {
        return;
    } // Multi-account has no native main equivalent.
    let Some(account) = object(&accounts[0], p, "config.discord.account") else {
        return;
    };
    unknown(
        account,
        &[
            "token_env_var",
            "token",
            "account_id",
            "application_id",
            "guild_policies",
            "dm_policy",
            "dm_allowlist",
            "owner_ids",
            "ack_reaction",
            "ignore_bots",
            "allow_bots_if_mention_only",
            "allowed_bot_author_ids",
            "auto_thread_threshold",
            "rate_limit_per_minute",
            "vault_dir",
        ],
        p,
        "config.discord.account",
    );
    if account
        .get("token")
        .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
    {
        p.error("raw_credential", "config.discord.account.token", "Direct/vault Discord token removed; use an explicit environment-variable reference in the offline copy");
    }
    if let Some(v) = account.get("token_env_var") {
        if let Some(s) = v.as_str().filter(|s| environment(s)) {
            cfg.discord.token_env = s.into();
        } else {
            p.error(
                "invalid_key_env",
                "config.discord.account.token_env_var",
                "environment-variable identifier required",
            );
        }
    } else {
        cfg.discord.token_env = "DISCORD_BOT_TOKEN".into();
    }
    let Some(guilds) = account.get("guild_policies").and_then(Value::as_object) else {
        return;
    };
    // Never union per-guild restrictions: that creates a cross-product grant.
    if guilds.len() != 1 {
        return;
    }
    let (guild_id, value) = guilds.iter().next().unwrap();
    if !snowflake(guild_id) {
        return;
    }
    let Some(guild) = object(value, p, "config.discord.guild") else {
        return;
    };
    unknown(
        guild,
        &[
            "enabled",
            "require_mention",
            "allowed_channels",
            "allowed_roles",
            "allowed_role_ids",
            "allowed_users",
            "mode",
        ],
        p,
        "config.discord.guild",
    );
    if !boolean(guild, "enabled", true, p, "config.discord.guild.enabled") {
        return;
    }
    let channels = texts(
        guild,
        "allowed_channels",
        p,
        "config.discord.guild.allowed_channels",
    );
    let users = texts(
        guild,
        "allowed_users",
        p,
        "config.discord.guild.allowed_users",
    );
    let restricted = ["allowed_roles", "allowed_role_ids"].iter().any(|k| {
        guild
            .get(*k)
            .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
    }) || guild
        .get("mode")
        .is_some_and(|v| v.as_str() != Some("full"));
    if restricted
        || channels.is_empty()
        || users.is_empty()
        || !channels.iter().chain(&users).all(|s| snowflake(s))
    {
        return;
    }
    cfg.discord.allowed_guilds = vec![guild_id.clone()];
    cfg.discord.allowed_channels = channels;
    cfg.discord.allowed_users = users;
    cfg.discord.require_mention = boolean(
        guild,
        "require_mention",
        false,
        p,
        "config.discord.guild.require_mention",
    );
    // DMs remain disabled: a single native user allowlist cannot independently
    // represent guild/DM principals, pairing state or open DM policy.
}

pub(crate) fn convert(bytes: &[u8], p: &mut MigrationPlan) -> Result<()> {
    // Parse directly to JSON values: no custom YAML tags, secret interpolation,
    // environment reads, vault resolution, Python hooks or credentials lookup.
    let value: Value = serde_yaml_ng::from_slice(bytes).map_err(|_| anyhow!("invalid offline YAML configuration; parse details suppressed to avoid credential disclosure"))?;
    let ordered_yaml: serde_yaml_ng::Value = serde_yaml_ng::from_slice(bytes)
        .map_err(|_| anyhow!("invalid offline YAML configuration; details suppressed"))?;
    let provider_order: Vec<String> = ordered_yaml
        .get("providers")
        .and_then(serde_yaml_ng::Value::as_mapping)
        .map(|m| {
            m.keys()
                .filter_map(serde_yaml_ng::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let Some(root) = object(&value, p, "config") else {
        return Ok(());
    };
    unknown(
        root,
        &[
            "config_version",
            "providers",
            "default_provider",
            "network",
            "filesystem",
            "shell",
            "plugins",
            "max_iterations",
            "workspace_path",
            "audit_log_path",
            "discord",
            "scheduling",
            "nomad",
            "heartbeat",
            "observability",
            "vault",
            "proactive",
            "sleeptime",
            "features",
            "retention",
            "sandbox",
            "container",
            "vision",
            "temperature",
            "max_sub_agents",
            "max_concurrent_agents",
            "max_sub_agent_depth",
            "max_spend_usd",
            "global_max_spend_usd",
            "global_budget_period",
            "landlock_enabled",
            "tools",
            "repoeval_foundry",
            "agents",
            "tool_intelligence",
            "obs",
            "vtube",
            "desktop",
            "system_prompt",
            "max_turn_seconds",
            "tool_timeout_seconds",
            "max_tokens",
            "max_guard_retries",
        ],
        p,
        "config",
    );
    let mut cfg = Config {
        state_dir: PathBuf::from("state"),
        ..Config::default()
    };
    cfg.policy.max_file_bytes = 1024 * 1024;
    cfg.runtime.max_steps = 10; // Legacy AgentConfig default, not native 32.
    main_knobs(root, &mut cfg, p);
    discord(root, &mut cfg, p);
    if let Some(v) = root.get("config_version") {
        if !v.is_u64() {
            p.error(
                "invalid_type",
                "config.config_version",
                "nonnegative legacy version required",
            );
        } else {
            p.review(
                "legacy_version",
                "config.config_version",
                "legacy schema stamp recorded, not used as the native schema version",
            );
        }
    }
    if let Some(v) = root.get("max_iterations") {
        if let Some(n) = v.as_u64().filter(|n| *n > 0 && *n <= 1000) {
            cfg.runtime.max_steps = n as u32;
        } else {
            p.error(
                "invalid_bounds",
                "config.max_iterations",
                "expected bounded positive iteration count",
            );
        }
    }
    if let Some(v) = root.get("providers")
        && let Some(providers) = object(v, p, "config.providers")
    {
        for (index, (key, value)) in providers.iter().enumerate() {
            let loc = format!("config.providers.entry_{index}");
            let Some(provider) = object(value, p, &loc) else {
                continue;
            };
            unknown(
                provider,
                &[
                    "name",
                    "model",
                    "base_url",
                    "timeout",
                    "enabled",
                    "api_key",
                    "api_key_env",
                    "api_keys",
                    "key_rotation_strategy",
                    "oauth_accounts",
                    "fast_model",
                    "premium_model",
                    "context_worker_provider",
                    "context_worker_model",
                    "requests_per_minute",
                    "tokens_per_minute",
                    "max_wait_seconds",
                    "circuit_breaker_threshold",
                    "circuit_breaker_cooldown_seconds",
                    "weight",
                    "account_weights",
                ],
                p,
                &loc,
            );
            let enabled = boolean(provider, "enabled", true, p, &loc);
            if !enabled {
                template(
                    p,
                    "disabled_provider",
                    "providers",
                    "Disabled provider omitted without validating inactive options or resolving credentials. Fresh reviewed setup required before enabling.",
                );
                continue;
            }
            if provider.get("name").is_some_and(|v| !v.is_string()) {
                p.error(
                    "invalid_type",
                    &loc,
                    "provider name must be a string; no implicit fallback",
                );
            }
            let kind = provider.get("name").and_then(Value::as_str).unwrap_or(key);
            if !["openai", "anthropic", "ollama"].contains(&kind) || !identifier(key) {
                template(
                    p,
                    "unsupported_provider",
                    "extensions.cli_provider_config",
                    "OAuth/CLI or unrecognized provider remains disabled. Review native kind, exact CLI argv, sandbox, account and authentication before enabling.",
                );
                continue;
            }
            let model = provider.get("model").and_then(Value::as_str).unwrap_or("");
            if !identifier(model) {
                p.error(
                    "invalid_model",
                    &loc,
                    "nonempty safe model identifier required; value suppressed",
                );
                continue;
            }
            let mut out = ProviderConfig {
                name: key.clone(),
                kind: kind.into(),
                model: model.into(),
                timeout_seconds: 30,
                ..ProviderConfig::default()
            };
            if [
                "requests_per_minute",
                "tokens_per_minute",
                "max_wait_seconds",
                "circuit_breaker_threshold",
                "circuit_breaker_cooldown_seconds",
                "weight",
                "fast_model",
                "premium_model",
                "context_worker_provider",
                "context_worker_model",
                "key_rotation_strategy",
                "oauth_accounts",
                "account_weights",
            ]
            .iter()
            .any(|k| provider.contains_key(*k))
            {
                template(
                    p,
                    "provider_operations",
                    "providers",
                    "Review RPM/TPM, wait/breaker, weighted routing, model tiers and worker egress before activation. No alternate model/account or egress inferred; native fallback preserves YAML priority only.",
                );
            }
            out.base_url = provider
                .get("base_url")
                .and_then(Value::as_str)
                .unwrap_or(match kind {
                    "openai" => "https://api.openai.com/v1",
                    "anthropic" => "https://api.anthropic.com/v1",
                    _ => "http://127.0.0.1:11434",
                })
                .into();
            if provider
                .get("base_url")
                .is_some_and(|v| !v.is_string() && !v.is_null())
                || !provider_url(&out.base_url)
            {
                p.error("unsafe_provider_url", &loc, "HTTP(S) provider URL must not contain credentials, query, fragment or wildcard");
                continue;
            }
            if let Some(v) = provider.get("timeout") {
                if let Some(n) = v.as_u64().filter(|n| *n > 0 && *n <= 3600) {
                    out.timeout_seconds = n;
                } else {
                    p.error(
                        "invalid_timeout",
                        &loc,
                        "provider timeout must be 1..3600 seconds",
                    );
                }
            }
            let raw = provider.get("api_key");
            let mut env = provider
                .get("api_key_env")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if provider
                .get("api_key_env")
                .is_some_and(|v| !v.as_str().is_some_and(environment))
            {
                p.error(
                    "invalid_key_env",
                    &loc,
                    "api_key_env must independently be an environment-variable identifier",
                );
            }
            if let Some(v) = raw.filter(|v| !v.is_null()) {
                if let Some(e) = v
                    .as_str()
                    .and_then(|s| s.strip_prefix("${").and_then(|s| s.strip_suffix('}')))
                    .filter(|s| environment(s))
                {
                    if env.as_ref().is_some_and(|given| given != e) {
                        p.error(
                            "conflicting_key_env",
                            &loc,
                            "conflicting environment-only credential references",
                        );
                    }
                    env = Some(e.into());
                } else if v.as_str() != Some("") {
                    // NEVER put the raw credential in a report, config or error.
                    p.error("raw_credential", &loc, "raw/vault credential removed and redacted; replace it with an explicit api_key_env reference in the offline copy");
                }
            }
            if provider
                .get("api_keys")
                .is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty()))
            {
                p.error("credential_rotation_unsupported", &loc, "multiple credentials removed and redacted; rotation cannot be silently converted to one account");
            }
            if provider.contains_key("api_key_env")
                && env.as_deref().is_none_or(|e| !environment(e))
            {
                p.error(
                    "invalid_key_env",
                    &loc,
                    "api_key_env must be an environment-variable identifier",
                );
                env = None;
            }
            out.api_key_env = env.or_else(|| match kind {
                "openai" => Some("OPENAI_API_KEY".into()),
                "anthropic" => Some("ANTHROPIC_API_KEY".into()),
                _ => None,
            });
            if enabled {
                cfg.providers.push(out);
            } else {
                p.review(
                    "disabled_provider",
                    &loc,
                    "disabled provider omitted; no native provider was enabled",
                );
            }
        }
    }
    cfg.providers.sort_by_key(|pr| {
        provider_order
            .iter()
            .position(|name| name == &pr.name)
            .unwrap_or(usize::MAX)
    });
    if root
        .get("default_provider")
        .is_none_or(|v| v.as_str() == Some(""))
    {
        let implicit = provider_order.iter().find(|key| {
            root.get("providers")
                .and_then(|v| v.get(*key))
                .and_then(|v| v.get("enabled"))
                .and_then(Value::as_bool)
                .unwrap_or(true)
        });
        if implicit.is_some_and(|key| !cfg.providers.iter().any(|pr| &pr.name == key)) {
            cfg.providers.clear();
            template(
                p,
                "default_provider",
                "providers",
                "Implicit YAML-priority default is unsupported. Entire provider pool disabled to avoid silent account/model substitution. Configure reviewed native default explicitly.",
            );
        }
    }
    if let Some(v) = root.get("default_provider") {
        if let Some(name) = v.as_str() {
            if !name.is_empty() {
                if let Some(i) = cfg.providers.iter().position(|pr| pr.name == name) {
                    let pr = cfg.providers.remove(i);
                    cfg.providers.insert(0, pr);
                } else {
                    cfg.providers.clear();
                    template(
                        p,
                        "default_provider",
                        "providers",
                        "Selected default is absent, disabled or unsupported. Entire provider pool disabled to prevent silent account/model substitution. Configure reviewed native default and fallback explicitly.",
                    );
                }
            }
        } else {
            p.error(
                "invalid_type",
                "config.default_provider",
                "provider name must be a string",
            );
        }
    }
    if let Some(v) = root.get("network")
        && let Some(net) = object(v, p, "config.network")
    {
        unknown(
            net,
            &[
                "default_deny",
                "allowed_origins",
                "allow_private_network",
                "allowed_cidrs",
                "allowed_domains",
                "allowed_hosts",
                "provider_allowed_hosts",
                "tool_allowed_hosts",
                "discord_allowed_hosts",
                "presets",
                "rest_policies",
            ],
            p,
            "config.network",
        );
        if !boolean(net, "default_deny", true, p, "config.network.default_deny") {
            p.error(
                "unrestricted_network",
                "config.network.default_deny",
                "default-allow network cannot convert; operator must specify exact origins",
            );
        }
        for key in [
            "allowed_cidrs",
            "allowed_domains",
            "allowed_hosts",
            "provider_allowed_hosts",
            "tool_allowed_hosts",
            "discord_allowed_hosts",
            "presets",
            "rest_policies",
        ] {
            if net
                .get(key)
                .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
            {
                if net.contains_key("allowed_origins")
                    || !boolean(net, "default_deny", true, p, "config.network.default_deny")
                {
                    p.error("network_requires_exact_origins", &format!("config.network.{key}"), "Cannot mix non-equivalent legacy/category/REST rules with native grants or default-allow. Replace the whole policy with reviewed exact origins first.");
                } else {
                    template(
                        p,
                        "network",
                        "policy.allowed_origins",
                        "Legacy domain/host/CIDR/category/REST rules not activated. Native network stays default-deny with NO inferred origins or private access. Review exact scheme, host, port and category restrictions before granting egress.",
                    );
                }
            }
        }
        for origin in texts(net, "allowed_origins", p, "config.network.allowed_origins") {
            if exact_origin(&origin) {
                cfg.policy.allowed_origins.push(origin);
            } else {
                p.error(
                    "invalid_origin",
                    "config.network.allowed_origins",
                    "exact HTTP(S) origins only; no wildcard, credentials, path, query or fragment",
                );
            }
        }
        cfg.policy.allow_private_network = boolean(
            net,
            "allow_private_network",
            false,
            p,
            "config.network.allow_private_network",
        );
        if cfg.policy.allowed_origins.is_empty() {
            cfg.policy.allow_private_network = false;
        }
        if net.contains_key("allowed_origins") || net.contains_key("allow_private_network") {
            p.review("native_network_optin", "config.network", "explicit native network fields retained without inferring origins or private-network access");
        }
    }
    if let Some(v) = root.get("filesystem")
        && let Some(fs) = object(v, p, "config.filesystem")
    {
        unknown(
            fs,
            &[
                "allowed_read_paths",
                "allowed_write_paths",
                "protected_write_paths",
            ],
            p,
            "config.filesystem",
        );
        if fs
            .get("protected_write_paths")
            .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
        {
            p.error(
                "protected_paths_unsupported",
                "config.filesystem.protected_write_paths",
                "protected-path exclusions cannot be discarded; redesign roots before import",
            );
        }
        for (key, target) in [
            ("allowed_read_paths", &mut cfg.policy.read_roots),
            ("allowed_write_paths", &mut cfg.policy.write_roots),
        ] {
            for path in texts(fs, key, p, &format!("config.filesystem.{key}")) {
                let path = PathBuf::from(path);
                if path.starts_with("~") || path.components().any(|c| c.as_os_str() == ".missy") {
                    template(
                        p,
                        "filesystem",
                        "policy.read_roots/write_roots",
                        "Home-relative and legacy .missy roots omitted, not expanded or opened. Supply explicit reviewed absolute offline/native roots before granting filesystem access; legacy state/credentials never become tool roots.",
                    );
                    continue;
                }
                if path.is_absolute()
                    && !path
                        .as_os_str()
                        .as_encoded_bytes()
                        .split(|b| *b == b'/')
                        .any(|s| s == b"." || s == b".." || s == b".missy")
                {
                    target.push(path);
                } else {
                    p.error("unsafe_root", &format!("config.filesystem.{key}"), "filesystem roots must be explicit absolute paths without dot components or Python .missy; no home expansion");
                }
            }
        }
    }
    if let Some(v) = root.get("shell")
        && let Some(shell) = object(v, p, "config.shell")
    {
        unknown(
            shell,
            &[
                "enabled",
                "unrestricted",
                "allowed_commands",
                "allowed_env_vars",
                "exact_argv",
            ],
            p,
            "config.shell",
        );
        cfg.policy.shell_enabled = boolean(shell, "enabled", false, p, "config.shell.enabled");
        if !cfg.policy.shell_enabled {
            template(
                p,
                "shell",
                "policy.allowed_commands",
                "Shell remains disabled. Inactive legacy command/environment/unrestricted options were not activated.",
            );
        }
        if cfg.policy.shell_enabled
            && boolean(shell, "unrestricted", false, p, "config.shell.unrestricted")
        {
            p.error("unrestricted_shell", "config.shell.unrestricted", "unrestricted shell cannot convert; remove it and provide explicit absolute exact_argv vectors");
        }
        if cfg.policy.shell_enabled
            && !texts(
                shell,
                "allowed_commands",
                p,
                "config.shell.allowed_commands",
            )
            .is_empty()
        {
            p.error("program_allowlist_unsupported", "config.shell.allowed_commands", "program names or shell command strings cannot convert to exact argv; operator must supply exact_argv vectors");
        }
        if cfg.policy.shell_enabled
            && !texts(
                shell,
                "allowed_env_vars",
                p,
                "config.shell.allowed_env_vars",
            )
            .is_empty()
        {
            p.error(
                "shell_environment_unsupported",
                "config.shell.allowed_env_vars",
                "environment inheritance grant cannot be silently dropped",
            );
        }
        if let Some(v) = shell.get("exact_argv") {
            let vectors: Option<Vec<Vec<String>>> = serde_json::from_value(v.clone()).ok();
            if let Some(vectors) = vectors.filter(|a| {
                a.iter().all(|v| {
                    !v.is_empty()
                        && Path::new(&v[0]).is_absolute()
                        && v.iter().all(|s| !s.contains('\0'))
                })
            }) {
                cfg.policy.allowed_commands = vectors;
                p.review("native_shell_optin", "config.shell.exact_argv", "operator-supplied native exact argv retained; shell enabled only if legacy enabled was explicitly true");
            } else {
                p.error(
                    "invalid_exact_argv",
                    "config.shell.exact_argv",
                    "exact argv requires nonempty vectors with absolute executable and no NUL",
                );
            }
        }
    }
    if let Some(v) = root.get("plugins")
        && let Some(plugins) = object(v, p, "config.plugins")
    {
        unknown(
            plugins,
            &["enabled", "allowed_plugins"],
            p,
            "config.plugins",
        );
        boolean(plugins, "enabled", false, p, "config.plugins.enabled");
        template(
            p,
            "plugins",
            "extensions.plugins_config",
            "Python plugins and inactive allowlists remain disabled, never imported or executed. Port to reviewed native tools with digest and capability policy before enabling.",
        );
    }
    let mut credentials = vec![];
    // A review receipt is not an installed kernel/container boundary.
    let isolation_required = root.get("landlock_enabled").and_then(Value::as_bool) == Some(true)
        || ["sandbox", "container"].iter().any(|k| {
            root.get(*k)
                .and_then(|v| v.get("enabled"))
                .and_then(Value::as_bool)
                == Some(true)
        });
    if isolation_required {
        cfg.policy.read_roots.clear();
        cfg.policy.write_roots.clear();
        cfg.policy.shell_enabled = false;
        cfg.policy.allowed_commands.clear();
        template(
            p,
            "local_isolation",
            "policy + extensions.sandbox_config",
            "Legacy isolation requires native enforcement review. All candidate filesystem and shell grants disabled until equivalent Landlock/sandbox/container boundaries are verified; a review template is not enforcement.",
        );
    }
    raw_credentials(&value, &mut credentials, false);
    if !credentials.is_empty() && contains_credential(&serde_json::to_value(&cfg)?, &credentials) {
        p.error("credential_in_converted_field", "config", "credential also appeared in a supported field; converted candidate suppressed entirely");
        p.converted_config = None;
        return Ok(());
    }
    cfg.validate()
        .map_err(|_| anyhow!("converted config validation failed; no source values included"))?;
    p.converted_config = Some(cfg);
    Ok(())
}
