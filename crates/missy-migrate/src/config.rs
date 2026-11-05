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
                        || k.contains("secret")
                        || k.contains("password")
                        || k.contains("token")
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
        Value::Object(o) => o.values().any(|v| contains_credential(v, credentials)),
        _ => false,
    }
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
                ],
                p,
                &loc,
            );
            let enabled = boolean(provider, "enabled", true, p, &loc);
            if provider.get("name").is_some_and(|v| !v.is_string()) {
                p.error(
                    "invalid_type",
                    &loc,
                    "provider name must be a string; no implicit fallback",
                );
            }
            let kind = provider.get("name").and_then(Value::as_str).unwrap_or(key);
            if !["openai", "anthropic", "ollama"].contains(&kind) || !identifier(key) {
                p.error("unsupported_provider", &loc, "only OpenAI, Anthropic and Ollama are convertible; OAuth/CLI providers require fresh native setup");
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
    if let Some(v) = root.get("default_provider") {
        if let Some(name) = v.as_str() {
            if !name.is_empty() {
                if let Some(i) = cfg.providers.iter().position(|pr| pr.name == name) {
                    let pr = cfg.providers.remove(i);
                    cfg.providers.insert(0, pr);
                } else {
                    p.error(
                        "missing_default_provider",
                        "config.default_provider",
                        "default provider is absent, disabled or unsupported",
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
                p.error("network_requires_exact_origins", &format!("config.network.{key}"), "legacy host/domain/CIDR/category/REST grants are not equivalent to exact origins; replace with explicit allowed_origins in the offline copy, review category widening");
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
        if boolean(shell, "unrestricted", false, p, "config.shell.unrestricted") {
            p.error("unrestricted_shell", "config.shell.unrestricted", "unrestricted shell cannot convert; remove it and provide explicit absolute exact_argv vectors");
        }
        if !texts(
            shell,
            "allowed_commands",
            p,
            "config.shell.allowed_commands",
        )
        .is_empty()
        {
            p.error("program_allowlist_unsupported", "config.shell.allowed_commands", "program names or shell command strings cannot convert to exact argv; operator must supply exact_argv vectors");
        }
        if !texts(
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
        if boolean(plugins, "enabled", false, p, "config.plugins.enabled")
            || !texts(
                plugins,
                "allowed_plugins",
                p,
                "config.plugins.allowed_plugins",
            )
            .is_empty()
        {
            p.error(
                "plugins_unsupported",
                "config.plugins",
                "Python plugins cannot become native tools/MCP; fresh audited setup required",
            );
        }
    }
    let mut credentials = vec![];
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
