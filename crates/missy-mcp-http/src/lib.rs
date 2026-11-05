//! Native, explicitly pinned remote MCP. No redirects, alternate HTTP clients,
//! environment forwarding, remote schema retrieval, or annotation-based trust.
mod transport;
use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolDispatchRefused, ToolExecutor};
use missy_policy::PolicyHttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};
use tokio_util::sync::CancellationToken;
use transport::Session;
use url::Url;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HttpMcpTransport {
    #[default]
    StreamableHttp,
    LegacySse,
}

/// Credentials are environment variable *names*, never inline values. They are
/// snapshotted at connect; reconnect after rotation to obtain a new fingerprint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HttpMcpConfig {
    pub name: String,
    pub endpoint: String,
    pub origin: String,
    pub manifest_sha256: String,
    pub timeout_seconds: u64,
    pub transport: HttpMcpTransport,
    pub credential_headers_env: BTreeMap<String, String>,
    pub max_response_bytes: usize,
    pub max_argument_bytes: usize,
    pub protocol_version: String,
    /// Legacy SSE's emitted endpoint must equal this operator-approved URL.
    /// Query-bearing legacy transports deliberately fail closed.
    pub message_endpoint: Option<String>,
}
impl Default for HttpMcpConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            endpoint: String::new(),
            origin: String::new(),
            manifest_sha256: String::new(),
            timeout_seconds: 30,
            transport: HttpMcpTransport::StreamableHttp,
            credential_headers_env: BTreeMap::new(),
            max_response_bytes: 1024 * 1024,
            max_argument_bytes: 64 * 1024,
            protocol_version: "2025-03-26".into(),
            message_endpoint: None,
        }
    }
}

fn checked_url(raw: &str) -> Result<Url> {
    if raw.len() > 4096 || raw.bytes().any(|b| b.is_ascii_control() || b == b'\\') {
        bail!("invalid MCP URL");
    }
    let u = Url::parse(raw).map_err(|_| anyhow!("invalid MCP URL"))?;
    if !matches!(u.scheme(), "https" | "http")
        || u.host_str().is_none()
        || !u.username().is_empty()
        || u.password().is_some()
        || u.query().is_some()
        || u.fragment().is_some()
    {
        bail!("MCP URL scheme, credentials, query or fragment denied");
    }
    Ok(u)
}
fn token(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
}
impl HttpMcpConfig {
    pub fn validate(&self) -> Result<()> {
        if !token(&self.name) || self.name.contains("__") {
            bail!("invalid MCP namespace");
        }
        let endpoint = checked_url(&self.endpoint)?;
        let origin = checked_url(&self.origin)?;
        if origin.path() != "/" || endpoint.origin() != origin.origin() {
            bail!("MCP endpoint must match configured exact origin");
        }
        if self.manifest_sha256.len() != 64
            || !self.manifest_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            bail!("MCP manifest SHA256 required");
        }
        if !(1..=600).contains(&self.timeout_seconds)
            || !(1024..=4 * 1024 * 1024).contains(&self.max_response_bytes)
            || !(2..=1024 * 1024).contains(&self.max_argument_bytes)
        {
            bail!("invalid MCP timeout or limits");
        }
        if !["2024-11-05", "2025-03-26", "2025-06-18"].contains(&self.protocol_version.as_str()) {
            bail!("unsupported MCP protocol version");
        }
        match (self.transport, self.message_endpoint.as_deref()) {
            (HttpMcpTransport::LegacySse, Some(raw))
                if checked_url(raw)?.origin() == origin.origin() => {}
            (HttpMcpTransport::StreamableHttp, None) => {}
            _ => bail!(
                "legacy MCP requires an exact same-origin message endpoint; streamable HTTP forbids one"
            ),
        }
        let mut headers = BTreeSet::new();
        if self.credential_headers_env.len() > 8 {
            bail!("too many MCP credential headers");
        }
        for (header, env) in &self.credential_headers_env {
            let header = header.to_ascii_lowercase();
            if !(header == "authorization" || header == "x-api-key" || header == "api-key")
                || !headers.insert(header)
                || env.is_empty()
                || env.len() > 128
                || !env.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            {
                bail!("MCP credential header or environment name denied");
            }
        }
        Ok(())
    }
}

/// Canonical JSON object keys sorted recursively, array order preserved. Pins
/// the complete tools/list result, including annotations and schemas.
pub fn manifest_digest(v: &Value) -> String {
    fn sorted(v: &Value) -> Value {
        match v {
            Value::Object(m) => Value::Object(
                m.iter()
                    .map(|(k, v)| (k.clone(), sorted(v)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
            v => v.clone(),
        }
    }
    hex::encode(Sha256::digest(
        serde_json::to_vec(&sorted(v)).expect("JSON value"),
    ))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Listing {
    tools: Vec<ListedTool>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListedTool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(rename = "inputSchema")]
    schema: Value,
    #[serde(default)]
    annotations: Option<Value>,
}

fn schema(v: &Value) -> Result<jsonschema::Validator> {
    fn keywords(v: &Value, depth: usize) -> Result<()> {
        if depth > 32 {
            bail!("MCP schema nesting exceeds limit");
        }
        if v.is_boolean() {
            return Ok(());
        }
        let object = v
            .as_object()
            .context("MCP schema must be object or boolean")?;
        for (k, value) in object {
            if ![
                "type",
                "title",
                "description",
                "default",
                "examples",
                "enum",
                "const",
                "properties",
                "required",
                "additionalProperties",
                "items",
                "prefixItems",
                "minItems",
                "maxItems",
                "uniqueItems",
                "contains",
                "minContains",
                "maxContains",
                "minLength",
                "maxLength",
                "pattern",
                "format",
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
                "minProperties",
                "maxProperties",
                "patternProperties",
                "propertyNames",
                "dependentRequired",
                "dependentSchemas",
                "allOf",
                "anyOf",
                "oneOf",
                "not",
                "if",
                "then",
                "else",
            ]
            .contains(&k.as_str())
            {
                bail!("unsupported MCP schema keyword");
            }
            match k.as_str() {
                "properties" | "patternProperties" | "dependentSchemas" => {
                    for sub in value.as_object().context("invalid schema map")?.values() {
                        keywords(sub, depth + 1)?;
                    }
                }
                "items"
                | "additionalProperties"
                | "contains"
                | "propertyNames"
                | "not"
                | "if"
                | "then"
                | "else" => keywords(value, depth + 1)?,
                "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                    for sub in value.as_array().context("invalid schema list")? {
                        keywords(sub, depth + 1)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    keywords(v, 0)?;
    if v.get("type").and_then(Value::as_str) != Some("object")
        || v.get("additionalProperties") != Some(&Value::Bool(false))
    {
        bail!("MCP argument schema requires closed object");
    }
    jsonschema::draft202012::options()
        .should_validate_formats(true)
        .should_ignore_unknown_formats(false)
        .build(v)
        .map_err(|_| anyhow!("invalid MCP input schema"))
}

struct Server {
    config: HttpMcpConfig,
    headers: Vec<(String, String)>,
    definitions: Vec<ToolDefinition>,
    validators: BTreeMap<String, jsonschema::Validator>,
}
pub struct HttpMcpTools {
    policy: PolicyHttpClient,
    servers: Vec<Server>,
    fingerprint: String,
}

/// An external effect may have occurred. Never downcast this to
/// ToolDispatchRefused or automatically replay it.
#[derive(Debug)]
pub struct HttpMcpOutcomeUnknown;
impl std::fmt::Display for HttpMcpOutcomeUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("remote MCP outcome unknown; do not replay")
    }
}
impl std::error::Error for HttpMcpOutcomeUnknown {}

impl HttpMcpTools {
    pub async fn connect(policy: PolicyHttpClient, configs: Vec<HttpMcpConfig>) -> Result<Self> {
        if configs.len() > 32 {
            bail!("too many remote MCP servers");
        }
        let mut names = BTreeSet::new();
        let mut servers = Vec::new();
        let mut identities = Vec::new();
        for config in configs {
            config.validate()?;
            if !names.insert(config.name.clone()) {
                bail!("duplicate MCP namespace");
            }
            let mut headers = Vec::new();
            let mut credentials = BTreeMap::new();
            for (header, name) in &config.credential_headers_env {
                let value = std::env::var(name)
                    .map_err(|_| anyhow!("MCP credential environment variable unavailable"))?;
                if value.is_empty()
                    || value.len() > 8192
                    || value.bytes().any(|b| !(32..=126).contains(&b))
                {
                    bail!("invalid MCP credential header value");
                }
                credentials.insert(
                    header.to_ascii_lowercase(),
                    hex::encode(Sha256::digest(value.as_bytes())),
                );
                headers.push((header.to_ascii_lowercase(), value));
            }
            let listing = bounded(&config, &CancellationToken::new(), async {
                let mut s = Session::open(&policy, &config, &headers).await?;
                s.verified_list().await
            })
            .await?;
            let listing: Listing = serde_json::from_value(listing)
                .map_err(|_| anyhow!("unsupported MCP tool listing"))?;
            if listing.tools.len() > 256 {
                bail!("MCP catalog exceeds tool limit");
            }
            let mut definitions = Vec::new();
            let mut validators = BTreeMap::new();
            for t in listing.tools {
                if !token(&t.name) {
                    bail!("invalid MCP tool name");
                }
                let name = format!("{}__{}", config.name, t.name);
                if validators
                    .insert(name.clone(), schema(&t.schema)?)
                    .is_some()
                {
                    bail!("duplicate MCP tool name");
                }
                let _ = t.annotations;
                definitions.push(ToolDefinition {
                    name,
                    description: t.description,
                    parameters: t.schema,
                    side_effect: true,
                });
            }
            identities.push(json!({"config":config,"credential_digests":credentials}));
            servers.push(Server {
                config,
                headers,
                definitions,
                validators,
            });
        }
        let fingerprint = manifest_digest(
            &json!({"kind":"native-http-mcp-v1","policy":policy.config(),"servers":identities}),
        );
        Ok(Self {
            policy,
            servers,
            fingerprint,
        })
    }
    fn gate<'a>(&'a self, ctx: &ToolContext, call: &ToolCall) -> Result<&'a Server> {
        ctx.scope.validate()?;
        if ctx.cancellation.is_cancelled() {
            bail!("MCP request cancelled before dispatch");
        }
        let server = self
            .servers
            .iter()
            .find(|s| s.validators.contains_key(&call.name))
            .context("unknown exact MCP tool namespace")?;
        fn bounded_tree(v: &Value, depth: usize, count: &mut usize) -> Result<()> {
            *count += 1;
            if depth > 32 || *count > 4096 {
                bail!("MCP argument tree exceeds limit");
            }
            match v {
                Value::Object(m) => {
                    for v in m.values() {
                        bounded_tree(v, depth + 1, count)?;
                    }
                }
                Value::Array(a) => {
                    for v in a {
                        bounded_tree(v, depth + 1, count)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        bounded_tree(&call.arguments, 0, &mut 0)?;
        if serde_json::to_vec(&call.arguments)?.len() > server.config.max_argument_bytes {
            bail!("MCP arguments exceed limit");
        }
        if !server.validators[&call.name].is_valid(&call.arguments) {
            bail!("MCP arguments failed schema validation");
        }
        Ok(server)
    }
}
#[async_trait]
impl ToolExecutor for HttpMcpTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.servers
            .iter()
            .flat_map(|s| s.definitions.clone())
            .collect()
    }
    fn configuration_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.gate(ctx, call).map(|_| ())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        let server = self.gate(ctx, call).map_err(|_| ToolDispatchRefused {
            reason: "remote MCP preflight denied".into(),
        })?;
        bounded(&server.config, &ctx.cancellation, async {
            let mut s = Session::open(&self.policy, &server.config, &server.headers).await?;
            s.verified_list().await?;
            let local = call
                .name
                .strip_prefix(&format!("{}__", server.config.name))
                .context("namespace mismatch")?;
            let result = s
                .rpc(
                    "tools/call",
                    json!({"name":local,"arguments":call.arguments}),
                )
                .await?;
            s.verified_list().await?;
            if !result.is_object()
                || result
                    .get("content")
                    .and_then(Value::as_array)
                    .is_none_or(|a| a.len() > 256)
                || result.get("isError").is_some_and(|v| !v.is_boolean())
                || result
                    .get("structuredContent")
                    .is_some_and(|v| !v.is_object())
            {
                bail!("invalid MCP tool result");
            }
            if serde_json::to_vec(&result)?.len() > server.config.max_response_bytes {
                bail!("MCP output exceeds limit");
            }
            if result
                .get("isError")
                .is_some_and(|v| v != &Value::Bool(false))
            {
                bail!("MCP tool reported error");
            }
            Ok(result)
        })
        .await
        .map_err(|_| HttpMcpOutcomeUnknown.into())
    }
}
async fn bounded<T>(
    c: &HttpMcpConfig,
    cancel: &CancellationToken,
    f: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! { biased; _ = cancel.cancelled()=>bail!("MCP cancelled"), result=tokio::time::timeout(Duration::from_secs(c.timeout_seconds),f)=>result.map_err(|_|anyhow!("MCP deadline exceeded"))? }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_references_unknown_keywords_and_invalid_shapes_rejected() {
        for bad in [
            json!({"type":"object","additionalProperties":false,"$ref":"https://attacker.invalid/schema"}),
            json!({"type":"object","additionalProperties":false,"properties":{"x":{"$ref":"#/x"}}}),
            json!({"type":"object","additionalProperties":false,"unknownKeyword":true}),
            json!({"type":"object","additionalProperties":true}),
            json!({"type":"object","additionalProperties":false,"required":4}),
        ] {
            assert!(schema(&bad).is_err(), "{bad}");
        }
        let valid = json!({"type":"object","additionalProperties":false,"properties":{"id":{"type":"integer"}}});
        assert!(schema(&valid).unwrap().is_valid(&json!({"id":12})));
    }
    #[test]
    fn canonical_digest_recurses_objects_and_keeps_array_order() {
        assert_eq!(
            manifest_digest(&json!({"z":{"b":1,"a":2},"a":0})),
            manifest_digest(&json!({"a":0,"z":{"a":2,"b":1}}))
        );
        assert_ne!(
            manifest_digest(&json!([1, 2])),
            manifest_digest(&json!([2, 1]))
        );
    }
}
