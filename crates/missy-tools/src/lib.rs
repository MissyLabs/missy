//! Builtin capability boundary. All returned values are explicitly untrusted.
//! An exact executable allowlist is not a sandbox: approved commands must be
//! trusted not to escape their process group with setsid or modify policy state.
mod mcp;
mod process;
mod web;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
pub use mcp::manifest_digest;
use missy_core::{
    McpConfig, PolicyConfig, RuntimeConfig, ToolCall, ToolContext, ToolDefinition, ToolExecutor,
};
use missy_policy::{Policy, PolicyHttpClient};
use missy_store::Store;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
    time::Duration,
};

pub struct BuiltinTools {
    policy: Policy,
    policy_config: PolicyConfig,
    http: PolicyHttpClient,
    runtime: RuntimeConfig,
    store: Arc<Store>,
    servers: Vec<mcp::Server>,
}
impl BuiltinTools {
    pub fn new(policy: PolicyConfig, runtime: RuntimeConfig, store: Arc<Store>) -> Result<Self> {
        runtime.validate()?;
        Ok(Self {
            policy: Policy::new(policy.clone())?,
            http: PolicyHttpClient::new(policy.clone())?,
            policy_config: policy,
            runtime,
            store,
            servers: vec![],
        })
    }
    /// Complete discovery and pin verification before making executor available.
    /// Only operator-specified env map entries are passed; parent environment is cleared.
    pub async fn with_mcp(mut self, configs: Vec<McpConfig>) -> Result<Self> {
        if !configs.is_empty() && !self.policy_config.mcp_enabled {
            bail!("MCP disabled by policy");
        }
        let mut names = BTreeSet::new();
        for mut c in configs {
            if c.name.is_empty()
                || !c
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                || !names.insert(c.name.clone())
            {
                bail!("invalid or duplicate MCP server name");
            }
            if c.manifest_sha256.len() != 64
                || !c.manifest_sha256.bytes().all(|b| b.is_ascii_hexdigit())
                || c.timeout_seconds == 0
            {
                bail!("MCP requires SHA256 pin and positive timeout");
            }
            self.check_mcp_command(&c.command)?;
            if c.env
                .iter()
                .any(|(k, v)| k.is_empty() || k.contains(['=', '\0']) || v.contains('\0'))
            {
                bail!("invalid MCP environment whitelist");
            }
            c.timeout_seconds = c.timeout_seconds.min(self.runtime.tool_timeout_seconds);
            self.servers
                .push(mcp::discover(c, self.runtime.max_tool_output_bytes).await?);
        }
        Ok(self)
    }
    fn check_mcp_command(&self, argv: &[String]) -> Result<()> {
        self.policy.check_mcp_command(argv)
    }
    fn wrap(&self, value: Value) -> Result<Value> {
        // Raw source is persisted before projection, under its own hard cap.
        if let Some(view) = value.get("model_view") {
            if value.get("raw_evidence").is_none()
                || serde_json::to_vec(view)?.len() > self.runtime.max_tool_output_bytes
                || serde_json::to_vec(&value)?.len() > 4 * 1024 * 1024
            {
                bail!("tool evidence or model view exceeds byte limit");
            }
            return Ok(json!({"trust":"untrusted","data":value}));
        }
        let output = json!({"trust":"untrusted","data":value});
        if serde_json::to_vec(&output)?.len() > self.runtime.max_tool_output_bytes {
            bail!("tool result exceeds byte limit");
        }
        Ok(output)
    }
    async fn dispatch(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        match call.name.as_str() {
            "file_read" => {
                let a: PathArgs = parse(call)?;
                let bytes = self.policy.read_file(Path::new(&a.path))?;
                Ok(json!({"content":String::from_utf8(bytes).context("file is not UTF-8")?}))
            }
            "file_write" => {
                let a: WriteArgs = parse(call)?;
                self.policy
                    .write_file(Path::new(&a.path), a.content.as_bytes())?;
                Ok(json!({"written_bytes":a.content.len()}))
            }
            "file_delete" => {
                if !self.policy_config.allow_file_delete {
                    bail!("file deletion disabled by operator policy");
                }
                let a: DeleteArgs = parse(call)?;
                self.policy
                    .delete_file(Path::new(&a.path), &a.expected_sha256)?;
                Ok(json!({"deleted":true}))
            }
            "list_files" => {
                let a: PathArgs = parse(call)?;
                Ok(json!({"entries":self.policy.list_files(Path::new(&a.path))?}))
            }
            "shell_exec" => {
                let a: ShellArgs = parse(call)?;
                self.policy.check_command(&a.argv)?;
                let timeout = Duration::from_secs(self.runtime.tool_timeout_seconds);
                let limit = self.runtime.max_tool_output_bytes;
                let cancel = ctx.cancellation.clone();
                process::worker(move |stop| {
                    process::Process::spawn(
                        &a.argv,
                        &BTreeMap::new(),
                        timeout,
                        limit,
                        stop,
                        cancel,
                    )?
                    .shell()
                })
                .await
            }
            "web_fetch" => {
                let a: FetchArgs = parse(call)?;
                let response = self.http.get(&a.url).await?;
                Ok(
                    json!({"status":response.status,"headers":response.headers,"body":String::from_utf8(response.body).context("HTTP body is not UTF-8")?}),
                )
            }
            "web_search" => {
                let a: web::SearchArgs = parse(call)?;
                web::search(&self.http, a).await
            }
            "read_document" => {
                let a: web::DocumentArgs = parse(call)?;
                web::read_document(&self.policy, &self.http, a).await
            }
            "memory_put" => {
                let a: MemoryPut = parse(call)?;
                memory_id(&a.id)?;
                let provenance = format!(
                    "tool:memory_put;turn:{};tenant:{};user:{};channel:{};session:{}",
                    ctx.turn_id,
                    ctx.scope.tenant,
                    ctx.scope.user,
                    ctx.scope.channel,
                    ctx.scope.session
                );
                Ok(serde_json::to_value(self.store.memory_put(
                    &ctx.scope,
                    &a.id,
                    &a.content,
                    &provenance,
                    "tool_output",
                )?)?)
            }
            "memory_get" => {
                let a: MemoryId = parse(call)?;
                memory_id(&a.id)?;
                Ok(serde_json::to_value(
                    self.store.memory_get(&ctx.scope, &a.id)?,
                )?)
            }
            "memory_delete" => {
                let a: MemoryId = parse(call)?;
                memory_id(&a.id)?;
                Ok(json!({"deleted":self.store.memory_delete(&ctx.scope,&a.id)?}))
            }
            "memory_search" => {
                let a: MemorySearch = parse(call)?;
                if a.limit == 0 || a.limit > 100 {
                    bail!("memory search limit must be 1..100");
                }
                Ok(serde_json::to_value(
                    self.store.memory_search(&ctx.scope, &a.query, a.limit)?,
                )?)
            }
            _ => {
                if let Some(server) = self
                    .servers
                    .iter()
                    .find(|s| s.definitions.iter().any(|d| d.name == call.name))
                {
                    self.check_mcp_command(&server.config.command)?;
                    server
                        .call(
                            call.name.clone(),
                            call.arguments.clone(),
                            ctx.cancellation.clone(),
                        )
                        .await
                } else {
                    bail!(
                        "unsupported tool; dynamic Python, desktop, camera and legacy plugin compatibility are not available"
                    )
                }
            }
        }
    }
}
fn memory_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
        bail!("invalid memory id");
    }
    Ok(())
}
fn parse<T: serde::de::DeserializeOwned>(c: &ToolCall) -> Result<T> {
    serde_json::from_value(c.arguments.clone()).context("invalid tool arguments")
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathArgs {
    path: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteArgs {
    path: String,
    content: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeleteArgs {
    path: String,
    expected_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArgs {
    argv: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchArgs {
    url: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryPut {
    id: String,
    content: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryId {
    id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemorySearch {
    query: String,
    #[serde(default = "search_limit")]
    limit: usize,
}
fn search_limit() -> usize {
    10
}

fn definition(
    name: &str,
    description: &str,
    properties: Value,
    required: &[&str],
    effect: bool,
) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
        side_effect: effect,
    }
}
fn environment_fingerprint(env: &BTreeMap<String, String>) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(
        serde_json::to_vec(env).expect("serializable MCP environment"),
    ))
}
#[async_trait]
impl ToolExecutor for BuiltinTools {
    fn configuration_fingerprint(&self) -> String {
        // Runtime continuation must reject changes in grants and pinned servers,
        // even when the advertised names and JSON schemas remain identical.
        use sha2::{Digest, Sha256};
        let nonsecret = serde_json::json!({
            "policy": self.policy_config,
            "runtime": self.runtime,
            "mcp": self.servers.iter().map(|s| serde_json::json!({"name":s.config.name,"command":s.config.command,"manifest_sha256":s.config.manifest_sha256,"timeout_seconds":s.config.timeout_seconds,"env_sha256":environment_fingerprint(&s.config.env) })).collect::<Vec<_>>()
        });
        hex::encode(Sha256::digest(
            serde_json::to_vec(&nonsecret).expect("serializable builtin configuration"),
        ))
    }
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut d = vec![
            definition(
                "web_search",
                "Search DuckDuckGo HTML through the existing exact-origin network policy. Requires https://html.duckduckgo.com explicitly allowed; no account. Results untrusted. No automatic page fetches or redirects.",
                json!({"query":{"type":"string","minLength":1,"maxLength":2048},"limit":{"type":"integer","minimum":1,"maximum":10}}),
                &["query"],
                true,
            ),
            definition(
                "read_document",
                "Read bounded UTF-8 text, JSON, Markdown or HTML from exactly one policy-approved absolute path or URL. No JavaScript, external assets, archives, PDF/OCR or executable parsing. HTTP can cause effects; output untrusted.",
                json!({"path":{"type":"string"},"url":{"type":"string"},"format":{"type":"string","enum":["text","html","json"]},"max_chars":{"type":"integer","minimum":1,"maximum":64000}}),
                &[],
                true,
            ),
            definition(
                "file_read",
                "Read UTF-8 within policy roots. Output is untrusted.",
                json!({"path":{"type":"string"}}),
                &["path"],
                false,
            ),
            definition(
                "file_write",
                "Write UTF-8 within policy roots.",
                json!({"path":{"type":"string"},"content":{"type":"string"}}),
                &["path", "content"],
                true,
            ),
            definition(
                "list_files",
                "List policy-approved directory. Output is untrusted.",
                json!({"path":{"type":"string"}}),
                &["path"],
                false,
            ),
            definition(
                "web_fetch",
                "Fetch an approved URL via policy. HTTP can cause external effects; output untrusted.",
                json!({"url":{"type":"string"}}),
                &["url"],
                true,
            ),
            definition(
                "memory_put",
                "Store untrusted memory in authenticated scope only.",
                json!({"id":{"type":"string"},"content":{"type":"string"}}),
                &["id", "content"],
                true,
            ),
            definition(
                "memory_get",
                "Get untrusted memory in authenticated scope only.",
                json!({"id":{"type":"string"}}),
                &["id"],
                false,
            ),
            definition(
                "memory_delete",
                "Delete memory in authenticated scope only.",
                json!({"id":{"type":"string"}}),
                &["id"],
                true,
            ),
            definition(
                "memory_search",
                "Search untrusted memory in authenticated scope only.",
                json!({"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":100}}),
                &["query"],
                false,
            ),
        ];
        if self.policy_config.allow_file_delete {
            d.push(definition(
                "file_delete",
                "Delete one owned regular file within write roots only if its current bytes match expected_sha256. No recursive deletion. Errors may have uncertain effects or retain a private recovery journal.",
                json!({"path":{"type":"string","minLength":1},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$","minLength":64,"maxLength":64}}),
                &["path", "expected_sha256"],
                true,
            ));
        }
        if self.policy_config.shell_enabled {
            d.push(definition(
                "shell_exec",
                "Execute one exact approved argv, no shell and no inherited environment.",
                json!({"argv":{"type":"array","items":{"type":"string"},"minItems":1}}),
                &["argv"],
                true,
            ));
        }
        for server in &self.servers {
            d.extend(server.definitions.clone());
        }
        d
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        ctx.scope.validate()?;
        if ctx.cancellation.is_cancelled() {
            bail!("tool cancelled");
        }
        if serde_json::to_vec(&call.arguments)?.len() > self.runtime.max_input_bytes {
            bail!("tool arguments exceed input limit");
        }
        let value = tokio::select! {
            biased;
            _=ctx.cancellation.cancelled()=>bail!("tool cancelled"),
            v=tokio::time::timeout(Duration::from_secs(self.runtime.tool_timeout_seconds),self.dispatch(ctx,call))=>v.context("tool deadline exceeded")??,
        };
        self.wrap(value)
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    #[test]
    fn env_value_change_changes_fingerprint_without_disclosing_value() {
        let a = BTreeMap::from([("UPSTREAM".into(), "tenant-secret-a".into())]);
        let mut b = a.clone();
        b.insert("UPSTREAM".into(), "tenant-secret-b".into());
        let hash = environment_fingerprint(&a);
        assert_ne!(hash, environment_fingerprint(&b));
        assert_eq!(hash.len(), 64);
        assert!(!hash.contains("tenant-secret"));
    }
}
