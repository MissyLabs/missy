//! Opt-in, pinned stdio MCP. Executables are trusted capabilities, not a sandbox.
use crate::process::{Process, worker};
use anyhow::{Context, Result, bail};
use missy_core::{McpConfig, ToolDefinition};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub(crate) struct Server {
    pub config: McpConfig,
    pub definitions: Vec<ToolDefinition>,
    pub limit: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListedTool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
    #[serde(default)]
    annotations: Option<Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Listing {
    tools: Vec<ListedTool>,
}

/// SHA256 of tools/list result, recursively sorted object keys, array order kept.
pub fn manifest_digest(listing: &Value) -> String {
    fn canonical(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let sorted: std::collections::BTreeMap<_, _> = m.iter().collect();
                Value::Object(
                    sorted
                        .into_iter()
                        .map(|(k, v)| (k.clone(), canonical(v)))
                        .collect(),
                )
            }
            Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
            x => x.clone(),
        }
    }
    hex::encode(Sha256::digest(
        serde_json::to_vec(&canonical(listing)).expect("JSON value"),
    ))
}
fn send(p: &mut Process, v: &Value) -> Result<()> {
    let mut b = serde_json::to_vec(v)?;
    b.push(b'\n');
    p.write(&b)
}
fn rpc(p: &mut Process, id: u64, method: &str, params: Value) -> Result<Value> {
    send(
        p,
        &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}),
    )?;
    let v: Value = serde_json::from_slice(&p.line()?)?;
    let object = v.as_object().context("MCP response must be object")?;
    if object
        .keys()
        .any(|k| !["jsonrpc", "id", "result", "error"].contains(&k.as_str()))
        || v["jsonrpc"] != "2.0"
        || v["id"] != id
    {
        bail!("unexpected MCP JSON-RPC response");
    }
    if object.contains_key("error") || !object.contains_key("result") {
        bail!("MCP request failed");
    }
    Ok(v["result"].clone())
}
fn start(
    config: &McpConfig,
    limit: usize,
    stop: Arc<AtomicBool>,
    cancel: CancellationToken,
) -> Result<Process> {
    let mut p = Process::spawn(
        &config.command,
        &config.env,
        Duration::from_secs(config.timeout_seconds),
        limit,
        stop,
        cancel,
    )?;
    let init = rpc(
        &mut p,
        1,
        "initialize",
        json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"missy-rust","version":"0.1.0"}}),
    )?;
    if init["protocolVersion"] != "2024-11-05" {
        bail!("unsupported MCP protocol version");
    }
    send(
        &mut p,
        &json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    )?;
    Ok(p)
}
fn verified_list(p: &mut Process, id: u64, config: &McpConfig) -> Result<Value> {
    let list = rpc(p, id, "tools/list", json!({}))?;
    if manifest_digest(&list) != config.manifest_sha256.to_ascii_lowercase() {
        bail!("MCP manifest digest mismatch");
    }
    Ok(list)
}
fn definitions(config: &McpConfig, list: Value) -> Result<Vec<ToolDefinition>> {
    let listing: Listing = serde_json::from_value(list)?;
    let mut names = BTreeSet::new();
    listing
        .tools
        .into_iter()
        .map(|t| {
            if t.name.is_empty()
                || !t
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
                || !names.insert(t.name.clone())
            {
                bail!("invalid or duplicate MCP tool name");
            }
            validate_schema(&t.input_schema)?;
            let _ = t.annotations;
            Ok(ToolDefinition {
                name: format!("{}__{}", config.name, t.name),
                description: t.description,
                parameters: t.input_schema,
                side_effect: true,
            })
        })
        .collect()
}
pub(crate) async fn discover(config: McpConfig, limit: usize) -> Result<Server> {
    let c = config.clone();
    let defs = worker(move |stop| {
        let mut p = start(&c, limit, stop, CancellationToken::new())?;
        let list = verified_list(&mut p, 2, &c)?;
        definitions(&c, list)
    })
    .await?;
    Ok(Server {
        config,
        definitions: defs,
        limit,
    })
}
impl Server {
    pub(crate) async fn call(
        &self,
        name: String,
        args: Value,
        cancel: CancellationToken,
    ) -> Result<Value> {
        let definition = self
            .definitions
            .iter()
            .find(|t| t.name == name)
            .context("unknown MCP tool")?;
        validate_value(&definition.parameters, &args)?;
        let c = self.config.clone();
        let limit = self.limit;
        let local_name = name
            .strip_prefix(&format!("{}__", c.name))
            .context("MCP namespace")?
            .to_owned();
        worker(move |stop| {
            let mut p = start(&c, limit, stop, cancel)?;
            let _ = verified_list(&mut p, 2, &c)?;
            let _ = verified_list(&mut p, 3, &c)?;
            let result = rpc(
                &mut p,
                4,
                "tools/call",
                json!({"name":local_name,"arguments":args}),
            )?;
            let _ = verified_list(&mut p, 5, &c)?;
            Ok(result)
        })
        .await
    }
}

fn validate_schema(s: &Value) -> Result<()> {
    let m = s.as_object().context("schema must be object")?;
    for k in m.keys() {
        if ![
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "description",
            "title",
            "default",
        ]
        .contains(&k.as_str())
        {
            bail!("unsupported MCP schema keyword: {k}");
        }
    }
    let kind = s["type"].as_str().context("schema requires single type")?;
    match kind {
        "object" => {
            if s["additionalProperties"] != false {
                bail!("MCP object schemas require additionalProperties=false");
            }
            if let Some(props) = s.get("properties") {
                for sub in props
                    .as_object()
                    .context("properties must be object")?
                    .values()
                {
                    validate_schema(sub)?;
                }
            }
            if let Some(required) = s.get("required") {
                for k in required.as_array().context("required must be array")? {
                    let k = k.as_str().context("required key must be string")?;
                    if s.get("properties").and_then(|p| p.get(k)).is_none() {
                        bail!("required property absent from schema");
                    }
                }
            }
        }
        "array" => validate_schema(s.get("items").context("array requires item schema")?)?,
        "string" | "number" | "integer" | "boolean" | "null" => {}
        _ => bail!("unsupported MCP schema type"),
    }
    if let Some(v) = s.get("enum")
        && v.as_array().is_none_or(|a| a.is_empty())
    {
        bail!("enum must be nonempty array");
    }
    Ok(())
}
fn validate_value(s: &Value, v: &Value) -> Result<()> {
    if let Some(values) = s.get("enum")
        && !values.as_array().context("enum")?.contains(v)
    {
        bail!("value outside enum");
    }
    match s["type"].as_str().context("missing type")? {
        "object" => {
            let m = v.as_object().context("expected object")?;
            for (k, v) in m {
                let schema = s
                    .get("properties")
                    .and_then(|p| p.get(k))
                    .context("unknown MCP argument")?;
                validate_value(schema, v)?;
            }
            if let Some(keys) = s.get("required").and_then(Value::as_array) {
                for k in keys {
                    if !m.contains_key(k.as_str().context("required key")?) {
                        bail!("missing MCP argument");
                    }
                }
            }
        }
        "array" => {
            for v in v.as_array().context("expected array")? {
                validate_value(&s["items"], v)?;
            }
        }
        "string" if v.is_string() => {}
        "number" if v.is_number() => {}
        "integer" if v.is_i64() || v.is_u64() => {}
        "boolean" if v.is_boolean() => {}
        "null" if v.is_null() => {}
        _ => bail!("MCP argument type mismatch"),
    }
    Ok(())
}
