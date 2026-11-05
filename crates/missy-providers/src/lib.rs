//! Bounded non-streaming provider adapters. All networking is policy mediated.
//! No adapter executes tools: failover only retries the same completion input.
use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use missy_core::{
    Completion, Message, PolicyConfig, Provider, ProviderConfig, ToolCall, ToolDefinition, Usage,
};
use missy_policy::PolicyHttpClient;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;

/// Base URL is an API root: OpenAI `/v1`, Anthropic `/v1`, Ollama origin.
/// No default external endpoint or implicit credential lookup is provided.
pub struct HttpProvider {
    config: ProviderConfig,
    client: PolicyHttpClient,
    endpoint: String,
}
impl HttpProvider {
    pub fn new(config: ProviderConfig, client: PolicyHttpClient) -> Result<Self> {
        ensure!(
            !config.name.is_empty() && !config.model.is_empty(),
            "provider name and model required"
        );
        ensure!(
            (1..=600).contains(&config.timeout_seconds),
            "provider timeout must be 1..600 seconds"
        );
        ensure!(
            (1..=1_000_000).contains(&config.max_output_tokens),
            "invalid output token bound"
        );
        let mut url = url::Url::parse(&config.base_url)
            .map_err(|_| anyhow::anyhow!("invalid provider URL"))?;
        ensure!(
            ["http", "https"].contains(&url.scheme()) && url.host_str().is_some(),
            "invalid provider URL scheme/host"
        );
        ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "provider URL cannot carry credentials, query or fragment"
        );
        let suffix = match config.kind.as_str() {
            "openai" => "/chat/completions",
            "anthropic" => "/messages",
            "ollama" => "/api/chat",
            _ => bail!("unsupported provider kind"),
        };
        let path = format!("{}{}", url.path().trim_end_matches('/'), suffix);
        url.set_path(&path);
        if let Some(key) = &config.api_key_env {
            validate_env_name(key)?;
        }
        let client = client.with_timeout(Duration::from_secs(config.timeout_seconds))?;
        Ok(Self {
            config,
            client,
            endpoint: url.into(),
        })
    }
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }
    pub fn costs_known(&self) -> bool {
        self.config.input_microusd_per_million.is_some()
            && self.config.output_microusd_per_million.is_some()
    }
    fn headers(&self) -> Result<Vec<(String, String)>> {
        let mut headers = vec![("content-type".into(), "application/json".into())];
        if self.config.kind == "anthropic" {
            headers.push(("anthropic-version".into(), "2023-06-01".into()));
        }
        if let Some(env) = &self.config.api_key_env {
            let key = std::env::var(env).map_err(|_| {
                anyhow::anyhow!("provider credential environment reference is unavailable")
            })?;
            ensure!(
                !key.is_empty() && key.len() <= 16_384 && key.bytes().all(|b| b.is_ascii_graphic()),
                "provider credential is empty, oversized or contains forbidden bytes"
            );
            if self.config.kind == "anthropic" {
                headers.push(("x-api-key".into(), key));
            } else {
                headers.push(("authorization".into(), format!("Bearer {key}")));
            }
        }
        Ok(headers)
    }
    fn payload(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Value> {
        validate_input(messages, tools)?;
        let tool_values: Vec<Value> = tools.iter().map(|t| json!({"type":"function", "function":{"name":t.name,"description":t.description,"parameters":t.parameters}})).collect();
        match self.config.kind.as_str() {
            "openai" => {
                let rows: Vec<Value> = messages.iter().map(|m| {
                    let mut row = json!({"role":m.role,"content":m.content});
                    if !m.tool_calls.is_empty() { row["tool_calls"] = Value::Array(m.tool_calls.iter().map(|c| json!({"id":c.id,"type":"function","function":{"name":c.name,"arguments":c.arguments.to_string()}})).collect()); }
                    if let Some(id) = &m.tool_call_id { row["tool_call_id"] = json!(id); }
                    row
                }).collect();
                let mut out = json!({"model":self.config.model,"messages":rows,"max_tokens":self.config.max_output_tokens,"stream":false,"n":1});
                if !tools.is_empty() {
                    out["tools"] = json!(tool_values);
                }
                Ok(out)
            }
            "anthropic" => {
                let mut rows: Vec<Value> = vec![];
                let mut system = vec![];
                for m in messages {
                    if m.role == "system" {
                        system.push(json!({"type":"text","text":m.content}));
                        continue;
                    }
                    let (role, blocks) = if m.role == "tool" {
                        (
                            "user",
                            vec![
                                json!({"type":"tool_result","tool_use_id":m.tool_call_id,"content":m.content}),
                            ],
                        )
                    } else {
                        let mut blocks = vec![];
                        if !m.content.is_empty() {
                            blocks.push(json!({"type":"text","text":m.content}));
                        }
                        for c in &m.tool_calls {
                            blocks.push(json!({"type":"tool_use","id":c.id,"name":c.name,"input":c.arguments}));
                        }
                        ensure!(!blocks.is_empty(), "Anthropic message has no content");
                        (m.role.as_str(), blocks)
                    };
                    if let Some(last) = rows.last_mut().filter(|r| r["role"] == role) {
                        last["content"].as_array_mut().unwrap().extend(blocks);
                    } else {
                        rows.push(json!({"role":role,"content":blocks}));
                    }
                }
                let mut out = json!({"model":self.config.model,"messages":rows,"max_tokens":self.config.max_output_tokens,"stream":false});
                if !system.is_empty() {
                    out["system"] = json!(system);
                }
                if !tools.is_empty() {
                    out["tools"] = json!(tools.iter().map(|t| json!({"name":t.name,"description":t.description,"input_schema":t.parameters})).collect::<Vec<_>>());
                }
                Ok(out)
            }
            "ollama" => {
                let names: HashMap<_, _> = messages
                    .iter()
                    .flat_map(|m| &m.tool_calls)
                    .map(|c| (c.id.as_str(), c.name.as_str()))
                    .collect();
                let rows: Vec<Value> = messages.iter().map(|m| {
                    let mut row = json!({"role":m.role,"content":m.content});
                    if !m.tool_calls.is_empty() { row["tool_calls"] = json!(m.tool_calls.iter().map(|c| json!({"function":{"name":c.name,"arguments":c.arguments}})).collect::<Vec<_>>()); }
                    if let Some(id) = &m.tool_call_id { row["tool_name"] = json!(names[id.as_str()]); }
                    row
                }).collect();
                let mut out = json!({"model":self.config.model,"messages":rows,"stream":false,"options":{"num_predict":self.config.max_output_tokens}});
                if !tools.is_empty() {
                    out["tools"] = json!(tool_values);
                }
                Ok(out)
            }
            _ => unreachable!(),
        }
    }
    fn decode(&self, value: Value, tools: &[ToolDefinition]) -> Result<Completion> {
        let (content, calls, input, output, finish) = match self.config.kind.as_str() {
            "openai" => {
                let choices = array(&value, "choices")?;
                ensure!(
                    choices.len() == 1,
                    "provider must return exactly one choice"
                );
                let choice = &choices[0];
                let msg = &choice["message"];
                ensure!(
                    string(msg, "role")? == "assistant",
                    "provider returned non-assistant role"
                );
                ensure!(
                    msg.get("refusal").is_none_or(Value::is_null),
                    "provider refused request"
                );
                ensure!(
                    msg.get("function_call").is_none_or(Value::is_null),
                    "legacy function_call response is unsupported"
                );
                let content = nullable_text(msg, "content")?;
                let mut calls = vec![];
                if let Some(v) = msg.get("tool_calls").filter(|v| !v.is_null()) {
                    for c in v
                        .as_array()
                        .ok_or_else(|| anyhow::anyhow!("tool_calls must be array"))?
                    {
                        ensure!(
                            string(c, "type")? == "function",
                            "unsupported tool call type"
                        );
                        let f = &c["function"];
                        let arguments =
                            strict_json(string(f, "arguments")?.as_bytes()).map_err(|_| {
                                anyhow::anyhow!("malformed or ambiguous tool arguments JSON")
                            })?;
                        calls.push(ToolCall {
                            id: string(c, "id")?.into(),
                            name: string(f, "name")?.into(),
                            arguments,
                        });
                    }
                }
                let reason = string(choice, "finish_reason")?;
                ensure!(
                    (reason == "stop" && calls.is_empty())
                        || (reason == "tool_calls" && !calls.is_empty()),
                    "incomplete, truncated or unsupported provider finish reason"
                );
                if let Some(details) = value["usage"].get("prompt_tokens_details") {
                    ensure!(details.is_object(), "invalid input token details");
                    if let Some(cached) = details.get("cached_tokens") {
                        ensure!(
                            cached.as_u64() == Some(0),
                            "cached input pricing is not configured"
                        );
                    }
                }
                (
                    content,
                    calls,
                    number(&value["usage"], "prompt_tokens")?,
                    number(&value["usage"], "completion_tokens")?,
                    reason.to_owned(),
                )
            }
            "anthropic" => {
                ensure!(
                    string(&value, "role")? == "assistant",
                    "provider returned non-assistant role"
                );
                ensure!(
                    string(&value, "type")? == "message",
                    "invalid Anthropic message type"
                );
                let mut text = String::new();
                let mut calls = vec![];
                for block in array(&value, "content")? {
                    match string(block, "type")? {
                        "text" => text.push_str(string(block, "text")?),
                        "tool_use" => calls.push(ToolCall {
                            id: string(block, "id")?.into(),
                            name: string(block, "name")?.into(),
                            arguments: block
                                .get("input")
                                .ok_or_else(|| anyhow::anyhow!("missing tool input"))?
                                .clone(),
                        }),
                        _ => bail!("unsupported Anthropic content block"),
                    }
                }
                let reason = string(&value, "stop_reason")?;
                ensure!(
                    (matches!(reason, "end_turn" | "stop_sequence") && calls.is_empty())
                        || (reason == "tool_use" && !calls.is_empty()),
                    "incomplete, truncated or unsupported Anthropic stop reason"
                );
                let usage = &value["usage"];
                // Requests never enable prompt caching; reject billed cache usage rather
                // than falsely price it using plain input rates.
                for key in ["cache_creation_input_tokens", "cache_read_input_tokens"] {
                    if let Some(n) = usage.get(key) {
                        ensure!(
                            n.as_u64() == Some(0),
                            "cached input pricing is not configured"
                        );
                    }
                }
                (
                    text,
                    calls,
                    number(usage, "input_tokens")?,
                    number(usage, "output_tokens")?,
                    reason.to_owned(),
                )
            }
            "ollama" => {
                ensure!(value["done"] == true, "incomplete Ollama response");
                ensure!(
                    string(&value, "done_reason")? == "stop",
                    "incomplete, truncated or unsupported Ollama stop reason"
                );
                let msg = &value["message"];
                ensure!(
                    string(msg, "role")? == "assistant",
                    "provider returned non-assistant role"
                );
                let text = string(msg, "content")?.to_owned();
                let mut calls = vec![];
                if let Some(v) = msg.get("tool_calls").filter(|v| !v.is_null()) {
                    for c in v
                        .as_array()
                        .ok_or_else(|| anyhow::anyhow!("tool_calls must be array"))?
                    {
                        let f = &c["function"];
                        let id = match c.get("id") {
                            None => format!("ollama_{}", uuid::Uuid::new_v4().simple()),
                            Some(v) => v
                                .as_str()
                                .ok_or_else(|| anyhow::anyhow!("invalid tool ID"))?
                                .into(),
                        };
                        calls.push(ToolCall {
                            id,
                            name: string(f, "name")?.into(),
                            arguments: f
                                .get("arguments")
                                .ok_or_else(|| anyhow::anyhow!("missing tool arguments"))?
                                .clone(),
                        });
                    }
                }
                (
                    text,
                    calls,
                    number(&value, "prompt_eval_count")?,
                    number(&value, "eval_count")?,
                    "stop".into(),
                )
            }
            _ => unreachable!(),
        };
        ensure!(
            output <= u64::from(self.config.max_output_tokens),
            "provider exceeded configured output token limit"
        );
        validate_calls(&calls, tools)?;
        ensure!(
            !content.is_empty() || !calls.is_empty(),
            "empty provider response"
        );
        let known = self.costs_known();
        let cost = cost_microusd(
            input,
            output,
            self.config.input_microusd_per_million.unwrap_or(0),
            self.config.output_microusd_per_million.unwrap_or(0),
        )?;
        Ok(Completion {
            message: Message {
                role: "assistant".into(),
                content,
                tool_calls: calls,
                tool_call_id: None,
            },
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
                cost_microusd: cost,
                cost_known: known,
            },
            finish_reason: finish,
        })
    }
}

#[async_trait]
impl Provider for HttpProvider {
    fn name(&self) -> &str {
        &self.config.name
    }
    fn configuration_fingerprint(&self) -> String {
        fingerprint(std::slice::from_ref(&self.config), self.client.config())
    }
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        cancellation: CancellationToken,
    ) -> Result<Completion> {
        ensure!(!cancellation.is_cancelled(), "provider request cancelled");
        let payload = serde_json::to_vec(&self.payload(messages, tools)?)?;
        ensure!(
            payload.len() <= MAX_REQUEST_BYTES,
            "provider request exceeds byte limit"
        );
        let headers = self.headers()?;
        let request = self
            .client
            .request("POST", &self.endpoint, headers, Some(payload));
        let response = tokio::select! {
            biased;
            _ = cancellation.cancelled() => bail!("provider request cancelled"),
            result = tokio::time::timeout(Duration::from_secs(self.config.timeout_seconds),request) => {
                result.map_err(|_| anyhow::anyhow!("provider request timed out"))?.map_err(|_| anyhow::anyhow!("provider HTTP transport or policy failure"))?
            }
        };
        ensure!(
            (200..300).contains(&response.status),
            "provider HTTP status {}",
            response.status
        );
        let value = strict_json(&response.body)
            .map_err(|_| anyhow::anyhow!("provider response is invalid or ambiguous JSON"))?;
        let completion = self.decode(value, tools)?;
        let historical_ids: HashSet<_> = messages
            .iter()
            .flat_map(|m| &m.tool_calls)
            .map(|c| c.id.as_str())
            .collect();
        ensure!(
            !completion
                .message
                .tool_calls
                .iter()
                .any(|c| historical_ids.contains(c.id.as_str())),
            "provider reused a historical tool call ID"
        );
        Ok(completion)
    }
}

macro_rules! adapter {
    ($name:ident, $kind:literal) => {
        pub struct $name(HttpProvider);
        impl $name {
            pub fn new(config: ProviderConfig, client: PolicyHttpClient) -> Result<Self> {
                ensure!(config.kind == $kind, "incorrect provider adapter kind");
                Ok(Self(HttpProvider::new(config, client)?))
            }
        }
        #[async_trait]
        impl Provider for $name {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn configuration_fingerprint(&self) -> String {
                self.0.configuration_fingerprint()
            }
            async fn complete(
                &self,
                messages: &[Message],
                tools: &[ToolDefinition],
                cancellation: CancellationToken,
            ) -> Result<Completion> {
                self.0.complete(messages, tools, cancellation).await
            }
        }
    };
}
adapter!(OpenAiProvider, "openai");
adapter!(AnthropicProvider, "anthropic");
adapter!(OllamaProvider, "ollama");

#[derive(Default)]
struct Circuit {
    failures: u32,
    retry_at: Option<Instant>,
}
pub struct ProviderPool {
    providers: Vec<HttpProvider>,
    circuits: Mutex<Vec<Circuit>>,
}
impl ProviderPool {
    pub fn new(configs: Vec<ProviderConfig>, policy: PolicyConfig) -> Result<Self> {
        Self::with_client(configs, PolicyHttpClient::new(policy)?)
    }
    pub fn with_client(configs: Vec<ProviderConfig>, client: PolicyHttpClient) -> Result<Self> {
        ensure!(
            !configs.is_empty() && configs.len() <= 16,
            "provider pool requires 1..16 providers"
        );
        let mut names = HashSet::new();
        let mut providers = vec![];
        for config in configs {
            ensure!(names.insert(config.name.clone()), "duplicate provider name");
            providers.push(HttpProvider::new(config, client.clone())?);
        }
        let circuits = Mutex::new((0..providers.len()).map(|_| Circuit::default()).collect());
        Ok(Self {
            providers,
            circuits,
        })
    }
    pub fn costs_known(&self) -> bool {
        self.providers.iter().all(HttpProvider::costs_known)
    }
    pub fn require_known_prices(&self) -> Result<()> {
        ensure!(
            self.costs_known(),
            "cost budget enforcement requires explicit input/output prices for every provider (including explicit zero for free models)"
        );
        Ok(())
    }
}
#[async_trait]
impl Provider for ProviderPool {
    fn name(&self) -> &str {
        "provider-pool"
    }
    fn configuration_fingerprint(&self) -> String {
        fingerprint(
            &self
                .providers
                .iter()
                .map(|p| p.config.clone())
                .collect::<Vec<_>>(),
            self.providers[0].client.config(),
        )
    }
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        cancellation: CancellationToken,
    ) -> Result<Completion> {
        validate_input(messages, tools)?;
        // A failed remote attempt may have been billed. The eventual successful
        // response cannot make the TOTAL call cost known again.
        let mut uncertain_billing = false;
        for (i, provider) in self.providers.iter().enumerate() {
            ensure!(!cancellation.is_cancelled(), "provider request cancelled");
            let retry = {
                self.circuits
                    .lock()
                    .map_err(|_| anyhow::anyhow!("provider circuit lock poisoned"))?[i]
                    .retry_at
            };
            if retry.is_some_and(|at| at > Instant::now()) {
                continue;
            }
            match provider
                .complete(messages, tools, cancellation.clone())
                .await
            {
                Ok(mut completion) => {
                    if uncertain_billing {
                        completion.usage.cost_known = false;
                    }
                    self.circuits
                        .lock()
                        .map_err(|_| anyhow::anyhow!("provider circuit lock poisoned"))?[i] =
                        Circuit::default();
                    return Ok(completion);
                }
                Err(_) => {
                    uncertain_billing = true;
                    ensure!(!cancellation.is_cancelled(), "provider request cancelled");
                    let mut circuits = self
                        .circuits
                        .lock()
                        .map_err(|_| anyhow::anyhow!("provider circuit lock poisoned"))?;
                    let c = &mut circuits[i];
                    c.failures = c.failures.saturating_add(1);
                    let backoff = Duration::from_millis((250u64 << c.failures.min(7)).min(30_000));
                    c.retry_at = Some(Instant::now() + backoff);
                }
            }
        }
        bail!("all configured providers failed or have open circuits; no completion produced")
    }
}

fn fingerprint(configs: &[ProviderConfig], policy: &PolicyConfig) -> String {
    // Includes credential reference NAME, never the resolved value.
    let encoded =
        serde_json::to_vec(&(configs, policy)).expect("provider configuration is serializable");
    format!("providers-v1:{}", hex::encode(Sha256::digest(encoded)))
}
fn validate_env_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= 128
            && name.bytes().enumerate().all(|(i, b)| b == b'_'
                || b.is_ascii_alphabetic()
                || (i > 0 && b.is_ascii_digit())),
        "credential must be an environment variable name, not a literal secret"
    );
    Ok(())
}
fn identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-.:".contains(&b))
}
fn validate_calls(calls: &[ToolCall], tools: &[ToolDefinition]) -> Result<()> {
    let mut ids = HashSet::new();
    ensure!(calls.len() <= 128, "too many provider tool calls");
    for call in calls {
        ensure!(
            identifier(&call.id) && ids.insert(&call.id),
            "invalid or duplicate tool call ID"
        );
        ensure!(
            identifier(&call.name) && call.arguments.is_object(),
            "invalid tool name or non-object arguments"
        );
        let tool = tools
            .iter()
            .find(|t| t.name == call.name)
            .ok_or_else(|| anyhow::anyhow!("provider requested an unknown tool"))?;
        validate_arguments(&call.arguments, &tool.parameters, 0)?;
    }
    Ok(())
}
fn validate_input(messages: &[Message], tools: &[ToolDefinition]) -> Result<()> {
    ensure!(!messages.is_empty(), "messages required");
    let mut names = HashSet::new();
    for t in tools {
        ensure!(
            identifier(&t.name) && names.insert(&t.name),
            "invalid or duplicate tool name"
        );
        ensure!(
            t.parameters.is_object() && t.parameters["type"] == "object",
            "tool schema must describe an object"
        );
        check_schema_refs(&t.parameters, 0)?;
        jsonschema::options()
            .with_retriever(NoSchemaRetrieval)
            .should_validate_formats(true)
            .build(&t.parameters)
            .map_err(|_| anyhow::anyhow!("invalid tool JSON schema"))?;
    }
    let mut ids = HashSet::new();
    let mut pending = HashSet::new();
    let mut seen_non_system = false;
    for m in messages {
        ensure!(
            ["system", "user", "assistant", "tool"].contains(&m.role.as_str()),
            "unknown message role"
        );
        if m.role == "system" {
            ensure!(
                !seen_non_system,
                "system messages must precede conversation"
            );
        } else {
            seen_non_system = true;
        }
        if m.role == "tool" {
            ensure!(m.tool_calls.is_empty(), "tool result cannot contain calls");
            let id = m
                .tool_call_id
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("tool result missing ID"))?;
            ensure!(pending.remove(id), "unknown or duplicate tool result ID");
        } else {
            ensure!(m.tool_call_id.is_none(), "tool_call_id on non-tool message");
            ensure!(
                pending.is_empty(),
                "unresolved tool calls before next message"
            );
            ensure!(
                m.role == "assistant" || m.tool_calls.is_empty(),
                "only assistant messages may contain calls"
            );
            for call in &m.tool_calls {
                ensure!(
                    identifier(&call.id)
                        && identifier(&call.name)
                        && call.arguments.is_object()
                        && ids.insert(call.id.clone()),
                    "invalid or duplicate historical tool call"
                );
                pending.insert(call.id.clone());
            }
        }
    }
    ensure!(
        pending.is_empty(),
        "unresolved tool calls at completion boundary"
    );
    Ok(())
}
// Disable all automatic network retrieval and additionally reject nonlocal refs.
// Tool schemas remain untrusted data; they cannot create a second HTTP route.
fn validate_arguments(value: &Value, schema: &Value, depth: usize) -> Result<()> {
    check_schema_refs(schema, depth)?;
    let validator = jsonschema::options()
        .with_retriever(NoSchemaRetrieval)
        .should_validate_formats(true)
        .build(schema)
        .map_err(|_| anyhow::anyhow!("invalid tool JSON schema"))?;
    ensure!(
        validator.is_valid(value),
        "tool arguments violate JSON schema"
    );
    Ok(())
}
struct NoSchemaRetrieval;
impl jsonschema::Retrieve for NoSchemaRetrieval {
    fn retrieve(
        &self,
        _: &jsonschema::Uri<String>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("external schema retrieval is forbidden".into())
    }
}
fn check_schema_refs(schema: &Value, depth: usize) -> Result<()> {
    ensure!(depth <= 32, "tool schema nesting exceeds limit");
    if let Some(object) = schema.as_object() {
        for (key, value) in object {
            if matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef") {
                ensure!(
                    value.as_str().is_some_and(|s| s.starts_with('#')),
                    "external schema references are forbidden"
                );
            }
            if key == "$schema" {
                ensure!(
                    value
                        .as_str()
                        .is_some_and(|s| s.starts_with("https://json-schema.org/")
                            || s.starts_with("http://json-schema.org/")),
                    "unrecognized JSON schema dialect"
                );
            }
            check_schema_refs(value, depth + 1)?;
        }
    } else if let Some(array) = schema.as_array() {
        for value in array {
            check_schema_refs(value, depth + 1)?;
        }
    }
    Ok(())
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("provider response missing or invalid {key}"))
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("provider response missing or invalid {key}"))
}
fn number(v: &Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("provider response missing or invalid {key}"))
}
fn nullable_text(v: &Value, key: &str) -> Result<String> {
    match v.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Null) => Ok(String::new()),
        _ => bail!("provider content must be text or null"),
    }
}
/// serde_json::Value otherwise silently accepts last-wins duplicate object keys.
fn strict_json(bytes: &[u8]) -> Result<Value> {
    use serde::{
        Deserialize, Deserializer,
        de::{self, MapAccess, SeqAccess, Visitor},
    };
    struct Strict(Value);
    struct JsonVisitor;
    impl<'de> Deserialize<'de> for Strict {
        fn deserialize<D: Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            deserializer.deserialize_any(JsonVisitor).map(Strict)
        }
    }
    impl<'de> Visitor<'de> for JsonVisitor {
        type Value = Value;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unambiguous JSON")
        }
        fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Value, E> {
            Ok(Value::Bool(v))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Value, E> {
            Ok(v.into())
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Value, E> {
            Ok(v.into())
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Value, E> {
            serde_json::Number::from_f64(v)
                .map(Value::Number)
                .ok_or_else(|| E::custom("nonfinite number"))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Value, E> {
            Ok(Value::String(v.into()))
        }
        fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Value, E> {
            Ok(Value::String(v))
        }
        fn visit_none<E: de::Error>(self) -> std::result::Result<Value, E> {
            Ok(Value::Null)
        }
        fn visit_unit<E: de::Error>(self) -> std::result::Result<Value, E> {
            Ok(Value::Null)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Value, A::Error> {
            let mut values = Vec::new();
            while let Some(Strict(v)) = seq.next_element()? {
                values.push(v);
            }
            Ok(Value::Array(values))
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
            let mut values = serde_json::Map::new();
            while let Some(k) = map.next_key::<String>()? {
                if values.contains_key(&k) {
                    return Err(de::Error::custom("duplicate object key"));
                }
                let Strict(v) = map.next_value()?;
                values.insert(k, v);
            }
            Ok(Value::Object(values))
        }
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let Strict(value) = Strict::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}
pub fn cost_microusd(input: u64, output: u64, input_rate: u64, output_rate: u64) -> Result<u64> {
    let amount = (input as u128 * input_rate as u128)
        .checked_add(output as u128 * output_rate as u128)
        .ok_or_else(|| anyhow::anyhow!("provider cost overflow"))?;
    u64::try_from(amount.div_ceil(1_000_000)).map_err(|_| anyhow::anyhow!("provider cost overflow"))
}
