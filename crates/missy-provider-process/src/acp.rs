//! Bounded ACP v1 framing for one audited adapter, not arbitrary agents.
//! NoTools relies on explicit adapter SDK options, NOT permission prompts.
use crate::{CliProviderConfig, process};
use anyhow::Result;
use missy_core::{Completion, Message, Usage};
use serde_json::{Value, json};
use std::path::Path;
use tokio_util::sync::CancellationToken;

const ADAPTER: &str = "@agentclientprotocol/claude-agent-acp";
const VERSION: &str = "0.93.0";
fn invalid() -> anyhow::Error {
    anyhow::anyhow!("ACP returned unsupported or unsafe protocol; diagnostics withheld")
}
fn frame(value: Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(&value)?;
    bytes.push(b'\n');
    Ok(bytes)
}
fn request(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params})
}
pub(crate) struct Session<'a> {
    config: &'a CliProviderConfig,
    cwd: &'a Path,
    system: String,
    prompt: String,
    phase: u64,
    session: Option<String>,
    text: String,
    stop: Option<String>,
    permission_denied: bool,
}
impl<'a> Session<'a> {
    pub(crate) fn new(
        config: &'a CliProviderConfig,
        cwd: &'a Path,
        system: String,
        prompt: String,
    ) -> Self {
        Self {
            config,
            cwd,
            system,
            prompt,
            phase: 1,
            session: None,
            text: String::new(),
            stop: None,
            permission_denied: false,
        }
    }
    pub(crate) fn initialize() -> Result<Vec<u8>> {
        frame(request(
            1,
            "initialize",
            json!({
                "protocolVersion":1,
                "clientInfo":{"name":"missy-native-chat", "version":env!("CARGO_PKG_VERSION")},
                "clientCapabilities":{"fs":{"readTextFile":false,"writeTextFile":false},"terminal":false}
            }),
        ))
    }
    pub(crate) fn cancel(&self) -> Vec<u8> {
        self.session
            .as_ref()
            .and_then(|id| {
                frame(json!({
                    "jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":id}
                }))
                .ok()
            })
            .unwrap_or_default()
    }
    /// `None` is a fully validated terminal response, never merely EOF.
    pub(crate) fn receive(&mut self, line: &[u8]) -> Result<Option<Vec<u8>>> {
        let event: Value = serde_json::from_slice(line).map_err(|_| invalid())?;
        if event["jsonrpc"] != "2.0" || self.stop.is_some() {
            return Err(invalid());
        }
        if let Some(method) = event.get("method").and_then(Value::as_str) {
            if event.get("result").is_some() || event.get("error").is_some() {
                return Err(invalid());
            }
            if method == "session/request_permission" {
                if self.phase != 4
                    || event["params"]["sessionId"].as_str() != self.session.as_deref()
                {
                    return Err(invalid());
                }
                let id = event
                    .get("id")
                    .filter(|v| v.is_string() || v.is_u64())
                    .ok_or_else(invalid)?;
                self.permission_denied = true;
                // Never choose the first option when reject is absent: it can be allow.
                return Ok(Some(frame(json!({"jsonrpc":"2.0","id":id,
                    "result":{"outcome":{"outcome":"cancelled"}}}))?));
            }
            if method != "session/update"
                || event.get("id").is_some()
                || event["params"]["sessionId"].as_str() != self.session.as_deref()
            {
                return Err(invalid());
            }
            let update = &event["params"]["update"];
            match update["sessionUpdate"].as_str() {
                Some("agent_message_chunk") if self.phase == 4 => {
                    if update["content"]["type"] != "text" {
                        return Err(invalid());
                    }
                    self.text
                        .push_str(update["content"]["text"].as_str().ok_or_else(invalid)?);
                }
                Some("agent_thought_chunk") if self.phase == 4 => {}
                Some("available_commands_update") => {
                    if update["availableCommands"].as_array().is_none() {
                        return Err(invalid());
                    }
                }
                Some("config_option_update") => {
                    let options = update["configOptions"].as_array().ok_or_else(invalid)?;
                    if self.phase == 4
                        && options
                            .iter()
                            .any(|o| o["id"] == "model" && o["currentValue"] != self.config.model)
                    {
                        return Err(invalid());
                    }
                }
                Some("session_info_update" | "usage_update") => {}
                // Native tool events, fs/terminal requests and protocol drift fail closed.
                _ => return Err(invalid()),
            }
            return Ok(Some(Vec::new()));
        }
        if event.get("error").is_some() || event["id"].as_u64() != Some(self.phase) {
            return Err(invalid());
        }
        let result = event
            .get("result")
            .filter(|v| v.is_object())
            .ok_or_else(invalid)?;
        let next = match self.phase {
            1 => {
                if result["protocolVersion"] != 1
                    || result["agentInfo"]["name"] != ADAPTER
                    || result["agentInfo"]["version"] != VERSION
                {
                    return Err(invalid());
                }
                request(
                    2,
                    "session/new",
                    json!({
                        "cwd":self.cwd, "mcpServers":[],
                        "_meta":{"systemPrompt":self.system,"disableBuiltInTools":true,
                            "claudeCode":{"options":{
                                "tools":[], "settingSources":[], "strictMcpConfig":true,
                                "mcpServers":{}, "persistSession":false, "maxTurns":1,
                                "promptSuggestions":false,
                                "settings":{"disableAllHooks":true,"switchModelsOnFlag":false},
                                "extraArgs":{"disable-slash-commands":"","no-chrome":""}
                            }}}
                    }),
                )
            }
            2 => {
                let id = result["sessionId"]
                    .as_str()
                    .filter(|s| !s.is_empty() && s.len() <= 256)
                    .ok_or_else(invalid)?;
                self.session = Some(id.to_owned());
                request(
                    3,
                    "session/set_config_option",
                    json!({"sessionId":id,"configId":"model","value":self.config.model}),
                )
            }
            3 => {
                let options = result["configOptions"].as_array().ok_or_else(invalid)?;
                let models: Vec<_> = options.iter().filter(|o| o["id"] == "model").collect();
                if models.len() != 1 || models[0]["currentValue"] != self.config.model {
                    return Err(invalid());
                }
                request(
                    4,
                    "session/prompt",
                    json!({"sessionId":self.session,
                    "prompt":[{"type":"text","text":self.prompt}]}),
                )
            }
            4 => {
                let stop = result["stopReason"].as_str().ok_or_else(invalid)?;
                if self.permission_denied
                    || self.text.is_empty()
                    || !matches!(stop, "end_turn" | "max_tokens")
                {
                    return Err(invalid());
                }
                self.stop = Some(stop.to_owned());
                return Ok(None);
            }
            _ => return Err(invalid()),
        };
        self.phase += 1;
        Ok(Some(frame(next)?))
    }
    fn completion(self) -> Result<Completion> {
        let stop = self.stop.ok_or_else(invalid)?;
        Ok(Completion {
            message: Message::text("assistant", self.text),
            // ACP context-window usage is not billable per-turn token usage.
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cost_microusd: 0,
                cost_known: false,
            },
            finish_reason: if stop == "max_tokens" {
                "length"
            } else {
                "stop"
            }
            .into(),
        })
    }
}
pub(crate) async fn complete(
    config: &CliProviderConfig,
    cwd: &Path,
    system: String,
    prompt: String,
    cancellation: CancellationToken,
) -> Result<Completion> {
    let mut session = Session::new(config, cwd, system, prompt);
    process::run_acp(config, cwd, &mut session, cancellation).await?;
    session.completion()
}
