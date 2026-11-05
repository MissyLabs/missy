//! Bounded SSE/NDJSON assembly. Partial output is never a completion or dispatch.
use super::*;
use missy_policy::{HttpResponse, PolicyHttpStream};

pub(crate) async fn collect(mut stream: PolicyHttpStream, kind: &str) -> Result<HttpResponse> {
    let status = stream.status;
    if !(200..300).contains(&status) {
        return Ok(HttpResponse {
            status,
            headers: vec![],
            body: vec![],
        });
    }
    let expected = if kind == "ollama" {
        "application/"
    } else {
        "text/event-stream"
    };
    ensure!(
        stream
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("content-type")
                && v.to_ascii_lowercase().starts_with(expected)),
        "provider stream has invalid content type"
    );
    let mut decoder = Decoder::new(kind);
    let mut pending = Vec::new();
    let mut event = String::new();
    while let Some(bytes) = stream.chunk().await? {
        pending.extend_from_slice(&bytes);
        ensure!(
            pending.len() <= MAX_REQUEST_BYTES,
            "provider stream line exceeds bound"
        );
        while let Some(end) = pending.iter().position(|b| *b == b'\n') {
            let mut bytes: Vec<_> = pending.drain(..=end).collect();
            bytes.pop();
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
            let line = std::str::from_utf8(&bytes)
                .map_err(|_| anyhow::anyhow!("invalid provider stream UTF-8"))?;
            if kind == "ollama" {
                if !line.trim().is_empty() {
                    decoder.data(line)?;
                }
            } else if line.is_empty() {
                if !event.is_empty() {
                    decoder.data(event.trim_end_matches('\n'))?;
                    event.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                event.push_str(data.strip_prefix(' ').unwrap_or(data));
                event.push('\n');
                ensure!(
                    event.len() <= MAX_REQUEST_BYTES,
                    "provider stream event exceeds bound"
                );
            } else {
                ensure!(
                    line.starts_with(':')
                        || line.starts_with("event:")
                        || line.starts_with("id:")
                        || line.starts_with("retry:"),
                    "invalid provider SSE framing"
                );
            }
        }
    }
    if kind == "ollama" && !pending.is_empty() {
        decoder.data(
            std::str::from_utf8(&pending)
                .map_err(|_| anyhow::anyhow!("invalid provider stream UTF-8"))?,
        )?;
    } else {
        ensure!(
            pending.is_empty() && event.is_empty(),
            "provider SSE ended inside event"
        );
    }
    let body = serde_json::to_vec(&decoder.finish()?)?;
    Ok(HttpResponse {
        status,
        headers: vec![],
        body,
    })
}

struct Decoder {
    kind: String,
    text: String,
    calls: std::collections::BTreeMap<u64, Value>,
    usage: Option<Value>,
    reason: Option<String>,
    final_value: Option<Value>,
    done: bool,
    started: bool,
}
impl Decoder {
    fn new(kind: &str) -> Self {
        Self {
            kind: kind.into(),
            text: String::new(),
            calls: Default::default(),
            usage: None,
            reason: None,
            final_value: None,
            done: false,
            started: false,
        }
    }
    fn data(&mut self, data: &str) -> Result<()> {
        ensure!(
            !self.done,
            "provider sent data after terminal stream marker"
        );
        if data == "[DONE]" {
            ensure!(self.kind == "openai", "unexpected provider stream marker");
            self.done = true;
            return Ok(());
        }
        let value = strict_json(data.as_bytes())
            .map_err(|_| anyhow::anyhow!("invalid provider stream JSON"))?;
        ensure!(
            value.get("error").is_none_or(Value::is_null),
            "provider stream returned error"
        );
        match self.kind.as_str() {
            "openai" => {
                if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
                    self.usage = Some(usage.clone());
                }
                let choices = array(&value, "choices")?;
                ensure!(choices.len() <= 1, "multiple streaming choices forbidden");
                if let Some(choice) = choices.first() {
                    ensure!(
                        number(choice, "index")? == 0,
                        "invalid streaming choice index"
                    );
                    let delta = &choice["delta"];
                    ensure!(
                        delta.get("role").is_none_or(|v| v == "assistant"),
                        "invalid streaming role"
                    );
                    ensure!(
                        delta.get("refusal").is_none_or(Value::is_null)
                            && delta.get("function_call").is_none_or(Value::is_null),
                        "unsupported streaming refusal or legacy call"
                    );
                    if let Some(text) = delta.get("content").filter(|v| !v.is_null()) {
                        self.text.push_str(
                            text.as_str()
                                .ok_or_else(|| anyhow::anyhow!("invalid streaming content"))?,
                        );
                    }
                    if let Some(calls) = delta.get("tool_calls") {
                        for call in calls
                            .as_array()
                            .ok_or_else(|| anyhow::anyhow!("invalid streaming calls"))?
                        {
                            let index = number(call, "index")?;
                            ensure!(index < 128, "stream call index exceeds bound");
                            let item = self.calls.entry(index).or_insert_with(||json!({"id":"","type":"function","function":{"name":"","arguments":""}}));
                            if let Some(kind) = call.get("type") {
                                ensure!(kind == "function", "native streaming tools forbidden");
                            }
                            append(item, "id", call.get("id"))?;
                            append(&mut item["function"], "name", call["function"].get("name"))?;
                            append(
                                &mut item["function"],
                                "arguments",
                                call["function"].get("arguments"),
                            )?;
                        }
                    }
                    if let Some(reason) = choice.get("finish_reason").filter(|v| !v.is_null()) {
                        ensure!(self.reason.is_none(), "multiple stream finish reasons");
                        self.reason = Some(
                            reason
                                .as_str()
                                .ok_or_else(|| anyhow::anyhow!("invalid stream finish reason"))?
                                .into(),
                        );
                    }
                }
            }
            "openai_responses" => match string(&value, "type")? {
                "response.completed" | "response.incomplete" => {
                    self.final_value = Some(value["response"].clone());
                    self.done = true;
                }
                "error" | "response.failed" | "response.cancelled" => {
                    bail!("Responses stream failed")
                }
                event if event.starts_with("response.") => {}
                _ => bail!("unsupported Responses stream event"),
            },
            "ollama" => {
                ensure!(
                    value["message"]["role"] == "assistant",
                    "invalid Ollama stream role"
                );
                self.text.push_str(string(&value["message"], "content")?);
                if let Some(calls) = value["message"].get("tool_calls") {
                    for call in calls
                        .as_array()
                        .ok_or_else(|| anyhow::anyhow!("invalid Ollama stream calls"))?
                    {
                        ensure!(self.calls.len() < 128, "too many streaming calls");
                        self.calls.insert(self.calls.len() as u64, call.clone());
                    }
                }
                if value["done"] == true {
                    let mut final_value = value.clone();
                    final_value["message"]["content"] = json!(self.text);
                    final_value["message"]["tool_calls"] =
                        json!(self.calls.values().collect::<Vec<_>>());
                    self.final_value = Some(final_value);
                    self.done = true;
                }
            }
            "anthropic" => self.anthropic(value)?,
            _ => bail!("unsupported streaming provider"),
        }
        ensure!(
            self.text.len() <= MAX_REQUEST_BYTES,
            "streamed provider text exceeds bound"
        );
        Ok(())
    }
    fn anthropic(&mut self, value: Value) -> Result<()> {
        match string(&value, "type")? {
            "ping" => {}
            "message_start" => {
                ensure!(
                    !self.started && value["message"]["role"] == "assistant",
                    "invalid Anthropic stream start"
                );
                self.started = true;
                self.usage = Some(value["message"]["usage"].clone());
            }
            "content_block_start" => {
                ensure!(self.started, "Anthropic block before message");
                let index = number(&value, "index")?;
                ensure!(
                    index < 128 && !self.calls.contains_key(&index),
                    "invalid Anthropic stream block index"
                );
                let block = &value["content_block"];
                ensure!(
                    ["text", "tool_use"].contains(&string(block, "type")?),
                    "unsupported Anthropic streaming block"
                );
                let mut block = block.clone();
                if block["type"] == "tool_use" {
                    block["_arguments"] = json!("");
                }
                self.calls.insert(index, block);
            }
            "content_block_delta" => {
                let block = self
                    .calls
                    .get_mut(&number(&value, "index")?)
                    .ok_or_else(|| anyhow::anyhow!("unknown Anthropic stream block"))?;
                match string(&value["delta"], "type")? {
                    "text_delta" => {
                        ensure!(block["type"] == "text", "Anthropic delta type mismatch");
                        append(block, "text", value["delta"].get("text"))?;
                    }
                    "input_json_delta" => {
                        ensure!(block["type"] == "tool_use", "Anthropic delta type mismatch");
                        append(block, "_arguments", value["delta"].get("partial_json"))?;
                    }
                    _ => bail!("unsupported Anthropic stream delta"),
                }
            }
            "content_block_stop" => {}
            "message_delta" => {
                let usage = self
                    .usage
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("Anthropic delta before start"))?;
                usage["output_tokens"] = json!(number(&value["usage"], "output_tokens")?);
                self.reason = Some(string(&value["delta"], "stop_reason")?.into());
            }
            "message_stop" => {
                ensure!(
                    self.reason.is_some(),
                    "Anthropic stream missing finish reason"
                );
                self.done = true;
            }
            _ => bail!("unsupported Anthropic stream event"),
        }
        Ok(())
    }
    fn finish(mut self) -> Result<Value> {
        ensure!(self.done, "provider stream ended without terminal marker");
        if let Some(value) = self.final_value {
            return Ok(value);
        }
        let usage = self
            .usage
            .ok_or_else(|| anyhow::anyhow!("provider stream missing final usage"))?;
        let reason = self
            .reason
            .ok_or_else(|| anyhow::anyhow!("provider stream missing finish reason"))?;
        if self.kind == "anthropic" {
            for block in self.calls.values_mut() {
                if block["type"] == "tool_use" {
                    let raw = block["_arguments"].as_str().unwrap_or("");
                    if !raw.is_empty() {
                        block["input"] = strict_json(raw.as_bytes())?;
                    }
                    block.as_object_mut().unwrap().remove("_arguments");
                }
            }
            return Ok(
                json!({"type":"message","role":"assistant","content":self.calls.values().collect::<Vec<_>>(),"stop_reason":reason,"usage":usage}),
            );
        }
        Ok(
            json!({"choices":[{"message":{"role":"assistant","content":self.text,"tool_calls":self.calls.values().collect::<Vec<_>>()},"finish_reason":reason}],"usage":usage}),
        )
    }
}
fn append(target: &mut Value, key: &str, value: Option<&Value>) -> Result<()> {
    if let Some(value) = value {
        let text = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("invalid streaming text field"))?;
        let old = target[key]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("invalid streaming accumulator"))?;
        ensure!(
            old.len().saturating_add(text.len()) <= MAX_REQUEST_BYTES,
            "stream field exceeds bound"
        );
        target[key] = json!(format!("{old}{text}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anthropic_ndjson_responses_and_terminal_validation() {
        let mut a = Decoder::new("anthropic");
        for v in [
            json!({"type":"message_start","message":{"role":"assistant","usage":{"input_tokens":10,"output_tokens":0}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"é𐐀"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}),
            json!({"type":"message_stop"}),
        ] {
            a.data(&v.to_string()).unwrap();
        }
        let value = a.finish().unwrap();
        assert_eq!(value["content"][0]["text"], "é𐐀");
        assert_eq!(value["usage"]["output_tokens"], 3);
        let mut o = Decoder::new("ollama");
        o.data(&json!({"done":false,"message":{"role":"assistant","content":"hel"}}).to_string())
            .unwrap();
        o.data(&json!({"done":true,"message":{"role":"assistant","content":"lo"},"eval_count":3,"prompt_eval_count":10}).to_string()).unwrap();
        assert_eq!(o.finish().unwrap()["message"]["content"], "hello");
        let mut r = Decoder::new("openai_responses");
        r.data(&json!({"type":"response.output_text.delta","delta":"partial"}).to_string())
            .unwrap();
        assert!(r.finish().is_err());
        let mut r = Decoder::new("openai_responses");
        r.data(&json!({"type":"response.completed","response":{"status":"completed"}}).to_string())
            .unwrap();
        assert_eq!(r.finish().unwrap()["status"], "completed");
    }
}
