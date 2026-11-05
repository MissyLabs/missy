use super::{HttpMcpConfig, HttpMcpTransport, checked_url, manifest_digest};
use anyhow::{Context, Result, anyhow, bail};
use missy_policy::{PolicyHttpClient, PolicyHttpStream};
use serde_json::{Value, json};

pub(super) struct Session<'a> {
    policy: PolicyHttpClient,
    config: &'a HttpMcpConfig,
    credentials: &'a [(String, String)],
    session: Option<String>,
    id: u64,
    legacy: Option<Events>,
    post_url: String,
}
impl<'a> Session<'a> {
    pub async fn open(
        policy: &'a PolicyHttpClient,
        config: &'a HttpMcpConfig,
        credentials: &'a [(String, String)],
    ) -> Result<Self> {
        let mut s = Self {
            policy: policy
                .clone()
                .with_timeout(std::time::Duration::from_secs(config.timeout_seconds))?
                .with_chunk_timeout(std::time::Duration::from_secs(config.timeout_seconds))?,
            config,
            credentials,
            session: None,
            id: 0,
            legacy: None,
            post_url: config.endpoint.clone(),
        };
        if config.transport == HttpMcpTransport::LegacySse {
            let stream = s
                .policy
                .request_stream("GET", &config.endpoint, s.headers(false), None)
                .await?;
            check_status(&stream)?;
            if content_type(&stream)? != "text/event-stream" {
                bail!("legacy MCP requires event stream");
            }
            let mut events = Events::new(stream, config.max_response_bytes);
            let event = events
                .next()
                .await?
                .context("missing legacy endpoint event")?;
            if event.kind != "endpoint" {
                bail!("legacy MCP endpoint event required first");
            }
            if event
                .data
                .bytes()
                .any(|b| b.is_ascii_control() || b == b'\\')
            {
                bail!("invalid legacy MCP endpoint");
            }
            let resolved = checked_url(&config.endpoint)?
                .join(&event.data)
                .map_err(|_| anyhow!("invalid legacy MCP endpoint"))?;
            let actual = checked_url(resolved.as_str())?;
            let approved = checked_url(
                config
                    .message_endpoint
                    .as_deref()
                    .context("missing approved message endpoint")?,
            )?;
            if actual != approved || actual.origin() != checked_url(&config.endpoint)?.origin() {
                bail!("legacy MCP endpoint differs from exact approved URL");
            }
            s.post_url = approved.to_string();
            s.legacy = Some(events);
        }
        let result=s.rpc("initialize",json!({"protocolVersion":config.protocol_version,"capabilities":{},"clientInfo":{"name":"missy-rust","version":env!("CARGO_PKG_VERSION")}})).await?;
        if result["protocolVersion"] != config.protocol_version
            || !result["capabilities"]["tools"].is_object()
        {
            bail!("MCP initialization protocol or tool capability mismatch");
        }
        if config.transport == HttpMcpTransport::StreamableHttp && s.session.is_none() {
            bail!("MCP session ID required");
        }
        s.notify(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
            .await?;
        Ok(s)
    }
    fn headers(&self, json: bool) -> Vec<(String, String)> {
        let mut h = self.credentials.to_vec();
        h.push((
            "accept".into(),
            "application/json, text/event-stream".into(),
        ));
        if json {
            h.push(("content-type".into(), "application/json".into()));
        }
        if self.id > 0 {
            h.push((
                "mcp-protocol-version".into(),
                self.config.protocol_version.clone(),
            ));
        }
        if let Some(id) = &self.session {
            h.push(("mcp-session-id".into(), id.clone()));
        }
        h
    }
    fn correlate(&mut self, response: &PolicyHttpStream) -> Result<()> {
        let ids: Vec<_> = response
            .headers
            .iter()
            .filter(|(k, _)| k.eq_ignore_ascii_case("mcp-session-id"))
            .collect();
        if ids.len() > 1 {
            bail!("duplicate MCP session header");
        }
        if let Some((_, id)) = ids.first() {
            if id.is_empty() || id.len() > 256 || !id.bytes().all(|b| b.is_ascii_graphic()) {
                bail!("invalid MCP session ID");
            }
            match &self.session {
                Some(old) if old != id => bail!("MCP session correlation mismatch"),
                None => self.session = Some(id.clone()),
                _ => {}
            }
        }
        Ok(())
    }
    async fn post(&mut self, body: Value) -> Result<PolicyHttpStream> {
        let response = self
            .policy
            .request_stream(
                "POST",
                &self.post_url,
                self.headers(true),
                Some(serde_json::to_vec(&body)?),
            )
            .await?;
        check_status(&response)?;
        self.correlate(&response)?;
        Ok(response)
    }
    async fn notify(&mut self, body: Value) -> Result<()> {
        let mut r = self.post(body).await?;
        if ![202, 204].contains(&r.status) {
            bail!("MCP notification must return empty acceptance");
        }
        if r.chunk().await?.is_some_and(|b| !b.is_empty()) {
            bail!("MCP notification returned unexpected body");
        }
        Ok(())
    }
    pub async fn rpc(&mut self, method: &str, params: Value) -> Result<Value> {
        self.id = self.id.checked_add(1).context("MCP request ID exhausted")?;
        let id = self.id;
        let mut response = self
            .post(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
            .await?;
        let message = if let Some(events) = &mut self.legacy {
            if response.status != 202 {
                bail!("legacy MCP POST must return 202");
            }
            if response.chunk().await?.is_some_and(|b| !b.is_empty()) {
                bail!("legacy MCP POST body denied");
            }
            let e = events
                .next()
                .await?
                .context("legacy MCP stream ended before response")?;
            if e.kind != "message" {
                bail!("unexpected legacy MCP event");
            }
            serde_json::from_str(&e.data).map_err(|_| anyhow!("invalid MCP event JSON"))?
        } else {
            match content_type(&response)?.as_str() {
                "application/json" => {
                    let mut b = Vec::new();
                    while let Some(chunk) = response.chunk().await? {
                        if b.len().saturating_add(chunk.len()) > self.config.max_response_bytes {
                            bail!("MCP JSON response exceeds limit");
                        }
                        b.extend_from_slice(&chunk);
                    }
                    serde_json::from_slice(&b).map_err(|_| anyhow!("invalid MCP JSON response"))?
                }
                "text/event-stream" => {
                    let mut events = Events::new(response, self.config.max_response_bytes);
                    let e = events
                        .next()
                        .await?
                        .context("MCP stream ended before response")?;
                    if e.kind != "message" {
                        bail!("unexpected MCP SSE event");
                    }
                    serde_json::from_str(&e.data).map_err(|_| anyhow!("invalid MCP event JSON"))?
                }
                _ => bail!("unsupported MCP response content type"),
            }
        };
        rpc_result(message, id)
    }
    pub async fn verified_list(&mut self) -> Result<Value> {
        let listing = self.rpc("tools/list", json!({})).await?;
        if manifest_digest(&listing) != self.config.manifest_sha256.to_ascii_lowercase() {
            bail!("MCP manifest pin mismatch");
        }
        Ok(listing)
    }
}
fn check_status(r: &PolicyHttpStream) -> Result<()> {
    if !(200..300).contains(&r.status) {
        bail!("MCP HTTP status denied");
    }
    Ok(())
}
fn content_type(r: &PolicyHttpStream) -> Result<String> {
    let values: Vec<_> = r
        .headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        .collect();
    if values.len() != 1 {
        bail!("MCP requires exactly one content type");
    }
    Ok(values[0]
        .1
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase())
}
fn rpc_result(v: Value, id: u64) -> Result<Value> {
    let m = v.as_object().context("MCP RPC response must be object")?;
    if v["jsonrpc"] != "2.0"
        || v["id"] != id
        || m.keys()
            .any(|k| !["jsonrpc", "id", "result", "error"].contains(&k.as_str()))
        || m.contains_key("error")
        || !m.contains_key("result")
    {
        bail!("MCP RPC error or correlation mismatch");
    }
    Ok(v["result"].clone())
}

struct Event {
    kind: String,
    data: String,
}
/// Complete-event reader, not a buffered EOF reader. Endless comments and
/// unterminated events cannot bypass cumulative session byte/deadline bounds.
struct Events {
    stream: PolicyHttpStream,
    pending: Vec<u8>,
    read: usize,
    limit: usize,
}
impl Events {
    fn new(stream: PolicyHttpStream, limit: usize) -> Self {
        Self {
            stream,
            pending: Vec::new(),
            read: 0,
            limit,
        }
    }
    async fn next(&mut self) -> Result<Option<Event>> {
        let mut data = Vec::new();
        let mut kind = String::from("message");
        let mut event_bytes = 0usize;
        loop {
            if let Some(end) = self.pending.iter().position(|b| *b == b'\n') {
                let mut line: Vec<u8> = self.pending.drain(..=end).collect();
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                event_bytes = event_bytes.saturating_add(line.len() + 1);
                if event_bytes > self.limit {
                    bail!("MCP SSE event exceeds limit");
                }
                let line =
                    std::str::from_utf8(&line).map_err(|_| anyhow!("MCP SSE invalid UTF8"))?;
                if line.is_empty() {
                    if !data.is_empty() {
                        return Ok(Some(Event {
                            kind,
                            data: data.join("\n"),
                        }));
                    }
                    continue;
                }
                if line.starts_with(':') {
                    continue;
                }
                let (field, value) = line.split_once(':').unwrap_or((line, ""));
                let value = value.strip_prefix(' ').unwrap_or(value);
                match field {
                    "event" => kind = value.to_owned(),
                    "data" => data.push(value.to_owned()),
                    "id" | "retry" => {}
                    _ => bail!("unsupported MCP SSE field"),
                }
            } else {
                match self.stream.chunk().await? {
                    Some(bytes) => {
                        self.read = self.read.saturating_add(bytes.len());
                        if self.read > self.limit {
                            bail!("MCP SSE stream exceeds limit");
                        }
                        self.pending.extend_from_slice(&bytes);
                    }
                    None => {
                        if !self.pending.is_empty() || !data.is_empty() {
                            bail!("truncated MCP SSE event");
                        }
                        return Ok(None);
                    }
                }
            }
        }
    }
}
