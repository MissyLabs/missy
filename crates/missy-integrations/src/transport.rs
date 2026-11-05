use crate::config::ConnectionConfig;
use anyhow::{Result, anyhow, bail};
use futures_util::{SinkExt, StreamExt};
use missy_policy::{PolicyHttpClient, PolicyWebSocket};
use serde_json::Value;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

// Intentionally no Debug, Serialize, Display or public accessor for secret values.
pub(crate) struct Connection {
    pub cfg: ConnectionConfig,
    pub http: PolicyHttpClient,
    secret: Option<String>,
}
impl Connection {
    pub fn new(http: &PolicyHttpClient, cfg: ConnectionConfig, ws: bool) -> Result<Self> {
        cfg.validate(ws)?;
        let secret = cfg
            .token_env
            .as_ref()
            .map(|name| {
                let s = std::env::var(name)
                    .map_err(|_| anyhow!("integration credential unavailable"))?;
                if s.is_empty() || s.len() > 4096 || s.chars().any(char::is_control) {
                    bail!("integration credential invalid");
                }
                Ok(s)
            })
            .transpose()?;
        Ok(Self {
            http: http
                .clone()
                .with_timeout(Duration::from_secs(cfg.timeout_seconds))?,
            cfg,
            secret,
        })
    }
    pub fn credential(&self) -> Option<&str> {
        self.secret.as_deref()
    }
    pub fn duration(&self) -> Duration {
        Duration::from_secs(self.cfg.timeout_seconds)
    }
    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.endpoint.trim_end_matches('/'), path)
    }
    /// Binary HTTP still uses the sole policy gateway and its 4 MiB hard limit.
    pub async fn bytes(
        &self,
        method: &str,
        path: &str,
        body: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<Vec<u8>> {
        let mut headers = vec![];
        if let Some(secret) = &self.secret {
            headers.push(("Authorization".into(), format!("Bearer {secret}")));
        }
        if body.is_some() {
            headers.push(("Content-Type".into(), "application/octet-stream".into()));
        }
        let mut response = self
            .http
            .request_stream(method, &self.url(path), headers, body)
            .await
            .map_err(|_| anyhow!("integration policy or HTTP transport rejected request"))?;
        if !(200..300).contains(&response.status) {
            bail!("integration upstream HTTP status {}", response.status);
        }
        let mut data = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| anyhow!("integration response read rejected"))?
        {
            if data.len().saturating_add(chunk.len()) > limit {
                bail!("integration binary response exceeds configured bound");
            }
            data.extend_from_slice(&chunk);
        }
        Ok(data)
    }
    pub async fn json(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        nomad: bool,
    ) -> Result<Value> {
        let mut headers = vec![];
        if let Some(secret) = &self.secret {
            headers.push(if nomad {
                ("X-Nomad-Token".into(), secret.clone())
            } else {
                ("Authorization".into(), format!("Bearer {secret}"))
            });
        }
        // Gateway errors may gain upstream text in future; never propagate credentials.
        let response = self
            .http
            .request(
                method,
                &self.url(path),
                {
                    if body.is_some() {
                        headers.push(("Content-Type".into(), "application/json".into()));
                    }
                    headers
                },
                body.map(|v| serde_json::to_vec(&v)).transpose()?,
            )
            .await
            .map_err(|_| anyhow!("integration policy or HTTP transport rejected request"))?;
        if !(200..300).contains(&response.status) {
            bail!("integration upstream HTTP status {}", response.status);
        }
        if response.body.len() > self.cfg.max_output_bytes {
            bail!("integration HTTP response exceeds bound");
        }
        if response.status == 204 && response.body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&response.body)
            .map_err(|_| anyhow!("integration returned invalid JSON"))
    }
    pub async fn socket(&self) -> Result<PolicyWebSocket> {
        self.http
            .connect_websocket(&self.cfg.endpoint, vec![])
            .await
            .map_err(|_| anyhow!("integration policy or WebSocket handshake rejected request"))
    }
    pub fn redact(&self, value: &mut Value) {
        match value {
            Value::String(s) => {
                if let Some(secret) = &self.secret {
                    *s = s.replace(secret, "[REDACTED]");
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|v| self.redact(v)),
            Value::Object(m) => {
                // Also redact values from unexpected upstream credential fields.
                let original = std::mem::take(m);
                for (mut k, mut v) in original {
                    let key = k.to_ascii_lowercase();
                    if [
                        "password",
                        "token",
                        "secret",
                        "authorization",
                        "authentication",
                    ]
                    .iter()
                    .any(|x| key.contains(x))
                    {
                        v = Value::String("[REDACTED]".into());
                    } else {
                        self.redact(&mut v);
                    }
                    if let Some(secret) = &self.secret {
                        k = k.replace(secret, "[REDACTED]");
                    }
                    m.insert(k, v);
                }
            }
            _ => {}
        }
    }
}

pub(crate) async fn send(ws: &mut PolicyWebSocket, value: Value) -> Result<()> {
    ws.send(Message::Text(value.to_string().into()))
        .await
        .map_err(|_| anyhow!("WebSocket write failed"))
}
pub(crate) async fn receive(ws: &mut PolicyWebSocket, limit: usize) -> Result<Value> {
    // Bound control-frame work as well as the enclosing operation deadline.
    for _ in 0..32 {
        match ws.next().await {
            Some(Ok(Message::Text(text))) if text.len() <= limit => {
                let v: Value =
                    serde_json::from_str(&text).map_err(|_| anyhow!("invalid WebSocket JSON"))?;
                if !v.is_object() {
                    bail!("WebSocket response must be an object");
                }
                return Ok(v);
            }
            Some(Ok(Message::Ping(p))) => ws
                .send(Message::Pong(p))
                .await
                .map_err(|_| anyhow!("WebSocket heartbeat failed"))?,
            Some(Ok(Message::Pong(_))) => {}
            _ => bail!("WebSocket response rejected or exceeds bounds"),
        }
    }
    bail!("WebSocket control-frame bound exceeded")
}
