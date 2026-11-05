//! Native CDP connection compatible with a Playwright-launched Chromium CDP
//! endpoint. No Python, node subprocess, arbitrary evaluate, or endpoint discovery.
use crate::parse;
use anyhow::{Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::{SinkExt, StreamExt};
use missy_core::ToolCall;
use missy_policy::{PolicyHttpClient, PolicyWebSocket};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserConfig {
    /// Exact HTTP(S) Chromium CDP service origin. No path, credentials or query.
    pub endpoint: String,
    /// Explicit page target ID. Never selects the first personal browser tab.
    pub target_id: String,
    pub allowed_origins: BTreeSet<String>,
    /// Required acknowledgement of an independently enforced browser/container
    /// network boundary. CDP's powerful remote peer can otherwise access ANY
    /// authority from its own host. Client policy is not that network boundary.
    pub isolated_browser_network: bool,
}
pub(crate) fn origin(raw: &str) -> Result<String> {
    let u = Url::parse(raw)?;
    if !matches!(u.scheme(), "http" | "https")
        || u.host_str().is_none()
        || !u.username().is_empty()
        || u.password().is_some()
    {
        bail!("browser URL must be credential-free HTTP(S)");
    }
    Ok(u.origin().ascii_serialization())
}
impl BrowserConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.isolated_browser_network
            || origin(&self.endpoint)? != self.endpoint.trim_end_matches('/')
            || self.target_id.is_empty()
            || self.target_id.len() > 128
            || !self
                .target_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
            || self.allowed_origins.is_empty()
        {
            bail!("browser requires explicit isolated CDP origin, target and URL origins");
        }
        for value in &self.allowed_origins {
            if origin(value)? != *value {
                bail!("browser allowlist requires canonical exact origins");
            }
        }
        Ok(())
    }
    fn check_url(&self, raw: &str) -> Result<()> {
        if raw.len() > 8192 || !self.allowed_origins.contains(&origin(raw)?) {
            bail!("browser URL origin denied");
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Navigate {
    url: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Click {
    selector: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fill {
    selector: String,
    value: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wait {
    selector: String,
    state: String,
    timeout_ms: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Evaluate {
    expression_id: String,
}
fn fixed_expression(id: &str) -> Result<&'static str> {
    match id {
        "url" => Ok("String(location.href).slice(0,8192)"),
        "title" => Ok("String(document.title).slice(0,1024)"),
        "ready_state" => Ok("String(document.readyState)"),
        _ => bail!("unknown fixed expression ID; arbitrary JavaScript is not supported"),
    }
}
fn selector(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > 1024 || s.contains('\0') {
        bail!("invalid/bounded selector required");
    }
    Ok(())
}
pub(crate) fn validate_call(call: &ToolCall, c: &BrowserConfig) -> Result<()> {
    match call.name.as_str() {
        "browser_read" | "browser_screenshot" | "browser_get_url" | "browser_close" => {
            parse::<crate::Empty>(call)?;
        }
        "browser_wait" => {
            let a: Wait = parse(call)?;
            selector(&a.selector)?;
            if !matches!(a.state.as_str(), "visible" | "hidden")
                || !(1..=15000).contains(&a.timeout_ms)
            {
                bail!("wait requires visible/hidden and timeout_ms 1..15000");
            }
        }
        "browser_evaluate" => {
            fixed_expression(&parse::<Evaluate>(call)?.expression_id)?;
        }
        "browser_navigate" => c.check_url(&parse::<Navigate>(call)?.url)?,
        "browser_click" => selector(&parse::<Click>(call)?.selector)?,
        "browser_fill" => {
            let a: Fill = parse(call)?;
            selector(&a.selector)?;
            if a.value.len() > 8192 {
                bail!("fill exceeds limit");
            }
        }
        _ => bail!("unknown browser operation"),
    }
    Ok(())
}
struct Cdp<'a> {
    socket: PolicyWebSocket,
    config: &'a BrowserConfig,
    next_id: u64,
}
impl Cdp<'_> {
    async fn send(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("CDP ID overflow"))?;
        self.socket
            .send(Message::Text(
                json!({"id":id,"method":method,"params":params})
                    .to_string()
                    .into(),
            ))
            .await?;
        Ok(id)
    }
    async fn rpc(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.send(method, params).await?;
        let mut auxiliary = BTreeSet::new();
        let mut response = None;
        // Bounded events per command. No unmatched or stale replies are ignored.
        for _ in 0..2048 {
            let message = self
                .socket
                .next()
                .await
                .ok_or_else(|| anyhow::anyhow!("CDP closed"))??;
            let bytes = match message {
                Message::Text(s) => s.as_bytes().to_vec(),
                Message::Ping(p) => {
                    self.socket.send(Message::Pong(p)).await?;
                    continue;
                }
                Message::Pong(_) => continue,
                _ => bail!("unexpected CDP frame"),
            };
            if bytes.len() > 8 * 1024 * 1024 {
                bail!("CDP frame too large");
            }
            let v: Value = serde_json::from_slice(&bytes)?;
            if let Some(reply) = v.get("id") {
                let reply = reply
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("noninteger CDP ID"))?;
                if reply == id && response.is_none() {
                    if v.get("error").is_some() {
                        bail!("CDP command failed");
                    }
                    response = Some(
                        v.get("result")
                            .cloned()
                            .ok_or_else(|| anyhow::anyhow!("missing CDP result"))?,
                    );
                } else if auxiliary.remove(&reply) {
                    if v.get("error").is_some() {
                        bail!("CDP request interception failed");
                    }
                } else {
                    bail!("unexpected/duplicate CDP response ID");
                }
            } else {
                let event = v["method"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("invalid CDP event"))?;
                let p = &v["params"];
                match event {
                    "Fetch.requestPaused" => {
                        let request = p["requestId"]
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("missing Fetch request ID"))?;
                        let raw = p["request"]["url"]
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("missing intercepted URL"))?;
                        let permitted = self.config.check_url(raw).is_ok();
                        let ack = if permitted {
                            self.send("Fetch.continueRequest", json!({"requestId":request}))
                                .await?
                        } else {
                            self.send(
                                "Fetch.failRequest",
                                json!({"requestId":request,"errorReason":"BlockedByClient"}),
                            )
                            .await?
                        };
                        auxiliary.insert(ack);
                        if !permitted {
                            bail!("browser request/redirect origin denied and blocked");
                        }
                    }
                    "Page.frameNavigated" => self.config.check_url(
                        p["frame"]["url"]
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("missing frame URL"))?,
                    )?,
                    "Page.navigatedWithinDocument" => self.config.check_url(
                        p["url"]
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("missing navigation URL"))?,
                    )?,
                    _ => {}
                }
            }
            if auxiliary.is_empty()
                && let Some(response) = response.take()
            {
                return Ok(response);
            }
        }
        bail!("CDP event budget exceeded")
    }
    async fn evaluate_fixed(&mut self, expression: String) -> Result<Value> {
        // Check origin in the very same renderer turn as the fixed action/read.
        // A pre-action RPC alone would permit a navigation race before click/fill.
        let origins = serde_json::to_string(&self.config.allowed_origins)?;
        let expression = format!(
            "(()=>{{if(!{origins}.includes(location.origin))throw Error('origin denied');return ({expression});}})()"
        );
        let response=self.rpc("Runtime.evaluate",json!({"expression":expression,"returnByValue":true,"awaitPromise":false,"timeout":3000,"userGesture":false})).await?;
        if response.get("exceptionDetails").is_some() {
            bail!("fixed browser script failed");
        }
        response["result"]
            .get("value")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fixed script returned no value"))
    }
    async fn check_location(&mut self) -> Result<()> {
        let location = self.evaluate_fixed("String(location.href)".into()).await?;
        self.config.check_url(
            location
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("invalid page location"))?,
        )
    }
}

pub(crate) async fn execute(
    c: &BrowserConfig,
    http: &PolicyHttpClient,
    call: &ToolCall,
    limit: usize,
) -> Result<Value> {
    validate_call(call, c)?;
    let response = http
        .get(&format!("{}/json/list", c.endpoint.trim_end_matches('/')))
        .await?;
    if response.status != 200 || response.body.len() > 1024 * 1024 {
        bail!("CDP discovery response denied");
    }
    let pages: Vec<Value> = serde_json::from_slice(&response.body)?;
    if pages.len() > 256 {
        bail!("CDP target list exceeds limit");
    }
    let mut matches = pages
        .iter()
        .filter(|v| v["id"].as_str() == Some(&c.target_id) && v["type"] == "page");
    let page = matches
        .next()
        .ok_or_else(|| anyhow::anyhow!("explicit CDP page target missing"))?;
    if matches.next().is_some() {
        bail!("ambiguous CDP target");
    }
    c.check_url(
        page["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("CDP page URL missing"))?,
    )?;
    let ws = page["webSocketDebuggerUrl"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("CDP page websocket missing"))?;
    let ws_url = Url::parse(ws)?;
    let equivalent = match ws_url.scheme() {
        "ws" => ws.replacen("ws://", "http://", 1),
        "wss" => ws.replacen("wss://", "https://", 1),
        _ => bail!("CDP websocket scheme denied"),
    };
    if origin(&equivalent)? != origin(&c.endpoint)?
        || ws_url.path() != format!("/devtools/page/{}", c.target_id)
        || ws_url.query().is_some()
        || ws_url.fragment().is_some()
    {
        bail!("CDP websocket endpoint differs from pinned page authority");
    }
    let mut client = Cdp {
        socket: http.connect_websocket(ws, vec![]).await?,
        config: c,
        next_id: 1,
    };
    client.rpc("Page.enable", json!({})).await?;
    // Browser-wide network isolation is still mandatory: service workers,
    // preexisting requests, WebSockets and other targets are outside this hook.
    client
        .rpc(
            "Fetch.enable",
            json!({"patterns":[{"urlPattern":"*","requestStage":"Request"}]}),
        )
        .await?;
    client.check_location().await?;
    if call.name == "browser_close" {
        // Page.close acts on this pinned page websocket only, not a process.
        client.rpc("Page.close", json!({})).await?;
        let _ = client.socket.close(None).await;
        return Ok(json!({"closed":true,"target_id":c.target_id}));
    }
    let result=match call.name.as_str() {
        "browser_get_url" => client.evaluate_fixed(fixed_expression("url")?.into()).await?,
        "browser_evaluate" => {
            let a: Evaluate = parse(call)?;
            client.evaluate_fixed(fixed_expression(&a.expression_id)?.into()).await?
        }
        "browser_wait" => {
            let a: Wait = parse(call)?;
            let expression = format!("((s)=>{{const e=document.querySelector(s);if(!e)return false;const r=e.getBoundingClientRect(),c=getComputedStyle(e);return r.width>0&&r.height>0&&c.display!=='none'&&c.visibility!=='hidden'&&c.visibility!=='collapse';}})({})",serde_json::to_string(&a.selector)?);
            tokio::time::timeout(std::time::Duration::from_millis(a.timeout_ms), async {
                loop {
                    let visible = client.evaluate_fixed(expression.clone()).await?;
                    let visible = visible.as_bool().ok_or_else(||anyhow::anyhow!("invalid visibility result"))?;
                    if visible == (a.state == "visible") { break Ok::<_,anyhow::Error>(json!({"matched":true,"state":a.state})); }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }).await.map_err(|_|anyhow::anyhow!("browser wait timed out"))??
        }
        "browser_read"=>client.evaluate_fixed("JSON.stringify({url:String(location.href),title:String(document.title).slice(0,1024),text:String(document.body?.innerText??'').slice(0,32768)})".into()).await?,
        "browser_navigate"=>{
            let a:Navigate=parse(call)?;
            let navigation=client.rpc("Page.navigate",json!({"url":a.url})).await?;
            if navigation.get("errorText").is_some() { bail!("browser navigation failed"); }
            let mut ready=false;
            for _ in 0..40 {
                let state=client.evaluate_fixed("JSON.stringify({url:String(location.href),ready:document.readyState})".into()).await?;
                let state:Value=serde_json::from_str(state.as_str().ok_or_else(||anyhow::anyhow!("invalid navigation state"))?)?;
                c.check_url(state["url"].as_str().ok_or_else(||anyhow::anyhow!("missing final URL"))?)?;
                if state["ready"]=="complete" && state["url"].as_str()==Some(a.url.as_str()) { ready=true;break; }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            if !ready { bail!("navigation not settled at requested URL; redirects may have occurred"); }
            json!({"navigated":true,"url":a.url})
        }
        "browser_click"=>{
            let a:Click=parse(call)?;
            client.evaluate_fixed(format!("((s)=>{{const e=document.querySelector(s);if(!e||!(e instanceof HTMLElement))throw Error('target missing');e.click();return true;}})({})",serde_json::to_string(&a.selector)?)).await?
        }
        "browser_fill"=>{
            let a:Fill=parse(call)?;
            client.evaluate_fixed(format!("((s,v)=>{{const e=document.querySelector(s);if(!(e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement))throw Error('not input');e.value=v;e.dispatchEvent(new Event('input',{{bubbles:true}}));e.dispatchEvent(new Event('change',{{bubbles:true}}));return true;}})({},{})",serde_json::to_string(&a.selector)?,serde_json::to_string(&a.value)?)).await?
        }
        "browser_screenshot"=>{
            let result=client.rpc("Page.captureScreenshot",json!({"format":"png","captureBeyondViewport":false,"fromSurface":true})).await?;
            let data=result["data"].as_str().ok_or_else(||anyhow::anyhow!("screenshot data missing"))?;
            if data.len()>limit.saturating_sub(1024) { bail!("browser screenshot exceeds output budget"); }
            let bytes=STANDARD.decode(data)?;
            if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") { bail!("browser returned non-PNG image"); }
            json!({"artifact":{"media_type":"image/png","encoding":"base64","data":data}})
        }
        _=>bail!("unknown browser operation"),
    };
    client.check_location().await?; // Final URL and queued frame events checked before releasing data.
    client.rpc("Fetch.disable", json!({})).await?;
    client.socket.close(None).await?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use missy_core::PolicyConfig;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    async fn fake(mode: &str) -> (BrowserConfig, PolicyHttpClient, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let authority = listener.local_addr().unwrap();
        let endpoint = format!("http://{authority}");
        let c = BrowserConfig {
            endpoint: endpoint.clone(),
            target_id: "test".into(),
            allowed_origins: BTreeSet::from(["https://example.org".into()]),
            isolated_browser_network: true,
        };
        let policy = PolicyConfig {
            allowed_origins: vec![endpoint],
            allow_private_network: true,
            ..Default::default()
        };
        let http = PolicyHttpClient::new(policy).unwrap();
        let mode = mode.to_string();
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0; 4096];
            let n = stream.read(&mut request).await.unwrap();
            assert!(
                std::str::from_utf8(&request[..n])
                    .unwrap()
                    .starts_with("GET /json/list ")
            );
            let body=json!([{"id":"test","type":"page","url":"https://example.org/","webSocketDebuggerUrl":format!("ws://{authority}/devtools/page/test")}]).to_string();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            drop(stream);
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let mut evaluations = 0;
            while let Some(Ok(message)) = ws.next().await {
                let Message::Text(text) = message else {
                    break;
                };
                let request: Value = serde_json::from_str(&text).unwrap();
                let id = request["id"].as_u64().unwrap();
                if mode == "bad-id" {
                    let _ = ws
                        .send(Message::Text(
                            json!({"id":id+1,"result":{}}).to_string().into(),
                        ))
                        .await;
                    break;
                }
                if mode == "bad-final"
                    && request["method"] == "Runtime.evaluate"
                    && evaluations == 2
                {
                    let _=ws.send(Message::Text(json!({"method":"Page.frameNavigated","params":{"frame":{"url":"https://evil.invalid/"}}}).to_string().into())).await;
                    break;
                }
                if mode == "redirect" && request["method"] == "Runtime.evaluate" {
                    ws.send(Message::Text(json!({"method":"Fetch.requestPaused","params":{"requestId":"r1","request":{"url":"https://evil.invalid/"}}}).to_string().into())).await.unwrap();
                    let blocked: Value =
                        serde_json::from_str(ws.next().await.unwrap().unwrap().to_text().unwrap())
                            .unwrap();
                    assert_eq!(blocked["method"], "Fetch.failRequest");
                    assert_eq!(blocked["params"]["errorReason"], "BlockedByClient");
                    break;
                }
                let result = if request["method"] == "Runtime.evaluate" {
                    let expression = request["params"]["expression"].as_str().unwrap();
                    assert!(expression.contains("includes(location.origin)"));
                    evaluations += 1;
                    let value = if expression.contains("innerText") {
                        json!(
                            "{\"url\":\"https://example.org/\",\"title\":\"Mock\",\"text\":\"bounded text\"}"
                        )
                    } else if expression.contains("getBoundingClientRect") {
                        assert!(expression.contains("querySelector(s)"));
                        json!(mode != "hidden" && mode != "timeout")
                    } else if expression.contains("String(document.title).slice") {
                        json!("Mock title")
                    } else if expression.contains("String(document.readyState)") {
                        json!("complete")
                    } else if expression.contains("document.readyState") {
                        json!("{\"url\":\"https://example.org/\",\"ready\":\"complete\"}")
                    } else if expression.contains("e.click()") || expression.contains("e.value=v") {
                        if mode == "fill" {
                            assert!(expression.contains("\\\";throw Error('injected');//"));
                        }
                        json!(true)
                    } else {
                        json!("https://example.org/")
                    };
                    json!({"result":{"type":"string","value":value}})
                } else if request["method"] == "Page.close" {
                    assert_eq!(request["params"], json!({}));
                    assert_eq!(evaluations, 1);
                    json!({})
                } else if request["method"] == "Page.captureScreenshot" {
                    json!({"data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=="})
                } else {
                    json!({})
                };
                if ws
                    .send(Message::Text(
                        json!({"id":id,"result":result}).to_string().into(),
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
        (c, http, task)
    }
    #[tokio::test]
    async fn native_http_websocket_read_and_fixed_scripts() {
        let (c, http, task) = fake("ok").await;
        let call = ToolCall {
            id: "1".into(),
            name: "browser_read".into(),
            arguments: json!({}),
        };
        let result = execute(&c, &http, &call, 65536).await.unwrap();
        assert!(result.as_str().unwrap().contains("bounded text"));
        task.await.unwrap();
    }
    #[tokio::test]
    async fn unexpected_response_ids_fail_closed() {
        let (c, http, task) = fake("bad-id").await;
        let call = ToolCall {
            id: "1".into(),
            name: "browser_read".into(),
            arguments: json!({}),
        };
        assert!(
            execute(&c, &http, &call, 65536)
                .await
                .unwrap_err()
                .to_string()
                .contains("response ID")
        );
        task.await.unwrap();
    }
    #[tokio::test]
    async fn fixed_wait_url_evaluate_and_exact_page_close_transactions() {
        for (mode, name, args, expected) in [
            (
                "url",
                "browser_get_url",
                json!({}),
                json!("https://example.org/"),
            ),
            (
                "title",
                "browser_evaluate",
                json!({"expression_id":"title"}),
                json!("Mock title"),
            ),
            (
                "ready",
                "browser_evaluate",
                json!({"expression_id":"ready_state"}),
                json!("complete"),
            ),
            (
                "visible",
                "browser_wait",
                json!({"selector":"button[data-x='quote\\\"']","state":"visible","timeout_ms":1000}),
                json!({"matched":true,"state":"visible"}),
            ),
            (
                "hidden",
                "browser_wait",
                json!({"selector":"#absent","state":"hidden","timeout_ms":1000}),
                json!({"matched":true,"state":"hidden"}),
            ),
            (
                "close",
                "browser_close",
                json!({}),
                json!({"closed":true,"target_id":"test"}),
            ),
        ] {
            let (c, http, task) = fake(mode).await;
            let call = ToolCall {
                id: "new".into(),
                name: name.into(),
                arguments: args,
            };
            assert_eq!(execute(&c, &http, &call, 65536).await.unwrap(), expected);
            task.await.unwrap();
        }
    }
    #[tokio::test]
    async fn wait_timeout_is_bounded_and_drops_session() {
        let (c, http, task) = fake("timeout").await;
        let call = ToolCall {
            id: "wait".into(),
            name: "browser_wait".into(),
            arguments: json!({"selector":"#missing","state":"visible","timeout_ms":30}),
        };
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            execute(&c, &http, &call, 65536),
        )
        .await
        .unwrap();
        assert!(result.unwrap_err().to_string().contains("wait timed out"));
        task.await.unwrap();
    }
    #[tokio::test]
    async fn redirects_to_unknown_origins_are_blocked() {
        let (c, http, task) = fake("redirect").await;
        let call = ToolCall {
            id: "1".into(),
            name: "browser_read".into(),
            arguments: json!({}),
        };
        assert!(
            execute(&c, &http, &call, 65536)
                .await
                .unwrap_err()
                .to_string()
                .contains("redirect origin denied")
        );
        task.await.unwrap();
    }
    #[tokio::test]
    async fn final_frame_navigation_denied_before_text_released() {
        let (c, http, task) = fake("bad-final").await;
        let call = ToolCall {
            id: "1".into(),
            name: "browser_read".into(),
            arguments: json!({}),
        };
        assert!(
            execute(&c, &http, &call, 65536)
                .await
                .unwrap_err()
                .to_string()
                .contains("origin denied")
        );
        task.await.unwrap();
    }
    #[tokio::test]
    async fn fixed_click_fill_navigate_and_screenshot_protocols() {
        fixed_click_fill_navigate_and_screenshot_transactions().await;
    }
    #[tokio::test]
    async fn new_fixed_reads_and_wait_reject_final_cross_origin_events() {
        for (name, args) in [
            ("browser_get_url", json!({})),
            ("browser_evaluate", json!({"expression_id":"title"})),
            (
                "browser_wait",
                json!({"selector":"input","state":"visible","timeout_ms":1000}),
            ),
        ] {
            let (c, http, task) = fake("bad-final").await;
            let call = ToolCall {
                id: "origin".into(),
                name: name.into(),
                arguments: args,
            };
            assert!(
                execute(&c, &http, &call, 65536)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("origin denied")
            );
            task.await.unwrap();
        }
    }
    async fn fixed_click_fill_navigate_and_screenshot_transactions() {
        for (mode, name, args) in [
            (
                "click",
                "browser_click",
                json!({"selector":"button[data-name='a\\b']"}),
            ),
            (
                "fill",
                "browser_fill",
                json!({"selector":"input","value":"\";throw Error('injected');//"}),
            ),
            (
                "navigate",
                "browser_navigate",
                json!({"url":"https://example.org/"}),
            ),
            ("screenshot", "browser_screenshot", json!({})),
        ] {
            let (c, http, task) = fake(mode).await;
            let call = ToolCall {
                id: "1".into(),
                name: name.into(),
                arguments: args,
            };
            let result = execute(&c, &http, &call, 65536).await.unwrap();
            if mode == "screenshot" {
                assert_eq!(result["artifact"]["media_type"], "image/png");
            } else if mode == "navigate" {
                assert_eq!(result["navigated"], true);
            } else {
                assert_eq!(result, true);
            }
            task.await.unwrap();
        }
    }
}
