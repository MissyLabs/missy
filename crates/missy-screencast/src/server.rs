use crate::*;
use axum::{
    Router,
    body::Body,
    extract::{
        DefaultBodyLimit, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::{HeaderMap, HeaderValue, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde_json::json;
use tokio::{
    net::TcpListener,
    sync::OwnedSemaphorePermit,
    task::JoinHandle,
    time::{sleep_until, timeout},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Authenticate {
    pub session_id: String,
    pub token: String,
    pub scope: Scope,
}

/// Mount ONLY on the exact configured loopback authority. Do not expose via proxies.
/// Static UI and JS are public; every socket authenticates before accepting any pixels.
pub fn router(channel: ScreencastChannel) -> Router {
    Router::new()
        .route("/", get(page))
        .route("/capture.js", get(script))
        .route("/ws", get(upgrade))
        .layer(DefaultBodyLimit::max(1024))
        .layer(middleware::from_fn_with_state(channel.clone(), boundary))
        .with_state(channel)
}
async fn boundary(
    State(channel): State<ScreencastChannel>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let expected_host = channel.config().bind.to_string();
    let headers = request.headers();
    let host_ok = headers.get_all("host").iter().count() == 1
        && headers.get("host").and_then(|v| v.to_str().ok()) == Some(expected_host.as_str());
    let origin_ok = !headers.contains_key("origin")
        || (headers.get_all("origin").iter().count() == 1
            && headers.get("origin").and_then(|v| v.to_str().ok())
                == Some(channel.config().origin().as_str()));
    let fetch_ok = headers
        .get("sec-fetch-site")
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| matches!(v, "none" | "same-origin"));
    let mut response = if !channel.config().enabled || channel.inner.shutdown.is_cancelled() {
        StatusCode::SERVICE_UNAVAILABLE.into_response()
    } else if !host_ok
        || !origin_ok
        || !fetch_ok
        || request.uri().query().is_some()
        || request.method() != Method::GET
    {
        StatusCode::FORBIDDEN.into_response()
    } else {
        next.run(request).await
    };
    let headers = response.headers_mut();
    for (name, value) in [
        ("cache-control", "no-store"),
        ("referrer-policy", "no-referrer"),
        ("x-content-type-options", "nosniff"),
        ("x-frame-options", "DENY"),
        ("cross-origin-resource-policy", "same-origin"),
        (
            "content-security-policy",
            "default-src 'none'; script-src 'self'; connect-src 'self'; media-src blob:; img-src blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        ),
        (
            "permissions-policy",
            "camera=(), microphone=(), display-capture=(self)",
        ),
    ] {
        headers.insert(name, HeaderValue::from_static(value));
    }
    response
}
async fn page() -> Html<&'static str> {
    Html(include_str!("capture.html"))
}
async fn script() -> impl IntoResponse {
    (
        [("content-type", "text/javascript; charset=utf-8")],
        include_str!("capture.js"),
    )
}
async fn upgrade(
    State(channel): State<ScreencastChannel>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    // Browser WS API cannot set Authorization. Require same-origin upgrade, then
    // a bounded first-message authentication exchange, never a query/subprotocol secret.
    if headers.get("origin").and_then(|v| v.to_str().ok())
        != Some(channel.config().origin().as_str())
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let Ok(permit) = channel.inner.connections.clone().try_acquire_owned() else {
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    };
    ws.max_frame_size(channel.config().max_frame_bytes)
        .max_message_size(channel.config().max_frame_bytes)
        .on_upgrade(move |socket| connection(channel, socket, permit))
}
struct Disconnect {
    channel: ScreencastChannel,
    session_id: String,
}
impl Drop for Disconnect {
    fn drop(&mut self) {
        self.channel.disconnect(&self.session_id);
    }
}
async fn connection(
    channel: ScreencastChannel,
    mut socket: WebSocket,
    _permit: OwnedSemaphorePermit,
) {
    let auth = tokio::select! {
        _ = channel.inner.shutdown.cancelled() => return,
        result = timeout(Duration::from_secs(channel.config().handshake_timeout_seconds), socket.recv()) => result,
    };
    let Ok(Some(Ok(Message::Text(text)))) = auth else {
        let _ = socket.send(Message::Close(None)).await;
        return;
    };
    if text.len() > 2048 {
        let _ = socket.send(Message::Close(None)).await;
        return;
    }
    let Ok(auth) = serde_json::from_str::<Authenticate>(&text) else {
        let _ = socket.send(Message::Close(None)).await;
        return;
    };
    let id = auth.session_id.clone();
    let Ok((cancel, expiry)) = channel.authenticate(auth) else {
        let _ = socket.send(Message::Close(None)).await;
        return;
    };
    let _guard = Disconnect {
        channel: channel.clone(),
        session_id: id.clone(),
    };
    if !send(&mut socket, json!({"type":"authenticated","session_id":id})).await {
        return;
    }
    loop {
        let message = tokio::select! {
            _ = cancel.cancelled() => break,
            _ = sleep_until(expiry.into()) => break,
            message = timeout(Duration::from_secs(channel.config().idle_timeout_seconds), socket.recv()) => message,
        };
        match message {
            Ok(Some(Ok(Message::Binary(bytes)))) => {
                // Backpressure is per socket plus global decode/analysis concurrency. No frame queues.
                let Ok(permit) = channel.inner.frames.clone().try_acquire_owned() else {
                    if !send(&mut socket, json!({"type":"frame","status":"busy"})).await {
                        break;
                    }
                    continue;
                };
                let result = tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = sleep_until(expiry.into()) => break,
                    result = timeout(Duration::from_secs(channel.config().analysis_timeout_seconds), process_frame(&channel, &id, bytes.to_vec(), permit)) => result,
                };
                let status = match result {
                    Ok(Ok(status)) => status,
                    Ok(Err(_)) => "rejected",
                    Err(_) => "timeout",
                };
                if !send(&mut socket, json!({"type":"frame","status":status})).await {
                    break;
                }
            }
            Ok(Some(Ok(Message::Text(text)))) if text == "stop" => break,
            Ok(Some(Ok(Message::Ping(_)))) | Ok(Some(Ok(Message::Pong(_)))) => {}
            _ => break,
        }
    }
    let _ = timeout(Duration::from_secs(1), socket.send(Message::Close(None))).await;
}
async fn send(socket: &mut WebSocket, value: serde_json::Value) -> bool {
    timeout(
        Duration::from_secs(2),
        socket.send(Message::Text(value.to_string().into())),
    )
    .await
    .is_ok_and(|r| r.is_ok())
}
async fn process_frame(
    channel: &ScreencastChannel,
    id: &str,
    bytes: Vec<u8>,
    permit: OwnedSemaphorePermit,
) -> Result<&'static str> {
    let (approval, frame_number, cancellation) = {
        let mut sessions = channel.inner.sessions.lock().unwrap();
        let s = sessions
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("session unavailable"))?;
        if s.cancel.is_cancelled() || s.expires <= Instant::now() {
            bail!("session closed");
        }
        let now = Instant::now();
        if s.last_frame.is_some_and(|last| {
            now.duration_since(last) < Duration::from_millis(channel.config().min_frame_interval_ms)
        }) {
            return Ok("rate_limited");
        }
        if s.status.frames_received >= channel.config().max_frames_per_session {
            close(s);
            bail!("session frame limit");
        }
        // Count all admitted attempts, including invalid frames, to bound abuse.
        s.last_frame = Some(now);
        s.status.frames_received += 1;
        (
            s.approval.clone(),
            s.status.frames_received,
            s.cancel.clone(),
        )
    };
    let config = channel.config().clone();
    // Hold the global permit IN the blocking task too: cancellation cannot release
    // it while a non-cancellable image decoder is still running.
    let (clean, _permit) = tokio::task::spawn_blocking(move || {
        let result = image_frame::sanitize(bytes, &config);
        (result, permit)
    })
    .await?;
    let clean = clean?;
    let analyze = {
        let mut sessions = channel.inner.sessions.lock().unwrap();
        let s = sessions
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("session unavailable"))?;
        if s.cancel.is_cancelled() || s.expires <= Instant::now() {
            bail!("session closed");
        }
        if s.sample.as_ref().is_some_and(|sample| {
            !image_frame::changed(sample, &clean.sample, channel.config().change_threshold)
        }) {
            return Ok("unchanged");
        }
        if s.last_analysis.is_some_and(|last| {
            last.elapsed() < Duration::from_millis(channel.config().analysis_cooldown_ms)
        }) {
            return Ok("cooldown");
        }
        s.sample = Some(clean.sample);
        s.last_analysis = Some(Instant::now());
        approval.analyze
    };
    let request_id = format!("screencast:{id}:{frame_number}");
    let sha256 = hex::encode(Sha256::digest(&clean.png));
    let frame = FrameRequest {
        context: ToolContext {
            scope: approval.scope.clone(),
            turn_id: request_id.clone(),
            cancellation: cancellation.clone(),
        },
        approval_receipt: approval.approval_receipt.clone(),
        session_id: id.into(),
        frame_number,
        sha256: sha256.clone(),
        width: clean.width,
        height: clean.height,
        png: clean.png,
    };
    let analysis = if analyze {
        let output = channel
            .inner
            .analyzer
            .as_ref()
            .expect("validated analyzer")
            .analyze(&frame)
            .await?;
        if output.len() > channel.config().max_analysis_bytes {
            bail!("analysis exceeds output bound");
        }
        Some(missy_security::redact_output(&output))
    } else {
        None
    };
    if cancellation.is_cancelled() {
        bail!("session closed");
    }
    {
        let mut sessions = channel.inner.sessions.lock().unwrap();
        let s = sessions
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("session unavailable"))?;
        if s.expires <= Instant::now() || s.cancel.is_cancelled() {
            bail!("session closed");
        }
        if let Some(text) = &analysis {
            s.status.analyses_completed += 1;
            s.status.latest_analysis = Some(text.clone());
        }
    }
    if approval.deliver_analysis || approval.deliver_frame_artifacts {
        channel
            .inner
            .sink
            .as_ref()
            .expect("validated sink")
            .deliver(ChannelEvent {
                request_id,
                scope: approval.scope,
                session_id: id.into(),
                frame_number,
                sha256,
                width: frame.width,
                height: frame.height,
                analysis: if approval.deliver_analysis {
                    analysis
                } else {
                    None
                },
                png_artifact: if approval.deliver_frame_artifacts {
                    Some(frame.png)
                } else {
                    None
                },
            })
            .await?;
    }
    Ok(if analyze { "analyzed" } else { "accepted" })
}

pub struct ServerHandle {
    pub local_addr: SocketAddr,
    channel: ScreencastChannel,
    task: Option<JoinHandle<std::io::Result<()>>>,
}
impl ServerHandle {
    pub async fn stop(mut self) -> Result<()> {
        self.channel.shutdown();
        if let Some(mut task) = self.task.take() {
            match timeout(Duration::from_secs(5), &mut task).await {
                Ok(result) => result??,
                Err(_) => {
                    task.abort();
                    let _ = task.await;
                    bail!("screencast shutdown exceeded deadline");
                }
            }
        }
        Ok(())
    }
    pub fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(|t| t.is_finished())
    }
}
impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.channel.shutdown();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
/// Owns a production lifecycle task; no detached OS process and no capture starts here.
pub async fn start(channel: ScreencastChannel) -> Result<ServerHandle> {
    if !channel.config().enabled || channel.inner.shutdown.is_cancelled() {
        bail!("screencast disabled or stopped");
    }
    let listener = TcpListener::bind(channel.config().bind).await?;
    let local_addr = listener.local_addr()?;
    let shutdown = channel.inner.shutdown.clone();
    let app = router(channel.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await
    });
    Ok(ServerHandle {
        local_addr,
        channel,
        task: Some(task),
    })
}
