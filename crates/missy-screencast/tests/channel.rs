use anyhow::{Result, bail};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use missy_core::Scope;
use missy_screencast::*;
use serde_json::{Value, json};
use std::{
    io::Cursor,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message, client::IntoClientRequest},
};

struct Verifier;
struct PendingAnalyzer(Arc<AtomicUsize>);
#[async_trait]
impl FrameAnalyzer for PendingAnalyzer {
    async fn analyze(&self, _: &FrameRequest) -> Result<String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        std::future::pending().await
    }
}
#[tokio::test]
async fn owner_cancel_interrupts_pending_analysis_and_prevents_delivery() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sink = Arc::new(Sink::default());
    let channel = ScreencastChannel::new(
        ScreencastConfig {
            allow_analysis: true,
            allow_analysis_delivery: true,
            ..config()
        },
        Arc::new(Verifier),
        Some(Arc::new(PendingAnalyzer(calls.clone()))),
        Some(sink.clone()),
    )
    .unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let pair = channel
        .create_session(SessionApproval {
            analyze: true,
            deliver_analysis: true,
            ..grant()
        })
        .await
        .unwrap();
    let mut socket = auth(&channel, &pair).await;
    socket.send(Message::Binary(png(0).into())).await.unwrap();
    for _ in 0..100 {
        if calls.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    channel.cancel_session(&scope(), &pair.session_id).unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap(),
        Some(Ok(Message::Close(_))) | None
    ));
    assert!(sink.0.lock().unwrap().is_empty());
    handle.stop().await.unwrap();
}
#[async_trait]
impl ApprovalVerifier for Verifier {
    async fn verify(&self, grant: &SessionApproval, fingerprint: &str) -> Result<()> {
        assert!(fingerprint.starts_with("screencast-v1:"));
        if grant.approval_receipt != "durable-human-receipt" {
            bail!("not approved");
        }
        Ok(())
    }
}
#[derive(Default)]
struct Analyzer {
    calls: AtomicUsize,
}
#[async_trait]
impl FrameAnalyzer for Analyzer {
    async fn analyze(&self, frame: &FrameRequest) -> Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(frame.png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(frame.approval_receipt, "durable-human-receipt");
        assert!(frame.context.turn_id.starts_with("screencast:"));
        Ok("Untrusted screen observation".into())
    }
}
#[derive(Default)]
struct Sink(Mutex<Vec<ChannelEvent>>);
#[async_trait]
impl EventSink for Sink {
    async fn deliver(&self, event: ChannelEvent) -> Result<()> {
        self.0.lock().unwrap().push(event);
        Ok(())
    }
}
fn scope() -> Scope {
    Scope {
        tenant: "fixture".into(),
        user: "owner".into(),
        channel: "discord:7".into(),
        session: "parent".into(),
    }
}
fn config() -> ScreencastConfig {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let bind = listener.local_addr().unwrap();
    drop(listener);
    ScreencastConfig {
        enabled: true,
        bind,
        min_frame_interval_ms: 100,
        analysis_cooldown_ms: 0,
        handshake_timeout_seconds: 1,
        ..Default::default()
    }
}
fn grant() -> SessionApproval {
    SessionApproval {
        scope: scope(),
        approval_receipt: "durable-human-receipt".into(),
        ttl_seconds: 30,
        analyze: false,
        deliver_analysis: false,
        deliver_frame_artifacts: false,
    }
}
fn pairing_value(pair: &Pairing) -> Value {
    json!({"session_id":pair.session_id,"token":pair.token,"scope":pair.scope})
}
type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;
async fn ws(channel: &ScreencastChannel) -> Ws {
    let mut req = format!("ws://{}/ws", channel.config().bind)
        .into_client_request()
        .unwrap();
    req.headers_mut()
        .insert("origin", channel.config().origin().parse().unwrap());
    connect_async(req).await.unwrap().0
}
async fn auth(channel: &ScreencastChannel, pair: &Pairing) -> Ws {
    let mut socket = ws(channel).await;
    socket
        .send(Message::Text(pairing_value(pair).to_string().into()))
        .await
        .unwrap();
    assert_eq!(receive(&mut socket).await["type"], "authenticated");
    socket
}
async fn receive(socket: &mut Ws) -> Value {
    let message = tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    serde_json::from_str(message.to_text().unwrap()).unwrap()
}
async fn frame(socket: &mut Ws, bytes: Vec<u8>) -> String {
    socket.send(Message::Binary(bytes.into())).await.unwrap();
    receive(socket).await["status"]
        .as_str()
        .unwrap()
        .to_string()
}
fn png(color: u8) -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([color, color, color])));
    let mut out = Cursor::new(vec![]);
    image.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}
async fn delay() {
    tokio::time::sleep(Duration::from_millis(120)).await;
}

#[tokio::test]
async fn handshake_connection_limit_cooldown_and_default_no_delivery() {
    let analyzer = Arc::new(Analyzer::default());
    let cfg = ScreencastConfig {
        max_connections: 1,
        allow_analysis: true,
        analysis_cooldown_ms: 10_000,
        ..config()
    };
    let channel =
        ScreencastChannel::new(cfg, Arc::new(Verifier), Some(analyzer.clone()), None).unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let mut idle = ws(&channel).await;
    let mut request = format!("ws://{}/ws", channel.config().bind)
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", channel.config().origin().parse().unwrap());
    assert!(connect_async(request).await.is_err());
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(2), idle.next())
            .await
            .unwrap(),
        Some(Ok(Message::Close(_))) | None
    ));
    drop(idle);
    let pair = channel
        .create_session(SessionApproval {
            analyze: true,
            ..grant()
        })
        .await
        .unwrap();
    let mut socket = auth(&channel, &pair).await;
    assert_eq!(frame(&mut socket, png(0)).await, "analyzed");
    delay().await;
    assert_eq!(frame(&mut socket, png(255)).await, "cooldown");
    assert_eq!(analyzer.calls.load(Ordering::SeqCst), 1);
    channel.cancel_session(&scope(), &pair.session_id).unwrap();
    handle.stop().await.unwrap();
}

#[tokio::test]
async fn real_loopback_intake_analysis_and_exact_scope_sink() {
    let analyzer = Arc::new(Analyzer::default());
    let sink = Arc::new(Sink::default());
    let config = ScreencastConfig {
        allow_analysis: true,
        allow_analysis_delivery: true,
        allow_frame_artifacts: true,
        ..config()
    };
    let channel = ScreencastChannel::new(
        config,
        Arc::new(Verifier),
        Some(analyzer.clone()),
        Some(sink.clone()),
    )
    .unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let pair = channel
        .create_session(SessionApproval {
            analyze: true,
            deliver_analysis: true,
            deliver_frame_artifacts: true,
            ..grant()
        })
        .await
        .unwrap();
    assert!(!pair.capture_url.contains(&pair.token));
    let mut socket = auth(&channel, &pair).await;
    assert_eq!(frame(&mut socket, png(0)).await, "analyzed");
    delay().await;
    assert_eq!(frame(&mut socket, png(0)).await, "unchanged");
    delay().await;
    assert_eq!(frame(&mut socket, png(255)).await, "analyzed");
    assert_eq!(analyzer.calls.load(Ordering::SeqCst), 2);
    {
        let delivered = sink.0.lock().unwrap();
        assert_eq!(delivered.len(), 2);
        assert_eq!(delivered[0].scope, scope());
        assert!(delivered[0].png_artifact.is_some());
        assert_eq!(
            delivered[0].analysis.as_deref(),
            Some("Untrusted screen observation")
        );
        assert_ne!(delivered[0].request_id, delivered[1].request_id);
    }
    let status = channel.status(&scope(), &pair.session_id).unwrap();
    assert_eq!(status.frames_received, 3);
    assert_eq!(status.analyses_completed, 2);
    socket.send(Message::Text("stop".into())).await.unwrap();
    delay().await;
    assert_eq!(
        channel.status(&scope(), &pair.session_id).unwrap().phase,
        SessionPhase::Closed
    );
    handle.stop().await.unwrap();
}

#[tokio::test]
async fn policy_defaults_approval_owner_locks_and_capacity() {
    let mut cfg = config();
    cfg.max_sessions = 1;
    let channel = ScreencastChannel::new(cfg, Arc::new(Verifier), None, None).unwrap();
    assert!(
        channel
            .create_session(SessionApproval {
                approval_receipt: "made-up".into(),
                ..grant()
            })
            .await
            .is_err()
    );
    assert!(
        channel
            .create_session(SessionApproval {
                analyze: true,
                ..grant()
            })
            .await
            .is_err()
    );
    assert!(
        channel
            .create_session(SessionApproval {
                deliver_frame_artifacts: true,
                ..grant()
            })
            .await
            .is_err()
    );
    let pair = channel.create_session(grant()).await.unwrap();
    assert!(channel.create_session(grant()).await.is_err());
    let mut alien = scope();
    alien.user = "attacker".into();
    assert!(channel.status(&alien, &pair.session_id).is_err());
    assert!(channel.cancel_session(&alien, &pair.session_id).is_err());
    channel.cancel_session(&scope(), &pair.session_id).unwrap();
    let new = channel.create_session(grant()).await.unwrap();
    assert_ne!(pair.token, new.token);
    channel.shutdown();
    assert!(channel.create_session(grant()).await.is_err());
}

#[tokio::test]
async fn http_rebinding_csrf_query_and_static_headers() {
    let channel = ScreencastChannel::new(config(), Arc::new(Verifier), None, None).unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let client = reqwest::Client::new();
    let url = channel.config().origin();
    let page = client.get(&url).send().await.unwrap();
    assert_eq!(page.status(), 200);
    assert_eq!(page.headers()["cache-control"], "no-store");
    assert!(!page.headers().contains_key("access-control-allow-origin"));
    assert!(page.text().await.unwrap().contains("Start sharing"));
    assert_eq!(
        client
            .get(&url)
            .header("Host", "attacker.test")
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        client
            .get(&url)
            .header("Origin", "https://attacker.test")
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        client
            .get(&url)
            .header("Sec-Fetch-Site", "cross-site")
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        client
            .get(format!("{url}/?token=do-not-log"))
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(client.post(&url).send().await.unwrap().status(), 403);
    let script = client
        .get(format!("{url}/capture.js"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!script.contains("localStorage"));
    assert!(!script.contains("sessionStorage"));
    assert_eq!(script.matches("getDisplayMedia({").count(), 1);
    let request = format!("ws://{}/ws", channel.config().bind)
        .into_client_request()
        .unwrap();
    assert!(connect_async(request).await.is_err());
    let mut request = format!("ws://{}/ws", channel.config().bind)
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", "http://attacker.test".parse().unwrap());
    assert!(connect_async(request).await.is_err());
    handle.stop().await.unwrap();
}

#[tokio::test]
async fn first_message_required_scope_bound_single_use_no_reconnect() {
    let channel = ScreencastChannel::new(config(), Arc::new(Verifier), None, None).unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let pair = channel.create_session(grant()).await.unwrap();
    let mut binary = ws(&channel).await;
    binary.send(Message::Binary(png(0).into())).await.unwrap();
    assert!(matches!(
        binary.next().await,
        Some(Ok(Message::Close(_))) | None
    ));
    let mut wrong = ws(&channel).await;
    let mut val = pairing_value(&pair);
    val["scope"]["user"] = json!("attacker");
    wrong
        .send(Message::Text(val.to_string().into()))
        .await
        .unwrap();
    assert!(matches!(
        wrong.next().await,
        Some(Ok(Message::Close(_))) | None
    ));
    let mut good = auth(&channel, &pair).await;
    let mut replay = ws(&channel).await;
    replay
        .send(Message::Text(pairing_value(&pair).to_string().into()))
        .await
        .unwrap();
    assert!(matches!(
        replay.next().await,
        Some(Ok(Message::Close(_))) | None
    ));
    assert_eq!(frame(&mut good, png(0)).await, "accepted");
    channel.cancel_session(&scope(), &pair.session_id).unwrap();
    assert!(matches!(
        good.next().await,
        Some(Ok(Message::Close(_))) | None
    ));
    let mut reconnect = ws(&channel).await;
    reconnect
        .send(Message::Text(pairing_value(&pair).to_string().into()))
        .await
        .unwrap();
    assert!(matches!(
        reconnect.next().await,
        Some(Ok(Message::Close(_))) | None
    ));
    handle.stop().await.unwrap();
}

#[tokio::test]
async fn invalid_dimension_payload_rate_bounds() {
    let config = ScreencastConfig {
        max_dimension: 32,
        max_frames_per_session: 2,
        ..config()
    };
    let channel = ScreencastChannel::new(config, Arc::new(Verifier), None, None).unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let pair = channel.create_session(grant()).await.unwrap();
    let mut socket = auth(&channel, &pair).await;
    assert_eq!(frame(&mut socket, png(0)).await, "rejected");
    assert_eq!(frame(&mut socket, png(0)).await, "rate_limited");
    delay().await;
    assert_eq!(frame(&mut socket, vec![1, 2, 3]).await, "rejected");
    assert_eq!(
        channel
            .status(&scope(), &pair.session_id)
            .unwrap()
            .frames_received,
        2
    );
    channel.cancel_session(&scope(), &pair.session_id).unwrap();
    handle.stop().await.unwrap();
}

#[tokio::test]
async fn expiry_and_restart_invalidate_credentials() {
    let config = config();
    let channel = ScreencastChannel::new(config.clone(), Arc::new(Verifier), None, None).unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let pair = channel
        .create_session(SessionApproval {
            ttl_seconds: 1,
            ..grant()
        })
        .await
        .unwrap();
    let mut socket = auth(&channel, &pair).await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    assert!(matches!(
        socket.next().await,
        Some(Ok(Message::Close(_))) | None
    ));
    assert_eq!(
        channel.status(&scope(), &pair.session_id).unwrap().phase,
        SessionPhase::Closed
    );
    handle.stop().await.unwrap();
    let restarted = ScreencastChannel::new(config, Arc::new(Verifier), None, None).unwrap();
    let handle = start(restarted.clone()).await.unwrap();
    let mut replay = ws(&restarted).await;
    replay
        .send(Message::Text(pairing_value(&pair).to_string().into()))
        .await
        .unwrap();
    assert!(matches!(
        replay.next().await,
        Some(Ok(Message::Close(_))) | None
    ));
    handle.stop().await.unwrap();
}

#[test]
fn invalid_config_and_missing_dependencies_fail_closed() {
    let cfg = ScreencastConfig {
        bind: "0.0.0.0:8780".parse().unwrap(),
        ..Default::default()
    };
    assert!(ScreencastChannel::new(cfg, Arc::new(Verifier), None, None).is_err());
    let cfg = ScreencastConfig {
        allow_analysis: true,
        ..config()
    };
    assert!(ScreencastChannel::new(cfg, Arc::new(Verifier), None, None).is_err());
    let cfg = ScreencastConfig {
        allow_frame_artifacts: true,
        ..config()
    };
    assert!(ScreencastChannel::new(cfg, Arc::new(Verifier), None, None).is_err());
}

/// Opt-in because Playwright is not a Rust dependency. CI operators may set the
/// interpreter path to a venv with Playwright and its headless Chromium installed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chromium_synthetic_browser_fixture() {
    let Ok(python) = std::env::var("MISSY_SCREENCAST_BROWSER_PYTHON") else {
        return;
    };
    let analyzer = Arc::new(Analyzer::default());
    let channel = ScreencastChannel::new(
        ScreencastConfig {
            allow_analysis: true,
            ..config()
        },
        Arc::new(Verifier),
        Some(analyzer.clone()),
        None,
    )
    .unwrap();
    let handle = start(channel.clone()).await.unwrap();
    let pair = channel
        .create_session(SessionApproval {
            analyze: true,
            ..grant()
        })
        .await
        .unwrap();
    let payload =
        json!({"url":channel.config().origin(),"pairing":pairing_value(&pair)}).to_string();
    let script = format!("{}/tests/browser_fixture.py", env!("CARGO_MANIFEST_DIR"));
    let output = tokio::task::spawn_blocking(move || {
        use std::io::Write;
        let mut child = std::process::Command::new(python)
            .arg(script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(payload.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("PASS"));
    assert_eq!(analyzer.calls.load(Ordering::SeqCst), 1);
    handle.stop().await.unwrap();
}
