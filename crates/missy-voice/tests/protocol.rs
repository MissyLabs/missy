use anyhow::Result;
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use missy_core::Scope;
use missy_voice::{
    AudioFormat, Device, DeviceRegistry, PolicyMode, SpeechAudio, Stt, Tts, VoiceAgent,
    VoiceConfig, VoiceServer,
};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::{
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};
use tokio_util::sync::CancellationToken;

type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;
fn private_dir() -> tempfile::TempDir {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    dir
}
#[derive(Default)]
struct Mocks {
    stt_calls: AtomicUsize,
    agent_calls: AtomicUsize,
    devices: Mutex<Vec<Device>>,
    cancelled: AtomicUsize,
    slow: bool,
}
#[async_trait]
impl Stt for Mocks {
    async fn transcribe(
        &self,
        pcm: &[u8],
        format: AudioFormat,
        cancellation: CancellationToken,
    ) -> Result<String> {
        self.stt_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(format.sample_rate, 16000);
        assert_eq!(format.channels, 1);
        assert_eq!(pcm, [0, 1, 2, 3]);
        if self.slow {
            let _guard = CancelMark(&self.cancelled);
            tokio::select! { _ = cancellation.cancelled() => {}, _ = tokio::time::sleep(Duration::from_secs(30)) => {} }
        }
        Ok("hello native voice".into())
    }
}
struct CancelMark<'a>(&'a AtomicUsize);
impl Drop for CancelMark<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[async_trait]
impl Tts for Mocks {
    async fn synthesize(&self, text: &str, _: CancellationToken) -> Result<SpeechAudio> {
        assert_eq!(text, "a native answer");
        // Sufficient protocol test WAV. Real adapter separately validates WAV.
        let mut wav = b"RIFF\x30\0\0\0WAVEfmt ".to_vec();
        wav.extend_from_slice(&[0; 64]);
        Ok(SpeechAudio {
            wav,
            sample_rate: 22050,
        })
    }
}
#[async_trait]
impl VoiceAgent for Mocks {
    async fn respond(&self, device: &Device, prompt: &str, _: CancellationToken) -> Result<String> {
        assert_eq!(prompt, "hello native voice");
        self.agent_calls.fetch_add(1, Ordering::SeqCst);
        self.devices.lock().unwrap().push(device.clone());
        Ok("a native answer".into())
    }
}
struct Harness {
    _dir: tempfile::TempDir,
    registry: Arc<DeviceRegistry>,
    node: Device,
    token: String,
    url: String,
    mocks: Arc<Mocks>,
    cancellation: CancellationToken,
    server: JoinHandle<Result<()>>,
}
impl Harness {
    async fn start(mode: PolicyMode, config: VoiceConfig, slow: bool) -> Self {
        let dir = private_dir();
        let registry = Arc::new(DeviceRegistry::open(dir.path().join("devices.json")).unwrap());
        let node = registry
            .request_pairing("Office speaker", "Office")
            .unwrap();
        let token = registry
            .approve(
                &node.node_id,
                Scope {
                    tenant: "private".into(),
                    user: "explicit-user".into(),
                    channel: "voice".into(),
                    session: node.node_id.clone(),
                },
                mode,
            )
            .unwrap();
        let node = registry.get(&node.node_id).unwrap();
        let mocks = Arc::new(Mocks {
            slow,
            ..Default::default()
        });
        let server = Arc::new(
            VoiceServer::new(
                config,
                registry.clone(),
                mocks.clone(),
                mocks.clone(),
                mocks.clone(),
            )
            .unwrap(),
        );
        let cancellation = CancellationToken::new();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let handle = tokio::spawn(server.serve(listener, cancellation.clone()));
        Self {
            _dir: dir,
            registry,
            node,
            token,
            url,
            mocks,
            cancellation,
            server: handle,
        }
    }
    async fn client(&self) -> Client {
        connect_async(&self.url).await.unwrap().0
    }
    async fn authenticated(&self) -> Client {
        let mut c = self.client().await;
        write(
            &mut c,
            json!({"type":"auth", "node_id":self.node.node_id, "token":self.token}),
        )
        .await;
        assert_eq!(read(&mut c).await["type"], "auth_ok");
        c
    }
    async fn shutdown(self) {
        self.cancellation.cancel();
        self.server.await.unwrap().unwrap();
    }
}
async fn write(client: &mut Client, json: Value) {
    client
        .send(Message::Text(json.to_string().into()))
        .await
        .unwrap();
}
async fn read(client: &mut Client) -> Value {
    let msg = tokio::time::timeout(Duration::from_secs(10), client.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    serde_json::from_str(msg.to_text().unwrap()).unwrap()
}
async fn is_closed(client: &mut Client) -> bool {
    matches!(
        tokio::time::timeout(Duration::from_secs(3), client.next()).await,
        Ok(None | Some(Err(_)) | Some(Ok(Message::Close(_))))
    )
}
async fn send_audio(c: &mut Client) {
    write(
        c,
        json!({"type":"audio_start", "sample_rate":16000, "channels":1, "format":"pcm_s16le"}),
    )
    .await;
    c.send(Message::Binary(vec![0, 1, 2, 3].into()))
        .await
        .unwrap();
    write(c, json!({"type":"audio_end"})).await;
}

#[tokio::test]
async fn real_loopback_protocol_streams_audio_and_binds_scope() {
    let config = VoiceConfig {
        debug_transcripts: true,
        audio_chunk_size: 8,
        ..Default::default()
    };
    let h = Harness::start(PolicyMode::SafeChat, config, false).await;
    let mut client = h.authenticated().await;
    send_audio(&mut client).await;
    assert_eq!(read(&mut client).await["type"], "transcript");
    assert_eq!(
        read(&mut client).await,
        json!({"type":"response_text", "text":"a native answer"})
    );
    let start = read(&mut client).await;
    assert_eq!(start["format"], "wav");
    let mut chunks = 0;
    loop {
        let message = client.next().await.unwrap().unwrap();
        match message {
            Message::Binary(bytes) => {
                assert!(bytes.len() <= 8);
                chunks += 1;
            }
            Message::Text(text) => {
                assert_eq!(
                    serde_json::from_str::<Value>(&text).unwrap()["type"],
                    "audio_end"
                );
                break;
            }
            _ => panic!("unexpected frame"),
        }
    }
    assert!(chunks > 1);
    assert_eq!(h.mocks.agent_calls.load(Ordering::SeqCst), 1);
    {
        let devices = h.mocks.devices.lock().unwrap();
        assert_eq!(devices[0].mode, PolicyMode::SafeChat);
        assert_eq!(devices[0].scope.as_ref().unwrap().user, "explicit-user");
        assert_eq!(devices[0].scope.as_ref().unwrap().session, h.node.node_id);
    }
    drop(client);
    h.shutdown().await;
}

#[tokio::test]
async fn pending_pairing_never_authenticates_or_returns_token() {
    let h = Harness::start(PolicyMode::Full, VoiceConfig::default(), false).await;
    let mut c = h.client().await;
    write(&mut c, json!({"type":"pair_request", "friendly_name":"New device", "room":"Lobby", "hardware_profile":{}})).await;
    let pending = read(&mut c).await;
    assert_eq!(pending["type"], "pair_pending");
    assert!(pending.get("token").is_none());
    let id = pending["node_id"].as_str().unwrap();
    let node = h.registry.get(id).unwrap();
    assert!(!node.paired && node.scope.is_none() && node.mode == PolicyMode::Muted);
    assert!(h.registry.authenticate(id, "").is_err());
    assert_eq!(h.mocks.stt_calls.load(Ordering::SeqCst), 0);
    h.shutdown().await;
}

#[tokio::test]
async fn invalid_token_preauth_audio_and_scope_injection_are_denied() {
    let h = Harness::start(PolicyMode::Full, VoiceConfig::default(), false).await;
    for value in [
        json!({"type":"auth", "node_id":h.node.node_id, "token":"invalid"}),
        json!({"type":"audio_start", "sample_rate":16000, "channels":1, "format":"pcm_s16le"}),
        json!({"type":"auth", "node_id":h.node.node_id, "token":h.token, "scope":{"user":"admin"}}),
        json!({"type":"auth", "node_id":h.node.node_id, "token":h.token, "mode":"full"}),
        json!({"type":"text", "text":"run something"}),
    ] {
        let mut c = h.client().await;
        write(&mut c, value).await;
        assert_eq!(read(&mut c).await["type"], "auth_fail");
    }
    let mut binary = h.client().await;
    binary
        .send(Message::Binary(vec![0, 1].into()))
        .await
        .unwrap();
    assert_eq!(read(&mut binary).await["type"], "auth_fail");
    assert_eq!(h.mocks.stt_calls.load(Ordering::SeqCst), 0);
    h.shutdown().await;
}

#[tokio::test]
async fn replay_and_concurrent_device_connections_rejected() {
    let h = Harness::start(PolicyMode::Full, VoiceConfig::default(), false).await;
    let mut c = h.authenticated().await;
    let mut duplicate = h.client().await;
    write(
        &mut duplicate,
        json!({"type":"auth", "node_id":h.node.node_id, "token":h.token}),
    )
    .await;
    assert_eq!(read(&mut duplicate).await["type"], "auth_fail");
    write(
        &mut c,
        json!({"type":"auth", "node_id":h.node.node_id, "token":h.token}),
    )
    .await;
    assert_eq!(read(&mut c).await["type"], "error");
    assert!(is_closed(&mut c).await);
    h.shutdown().await;
}

#[tokio::test]
async fn muted_and_operator_changes_are_enforced() {
    let h = Harness::start(PolicyMode::Muted, VoiceConfig::default(), false).await;
    let mut c = h.authenticated().await;
    write(
        &mut c,
        json!({"type":"audio_start", "sample_rate":16000, "channels":1, "format":"pcm_s16le"}),
    )
    .await;
    assert_eq!(read(&mut c).await["type"], "muted");
    assert_eq!(h.mocks.stt_calls.load(Ordering::SeqCst), 0);
    // A separately opened registry is the CLI's view; server must refresh.
    let cli = DeviceRegistry::open(h._dir.path().join("devices.json")).unwrap();
    cli.set_mode(&h.node.node_id, PolicyMode::SafeChat).unwrap();
    write(
        &mut c,
        json!({"type":"heartbeat", "node_id":h.node.node_id}),
    )
    .await;
    assert_eq!(read(&mut c).await["mode"], "safe-chat");
    cli.revoke(&h.node.node_id).unwrap();
    write(
        &mut c,
        json!({"type":"heartbeat", "node_id":h.node.node_id}),
    )
    .await;
    assert!(is_closed(&mut c).await);
    h.shutdown().await;
}

#[tokio::test]
async fn audio_bounds_alignment_sequence_and_payload_limit() {
    let config = VoiceConfig {
        max_audio_bytes: 4,
        ..Default::default()
    };
    let h = Harness::start(PolicyMode::Full, config, false).await;
    for start in [
        json!({"type":"audio_start", "sample_rate":7999, "channels":1, "format":"pcm_s16le"}),
        json!({"type":"audio_start", "sample_rate":48001, "channels":1, "format":"pcm_s16le"}),
        json!({"type":"audio_start", "sample_rate":16000, "channels":0, "format":"pcm_s16le"}),
        json!({"type":"audio_start", "sample_rate":16000, "channels":3, "format":"pcm_s16le"}),
        json!({"type":"audio_start", "sample_rate":16000, "channels":1, "format":"wav"}),
        json!({"type":"audio_end"}),
    ] {
        let mut c = h.authenticated().await;
        write(&mut c, start).await;
        assert!(is_closed(&mut c).await);
    }
    let mut c = h.authenticated().await;
    write(
        &mut c,
        json!({"type":"audio_start", "sample_rate":16000, "channels":1, "format":"pcm_s16le"}),
    )
    .await;
    c.send(Message::Binary(vec![1, 2, 3, 4, 5].into()))
        .await
        .unwrap();
    assert!(is_closed(&mut c).await);
    let mut c = h.authenticated().await;
    write(
        &mut c,
        json!({"type":"audio_start", "sample_rate":16000, "channels":1, "format":"pcm_s16le"}),
    )
    .await;
    c.send(Message::Binary(vec![1, 2, 3].into())).await.unwrap();
    write(&mut c, json!({"type":"audio_end"})).await;
    assert!(is_closed(&mut c).await);
    assert_eq!(h.mocks.stt_calls.load(Ordering::SeqCst), 0);
    h.shutdown().await;
}

#[tokio::test]
async fn disconnect_cancels_inflight_stt() {
    let h = Harness::start(PolicyMode::Full, VoiceConfig::default(), true).await;
    let mut c = h.authenticated().await;
    send_audio(&mut c).await;
    tokio::time::timeout(Duration::from_secs(3), async {
        while h.mocks.stt_calls.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    c.close(None).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while h.mocks.cancelled.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(h.mocks.agent_calls.load(Ordering::SeqCst), 0);
    h.shutdown().await;
}

#[tokio::test]
async fn auth_timeout_connection_and_frame_limits() {
    let config = VoiceConfig {
        auth_timeout_seconds: 1,
        max_connections: 1,
        max_frame_bytes: 128,
        ..Default::default()
    };
    // chunk size must fit max_frame_bytes.
    let config = VoiceConfig {
        audio_chunk_size: 64,
        ..config
    };
    let h = Harness::start(PolicyMode::Full, config, false).await;
    let mut idle = h.client().await;
    assert!(connect_async(&h.url).await.is_err());
    assert!(is_closed(&mut idle).await);
    // More than 128 bytes rejected before parsing/auth callbacks.
    let mut oversized = h.client().await;
    oversized
        .send(Message::Text("x".repeat(129).into()))
        .await
        .unwrap();
    assert!(is_closed(&mut oversized).await);
    h.shutdown().await;
}

#[tokio::test]
async fn frame_rate_and_client_mode_changes_fail_closed() {
    let h = Harness::start(
        PolicyMode::Full,
        VoiceConfig {
            frames_per_second: 1,
            ..Default::default()
        },
        false,
    )
    .await;
    let mut c = h.authenticated().await;
    write(
        &mut c,
        json!({"type":"heartbeat", "node_id":h.node.node_id}),
    )
    .await;
    assert_eq!(read(&mut c).await["type"], "heartbeat_ack");
    write(
        &mut c,
        json!({"type":"heartbeat", "node_id":h.node.node_id}),
    )
    .await;
    assert!(is_closed(&mut c).await);
    let mut c = h.authenticated().await;
    write(&mut c, json!({"type":"audio_start", "sample_rate":16000, "channels":1, "format":"pcm_s16le", "mode":"full"})).await;
    assert_eq!(read(&mut c).await["type"], "error");
    assert!(is_closed(&mut c).await);
    let mut c = h.authenticated().await;
    write(
        &mut c,
        json!({"type":"heartbeat", "node_id":"other-device"}),
    )
    .await;
    assert!(is_closed(&mut c).await);
    assert_eq!(h.mocks.agent_calls.load(Ordering::SeqCst), 0);
    h.shutdown().await;
}

#[tokio::test]
async fn operator_mute_cancels_inflight_stt_before_agent_call() {
    let h = Harness::start(PolicyMode::Full, VoiceConfig::default(), true).await;
    let mut c = h.authenticated().await;
    send_audio(&mut c).await;
    tokio::time::timeout(Duration::from_secs(3), async {
        while h.mocks.stt_calls.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    h.registry
        .set_mode(&h.node.node_id, PolicyMode::Muted)
        .unwrap();
    assert!(is_closed(&mut c).await);
    assert_eq!(h.mocks.cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(h.mocks.agent_calls.load(Ordering::SeqCst), 0);
    h.shutdown().await;
}

#[tokio::test]
async fn supplied_wildcard_listener_is_rejected() {
    let h = Harness::start(PolicyMode::Full, VoiceConfig::default(), false).await;
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let server = Arc::new(
        VoiceServer::new(
            VoiceConfig::default(),
            h.registry.clone(),
            h.mocks.clone(),
            h.mocks.clone(),
            h.mocks.clone(),
        )
        .unwrap(),
    );
    assert!(
        server
            .serve(listener, CancellationToken::new())
            .await
            .is_err()
    );
    h.shutdown().await;
}

#[test]
fn registry_is_private_persistent_hashed_and_fixed_scope() {
    use std::{fs, os::unix::fs::PermissionsExt};
    let dir = private_dir();
    let path = dir.path().join("devices.json");
    let registry = DeviceRegistry::open(&path).unwrap();
    let node = registry.request_pairing("Desk", "Room").unwrap();
    let bad = Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "api".into(),
        session: node.node_id.clone(),
    };
    assert!(
        registry
            .approve(&node.node_id, bad.clone(), PolicyMode::Full)
            .is_err()
    );
    let scope = Scope {
        channel: "voice".into(),
        ..bad
    };
    let token = registry
        .approve(&node.node_id, scope.clone(), PolicyMode::SafeChat)
        .unwrap();
    let bytes = fs::read_to_string(&path).unwrap();
    assert!(!bytes.contains(&token));
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let loaded = DeviceRegistry::open(&path).unwrap();
    assert_eq!(
        loaded.authenticate(&node.node_id, &token).unwrap().scope,
        Some(scope.clone())
    );
    assert!(
        loaded
            .approve(&node.node_id, scope, PolicyMode::Full)
            .is_err()
    );
    assert!(loaded.authenticate("unknown", &token).is_err());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(DeviceRegistry::open(&path).is_err());
}

#[test]
fn legacy_corrupt_symlink_and_insecure_remote_configs_fail_closed() {
    use std::{fs, os::unix::fs::symlink};
    let dir = private_dir();
    let path = dir.path().join("devices.json");
    fs::write(&path, r#"{"node":{"paired":true,"token_hash":"legacy"}}"#).unwrap();
    assert!(DeviceRegistry::open(&path).is_err());
    fs::remove_file(&path).unwrap();
    symlink("other", &path).unwrap();
    assert!(DeviceRegistry::open(&path).is_err());
    assert!(
        VoiceConfig {
            bind: "0.0.0.0:8765".parse().unwrap(),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(serde_json::from_value::<VoiceConfig>(json!({"allow_insecure_remote":true})).is_err());
    assert!(serde_json::from_value::<VoiceConfig>(json!({"bind":"localhost:8765"})).is_err());
}
