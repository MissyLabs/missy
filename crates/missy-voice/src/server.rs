use crate::{AudioFormat, Device, DeviceRegistry, PolicyMode, Stt, Tts, VoiceAgent};
use anyhow::{Context, Result, bail, ensure};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::Semaphore,
    task::JoinSet,
};
use tokio_tungstenite::{
    WebSocketStream, accept_async_with_config,
    tungstenite::{Message, protocol::WebSocketConfig},
};
use tokio_util::sync::{CancellationToken, DropGuard};

const MAX_AUDIO: usize = 10 * 1024 * 1024;
const MAX_FRAME: usize = 1024 * 1024;

/// Independent voice config; never deserialize voice input into application config.
/// TLS/remote and the legacy insecure-remote escape hatch are deliberately absent.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoiceConfig {
    pub bind: SocketAddr,
    pub auth_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
    pub turn_timeout_seconds: u64,
    pub max_connections: usize,
    pub max_frame_bytes: usize,
    pub max_audio_bytes: usize,
    pub frames_per_second: usize,
    pub max_turns_per_minute: usize,
    pub audio_chunk_size: usize,
    pub debug_transcripts: bool,
}
impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8765".parse().unwrap(),
            auth_timeout_seconds: 10,
            idle_timeout_seconds: 120,
            turn_timeout_seconds: 180,
            max_connections: 50,
            max_frame_bytes: MAX_FRAME,
            max_audio_bytes: MAX_AUDIO,
            frames_per_second: 64,
            max_turns_per_minute: 10,
            audio_chunk_size: 4096,
            debug_transcripts: false,
        }
    }
}
impl VoiceConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.bind.ip().is_loopback(),
            "voice supports numeric loopback only; remote TLS not implemented"
        );
        ensure!(
            (1..=50).contains(&self.max_connections),
            "invalid connection limit"
        );
        ensure!(
            (128..=MAX_FRAME).contains(&self.max_frame_bytes),
            "invalid frame limit"
        );
        ensure!(
            (2..=MAX_AUDIO).contains(&self.max_audio_bytes),
            "invalid audio limit"
        );
        ensure!(
            (1..=128).contains(&self.frames_per_second)
                && (1..=60).contains(&self.max_turns_per_minute),
            "invalid rate limit"
        );
        ensure!(
            (1..=self.max_frame_bytes).contains(&self.audio_chunk_size),
            "invalid chunk size"
        );
        ensure!(
            (1..=30).contains(&self.auth_timeout_seconds)
                && (1..=600).contains(&self.idle_timeout_seconds)
                && (1..=600).contains(&self.turn_timeout_seconds),
            "invalid timeout"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum ClientMessage {
    #[serde(rename = "auth")]
    Auth { node_id: String, token: String },
    #[serde(rename = "pair_request")]
    PairRequest {
        friendly_name: String,
        #[serde(default)]
        room: String,
        #[serde(default)]
        hardware_profile: Value,
    },
    #[serde(rename = "audio_start")]
    AudioStart {
        sample_rate: u32,
        channels: u16,
        format: String,
    },
    #[serde(rename = "audio_end")]
    AudioEnd,
    #[serde(rename = "heartbeat")]
    Heartbeat {
        node_id: String,
        #[serde(default)]
        occupancy: Option<bool>,
        #[serde(default)]
        noise_level: Option<f64>,
        #[serde(default)]
        wake_word_fp: bool,
    },
}

type Ws = WebSocketStream<TcpStream>;
type Sender = futures_util::stream::SplitSink<Ws, Message>;

pub struct VoiceServer {
    config: VoiceConfig,
    registry: Arc<DeviceRegistry>,
    stt: Arc<dyn Stt>,
    tts: Arc<dyn Tts>,
    agent: Arc<dyn VoiceAgent>,
    connected: Arc<Mutex<HashSet<String>>>,
}
impl VoiceServer {
    pub fn new(
        config: VoiceConfig,
        registry: Arc<DeviceRegistry>,
        stt: Arc<dyn Stt>,
        tts: Arc<dyn Tts>,
        agent: Arc<dyn VoiceAgent>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            registry,
            stt,
            tts,
            agent,
            connected: Arc::new(Mutex::new(HashSet::new())),
        })
    }

    pub async fn bind_and_serve(self: Arc<Self>, cancellation: CancellationToken) -> Result<()> {
        let listener = TcpListener::bind(self.config.bind).await?;
        self.serve(listener, cancellation).await
    }

    /// Embedders may bind port 0 for tests. Even a supplied listener is checked:
    /// a wildcard/non-loopback listener cannot bypass the config restriction.
    pub async fn serve(
        self: Arc<Self>,
        listener: TcpListener,
        cancellation: CancellationToken,
    ) -> Result<()> {
        self.config.validate()?;
        ensure!(
            listener.local_addr()?.ip().is_loopback(),
            "listener must be loopback"
        );
        let slots = Arc::new(Semaphore::new(self.config.max_connections));
        let mut tasks = JoinSet::new();
        loop {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => break,
                Some(_) = tasks.join_next(), if !tasks.is_empty() => {},
                result = listener.accept() => {
                    let (stream, peer) = result?;
                    if !peer.ip().is_loopback() { continue; }
                    let Ok(permit) = slots.clone().try_acquire_owned() else { drop(stream); continue; };
                    let server = self.clone();
                    let child = cancellation.child_token();
                    tasks.spawn(async move {
                        let _permit = permit;
                        let _cancel_on_drop = child.clone().drop_guard();
                        // Protocol and credential details never enter logs or network errors.
                        let _ = server.connection(stream, child).await;
                    });
                }
            }
        }
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        Ok(())
    }

    async fn connection(&self, stream: TcpStream, cancellation: CancellationToken) -> Result<()> {
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.config.auth_timeout_seconds);
        let ws_config = WebSocketConfig::default()
            .max_frame_size(Some(self.config.max_frame_bytes))
            .max_message_size(Some(self.config.max_frame_bytes));
        let mut ws = tokio::select! {
            _ = cancellation.cancelled() => return Ok(()),
            result = tokio::time::timeout_at(deadline, accept_async_with_config(stream, Some(ws_config))) => result??,
        };
        let first = tokio::select! {
            _ = cancellation.cancelled() => return Ok(()),
            result = tokio::time::timeout_at(deadline, ws.next()) => result?.context("auth frame missing")??,
        };
        let parsed = match first {
            Message::Text(t) => serde_json::from_str::<ClientMessage>(&t),
            _ => {
                send_ws(
                    &mut ws,
                    json!({"type":"auth_fail", "reason":"authentication required"}),
                )
                .await?;
                return Ok(());
            }
        };
        let device = match parsed {
            Ok(ClientMessage::Auth { node_id, token }) => {
                // PBKDF2 is intentionally expensive. Keep CPU-bound work off the
                // async reactor; the admission permit bounds concurrent hashes.
                let registry = self.registry.clone();
                let result =
                    tokio::task::spawn_blocking(move || registry.authenticate(&node_id, &token))
                        .await?;
                match result {
                    Ok(device) => device,
                    Err(_) => {
                        send_ws(
                            &mut ws,
                            json!({"type":"auth_fail", "reason":"invalid credentials"}),
                        )
                        .await?;
                        return Ok(());
                    }
                }
            }
            Ok(ClientMessage::PairRequest {
                friendly_name,
                room,
                hardware_profile,
            }) => {
                ensure!(
                    hardware_profile.is_null() || hardware_profile.is_object(),
                    "invalid hardware profile"
                );
                match self.registry.request_pairing(&friendly_name, &room) {
                    Ok(device) => {
                        send_ws(
                            &mut ws,
                            json!({"type":"pair_pending", "node_id":device.node_id}),
                        )
                        .await?
                    }
                    Err(_) => {
                        send_ws(
                            &mut ws,
                            json!({"type":"error", "message":"pairing request rejected"}),
                        )
                        .await?
                    }
                }
                // Never automatically approve or expose the operator-issued token.
                let _ = ws.close(None).await;
                return Ok(());
            }
            _ => {
                send_ws(
                    &mut ws,
                    json!({"type":"auth_fail", "reason":"authentication required"}),
                )
                .await?;
                return Ok(());
            }
        };
        let guard = match ActiveDevice::claim(self.connected.clone(), &device.node_id) {
            Ok(guard) => guard,
            Err(_) => {
                send_ws(
                    &mut ws,
                    json!({"type":"auth_fail", "reason":"device already connected"}),
                )
                .await?;
                return Ok(());
            }
        };
        send_ws(
            &mut ws,
            json!({"type":"auth_ok", "node_id":device.node_id, "room":device.room}),
        )
        .await?;
        let (mut sender, mut receiver) = ws.split();
        let mut audio: Option<(AudioFormat, Vec<u8>)> = None;
        let mut frame_rate = Rate::new(Duration::from_secs(1), self.config.frames_per_second);
        let mut turn_rate = Rate::new(Duration::from_secs(60), self.config.max_turns_per_minute);
        loop {
            let message = tokio::select! {
                _ = cancellation.cancelled() => break,
                message = tokio::time::timeout(Duration::from_secs(self.config.idle_timeout_seconds), receiver.next()) => message?.context("connection closed")??,
            };
            frame_rate.check()?;
            // Re-read every frame so operator revocation/muting is not lost behind
            // an old gateway snapshot. No client field is used as authority.
            let trusted = self.registry.get(&device.node_id)?;
            ensure!(
                trusted.paired && same_scope(&trusted, &device),
                "device scope changed or revoked"
            );
            if trusted.mode == PolicyMode::Muted {
                audio = None;
                send(&mut sender, json!({"type":"muted"})).await?;
                if matches!(message, Message::Close(_)) {
                    break;
                }
                continue;
            }
            match message {
                Message::Text(text) => {
                    let request: ClientMessage = match serde_json::from_str(&text) {
                        Ok(request) => request,
                        Err(_) => {
                            send(
                                &mut sender,
                                json!({"type":"error", "message":"invalid protocol message"}),
                            )
                            .await?;
                            break;
                        }
                    };
                    match request {
                        ClientMessage::AudioStart {
                            sample_rate,
                            channels,
                            format,
                        } => {
                            ensure!(
                                audio.is_none() && format == "pcm_s16le",
                                "invalid audio_start"
                            );
                            let format = AudioFormat {
                                sample_rate,
                                channels,
                            };
                            format.validate()?;
                            audio = Some((format, Vec::new()));
                        }
                        ClientMessage::AudioEnd => {
                            let (format, pcm) =
                                audio.take().context("audio_end without audio_start")?;
                            ensure!(
                                !pcm.is_empty() && pcm.len() % (2 * format.channels as usize) == 0,
                                "invalid PCM alignment"
                            );
                            turn_rate.check()?;
                            let turn_cancel = cancellation.child_token();
                            let _drop_guard: DropGuard = turn_cancel.clone().drop_guard();
                            let result = self.process(
                                &trusted,
                                &pcm,
                                format,
                                &mut sender,
                                turn_cancel.clone(),
                            );
                            tokio::pin!(result);
                            let turn_deadline = tokio::time::Instant::now()
                                + Duration::from_secs(self.config.turn_timeout_seconds);
                            let mut authority_poll =
                                tokio::time::interval(Duration::from_millis(250));
                            // Observe a close while STT/provider/TTS run. Cancellation
                            // drops the future and kills command process groups.
                            loop {
                                tokio::select! {
                                    biased;
                                    _ = cancellation.cancelled() => return Ok(()),
                                    _ = tokio::time::sleep_until(turn_deadline) => bail!("voice turn timed out"),
                                    _ = authority_poll.tick() => { self.check_output_authority(&trusted)?; },
                                    incoming = receiver.next() => {
                                        frame_rate.check()?;
                                        match incoming {
                                            None | Some(Ok(Message::Close(_))) | Some(Err(_)) => return Ok(()),
                                            Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {},
                                            _ => bail!("pipelining during voice turn is not supported"),
                                        }
                                    }
                                    output = &mut result => { output?; break; }
                                }
                            }
                        }
                        ClientMessage::Heartbeat {
                            node_id,
                            occupancy,
                            noise_level,
                            wake_word_fp,
                        } => {
                            ensure!(node_id == trusted.node_id, "heartbeat identity mismatch");
                            ensure!(
                                noise_level
                                    .is_none_or(|n| n.is_finite() && (0.0..=200.0).contains(&n)),
                                "invalid noise level"
                            );
                            let _ = (occupancy, wake_word_fp); // No cross-device presence authority.
                            send(&mut sender, json!({"type":"heartbeat_ack", "node_id":trusted.node_id, "mode":trusted.mode})).await?;
                        }
                        _ => {
                            send(&mut sender, json!({"type":"error", "message":"authentication cannot be replayed"})).await?;
                            break;
                        }
                    }
                }
                Message::Binary(bytes) => {
                    let (_, buffer) = audio.as_mut().context("audio before audio_start")?;
                    ensure!(
                        bytes.len() <= self.config.max_audio_bytes.saturating_sub(buffer.len()),
                        "audio limit exceeded"
                    );
                    buffer.extend_from_slice(&bytes);
                }
                Message::Close(_) => break,
                Message::Ping(data) => sender.send(Message::Pong(data)).await?,
                Message::Pong(_) => {}
                _ => bail!("unsupported websocket frame"),
            }
        }
        drop(guard);
        Ok(())
    }

    async fn process(
        &self,
        device: &Device,
        pcm: &[u8],
        format: AudioFormat,
        sender: &mut Sender,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let prompt = self
            .stt
            .transcribe(pcm, format, cancellation.child_token())
            .await?;
        ensure!(prompt.len() <= 64 * 1024, "transcript too long");
        if prompt.trim().is_empty() {
            return Ok(());
        }
        let trusted = self.registry.get(&device.node_id)?;
        ensure!(
            same_scope(&trusted, device) && trusted.paired,
            "scope changed"
        );
        if trusted.mode == PolicyMode::Muted {
            return send(sender, json!({"type":"muted"})).await;
        }
        if self.config.debug_transcripts {
            send(sender, json!({"type":"transcript", "text":prompt})).await?;
        }
        let response = self
            .agent
            .respond(&trusted, &prompt, cancellation.child_token())
            .await?;
        ensure!(response.len() <= 256 * 1024, "response too long");
        self.check_output_authority(&trusted)?;
        send(sender, json!({"type":"response_text", "text":response})).await?;
        let speech = self
            .tts
            .synthesize(&response, cancellation.child_token())
            .await?;
        ensure!(speech.wav.len() <= MAX_AUDIO, "synthesized audio too large");
        ensure!(
            (8_000..=48_000).contains(&speech.sample_rate),
            "invalid TTS sample rate"
        );
        ensure!(
            speech.wav.starts_with(b"RIFF") && speech.wav.get(8..12) == Some(b"WAVE"),
            "TTS must provide WAV"
        );
        self.check_output_authority(&trusted)?;
        send(
            sender,
            json!({"type":"audio_start", "sample_rate":speech.sample_rate, "format":"wav"}),
        )
        .await?;
        for chunk in speech.wav.chunks(self.config.audio_chunk_size) {
            self.check_output_authority(&trusted)?;
            sender.send(Message::Binary(chunk.to_vec().into())).await?;
        }
        send(sender, json!({"type":"audio_end"})).await
    }
    fn check_output_authority(&self, device: &Device) -> Result<()> {
        let current = self.registry.get(&device.node_id)?;
        ensure!(
            current.paired
                && same_scope(&current, device)
                && current.mode == device.mode
                && current.mode != PolicyMode::Muted,
            "device authority changed during turn"
        );
        Ok(())
    }
}

fn same_scope(a: &Device, b: &Device) -> bool {
    a.node_id == b.node_id && a.scope.is_some() && a.scope == b.scope
}
async fn send(sender: &mut Sender, value: Value) -> Result<()> {
    sender.send(Message::Text(value.to_string().into())).await?;
    Ok(())
}
async fn send_ws(ws: &mut Ws, value: Value) -> Result<()> {
    ws.send(Message::Text(value.to_string().into())).await?;
    Ok(())
}
struct ActiveDevice {
    set: Arc<Mutex<HashSet<String>>>,
    id: String,
}
impl ActiveDevice {
    fn claim(set: Arc<Mutex<HashSet<String>>>, id: &str) -> Result<Self> {
        ensure!(
            set.lock()
                .map_err(|_| anyhow::anyhow!("connection lock poisoned"))?
                .insert(id.into()),
            "device already connected"
        );
        Ok(Self { set, id: id.into() })
    }
}
impl Drop for ActiveDevice {
    fn drop(&mut self) {
        if let Ok(mut set) = self.set.lock() {
            set.remove(&self.id);
        }
    }
}
struct Rate {
    since: Instant,
    count: usize,
    window: Duration,
    limit: usize,
}
impl Rate {
    fn new(window: Duration, limit: usize) -> Self {
        Self {
            since: Instant::now(),
            count: 0,
            window,
            limit,
        }
    }
    fn check(&mut self) -> Result<()> {
        if self.since.elapsed() >= self.window {
            self.since = Instant::now();
            self.count = 0;
        }
        self.count += 1;
        ensure!(self.count <= self.limit, "voice rate limit exceeded");
        Ok(())
    }
}
