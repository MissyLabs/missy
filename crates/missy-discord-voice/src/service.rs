use crate::{
    DiscordVoiceConfig, VoiceBinding, audio,
    protocol::{self, Dave, Heartbeat, MODE, RtpCipher, SILENCE, message},
};
use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use missy_policy::{PolicyHttpClient, PolicyWebSocket};
use missy_voice::{AudioFormat, Stt, Tts};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::Arc,
    time::Duration,
};
use tokio::{
    net::UdpSocket,
    sync::{Mutex, mpsc, oneshot},
    task::JoinHandle,
    time::{Instant, timeout},
};
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;
use zeroize::Zeroize;
#[cfg(test)]
#[path = "native_acceptance.rs"]
mod native_acceptance;
#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;

/// Produced only by authenticated Gateway events, never model arguments.
pub struct GatewayVoiceCredentials {
    pub guild_id: u64,
    pub channel_id: u64,
    pub user_id: u64,
    pub endpoint: String,
    pub session_id: String,
    pub token: String,
    pub session_cancel: CancellationToken,
}
impl fmt::Debug for GatewayVoiceCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatewayVoiceCredentials")
            .field("guild_id", &self.guild_id)
            .field("channel_id", &self.channel_id)
            .finish_non_exhaustive()
    }
}
impl Drop for GatewayVoiceCredentials {
    fn drop(&mut self) {
        self.token.zeroize();
        self.session_id.zeroize();
    }
}
#[async_trait]
pub trait GatewayVoiceAdapter: Send + Sync {
    async fn negotiate(
        &self,
        guild: u64,
        channel: u64,
        bot: u64,
    ) -> Result<GatewayVoiceCredentials>;
    async fn leave(&self, guild: u64, bot: u64) -> Result<()>;
}
#[async_trait]
pub trait DiscordVoiceAgent: Send + Sync {
    /// Parent selects a separately capability-restricted runtime by binding.mode.
    async fn respond(
        &self,
        binding: &VoiceBinding,
        prompt: &str,
        cancellation: CancellationToken,
    ) -> Result<String>;
}
pub struct DiscordVoiceService {
    config: Arc<DiscordVoiceConfig>,
    http: Arc<PolicyHttpClient>,
    gateway: Arc<dyn GatewayVoiceAdapter>,
    stt: Arc<dyn Stt>,
    tts: Arc<dyn Tts>,
    agent: Arc<dyn DiscordVoiceAgent>,
    sessions: Mutex<HashMap<u64, Session>>,
}
struct Session {
    channel: u64,
    text_output: mpsc::Sender<Vec<Vec<i16>>>,
    cancel: CancellationToken,
    task: JoinHandle<Result<()>>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceSessionStatus {
    ConnectingOrActive,
    Ended,
}
impl DiscordVoiceService {
    pub fn new(
        config: DiscordVoiceConfig,
        http: Arc<PolicyHttpClient>,
        gateway: Arc<dyn GatewayVoiceAdapter>,
        stt: Arc<dyn Stt>,
        tts: Arc<dyn Tts>,
        agent: Arc<dyn DiscordVoiceAgent>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: Arc::new(config),
            http,
            gateway,
            stt,
            tts,
            agent,
            sessions: Mutex::new(HashMap::new()),
        })
    }
    pub fn config(&self) -> &DiscordVoiceConfig {
        &self.config
    }
    pub fn join_approval(guild: u64, channel: u64) -> String {
        format!("JOIN DISCORD VOICE {guild}/{channel}")
    }
    pub fn leave_approval(guild: u64, channel: u64) -> String {
        format!("LEAVE DISCORD VOICE {guild}/{channel}")
    }
    pub async fn join(&self, guild: u64, channel: u64, actor: u64, approval: &str) -> Result<()> {
        self.join_cancellable(guild, channel, actor, approval, CancellationToken::new())
            .await
    }
    /// Cancels startup and active session without automatic rejoin or replay.
    pub async fn join_cancellable(
        &self,
        guild: u64,
        channel: u64,
        actor: u64,
        approval: &str,
        cancellation: CancellationToken,
    ) -> Result<()> {
        ensure!(!cancellation.is_cancelled(), "voice startup cancelled");
        ensure!(
            approval == Self::join_approval(guild, channel),
            "exact voice join approval required"
        );
        ensure!(
            self.config.binding(guild, channel, actor).is_some(),
            "actor lacks voice consent"
        );
        let mut sessions = tokio::select! {biased;_=cancellation.cancelled()=>bail!("voice startup cancelled"),s=self.sessions.lock()=>s};
        if let Some(s) = sessions.get(&guild) {
            ensure!(s.task.is_finished(), "guild has active voice");
        }
        if let Some(s) = sessions.remove(&guild) {
            s.cancel.cancel();
            let _ = s.task.await;
        }
        ensure!(sessions.len() < 16, "too many voice sessions");
        let negotiated = tokio::select! {biased;_=cancellation.cancelled()=>Err(anyhow::anyhow!("voice startup cancelled")),r=timeout(
            Duration::from_secs(45),
            self.gateway
                .negotiate(guild, channel, self.config.bot_user_id),
        )=>r.map_err(|_| anyhow::anyhow!("Gateway negotiation timeout")).and_then(|r|r)};
        let credentials = match negotiated {
            Ok(c) => c,
            Err(e) => {
                let _ = timeout(
                    Duration::from_secs(3),
                    self.gateway.leave(guild, self.config.bot_user_id),
                )
                .await;
                return Err(e);
            }
        };
        ensure!(
            credentials.guild_id == guild
                && credentials.channel_id == channel
                && credentials.user_id == self.config.bot_user_id,
            "voice credential scope mismatch"
        );
        let (config, http, gateway, stt, tts, agent) = (
            self.config.clone(),
            self.http.clone(),
            self.gateway.clone(),
            self.stt.clone(),
            self.tts.clone(),
            self.agent.clone(),
        );
        let cancel = cancellation.child_token();
        let child = cancel.clone();
        let (tx, rx) = oneshot::channel();
        let bot = self.config.bot_user_id;
        let (text_output, text_rx) = mpsc::channel(2);
        let task = tokio::spawn(async move {
            let result = run(
                config,
                http,
                credentials,
                stt,
                tts,
                agent,
                child,
                tx,
                text_rx,
            )
            .await;
            let _ = timeout(Duration::from_secs(3), gateway.leave(guild, bot)).await;
            result
        });
        sessions.insert(
            guild,
            Session {
                channel,
                text_output,
                cancel,
                task,
            },
        );
        let ready = tokio::select! {biased;_=cancellation.cancelled()=>false,r=timeout(Duration::from_secs(20),rx)=>matches!(r,Ok(Ok(())))};
        if ready {
            Ok(())
        } else {
            if let Some(s) = sessions.remove(&guild) {
                s.cancel.cancel();
                s.task.abort();
                let _ = s.task.await;
            }
            let _ = timeout(Duration::from_secs(3), self.gateway.leave(guild, bot)).await;
            bail!("encrypted voice negotiation failed")
        }
    }
    pub async fn leave(&self, guild: u64, channel: u64, actor: u64, approval: &str) -> Result<()> {
        ensure!(
            approval == Self::leave_approval(guild, channel),
            "exact voice leave approval required"
        );
        ensure!(
            self.config.binding(guild, channel, actor).is_some(),
            "actor lacks voice consent"
        );
        let mut sessions = self.sessions.lock().await;
        if let Some(s) = sessions.get(&guild) {
            ensure!(s.channel == channel, "channel mismatch");
        }
        if let Some(s) = sessions.remove(&guild) {
            s.cancel.cancel();
            let mut task = s.task;
            if timeout(Duration::from_secs(3), &mut task).await.is_err() {
                task.abort();
                let _ = task.await;
            }
        }
        timeout(
            Duration::from_secs(3),
            self.gateway.leave(guild, self.config.bot_user_id),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Gateway leave timeout"))?
    }
    pub async fn status(&self, guild: u64) -> Option<VoiceSessionStatus> {
        self.sessions.lock().await.get(&guild).map(|s| {
            if s.task.is_finished() {
                VoiceSessionStatus::Ended
            } else {
                VoiceSessionStatus::ConnectingOrActive
            }
        })
    }
    /// Queue only into an already human-approved joined session. Actor must
    /// come from authenticated events, never model JSON. Queued is not played.
    pub async fn speak_text(
        &self,
        guild: u64,
        channel: u64,
        actor: u64,
        text: &str,
        cancel: CancellationToken,
    ) -> Result<()> {
        ensure!(
            self.config.binding(guild, channel, actor).is_some(),
            "actor lacks voice consent"
        );
        ensure!(
            !text.trim().is_empty() && text.len() <= 32_768,
            "invalid voice text"
        );
        let (output, session_cancel) = {
            let sessions = self.sessions.lock().await;
            let s = sessions
                .get(&guild)
                .ok_or_else(|| anyhow::anyhow!("voice session not joined"))?;
            ensure!(
                s.channel == channel && !s.task.is_finished(),
                "voice session unavailable"
            );
            (s.text_output.clone(), s.cancel.clone())
        };
        tokio::select! {biased;_=cancel.cancelled()=>bail!("text speech cancelled"),_=session_cancel.cancelled()=>bail!("voice session ended"),r=timeout(Duration::from_secs(45),async{let audio=self.tts.synthesize(text,cancel.child_token()).await?;output.send(audio::wav_frames(&audio.wav,self.config.max_reply_seconds)?).await.map_err(|_|anyhow::anyhow!("voice output closed"))})=>r.map_err(|_|anyhow::anyhow!("text speech timed out"))?}
    }
    pub async fn shutdown(&self) {
        let sessions: Vec<_> = self.sessions.lock().await.drain().collect();
        for (_, s) in &sessions {
            s.cancel.cancel();
        }
        for (guild, s) in sessions {
            s.cancel.cancel();
            let mut task = s.task;
            if timeout(Duration::from_secs(3), &mut task).await.is_err() {
                task.abort();
                let _ = task.await;
            }
            let _ = timeout(
                Duration::from_secs(3),
                self.gateway.leave(guild, self.config.bot_user_id),
            )
            .await;
        }
    }
}
impl Drop for DiscordVoiceService {
    fn drop(&mut self) {
        if let Ok(mut sessions) = self.sessions.try_lock() {
            for (_, s) in sessions.drain() {
                s.cancel.cancel();
                s.task.abort();
            }
        }
    }
}
fn endpoint(config: &DiscordVoiceConfig, raw: &str) -> Result<String> {
    ensure!(
        !raw.contains('/') && !raw.contains('@') && !raw.contains('?') && !raw.contains('#'),
        "invalid voice endpoint"
    );
    let url = url::Url::parse(&format!("wss://{raw}/?v=8"))?;
    ensure!(
        url.host_str()
            .is_some_and(|h| config.endpoint_hosts.contains(h)),
        "voice endpoint denied"
    );
    Ok(url.to_string())
}
async fn resume_voice(
    http: &PolicyHttpClient,
    url: &str,
    c: &GatewayVoiceCredentials,
    seq: i64,
) -> Result<(PolicyWebSocket, Heartbeat)> {
    let mut ws = http.connect_websocket(url, vec![]).await?;
    send(&mut ws,message(7,json!({"server_id":c.guild_id.to_string(),"session_id":c.session_id,"token":c.token,"seq_ack":seq}))).await?;
    let mut heartbeat = Heartbeat::default();
    let start = Instant::now();
    loop {
        ensure!(
            start.elapsed() < Duration::from_secs(5),
            "voice resume timeout"
        );
        let frame = timeout(Duration::from_secs(5), ws.next())
            .await?
            .ok_or_else(|| anyhow::anyhow!("voice resume disconnected"))??;
        if let Message::Text(text) = frame {
            let v: Value =
                serde_json::from_str(&text).map_err(|_| anyhow::anyhow!("invalid resume JSON"))?;
            match v["op"].as_u64() {
                Some(8) => heartbeat.hello(
                    v["d"]["heartbeat_interval"]
                        .as_f64()
                        .filter(|n| n.is_finite() && *n >= 100.0)
                        .ok_or_else(|| anyhow::anyhow!("invalid resume Hello"))?
                        as u64,
                )?,
                Some(9) => {
                    ensure!(
                        heartbeat.interval_ms.is_some(),
                        "voice Resumed before Hello"
                    );
                    return Ok((ws, heartbeat));
                }
                Some(6) => heartbeat.ack(&v["d"])?,
                _ => bail!("unexpected voice resume event; full renegotiation required"),
            }
        } else if matches!(frame, Message::Close(_)) {
            bail!("voice resume refused");
        }
    }
}
/// Bounded transport-only retry, preserving RTP/DAVE counters and replay state.
/// Never re-identifies, starts another turn, or replays any synthesized speech.
async fn retry_resume(
    http: &PolicyHttpClient,
    url: &str,
    c: &GatewayVoiceCredentials,
    seq: i64,
    cancel: &CancellationToken,
) -> Result<(PolicyWebSocket, Heartbeat)> {
    for attempt in 0..3 {
        let operation = async {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(attempt * 200)).await;
            }
            resume_voice(http, url, c, seq).await
        };
        let result = tokio::select! {biased;_=cancel.cancelled()=>bail!("voice resume cancelled"),_=c.session_cancel.cancelled()=>bail!("Gateway voice session invalidated"),r=operation=>r};
        if result.is_ok() || attempt == 2 {
            return result;
        }
    }
    unreachable!("bounded resume attempts")
}
async fn send(ws: &mut PolicyWebSocket, msg: Message) -> Result<()> {
    timeout(Duration::from_secs(3), ws.send(msg))
        .await
        .map_err(|_| anyhow::anyhow!("voice WS timeout"))?
        .map_err(|_| anyhow::anyhow!("voice WS write failed"))
}
#[allow(clippy::too_many_arguments)]
async fn handle_json(
    ws: &mut PolicyWebSocket,
    text: &str,
    config: &DiscordVoiceConfig,
    credentials: &GatewayVoiceCredentials,
    heartbeat: &mut Heartbeat,
    udp: &mut Option<UdpSocket>,
    cipher: &mut Option<RtpCipher>,
    dave: &mut Option<Dave>,
    own_ssrc: &mut u32,
    senders: &mut HashMap<u32, u64>,
    roster: &mut std::collections::BTreeSet<u64>,
    collectors: &mut HashMap<u64, Collector>,
    seq: &mut i64,
) -> Result<()> {
    ensure!(text.len() <= 1_048_576, "signalling exceeds bound");
    let mut value: Value =
        serde_json::from_str(text).map_err(|_| anyhow::anyhow!("invalid voice JSON"))?;
    if let Some(s) = value["seq"].as_i64() {
        *seq = s;
    }
    let op = protocol::number(&value, "op")?;
    let data = &mut value["d"];
    match op {
        8 => heartbeat.hello(
            data["heartbeat_interval"]
                .as_f64()
                .filter(|f| f.is_finite() && *f >= 100.0)
                .ok_or_else(|| anyhow::anyhow!("invalid Hello"))? as u64,
        )?,
        6 => heartbeat.ack(data)?,
        2 => {
            ensure!(udp.is_none(), "duplicate Ready");
            *own_ssrc = u32::try_from(protocol::number(data, "ssrc")?)?;
            ensure!(
                data["modes"]
                    .as_array()
                    .is_some_and(|m| m.iter().any(|m| m == MODE)),
                "AEAD unavailable"
            );
            let address = config.udp_address(
                data["ip"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("missing UDP IP"))?,
                u16::try_from(protocol::number(data, "port")?)?,
            )?;
            let socket = UdpSocket::bind(if address.is_ipv4() {
                "0.0.0.0:0"
            } else {
                "[::]:0"
            })
            .await?;
            socket.connect(address).await?;
            socket.send(&protocol::discovery_request(*own_ssrc)).await?;
            let mut response = [0; 128];
            let len = timeout(Duration::from_secs(2), socket.recv(&mut response))
                .await
                .map_err(|_| anyhow::anyhow!("discovery timeout"))??;
            let (ip, port) = protocol::discovery_response(&response[..len], *own_ssrc)?;
            send(
                ws,
                message(
                    1,
                    json!({"protocol":"udp","data":{"address":ip,"port":port,"mode":MODE}}),
                ),
            )
            .await?;
            *udp = Some(socket);
        }
        4 => {
            ensure!(udp.is_some() && cipher.is_none(), "unexpected description");
            ensure!(data["mode"] == MODE, "transport downgrade denied");
            let mut session = Dave::new(
                &config.native_dave,
                u16::try_from(protocol::number(data, "dave_protocol_version")?)?,
                credentials.user_id,
                credentials.channel_id,
            )?;
            session.recognized.extend(roster.iter());
            let mut key: Vec<u8> = data["secret_key"]
                .as_array()
                .filter(|k| k.len() == 32)
                .ok_or_else(|| anyhow::anyhow!("invalid key"))?
                .iter()
                .map(|n| {
                    n.as_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| anyhow::anyhow!("invalid key byte"))
                })
                .collect::<Result<_>>()?;
            *cipher = Some(RtpCipher::new(&key, *own_ssrc)?);
            key.zeroize();
            data["secret_key"] = Value::Null;
            send(ws, session.key_package()?).await?;
            *dave = Some(session);
        }
        5 => {
            let user = protocol::id(&data["user_id"])?;
            let ssrc = u32::try_from(protocol::number(data, "ssrc")?)?;
            if collectors.contains_key(&user) {
                senders.retain(|_, u| *u != user);
                senders.insert(ssrc, user);
            }
        }
        11 => {
            if let Some(users) = data["user_ids"].as_array() {
                ensure!(
                    users.len() <= 1024 && roster.len() + users.len() <= 2048,
                    "roster too large"
                );
                for u in users {
                    roster.insert(protocol::id(u)?);
                }
            }
        }
        13 => {
            let user = protocol::id(&data["user_id"])?;
            senders.retain(|ssrc, u| {
                if *u == user {
                    if let Some(cipher) = cipher {
                        cipher.forget(*ssrc);
                    }
                    false
                } else {
                    true
                }
            });
            if let Some(c) = collectors.get_mut(&user) {
                c.pcm.clear();
            }
            roster.remove(&user);
        }
        _ => {}
    }
    if let Some(dave) = dave {
        for reply in dave.json(op, data)? {
            send(ws, reply).await?;
        }
    }
    Ok(())
}
struct Collector {
    decoder: opus2::Decoder,
    pcm: Vec<u8>,
    last: Instant,
    tx: mpsc::Sender<Vec<u8>>,
}
struct Workers {
    tasks: Vec<JoinHandle<()>>,
    cancel: CancellationToken,
}
impl Drop for Workers {
    fn drop(&mut self) {
        self.cancel.cancel();
        for t in &self.tasks {
            t.abort();
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn workers(
    config: &DiscordVoiceConfig,
    guild: u64,
    channel: u64,
    stt: Arc<dyn Stt>,
    tts: Arc<dyn Tts>,
    agent: Arc<dyn DiscordVoiceAgent>,
    cancel: CancellationToken,
    output: mpsc::Sender<Vec<Vec<i16>>>,
) -> Result<(HashMap<u64, Collector>, Workers)> {
    let mut collectors = HashMap::new();
    let mut tasks = Vec::new();
    let bindings: Vec<_> = config
        .bindings
        .iter()
        .filter(|b| b.guild_id == guild && b.channel_id == channel && b.consent)
        .cloned()
        .collect();
    ensure!(
        !bindings.is_empty() && bindings.len() <= config.max_users as usize,
        "voice user limit"
    );
    for binding in bindings {
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(2);
        collectors.insert(
            binding.user_id,
            Collector {
                decoder: opus2::Decoder::new(48_000, opus2::Channels::Stereo)?,
                pcm: Vec::new(),
                last: Instant::now(),
                tx,
            },
        );
        let (stt, tts, agent, output, cancel) = (
            stt.clone(),
            tts.clone(),
            agent.clone(),
            output.clone(),
            cancel.clone(),
        );
        let max_reply = config.max_reply_seconds;
        tasks.push(tokio::spawn(async move{loop{let pcm=tokio::select!{biased;_=cancel.cancelled()=>break,v=rx.recv()=>match v{Some(v)=>v,None=>break}};let turn=async{
            let prompt=stt.transcribe(&pcm,AudioFormat{sample_rate:48_000,channels:1},cancel.child_token()).await?;if prompt.trim().is_empty(){return Ok::<(),anyhow::Error>(());}ensure!(prompt.len()<=32_768,"transcript too large");let reply=agent.respond(&binding,&prompt,cancel.child_token()).await?;ensure!(reply.len()<=32_768,"reply too large");let speech=tts.synthesize(&reply,cancel.child_token()).await?;output.send(audio::wav_frames(&speech.wav,max_reply)?).await.map_err(|_|anyhow::anyhow!("output closed"))?;Ok(())
        };tokio::select!{biased;_=cancel.cancelled()=>break,_=timeout(Duration::from_secs(90),turn)=>{}}}}));
    }
    Ok((collectors, Workers { tasks, cancel }))
}
#[allow(clippy::too_many_arguments)]
async fn run(
    config: Arc<DiscordVoiceConfig>,
    http: Arc<PolicyHttpClient>,
    credentials: GatewayVoiceCredentials,
    stt: Arc<dyn Stt>,
    tts: Arc<dyn Tts>,
    agent: Arc<dyn DiscordVoiceAgent>,
    cancel: CancellationToken,
    ready: oneshot::Sender<()>,
    text_output: mpsc::Receiver<Vec<Vec<i16>>>,
) -> Result<()> {
    let url = endpoint(&config, &credentials.endpoint)?;
    let ws = tokio::select! {_=cancel.cancelled()=>return Ok(()),v=http.connect_websocket(&url,vec![])=>v?};
    run_connected(
        config,
        http,
        credentials,
        stt,
        tts,
        agent,
        cancel,
        ready,
        url,
        ws,
        Some(text_output),
    )
    .await
}

// There is deliberately no production URL override. The test adapter still goes
// through PolicyHttpClient, with a literal loopback URL and exact HTTP origin.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn run_at_loopback_url(
    config: Arc<DiscordVoiceConfig>,
    http: Arc<PolicyHttpClient>,
    credentials: GatewayVoiceCredentials,
    stt: Arc<dyn Stt>,
    tts: Arc<dyn Tts>,
    agent: Arc<dyn DiscordVoiceAgent>,
    cancel: CancellationToken,
    ready: oneshot::Sender<()>,
    url: String,
) -> Result<()> {
    let parsed = url::Url::parse(&url)?;
    ensure!(
        parsed.scheme() == "ws"
            && parsed
                .host_str()
                .and_then(|h| h.parse::<std::net::IpAddr>().ok())
                .is_some_and(|ip| ip.is_loopback()),
        "test URL must be literal loopback ws"
    );
    ensure!(
        parsed.username().is_empty()
            && parsed.password().is_none()
            && parsed.query().is_none()
            && parsed.fragment().is_none(),
        "invalid test URL metadata"
    );
    config.validate()?;
    let ws = http.connect_websocket(&url, vec![]).await?;
    run_connected(
        config,
        http,
        credentials,
        stt,
        tts,
        agent,
        cancel,
        ready,
        url,
        ws,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_connected(
    config: Arc<DiscordVoiceConfig>,
    http: Arc<PolicyHttpClient>,
    credentials: GatewayVoiceCredentials,
    stt: Arc<dyn Stt>,
    tts: Arc<dyn Tts>,
    agent: Arc<dyn DiscordVoiceAgent>,
    cancel: CancellationToken,
    ready: oneshot::Sender<()>,
    url: String,
    mut ws: PolicyWebSocket,
    mut text_output: Option<mpsc::Receiver<Vec<Vec<i16>>>>,
) -> Result<()> {
    send(&mut ws,message(0,json!({"server_id":credentials.guild_id.to_string(),"user_id":credentials.user_id.to_string(),"session_id":credentials.session_id,"token":credentials.token,"max_dave_protocol_version":missy_dave::DAVE_PROTOCOL_VERSION}))).await?;
    let mut heartbeat = Heartbeat::default();
    let mut udp: Option<UdpSocket> = None;
    let mut cipher: Option<RtpCipher> = None;
    let mut dave: Option<Dave> = None;
    let mut own_ssrc = 0;
    let mut senders: HashMap<u32, u64> = HashMap::new();
    let mut roster = std::collections::BTreeSet::new();
    let (out_tx, mut out_rx) = mpsc::channel(2);
    let (mut collectors, _workers) = workers(
        &config,
        credentials.guild_id,
        credentials.channel_id,
        stt,
        tts,
        agent,
        cancel.child_token(),
        out_tx,
    )?;
    let mut play: VecDeque<Vec<i16>> = VecDeque::new();
    let mut silence = 0u8;
    let mut speaking = false;
    let mut encoder =
        opus2::Encoder::new(48_000, opus2::Channels::Stereo, opus2::Application::Voip)?;
    let mut tick = tokio::time::interval(Duration::from_millis(20));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let started = Instant::now();
    let mut ready = Some(ready);
    let mut seq = -1i64;
    let mut datagram = [0u8; 4096];
    let mut rate_start = Instant::now();
    let mut received = 0u32;
    loop {
        tokio::select! {biased;
            _=credentials.session_cancel.cancelled()=>bail!("Gateway voice session invalidated"),
            _=cancel.cancelled()=>{let _=send(&mut ws,message(5,json!({"speaking":0,"delay":0,"ssrc":own_ssrc}))).await;let _=ws.close(None).await;return Ok(());},
            _=tick.tick()=>{
                if ready.is_some()&&started.elapsed()>Duration::from_secs(18){bail!("encrypted negotiation timeout");}
                if let Some(msg)=heartbeat.tick(seq)?{send(&mut ws,msg).await?;}
                for c in collectors.values_mut(){if !c.pcm.is_empty()&&c.last.elapsed()>=Duration::from_millis(config.silence_ms as u64){let pcm=std::mem::take(&mut c.pcm);if pcm.len()>=28_800{let _=c.tx.try_send(pcm);}}}
                if let(Some(udp),Some(cipher),Some(dave))=(&udp,&mut cipher,&mut dave) && dave.session.is_ready(){if let Some(ready)=ready.take(){let _=ready.send(());}let packet=if let Some(pcm)=play.pop_front(){silence=5;Some(encoder.encode_vec(&pcm,4000)?)}else if silence>0{silence-=1;Some(SILENCE.to_vec())}else{None};
                    if let Some(packet)=packet{if !speaking{send(&mut ws,message(5,json!({"speaking":1,"delay":0,"ssrc":own_ssrc}))).await?;speaking=true;}let encrypted=dave.session.encrypt_opus(own_ssrc,&packet).map_err(|_|anyhow::anyhow!("DAVE encryption failed"))?;udp.send(&cipher.encode(&encrypted)?).await?;}else if speaking{send(&mut ws,message(5,json!({"speaking":0,"delay":0,"ssrc":own_ssrc}))).await?;speaking=false;}
                }
            },
            Some(frames)=out_rx.recv(),if play.is_empty()=>{play=frames.into();},
            Some(frames)=async{match &mut text_output{Some(rx)=>rx.recv().await,None=>std::future::pending().await}},if play.is_empty()=>{play=frames.into();},
            received_frame=async{match &udp{Some(s)=>s.recv(&mut datagram).await,None=>std::future::pending().await}}=>{
                let len=received_frame?;if rate_start.elapsed()>=Duration::from_secs(1){rate_start=Instant::now();received=0;}received+=1;if received>config.max_users as u32*100||len<12{continue;}let ssrc=u32::from_be_bytes(datagram[8..12].try_into()?);let Some(user)=senders.get(&ssrc).copied()else{continue;};let Some(c)=collectors.get_mut(&user)else{continue;};let(Some(cipher),Some(dave))=(&mut cipher,&mut dave)else{continue;};if !dave.session.is_ready(){continue;}
                let Ok((_,transport))=cipher.decode(&datagram[..len])else{continue;};if transport==SILENCE{continue;}let Ok(opus)=dave.session.decrypt_opus(user,&transport)else{continue;};let mut pcm=[0i16;11_520];let Ok(samples)=c.decoder.decode(&opus,&mut pcm,false)else{continue;};
                let active=audio::speech_energy(&pcm[..samples*2]);
                if active {c.last=Instant::now();} else if c.pcm.is_empty() {continue;}
                let mono=audio::mono(&pcm[..samples*2]);let max=config.max_utterance_seconds as usize*96_000;if c.pcm.len()+mono.len()>max{c.pcm.clear();continue;}c.pcm.extend_from_slice(&mono);if c.pcm.len()>=max{let pcm=std::mem::take(&mut c.pcm);let _=c.tx.try_send(pcm);}
            },
            frame=ws.next()=>{
                let frame=match frame{
                    Some(Ok(frame))=>frame,
                    _=>{ensure!(dave.is_some(),"voice disconnected during negotiation");let(next,hb)=retry_resume(&http,&url,&credentials,seq,&cancel).await?;ws=next;heartbeat=hb;continue;}
                };
                match frame{
                    Message::Text(text)=>{handle_json(&mut ws,&text,&config,&credentials,&mut heartbeat,&mut udp,&mut cipher,&mut dave,&mut own_ssrc,&mut senders,&mut roster,&mut collectors,&mut seq).await?;},
                    Message::Binary(bytes)=>{let session=dave.as_mut().ok_or_else(||anyhow::anyhow!("DAVE before description"))?;for reply in session.binary(&bytes)?{send(&mut ws,reply).await?;}seq=session.sequence;},Message::Ping(bytes)=>send(&mut ws,Message::Pong(bytes)).await?,Message::Close(close)=>{
                        let code=close.map(|c|u16::from(c.code)).unwrap_or(1006);
                        ensure!(!matches!(code,4004|4006|4009|4011|4012|4014|4016|4017),"voice closed; fresh negotiation required");
                        let(next,hb)=retry_resume(&http,&url,&credentials,seq,&cancel).await?;ws=next;heartbeat=hb;
                    },_=>{}
                }
            }
        }
    }
}
