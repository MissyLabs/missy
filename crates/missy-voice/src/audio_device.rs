//! Explicit PulseAudio/PipeWire adapter. No ambient desktop server, default
//! device changes, or microphone use without trusted consent and exact approval.
use crate::{AudioFormat, SpeechAudio, SpeechOutput, Stt, Tts, speech::decode_pcm_wav};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::os::unix::process::CommandExt;
use std::{
    collections::BTreeMap, os::unix::fs::MetadataExt, path::PathBuf, process::Stdio, sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioExecutable {
    pub path: PathBuf,
    pub sha256: String,
}
impl AudioExecutable {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.path.is_absolute() && self.sha256.len() == 64,
            "audio executable requires absolute path and SHA256 pin"
        );
        let m = std::fs::metadata(&self.path)?;
        ensure!(
            m.is_file()
                && m.mode() & 0o111 != 0
                && m.mode() & 0o022 == 0
                && m.len() <= 64 * 1024 * 1024,
            "unsafe audio executable"
        );
        let bytes = std::fs::read(&self.path)?;
        ensure!(
            bytes.starts_with(b"\x7fELF") && format!("{:x}", Sha256::digest(&bytes)) == self.sha256,
            "native audio executable pin mismatch"
        );
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AudioDeviceKind {
    Virtual,
    Physical,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioDeviceConfig {
    pub enabled: bool,
    /// Explicit local Unix socket. No ambient PULSE_SERVER/DISPLAY/runtime.
    pub pulse_server: String,
    pub pactl: Option<AudioExecutable>,
    pub paplay: Option<AudioExecutable>,
    pub parec: Option<AudioExecutable>,
    pub sinks: BTreeMap<String, AudioDeviceKind>,
    pub sources: BTreeMap<String, AudioDeviceKind>,
    pub physical_output_consent: bool,
    pub physical_input_consent: bool,
    pub virtual_routes: Vec<String>,
    pub max_volume_pct: u8,
}
#[async_trait]
pub trait AudioApprovalGate: Send + Sync {
    /// MUST verify durable single-use approval for exact scope/turn/call/args.
    async fn approve(&self, ctx: &ToolContext, call: &ToolCall) -> Result<bool>;
}
struct Deny;
#[async_trait]
impl AudioApprovalGate for Deny {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> Result<bool> {
        Ok(false)
    }
}
pub struct AudioDeviceTools {
    cfg: AudioDeviceConfig,
    approval: Arc<dyn AudioApprovalGate>,
    tts: Option<Arc<dyn Tts>>,
    stt: Option<Arc<dyn Stt>>,
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Discover {
        direction: Direction,
    },
    TestRoute {
        sink: String,
    },
    CreateRoute {
        sink: String,
    },
    SetVolume {
        sink: String,
        volume_pct: u8,
    },
    Speak {
        sink: String,
        volume_pct: u8,
        text: String,
    },
    Listen {
        source: String,
        seconds: u8,
    },
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Direction {
    Input,
    Output,
}
fn identifier(s: &str) -> Result<()> {
    ensure!(
        !s.is_empty()
            && s.len() <= 128
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            && !s.starts_with('-'),
        "invalid audio device name"
    );
    Ok(())
}
impl AudioDeviceTools {
    pub fn new(cfg: AudioDeviceConfig) -> Result<Self> {
        if cfg.enabled {
            ensure!(
                cfg.pulse_server.starts_with("unix:/")
                    && !cfg.pulse_server.contains(['\0', '\n'])
                    && !cfg.pulse_server[5..].split('/').any(|s| s == ".."),
                "explicit absolute local Pulse Unix socket required"
            );
            cfg.pactl
                .as_ref()
                .context("pactl pin required")?
                .validate()?;
            for e in [&cfg.paplay, &cfg.parec].into_iter().flatten() {
                e.validate()?;
            }
            ensure!(
                (1..=80).contains(&cfg.max_volume_pct),
                "audio maximum volume must be 1..80"
            );
            for s in cfg
                .sinks
                .keys()
                .chain(cfg.sources.keys())
                .chain(cfg.virtual_routes.iter())
            {
                identifier(s)?;
            }
            for s in &cfg.virtual_routes {
                ensure!(
                    cfg.sinks.get(s) == Some(&AudioDeviceKind::Virtual),
                    "route must be granted as virtual sink"
                );
            }
        }
        Ok(Self {
            cfg,
            approval: Arc::new(Deny),
            tts: None,
            stt: None,
        })
    }
    pub fn with_approval(mut self, gate: Arc<dyn AudioApprovalGate>) -> Self {
        self.approval = gate;
        self
    }
    pub fn with_tts(mut self, tts: Arc<dyn Tts>) -> Self {
        self.tts = Some(tts);
        self
    }
    pub fn with_stt(mut self, stt: Arc<dyn Stt>) -> Self {
        self.stt = Some(stt);
        self
    }
    fn sink(&self, sink: &str) -> Result<()> {
        identifier(sink)?;
        let kind = self
            .cfg
            .sinks
            .get(sink)
            .context("output sink not granted")?;
        ensure!(
            *kind != AudioDeviceKind::Physical || self.cfg.physical_output_consent,
            "physical output consent required"
        );
        Ok(())
    }
    fn parse(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Request> {
        ctx.scope.validate()?;
        ensure!(
            self.cfg.enabled && call.name == "audio_device",
            "audio devices disabled"
        );
        ensure!(
            serde_json::to_vec(&call.arguments)?.len() <= 16_384,
            "audio arguments too large"
        );
        let r: Request = serde_json::from_value(call.arguments.clone())?;
        match &r {
            Request::Discover { .. } => {}
            Request::TestRoute { sink } => self.sink(sink)?,
            Request::CreateRoute { sink } => {
                self.sink(sink)?;
                ensure!(
                    self.cfg.virtual_routes.contains(sink),
                    "route creation denied"
                );
            }
            Request::SetVolume { sink, volume_pct } => {
                self.sink(sink)?;
                ensure!(*volume_pct <= self.cfg.max_volume_pct, "volume exceeds cap");
            }
            Request::Speak {
                sink,
                volume_pct,
                text,
            } => {
                self.validate_output(sink, *volume_pct)?;
                ensure!(
                    self.tts.is_some()
                        && !text.trim().is_empty()
                        && text.len() <= 8192
                        && !text.contains('\0'),
                    "invalid/unconfigured speech"
                );
            }
            Request::Listen { source, seconds } => {
                identifier(source)?;
                let kind = self
                    .cfg
                    .sources
                    .get(source)
                    .context("input source not granted")?;
                ensure!(
                    *kind != AudioDeviceKind::Physical || self.cfg.physical_input_consent,
                    "physical input consent required"
                );
                ensure!(
                    (1..=30).contains(seconds) && self.cfg.parec.is_some() && self.stt.is_some(),
                    "invalid/unconfigured listening"
                );
            }
        }
        Ok(r)
    }
    async fn command(
        &self,
        pin: &AudioExecutable,
        args: &[String],
        input: Option<&[u8]>,
        capture_for: Option<Duration>,
        cancel: CancellationToken,
        started: Option<tokio::sync::oneshot::Sender<tokio::time::Instant>>,
    ) -> Result<Vec<u8>> {
        pin.validate()?;
        ensure!(!cancel.is_cancelled(), "audio cancelled");
        let mut cmd = Command::new(&pin.path);
        cmd.args(args)
            .env_clear()
            .env("LC_ALL", "C")
            .env("PULSE_SERVER", &self.cfg.pulse_server)
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        cmd.as_std_mut().process_group(0);
        let mut child = cmd.spawn()?;
        if let Some(started) = started {
            let _ = started.send(tokio::time::Instant::now());
        }
        let _group = Group(child.id().context("audio child PID missing")?);
        let stdout = child.stdout.take().context("audio stdout missing")?;
        let reader = tokio::spawn(async move {
            let mut out = Vec::new();
            stdout
                .take(4 * 1024 * 1024 + 1)
                .read_to_end(&mut out)
                .await?;
            ensure!(out.len() <= 4 * 1024 * 1024, "audio output exceeds bound");
            Ok::<_, anyhow::Error>(out)
        });
        let operation = async {
            if let Some(input) = input {
                let mut stdin = child.stdin.take().context("audio stdin missing")?;
                stdin.write_all(input).await?;
                stdin.shutdown().await?;
            }
            if let Some(duration) = capture_for {
                tokio::select! {status=child.wait()=>{ensure!(status?.success(),"audio capture failed");},_=tokio::time::sleep(duration)=>{child.kill().await?;}}
            } else {
                ensure!(child.wait().await?.success(), "audio command failed");
            }
            Ok::<_, anyhow::Error>(())
        };
        let result = tokio::select! {biased;_=cancel.cancelled()=>Err(anyhow::anyhow!("audio cancelled")),r=tokio::time::timeout(Duration::from_secs(125),operation)=>r.context("audio command timeout").and_then(|v|v)};
        if let Err(e) = result {
            let _ = child.kill().await;
            reader.abort();
            return Err(e);
        }
        // A worker that exits after forking must not keep our pipe open via
        // its descendants. Kill only this child-owned process group first.
        unsafe {
            libc::kill(-(_group.0 as i32), libc::SIGKILL);
        }
        reader.await?
    }
    async fn pactl(&self, args: Vec<String>, cancel: CancellationToken) -> Result<Vec<u8>> {
        self.command(
            self.cfg.pactl.as_ref().context("pactl missing")?,
            &args,
            None,
            None,
            cancel,
            None,
        )
        .await
    }
    async fn devices(&self, direction: Direction, cancel: CancellationToken) -> Result<Vec<Value>> {
        let input = matches!(direction, Direction::Input);
        let bytes = self
            .pactl(
                vec![
                    "--format=json".into(),
                    "list".into(),
                    if input { "sources" } else { "sinks" }.into(),
                ],
                cancel,
            )
            .await?;
        let rows: Vec<Value> = serde_json::from_slice(&bytes)?;
        ensure!(rows.len() <= 1024, "audio device list too large");
        let grants = if input {
            &self.cfg.sources
        } else {
            &self.cfg.sinks
        };
        Ok(rows.into_iter().filter(|r|r["name"].as_str().is_some_and(|n|grants.contains_key(n))).map(|r|json!({"name":r["name"],"description":r["description"],"mute":r["mute"],"volume":r["volume"],"state":r["state"],"driver":r["driver"],"monitor_of_sink":r["monitor_of_sink"]})).collect())
    }
    async fn approved(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        ensure!(
            self.approval.approve(ctx, call).await?,
            "durable exact audio approval required"
        );
        Ok(())
    }
    async fn play_approved(
        &self,
        sink: &str,
        volume_pct: u8,
        audio: &SpeechAudio,
        cancel: CancellationToken,
        started: Option<tokio::sync::oneshot::Sender<tokio::time::Instant>>,
    ) -> Result<()> {
        self.validate_output(sink, volume_pct)?;
        let wav = decode_pcm_wav(&audio.wav)?;
        ensure!(
            wav.duration() <= Duration::from_secs(120) && wav.sample_rate == audio.sample_rate,
            "speech audio duration/rate invalid"
        );
        let volume = (volume_pct as u32 * 65536 / 100).to_string();
        self.command(
            self.cfg.paplay.as_ref().context("paplay missing")?,
            &[
                "--device".into(),
                sink.into(),
                "--volume".into(),
                volume,
                "--raw".into(),
                "--format=s16le".into(),
                format!("--rate={}", wav.sample_rate),
                format!("--channels={}", wav.channels),
            ],
            Some(wav.pcm),
            None,
            cancel,
            started,
        )
        .await?;
        Ok(())
    }
}
struct Group(u32);
impl Drop for Group {
    fn drop(&mut self) {
        unsafe {
            libc::kill(-(self.0 as i32), libc::SIGKILL);
        }
    }
}
#[async_trait]
impl SpeechOutput for AudioDeviceTools {
    fn configuration_fingerprint(&self) -> String {
        <Self as ToolExecutor>::configuration_fingerprint(self)
    }
    fn validate_output(&self, sink: &str, volume_pct: u8) -> Result<()> {
        ensure!(
            self.cfg.enabled && self.cfg.paplay.is_some(),
            "playback disabled"
        );
        self.sink(sink)?;
        ensure!(volume_pct <= self.cfg.max_volume_pct, "volume exceeds cap");
        Ok(())
    }
    async fn play(
        &self,
        ctx: &ToolContext,
        call: &ToolCall,
        sink: &str,
        volume_pct: u8,
        audio: &SpeechAudio,
        cancel: CancellationToken,
        started: tokio::sync::oneshot::Sender<tokio::time::Instant>,
    ) -> Result<()> {
        ctx.scope.validate()?;
        self.validate_output(sink, volume_pct)?;
        self.approved(ctx, call).await?;
        self.play_approved(sink, volume_pct, audio, cancel, Some(started))
            .await
    }
}
#[async_trait]
impl ToolExecutor for AudioDeviceTools {
    fn configuration_fingerprint(&self) -> String {
        format!(
            "audio-v1:{:x}:tts={}:stt={}",
            Sha256::digest(serde_json::to_vec(&self.cfg).unwrap_or_default()),
            self.tts.is_some(),
            self.stt.is_some()
        )
    }
    fn definitions(&self) -> Vec<ToolDefinition> {
        if !self.cfg.enabled {
            return vec![];
        }
        vec![ToolDefinition{name:"audio_device".into(),description:"Explicit approved audio metadata discovery, bounded microphone STT, local TTS playback, virtual route creation/test and capped sink volume. Never changes default audio or other clients.".into(),side_effect:true,parameters:json!({"type":"object","additionalProperties":false,"properties":{"action":{"enum":["discover","test_route","create_route","set_volume","speak","listen"]},"direction":{"enum":["input","output"]},"sink":{"type":"string"},"source":{"type":"string"},"volume_pct":{"type":"integer"},"seconds":{"type":"integer"},"text":{"type":"string"}},"required":["action"]})}]
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.parse(ctx, call).map(|_| ()).map_err(|e| {
            missy_core::ToolDispatchRefused {
                reason: e.to_string(),
            }
            .into()
        })
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        let req = self.parse(ctx, call)?;
        let cancel = ctx.cancellation.child_token();
        self.approved(ctx, call).await?;
        match req {
            Request::Discover { direction } => {
                Ok(json!({"devices":self.devices(direction,cancel).await?}))
            }
            Request::TestRoute { sink } => {
                let rows = self.devices(Direction::Output, cancel).await?;
                let row = rows
                    .into_iter()
                    .find(|r| r["name"] == sink)
                    .context("sink not present")?;
                Ok(json!({"sink":row,"played":false}))
            }
            Request::CreateRoute { sink } => {
                let rows = self
                    .devices(Direction::Output, cancel.child_token())
                    .await?;
                let existing = rows.iter().find(|r| r["name"] == sink);
                if let Some(row) = existing {
                    ensure!(
                        row["driver"]
                            .as_str()
                            .is_some_and(|s| s.contains("null-sink")),
                        "existing route is not a null sink; refusing volume mutation"
                    );
                }
                let exists = existing.is_some();
                if !exists {
                    self.pactl(
                        vec![
                            "load-module".into(),
                            "module-null-sink".into(),
                            format!("sink_name={sink}"),
                            format!("sink_properties=device.description={sink}"),
                        ],
                        cancel.child_token(),
                    )
                    .await?;
                }
                self.pactl(
                    vec![
                        "set-sink-volume".into(),
                        sink.clone(),
                        format!("{}%", self.cfg.max_volume_pct),
                    ],
                    cancel,
                )
                .await?;
                Ok(
                    json!({"sink":sink,"monitor_source":format!("{sink}.monitor"),"created":!exists,"default_changed":false}),
                )
            }
            Request::SetVolume { sink, volume_pct } => {
                self.pactl(
                    vec!["set-sink-volume".into(), sink, format!("{volume_pct}%")],
                    cancel,
                )
                .await?;
                Ok(json!({"volume_pct":volume_pct}))
            }
            Request::Speak {
                sink,
                volume_pct,
                text,
            } => {
                let audio = self
                    .tts
                    .as_ref()
                    .context("TTS missing")?
                    .synthesize(&text, cancel.child_token())
                    .await?;
                self.play_approved(&sink, volume_pct, &audio, cancel, None)
                    .await?;
                Ok(json!({"played":true,"sink":sink}))
            }
            Request::Listen { source, seconds } => {
                let bytes = self
                    .command(
                        self.cfg.parec.as_ref().context("parec missing")?,
                        &[
                            "--device".into(),
                            source.clone(),
                            "--raw".into(),
                            "--format=s16le".into(),
                            "--rate=16000".into(),
                            "--channels=1".into(),
                        ],
                        None,
                        Some(Duration::from_secs(seconds as u64)),
                        cancel.child_token(),
                        None,
                    )
                    .await?;
                ensure!(
                    !bytes.is_empty()
                        && bytes.len() <= seconds as usize * 32000 + 32000
                        && bytes.len().is_multiple_of(2),
                    "invalid captured PCM"
                );
                let text = self
                    .stt
                    .as_ref()
                    .context("STT missing")?
                    .transcribe(
                        &bytes,
                        AudioFormat {
                            sample_rate: 16000,
                            channels: 1,
                        },
                        cancel,
                    )
                    .await?;
                Ok(json!({"transcript":text,"source":source,"pcm_bytes":bytes.len()}))
            }
        }
    }
}
