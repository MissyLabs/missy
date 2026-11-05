//! Explicit, approval-gated native live capture. No ambient device/display/audio
//! discovery, no shells and no production synthetic-source escape hatch.
#![cfg(target_os = "linux")]
mod capture;
mod config;
mod process;
pub use config::{CaptureConfig, ExecutablePin, TrustedX11};

use anyhow::{Result, ensure};
use async_trait::async_trait;
use missy_core::{
    PolicyConfig, ToolCall, ToolContext, ToolDefinition, ToolDispatchRefused, ToolExecutor,
};
use missy_policy::Policy;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{os::unix::fs::MetadataExt, path::PathBuf, sync::Arc};

/// The embedding application's durable human approval gate. Must authorize this
/// exact call AND scope/turn. A model-supplied consent=true is necessary but is
/// never sufficient. There is deliberately no built-in permissive implementation.
#[async_trait]
pub trait CaptureApproval: Send + Sync {
    async fn authorize(&self, context: &ToolContext, call: &ToolCall) -> Result<()>;
    fn configuration_fingerprint(&self) -> String;
}
pub struct CaptureTools {
    policy: Policy,
    config: CaptureConfig,
    approval: Arc<dyn CaptureApproval>,
    slots: tokio::sync::Semaphore,
}
impl CaptureTools {
    /// Validates configuration without opening cameras or connecting to X11.
    pub fn new(
        policy: PolicyConfig,
        config: CaptureConfig,
        approval: Arc<dyn CaptureApproval>,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.max_output_bytes <= policy.max_file_bytes,
            "capture exceeds policy file budget"
        );
        Ok(Self {
            policy: Policy::new(policy)?,
            config,
            approval,
            slots: tokio::sync::Semaphore::new(1),
        })
    }
    fn parse(&self, call: &ToolCall) -> Result<Request> {
        let r: Request = serde_json::from_value(call.arguments.clone())?;
        r.validate(&self.config, &call.name)?;
        ensure!(
            self.definitions().iter().any(|d| d.name == call.name),
            "capture disabled or unknown"
        );
        self.output_snapshot(&r)?;
        Ok(r)
    }
    fn output_snapshot(&self, request: &Request) -> Result<Vec<(u64, u64, u64)>> {
        let p = &request.output;
        config::safe_path(p)?;
        let root = self
            .policy
            .config()
            .write_roots
            .iter()
            .filter(|r| p.starts_with(r))
            .max_by_key(|r| r.components().count())
            .ok_or_else(|| anyhow::anyhow!("output outside policy write roots"))?;
        config::canonical(root)?;
        let parent = p
            .parent()
            .ok_or_else(|| anyhow::anyhow!("output parent missing"))?;
        config::canonical(parent)?;
        let mut stamps = Vec::new();
        let mut path = PathBuf::from("/");
        for c in parent.components().skip(1) {
            path.push(c);
            let m = std::fs::symlink_metadata(&path)?;
            stamps.push((m.dev(), m.ino(), m.mode() as u64));
        }
        match std::fs::symlink_metadata(p) {
            Ok(m) => {
                ensure!(
                    m.is_file()
                        && !m.file_type().is_symlink()
                        && m.nlink() == 1
                        && m.uid() == unsafe { libc::geteuid() },
                    "output must be a regular nonsymlink file"
                );
                stamps.push((m.dev(), m.ino(), m.len()));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => stamps.push((0, 0, 0)),
            Err(e) => return Err(e.into()),
        }
        Ok(stamps)
    }
    fn snapshot(&self, r: &Request, name: &str) -> Result<Vec<u8>> {
        self.config.ffmpeg.open()?;
        self.config.ffprobe.open()?;
        config::private_directory(&self.config.staging_root)?;
        let output = self.output_snapshot(r)?;
        let source = if name == "screencast" {
            let x = self.config.x11.as_ref().expect("validated X11");
            x.validate_live()?;
            let m = std::fs::symlink_metadata(x.socket()?)?;
            json!({"socket":[m.dev(),m.ino(),m.uid() as u64],"cookie":hex::encode(Sha256::digest(x.cookie()?))})
        } else {
            let p = r.device.as_ref().expect("validated camera");
            config::camera_live(p)?;
            let m = std::fs::symlink_metadata(p)?;
            json!({"device":[m.dev(),m.ino(),m.rdev(),m.uid() as u64,m.mode() as u64]})
        };
        Ok(serde_json::to_vec(&(output, source))?)
    }
    /// Same exact ToolExecutor gate; no typed bypass around human approval.
    pub async fn capture_run(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.execute(ctx, call).await
    }
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    output: PathBuf,
    #[serde(default)]
    device: Option<PathBuf>,
    consent: bool,
    #[serde(default)]
    audio: bool,
    #[serde(default)]
    audio_consent: bool,
    #[serde(default = "one")]
    duration_seconds: u32,
    #[serde(default = "width")]
    width: u32,
    #[serde(default = "height")]
    height: u32,
    #[serde(default = "rate")]
    fps: u32,
}
fn one() -> u32 {
    1
}
fn width() -> u32 {
    640
}
fn height() -> u32 {
    480
}
fn rate() -> u32 {
    15
}
impl Request {
    fn validate(&self, c: &CaptureConfig, name: &str) -> Result<()> {
        ensure!(
            self.consent,
            "explicit caller consent is required in addition to durable human approval"
        );
        ensure!(
            !self.audio && !self.audio_consent,
            "audio capture is disabled; no microphone/system audio backend exists"
        );
        ensure!(
            (1..=90).contains(&self.duration_seconds),
            "duration must be 1..90 seconds"
        );
        ensure!(
            (1..=c.max_fps).contains(&self.fps),
            "fps outside configured bounds"
        );
        ensure!(
            (2..=1920).contains(&self.width)
                && (2..=1080).contains(&self.height)
                && self.width.is_multiple_of(2)
                && self.height.is_multiple_of(2)
                && self.width * self.height <= c.max_pixels,
            "dimensions outside configured bounds or not even"
        );
        config::safe_path(&self.output)?;
        let photo = name == "camera_photo";
        ensure!(
            self.output.extension().and_then(|s| s.to_str())
                == Some(if photo { "png" } else { "mp4" }),
            "output extension must match capture format"
        );
        if photo {
            ensure!(
                self.duration_seconds == 1,
                "photos capture exactly one frame"
            );
        }
        if name == "screencast" {
            ensure!(self.device.is_none(), "screencast takes no camera device");
        } else {
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("exact camera device required"))?;
            config::camera_path(device)?;
            ensure!(c.cameras.contains(device), "camera not allowlisted");
        }
        Ok(())
    }
}
fn refused(e: anyhow::Error) -> anyhow::Error {
    ToolDispatchRefused {
        reason: e.to_string(),
    }
    .into()
}
#[async_trait]
impl ToolExecutor for CaptureTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut out = Vec::new();
        for (enabled, name, description) in [
            (
                self.config.enable_camera,
                "camera_photo",
                "Capture exactly one silent PNG frame from an explicit allowlisted V4L2 camera. Requires exact durable human approval.",
            ),
            (
                self.config.enable_camera,
                "camera_clip",
                "Capture a bounded silent MP4 from an explicit allowlisted V4L2 camera. Requires exact durable human approval.",
            ),
            (
                self.config.enable_screencast,
                "screencast",
                "Capture a bounded silent MP4 of the explicitly configured trusted isolated X11 session. Never the ambient desktop. Requires exact durable human approval.",
            ),
        ] {
            if !enabled {
                continue;
            }
            let mut properties = json!({"output":{"type":"string"},"consent":{"type":"boolean","const":true},"audio":{"type":"boolean","const":false,"default":false},"audio_consent":{"type":"boolean","const":false,"default":false},"duration_seconds":{"type":"integer","minimum":1,"maximum":if name=="camera_photo" {1} else {90},"default":1},"fps":{"type":"integer","minimum":1,"maximum":self.config.max_fps,"default":15},"width":{"type":"integer","minimum":2,"maximum":1920,"multipleOf":2,"default":640},"height":{"type":"integer","minimum":2,"maximum":1080,"multipleOf":2,"default":480}});
            let mut required = vec!["output", "consent"];
            if name != "screencast" {
                properties["device"] = json!({"type":"string","enum":self.config.cameras});
                required.push("device");
            }
            out.push(ToolDefinition {name:name.into(),description:description.into(),parameters:json!({"type":"object","additionalProperties":false,"properties":properties,"required":required}),side_effect:true});
        }
        out
    }
    fn configuration_fingerprint(&self) -> String {
        hex::encode(Sha256::digest(
            serde_json::to_vec(&(
                self.policy.config(),
                &self.config,
                self.approval.configuration_fingerprint(),
                self.definitions(),
            ))
            .expect("serializable capture config"),
        ))
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        ensure!(
            !ctx.cancellation.is_cancelled(),
            "capture cancelled before dispatch"
        );
        let r = self.parse(call)?;
        self.snapshot(&r, &call.name)?;
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        let _slot = tokio::select! { biased; _ = ctx.cancellation.cancelled() => return Err(refused(anyhow::anyhow!("capture cancelled before dispatch"))), s = self.slots.acquire() => s? };
        let r = self.parse(call).map_err(refused)?;
        let before = self.snapshot(&r, &call.name).map_err(refused)?;
        tokio::select! {
            biased;
            _ = ctx.cancellation.cancelled() => return Err(refused(anyhow::anyhow!("capture cancelled while awaiting approval"))),
            result = self.approval.authorize(ctx,call) => result?,
        }
        let after = self.snapshot(&r, &call.name).map_err(refused)?;
        if before != after || ctx.cancellation.is_cancelled() {
            return Err(refused(anyhow::anyhow!(
                "capture inputs/output changed or cancelled after approval"
            )));
        }
        // Any error from here is intentionally NOT ToolDispatchRefused. Capture
        // may already have occurred. Runtime must journal UNKNOWN, never retry.
        capture::execute(&self.policy, &self.config, ctx, &r, &call.name, None).await
    }
}

#[cfg(test)]
mod tests;
