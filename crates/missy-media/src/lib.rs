//! Native file-only editing. No devices, playback, shell, or Python bridge.
//! Embedders MUST use the runtime's durable approval/journal gate.
#![cfg(target_os = "linux")]
mod config;
mod image_ops;
mod process;
mod storyboard;
mod validation;
mod video;
pub use config::{ExecutablePin, MediaConfig};
pub use image_ops::{ImageOperation, ImageRequest, Rotation};
pub use storyboard::{SceneGenerationRequest, SceneGenerator};
pub use video::{
    CaptionColor, MediaMetadata, StreamMetadata, TextPosition, VideoOperation, VideoRequest,
};

use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use missy_core::{PolicyConfig, ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use missy_policy::Policy;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc};

pub struct MediaTools {
    policy: Policy,
    config: Arc<MediaConfig>,
    scene_generator: Option<Arc<dyn SceneGenerator>>,
}
impl MediaTools {
    pub fn new(policy: PolicyConfig, config: MediaConfig) -> Result<Self> {
        config.validate()?;
        ensure!(
            missy_sandbox::landlock_abi()? >= 4,
            "media requires Landlock ABI >= 4; no fallback"
        );
        Ok(Self {
            policy: Policy::new(policy)?,
            config: Arc::new(config),
            scene_generator: None,
        })
    }
    /// Generator must dispatch through durable integration approval/journaling.
    pub fn with_scene_generator(mut self, generator: Arc<dyn SceneGenerator>) -> Self {
        self.scene_generator = Some(generator);
        self
    }
    /// Typed API. Caller remains responsible for approval/journaling.
    pub async fn edit_video(&self, ctx: &ToolContext, request: VideoRequest) -> Result<Value> {
        request.validate(&self.config)?;
        video::execute(&self.policy, &self.config, ctx, request).await
    }
    /// Pinned sandboxed FFprobe plus full FFmpeg decode. Never publishes/plays.
    pub async fn validate_mp4(&self, ctx: &ToolContext, bytes: &[u8]) -> Result<MediaMetadata> {
        validation::validate_mp4(&self.config, ctx, bytes).await
    }
    pub fn edit_image(&self, request: ImageRequest) -> Result<Value> {
        image_ops::execute(&self.policy, &self.config, request)
    }
    fn parse(&self, call: &ToolCall) -> Result<Request> {
        match call.name.as_str() {
            "video_edit" => {
                let r: VideoRequest = serde_json::from_value(call.arguments.clone())?;
                r.validate(&self.config)?;
                Ok(Request::Video(r))
            }
            "video_generate_storyboard" | "video_storyboard"
                if self.scene_generator.is_some() && call.arguments.get("scenes").is_some() =>
            {
                let r: storyboard::GenerateStoryboard =
                    serde_json::from_value(call.arguments.clone())?;
                r.validate(&self.config)?;
                r.validate_policy(&self.policy)?;
                Ok(Request::Generate(r))
            }
            "video_storyboard" => {
                let r: StoryboardRequest = serde_json::from_value(call.arguments.clone())?;
                let r = VideoRequest {
                    output: r.output,
                    operation: VideoOperation::Storyboard {
                        images: r.images,
                        seconds_per_image: r.seconds_per_image,
                        width: r.width,
                        height: r.height,
                    },
                };
                r.validate(&self.config)?;
                Ok(Request::Video(r))
            }
            "vision_capture" => {
                let r: CaptureRequest = serde_json::from_value(call.arguments.clone())?;
                let r = ImageRequest {
                    input: r.input,
                    output: r.output,
                    operation: ImageOperation::Copy,
                };
                r.validate(&self.config)?;
                Ok(Request::Image(r))
            }
            "image_edit" => {
                let r: ImageRequest = serde_json::from_value(call.arguments.clone())?;
                r.validate(&self.config)?;
                Ok(Request::Image(r))
            }
            "vision_devices" => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Empty {}
                let _: Empty = serde_json::from_value(call.arguments.clone())?;
                Ok(Request::Devices)
            }
            _ => bail!("unknown media tool"),
        }
    }
}
enum Request {
    Generate(storyboard::GenerateStoryboard),
    Video(VideoRequest),
    Image(ImageRequest),
    Devices,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureRequest {
    input: PathBuf,
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StoryboardRequest {
    images: Vec<PathBuf>,
    output: PathBuf,
    seconds_per_image: u32,
    width: u32,
    height: u32,
}

fn definition(
    name: &str,
    description: &str,
    parameters: Value,
    side_effect: bool,
) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters,
        side_effect,
    }
}
fn path_schema() -> Value {
    json!({"type":"string","description":"Absolute file path within policy roots"})
}
fn object_schema(properties: Value, required: &[&str]) -> Value {
    json!({"type":"object","additionalProperties":false,"properties":properties,"required":required})
}
#[async_trait]
impl ToolExecutor for MediaTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = vec![
            definition(
                "video_edit",
                "Native bounded file-only FFmpeg editing. Existing edits stay silent; audio_mux/audio_mix explicitly add AAC, speed/text/crossfade require preserve_audio. Text requires a policy-readable font_file. No arbitrary filters or URLs.",
                video::schema(),
                true,
            ),
            definition(
                "video_storyboard",
                "Assemble existing PNG/JPEG files, including separately approved Comfy artifacts, into silent MP4. No image generation or external calls.",
                object_schema(
                    json!({"images":{"type":"array","items":path_schema(),"minItems":1,"maxItems":8},"output":path_schema(),"seconds_per_image":{"type":"integer","minimum":1,"maximum":30},"width":{"type":"integer","minimum":2,"maximum":1920},"height":{"type":"integer","minimum":2,"maximum":1920}}),
                    &["images", "output", "seconds_per_image", "width", "height"],
                ),
                true,
            ),
            definition(
                "image_edit",
                "Bounded native PNG/JPEG copy, resize, crop, rotate or overlay. Existing pixels only.",
                image_ops::schema(),
                true,
            ),
            definition(
                "vision_capture",
                "Decode/copy an explicitly supplied image FILE to PNG/JPEG. Never opens a camera, microphone, screen or device.",
                object_schema(
                    json!({"input":path_schema(),"output":path_schema()}),
                    &["input", "output"],
                ),
                true,
            ),
            definition(
                "vision_devices",
                "Read-only sysfs camera discovery; never opens /dev or captures.",
                object_schema(json!({}), &[]),
                false,
            ),
        ];
        if self.scene_generator.is_some() {
            let generated = storyboard::definition();
            if let Some(existing) = definitions
                .iter_mut()
                .find(|d| d.name == "video_storyboard")
            {
                existing.description="Assemble supplied images OR generate prompted scenes through the approved integration then assemble a fully decoded MP4. Scene generation is real billable external work, not an images-only substitution.".into();
                existing.parameters = json!({"oneOf":[existing.parameters,generated.parameters]});
            }
            definitions.push(generated);
        }
        definitions
    }
    fn configuration_fingerprint(&self) -> String {
        let bytes = serde_json::to_vec(&(
            self.policy.config(),
            self.config.as_ref(),
            self.definitions(),
            self.scene_generator
                .as_ref()
                .map(|g| g.configuration_fingerprint()),
        ))
        .expect("serializable config");
        hex::encode(Sha256::digest(bytes))
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        ensure!(!ctx.cancellation.is_cancelled(), "media cancelled");
        self.parse(call)?;
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        ensure!(!ctx.cancellation.is_cancelled(), "media cancelled");
        match self.parse(call)? {
            Request::Generate(r) => storyboard::execute(self, ctx, r).await,
            Request::Video(r) => self.edit_video(ctx, r).await,
            Request::Image(r) => self.edit_image(r),
            Request::Devices => discover_cameras(),
        }
    }
}
/// Fixed sysfs path, READ ONLY. Expected kernel symlinks, no model-supplied path.
pub fn discover_cameras() -> Result<Value> {
    use std::io::Read;
    let entries = match std::fs::read_dir("/sys/class/video4linux") {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(json!({"devices":[],"capture":false}));
        }
        Err(e) => return Err(e.into()),
    };
    let mut devices = Vec::new();
    for entry in entries.take(64) {
        let entry = entry?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !id
            .strip_prefix("video")
            .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
        {
            continue;
        }
        let mut name = String::new();
        std::fs::File::open(entry.path().join("name"))?
            .take(256)
            .read_to_string(&mut name)?;
        devices.push(json!({"id":id,"name":name.trim()}));
    }
    Ok(json!({"devices":devices,"capture":false}))
}
#[derive(Debug, Serialize)]
pub struct Artifact {
    pub path: PathBuf,
    pub bytes: usize,
    pub sha256: String,
    pub media: MediaMetadata,
}
pub(crate) fn publish(
    policy: &Policy,
    output: PathBuf,
    bytes: &[u8],
    media: MediaMetadata,
) -> Result<Value> {
    // Descriptor-relative O_NOFOLLOW and atomic mode-0600 replacement. FFmpeg
    // never receives the actual destination or access to policy write roots.
    policy.write_file(&output, bytes)?;
    Ok(serde_json::to_value(Artifact {
        path: output,
        bytes: bytes.len(),
        sha256: hex::encode(Sha256::digest(bytes)),
        media,
    })?)
}
pub(crate) fn extension(path: &std::path::Path) -> Result<&str> {
    ensure!(path.is_absolute(), "media paths must be absolute");
    path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("media extension required"))
}
#[cfg(test)]
mod tests;
