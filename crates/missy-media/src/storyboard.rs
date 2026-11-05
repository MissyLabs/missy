//! Scene generation is a real integration action, not images-only assembly.
use super::*;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneGenerationRequest {
    pub prompt: String,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
}
#[async_trait]
pub trait SceneGenerator: Send + Sync {
    async fn generate(
        &self,
        ctx: &ToolContext,
        request: &SceneGenerationRequest,
    ) -> Result<PathBuf>;
    fn configuration_fingerprint(&self) -> String;
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScenePrompt {
    prompt: String,
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenerateStoryboard {
    scenes: Vec<ScenePrompt>,
    output: PathBuf,
    seconds_per_image: u32,
    width: u32,
    height: u32,
}
impl GenerateStoryboard {
    pub(crate) fn validate_policy(&self, policy: &Policy) -> Result<()> {
        for path in self
            .scenes
            .iter()
            .map(|s| &s.output)
            .chain(std::iter::once(&self.output))
        {
            ensure!(
                path.is_absolute()
                    && !path
                        .as_os_str()
                        .as_encoded_bytes()
                        .split(|b| *b == b'/')
                        .any(|p| p == b"." || p == b".."),
                "invalid storyboard output path"
            );
            ensure!(
                policy
                    .config()
                    .write_roots
                    .iter()
                    .any(|root| path.starts_with(root) && path != root),
                "storyboard output outside write roots"
            );
            let parent = path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("storyboard output parent missing"))?;
            ensure!(
                parent.canonicalize()? == parent,
                "storyboard output parent must exist without symlinks"
            );
        }
        for scene in &self.scenes {
            ensure!(
                policy
                    .config()
                    .read_roots
                    .iter()
                    .any(|root| scene.output.starts_with(root) && scene.output != *root),
                "generated scene must be readable under media policy"
            );
        }
        Ok(())
    }
    pub fn validate(&self, c: &MediaConfig) -> Result<()> {
        ensure!(
            !self.scenes.is_empty() && self.scenes.len() <= 8,
            "requires 1..8 scenes"
        );
        let mut paths = std::collections::BTreeSet::new();
        paths.insert(self.output.clone());
        for scene in &self.scenes {
            ensure!(
                !scene.prompt.trim().is_empty() && scene.prompt.len() <= 4096,
                "scene prompt invalid"
            );
            ensure!(
                paths.insert(scene.output.clone()),
                "scene and video paths must be unique"
            );
        }
        self.video().validate(c)
    }
    fn video(&self) -> VideoRequest {
        VideoRequest {
            output: self.output.clone(),
            operation: VideoOperation::Storyboard {
                images: self.scenes.iter().map(|s| s.output.clone()).collect(),
                seconds_per_image: self.seconds_per_image,
                width: self.width,
                height: self.height,
            },
        }
    }
}
pub(crate) fn definition() -> ToolDefinition {
    super::definition(
        "video_generate_storyboard",
        "Generate prompted scenes through the operator-bound approved image integration, validate every decoded artifact, then assemble a fully decoded silent MP4. Partial external jobs are never implicitly retried.",
        object_schema(
            json!({"scenes":{"type":"array","minItems":1,"maxItems":8,"items":{"type":"object","additionalProperties":false,"properties":{"prompt":{"type":"string","maxLength":4096},"output":path_schema()},"required":["prompt","output"]}},"output":path_schema(),"seconds_per_image":{"type":"integer","minimum":1,"maximum":30},"width":{"type":"integer","minimum":2,"maximum":1920},"height":{"type":"integer","minimum":2,"maximum":1920}}),
            &["scenes", "output", "seconds_per_image", "width", "height"],
        ),
        true,
    )
}
pub(crate) async fn execute(
    tools: &MediaTools,
    ctx: &ToolContext,
    r: GenerateStoryboard,
) -> Result<Value> {
    r.validate(&tools.config)?;
    r.validate_policy(&tools.policy)?;
    let generator = tools
        .scene_generator
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("scene generation disabled"))?;
    let mut generated = Vec::new();
    for scene in &r.scenes {
        ensure!(
            !ctx.cancellation.is_cancelled(),
            "storyboard cancelled; generated scenes may already exist"
        );
        let request = SceneGenerationRequest {
            prompt: scene.prompt.clone(),
            output: scene.output.clone(),
            width: r.width,
            height: r.height,
        };
        let path = generator.generate(ctx, &request).await?;
        ensure!(
            path == scene.output,
            "generator returned unexpected artifact path"
        );
        let bytes = tools.policy.read_file(&path)?;
        let decoded = image_ops::decode(&bytes, &tools.config)?;
        ensure!(
            decoded.width() == r.width && decoded.height() == r.height,
            "generated scene dimensions mismatch"
        );
        generated.push(
            json!({"path":path,"sha256":hex::encode(Sha256::digest(&bytes)),"bytes":bytes.len()}),
        );
    }
    let video = tools.edit_video(ctx, r.video()).await?;
    Ok(json!({"scenes":generated,"video":video,"audio":false,"generated":true}))
}
