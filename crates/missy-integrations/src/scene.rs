//! Trusted host composition for already-approved storyboard actions. This is
//! not a tool and must not be exposed without the parent media approval gate.
use crate::{
    IntegrationTools,
    comfy::{Download, History, Submit},
};
use anyhow::{Context, Result, ensure};
use missy_core::ToolContext;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Duration};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneGenerationGrant {
    pub workflow: String,
    pub output_node: String,
    pub width: u32,
    pub height: u32,
}
pub struct SceneImageRequest {
    pub prompt: String,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
}
impl IntegrationTools {
    /// Invoke ONLY after durable approval of the exact parent storyboard call.
    /// The grant is operator-owned; never derive it from model arguments.
    /// One submit, bounded status polling, scope-owned exact image descriptor.
    /// Cancellation leaves the external job uncertain, never replayed/cancelled
    /// globally (Comfy's interrupt endpoint can affect another user's job).
    pub async fn approved_scene(
        &self,
        ctx: &ToolContext,
        grant: &SceneGenerationGrant,
        request: SceneImageRequest,
    ) -> Result<PathBuf> {
        ctx.scope.validate()?;
        let comfy = self
            .comfy
            .as_ref()
            .context("ComfyUI scene integration disabled")?;
        ensure!(comfy.conn.cfg.allow_mutations, "ComfyUI mutations disabled");
        let workflow = comfy
            .cfg
            .workflows
            .get(&grant.workflow)
            .context("scene workflow not granted")?;
        ensure!(
            workflow.prompt_node.is_some()
                && workflow.graph[&grant.output_node]["class_type"].as_str() == Some("SaveImage"),
            "scene workflow needs prompt binding and SaveImage output node"
        );
        ensure!(
            grant.width >= 2
                && grant.height >= 2
                && grant.width <= 1920
                && grant.height <= 1920
                && request.width == grant.width
                && request.height == grant.height,
            "scene dimensions outside grant"
        );
        ensure!(
            !request.prompt.trim().is_empty()
                && request.prompt.len() <= 4096
                && request.output.is_absolute(),
            "invalid scene request"
        );
        let path = request
            .output
            .to_str()
            .context("scene output must be UTF-8")?
            .to_owned();
        let operation = async {
            let submitted = comfy
                .submit(
                    &ctx.scope,
                    Submit {
                        workflow: grant.workflow.clone(),
                        prompt: Some(request.prompt),
                        conditioning: None,
                    },
                )
                .await?;
            let id = submitted["prompt_id"]
                .as_str()
                .context("scene prompt ID missing")?
                .to_owned();
            loop {
                let history = comfy
                    .history(
                        &ctx.scope,
                        History {
                            prompt_id: id.clone(),
                        },
                    )
                    .await?;
                ensure!(history["status"] != "error", "scene backend failed");
                if history["completed"] == true {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            let result = comfy
                .download(
                    ctx,
                    Download {
                        prompt_id: id,
                        node: grant.output_node.clone(),
                        kind: "images".into(),
                        output_index: 0,
                        output_path: path,
                    },
                    self.artifact_validator.as_deref(),
                )
                .await?;
            ensure!(
                result["media"]["decoded"] == true
                    && result["media"]["width"] == grant.width
                    && result["media"]["height"] == grant.height,
                "generated scene dimensions mismatch"
            );
            Ok::<_, anyhow::Error>(request.output)
        };
        tokio::select! {biased;_=ctx.cancellation.cancelled()=>anyhow::bail!("storyboard generation cancelled; external scene job may remain queued, do not replay"),r=tokio::time::timeout(Duration::from_secs(180),operation)=>r.context("scene generation deadline; external job may remain queued, do not replay")?}
    }
}
