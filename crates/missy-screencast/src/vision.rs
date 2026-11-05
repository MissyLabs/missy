//! Reuses the configured vision tool's own pricing ledger and policy HTTP client.
//! There is no second provider implementation or independent inference budget here.
use crate::{FrameAnalyzer, FrameRequest};
use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_vision::VisionTools;
use std::sync::Arc;

#[async_trait]
pub trait FrameGovernor: Send + Sync {
    /// Durably reserve the parent operation before ANY provider dispatch. Must bind
    /// frame.context.turn_id, exact scope, session approval, image digest and prompt.
    /// Repeated IDs must refuse dispatch (including unknown/uncertain prior receipts).
    async fn reserve(&self, frame: &FrameRequest, prompt: &str) -> Result<String>;
    /// `false` means uncertain/error, never safe to replay. A dropped/cancelled future
    /// intentionally leaves the reservation unresolved and must also block replay.
    async fn finish(&self, reservation: &str, succeeded: bool) -> Result<()>;
}
pub struct GovernedVisionAnalyzer {
    vision: Arc<VisionTools>,
    governor: Arc<dyn FrameGovernor>,
    prompt: String,
}
impl GovernedVisionAnalyzer {
    pub fn new(
        vision: Arc<VisionTools>,
        governor: Arc<dyn FrameGovernor>,
        prompt: String,
    ) -> Result<Self> {
        if prompt.trim().is_empty() || prompt.len() > 4096 {
            bail!("invalid screencast analysis prompt");
        }
        Ok(Self {
            vision,
            governor,
            prompt,
        })
    }
}
#[async_trait]
impl FrameAnalyzer for GovernedVisionAnalyzer {
    async fn analyze(&self, frame: &FrameRequest) -> Result<String> {
        let receipt = self.governor.reserve(frame, &self.prompt).await?;
        if receipt.is_empty() || receipt.len() > 256 {
            bail!("invalid durable frame reservation");
        }
        let result = self
            .vision
            .analyze_png(&frame.context, &frame.png, &self.prompt)
            .await;
        let ok = result.as_ref().is_ok_and(|v| v["status"] == "ok");
        self.governor.finish(&receipt, ok).await?;
        let result = result?;
        if !ok {
            bail!("vision accounting/output did not settle successfully");
        }
        let text = result["analysis"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing vision analysis"))?;
        Ok(text.into())
    }
}
