use crate::{EvolutionPolicy, EvolutionRegistry};
use anyhow::{Result, ensure};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc};

/// Proposal-only model surface. Policy and repository are operator-selected;
/// review/apply/rollback remain exact-hash authenticated operator operations.
pub struct EvolutionTools {
    inner: Arc<dyn ToolExecutor>,
    registry: EvolutionRegistry,
    policy: EvolutionPolicy,
    repo: PathBuf,
}
impl EvolutionTools {
    pub fn new(
        inner: Arc<dyn ToolExecutor>,
        registry: EvolutionRegistry,
        policy: EvolutionPolicy,
        repo: PathBuf,
    ) -> Result<Self> {
        ensure!(
            repo.is_absolute(),
            "explicit absolute proposal repository required"
        );
        ensure!(
            !inner.definitions().iter().any(|d| d.name == "code_evolve"),
            "code_evolve name collision"
        );
        Ok(Self {
            inner,
            registry,
            policy,
            repo,
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Proposal {
    title: String,
    diff: String,
}
#[async_trait]
impl ToolExecutor for EvolutionTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut d = self.inner.definitions();
        d.push(ToolDefinition{name:"code_evolve".into(),description:"Propose a unified patch to the operator-configured repository. Does not change source or grant approval. Exact plan-hash review and apply are separate operator-only operations.".into(),side_effect:true,parameters:json!({"type":"object","properties":{"title":{"type":"string","minLength":1,"maxLength":512},"diff":{"type":"string","maxLength":1048576}},"required":["title","diff"],"additionalProperties":false})});
        d
    }
    fn configuration_fingerprint(&self) -> String {
        format!(
            "evolution-proposal-v1:{}:{}",
            self.inner.configuration_fingerprint(),
            crate::binding(&(&self.policy, &self.repo)).unwrap_or_else(|_| "invalid".into())
        )
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        if call.name == "code_evolve" {
            ctx.scope.validate()?;
            let p: Proposal = serde_json::from_value(call.arguments.clone())?;
            ensure!(p.diff.len() <= 1_048_576, "diff exceeds byte bound");
            Ok(())
        } else {
            self.inner.preflight(ctx, call).await
        }
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        if call.name == "code_evolve" {
            ctx.scope.validate()?;
            let p: Proposal = serde_json::from_value(call.arguments.clone())?;
            ensure!(p.diff.len() <= 1_048_576, "diff exceeds byte bound");
            let proposal =
                self.registry
                    .propose(&ctx.scope, &self.policy, &self.repo, &p.diff, &p.title)?;
            Ok(
                json!({"proposal_id":proposal.plan.id,"plan_hash":proposal.plan_hash,"status":proposal.status,"applied":false,"requires_operator_review":true}),
            )
        } else {
            self.inner.execute(ctx, call).await
        }
    }
}
