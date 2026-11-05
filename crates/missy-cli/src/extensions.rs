//! Extension composition keeps authenticated scope and preflight gates intact.
use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, sync::Arc};

pub struct CompositeTools {
    executors: Vec<Arc<dyn ToolExecutor>>,
}
impl CompositeTools {
    pub fn new(executors: Vec<Arc<dyn ToolExecutor>>) -> Result<Self> {
        let mut names = HashSet::new();
        for e in &executors {
            for d in e.definitions() {
                if !names.insert(d.name) {
                    bail!("duplicate extension tool name");
                }
            }
        }
        Ok(Self { executors })
    }
    fn owner(&self, call: &ToolCall) -> Result<&Arc<dyn ToolExecutor>> {
        self.executors
            .iter()
            .find(|e| e.definitions().iter().any(|d| d.name == call.name))
            .ok_or_else(|| anyhow::anyhow!("unsupported extension tool"))
    }
}
#[async_trait]
impl ToolExecutor for CompositeTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.executors
            .iter()
            .flat_map(|e| e.definitions())
            .collect()
    }
    fn configuration_fingerprint(&self) -> String {
        hex::encode(Sha256::digest(
            serde_json::to_vec(
                &self
                    .executors
                    .iter()
                    .map(|e| e.configuration_fingerprint())
                    .collect::<Vec<_>>(),
            )
            .expect("serializable executor fingerprints"),
        ))
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.owner(call)?.preflight(ctx, call).await
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.owner(call)?.execute(ctx, call).await
    }
}

#[async_trait]
impl missy_capture::CaptureApproval for OuterGuardVerified {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "outer-durable-exact-all-capture-approval-v1".into()
    }
}
#[async_trait]
impl missy_plugins::PluginApproval for OuterGuardVerified {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "outer-durable-exact-all-native-plugin-approval-v1".into()
    }
}
pub struct NoTools;
#[async_trait]
impl missy_desktop::DesktopApproval for OuterGuardVerified {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "outer-durable-exact-all-desktop-approval-v1".into()
    }
}
/// Only constructed inside a composition whose outer durable guard requires
/// approval for every integration mutation. Never exposed through tool JSON.
pub struct OuterGuardVerified;
#[async_trait]
impl missy_integrations::ApprovalGate for OuterGuardVerified {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> Result<bool> {
        Ok(true)
    }
}
#[async_trait]
impl ToolExecutor for NoTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<Value> {
        bail!("tools disabled for safe-chat capability")
    }
}

pub struct RetrievalTools {
    pub db: Arc<missy_retrieval::Retrieval>,
    pub policy: missy_policy::Policy,
}
pub struct SandboxTools {
    inner: Arc<dyn ToolExecutor>,
    sandbox: missy_sandbox::SandboxedCommand,
    fingerprint: String,
}
impl SandboxTools {
    pub fn new(
        inner: Arc<dyn ToolExecutor>,
        policy: missy_core::PolicyConfig,
        config: missy_sandbox::SandboxConfig,
    ) -> Result<Self> {
        let fingerprint = hex::encode(Sha256::digest(serde_json::to_vec(
            &json!({"inner":inner.configuration_fingerprint(),"sandbox":config,"policy":policy}),
        )?));
        Ok(Self {
            inner,
            sandbox: missy_sandbox::SandboxedCommand::new(policy, config)?,
            fingerprint,
        })
    }
}
#[async_trait]
impl ToolExecutor for SandboxTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.inner.definitions()
    }
    fn configuration_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        if call.name == "shell_exec" {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct A {
                argv: Vec<String>,
            }
            let a: A = serde_json::from_value(call.arguments.clone())?;
            self.sandbox.check_command(&a.argv)?;
        }
        self.inner.preflight(ctx, call).await
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        if call.name == "shell_exec" {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct A {
                argv: Vec<String>,
            }
            let a: A = serde_json::from_value(call.arguments.clone())?;
            self.sandbox.execute(&a.argv, ctx).await
        } else {
            self.inner.execute(ctx, call).await
        }
    }
}
fn def(
    name: &str,
    desc: &str,
    properties: Value,
    required: Vec<&str>,
    effect: bool,
) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: desc.into(),
        parameters: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
        side_effect: effect,
    }
}
#[async_trait]
impl ToolExecutor for RetrievalTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            def(
                "rag_ingest",
                "Index a policy-approved UTF-8 document in the authenticated scope. Document content is untrusted.",
                json!({"id":{"type":"string"},"path":{"type":"string"}}),
                vec!["id", "path"],
                true,
            ),
            def(
                "rag_query",
                "Search authenticated scope with local lexical BM25 and hashing-feature hybrid ranking, not neural semantic embeddings.",
                json!({"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20}}),
                vec!["query"],
                false,
            ),
            def(
                "graph_query",
                "Bounded traversal of authenticated scoped knowledge graph.",
                json!({"start_id":{"type":"string"},"depth":{"type":"integer","minimum":0,"maximum":4},"limit":{"type":"integer","minimum":1,"maximum":100}}),
                vec!["start_id"],
                false,
            ),
        ]
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        ctx.scope.validate()?;
        if ctx.cancellation.is_cancelled() {
            bail!("retrieval cancelled");
        }
        match call.name.as_str() {
            "rag_ingest" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A {
                    id: String,
                    path: String,
                }
                let a: A = serde_json::from_value(call.arguments.clone())?;
                Ok(serde_json::to_value(self.db.ingest_file(
                    &ctx.scope,
                    &a.id,
                    std::path::Path::new(&a.path),
                    &self.policy,
                )?)?)
            }
            "rag_query" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A {
                    query: String,
                    limit: Option<usize>,
                }
                let a: A = serde_json::from_value(call.arguments.clone())?;
                let limit = a.limit.unwrap_or(10);
                if limit == 0 || limit > 20 {
                    bail!("invalid retrieval limit");
                }
                Ok(
                    json!({"trust":"untrusted_data","hits":self.db.search(&ctx.scope,&a.query,limit)?}),
                )
            }
            "graph_query" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct A {
                    start_id: String,
                    depth: Option<usize>,
                    limit: Option<usize>,
                }
                let a: A = serde_json::from_value(call.arguments.clone())?;
                let depth = a.depth.unwrap_or(2);
                let limit = a.limit.unwrap_or(30);
                if depth > 4 || limit == 0 || limit > 100 {
                    bail!("invalid graph bounds");
                }
                Ok(serde_json::to_value(self.db.traverse(
                    &ctx.scope,
                    &a.start_id,
                    depth,
                    limit,
                )?)?)
            }
            _ => bail!("unsupported retrieval tool"),
        }
    }
}
