//! Data-only, exact-hash approved native plugins. No interpreter or compiler.
#![cfg(target_os = "linux")]
mod benchmark;
mod proposal;
mod registry;
mod sandbox;
mod schema;
pub use benchmark::*;
pub use registry::*;

use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use missy_core::{Scope, ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use missy_policy::Policy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandTemplate {
    pub executable: PathBuf,
    pub sha256: String,
    /// Fixed literal argv entries or a whole-entry `{property}` substitution.
    pub args: Vec<String>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PluginCapabilities {
    pub read_roots: Vec<PathBuf>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginManifest {
    pub id: String,
    pub definition: ToolDefinition,
    pub command: CommandTemplate,
    pub capabilities: PluginCapabilities,
}
fn timeout() -> u64 {
    30_000
}
fn output_limit() -> usize {
    65_536
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginConfig {
    #[serde(default)]
    pub enabled: bool,
    pub scope: Scope,
    #[serde(default)]
    pub allowed_ids: Vec<String>,
    /// Explicit loader/library directories, never implicit host defaults.
    #[serde(default)]
    pub runtime_read_roots: Vec<PathBuf>,
    #[serde(default = "timeout")]
    pub timeout_ms: u64,
    #[serde(default = "output_limit")]
    pub max_output_bytes: usize,
}
impl PluginConfig {
    pub fn validate(&self) -> Result<()> {
        self.scope.validate()?;
        ensure!(
            (1..=60_000).contains(&self.timeout_ms),
            "plugin timeout outside 1..60000ms"
        );
        ensure!(
            (1..=1_048_576).contains(&self.max_output_bytes),
            "plugin output outside 1..1MiB"
        );
        ensure!(self.allowed_ids.len() <= 64, "too many plugins");
        ensure!(
            self.runtime_read_roots.len() <= 16,
            "too many runtime read roots"
        );
        let mut seen = std::collections::BTreeSet::new();
        for id in &self.allowed_ids {
            validate_id(id)?;
            ensure!(seen.insert(id), "duplicate plugin id");
        }
        for root in &self.runtime_read_roots {
            sandbox::validate_root(root)?;
        }
        Ok(())
    }
}
pub(crate) fn validate_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty()
            && id.len() <= 48
            && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
        "plugin id must be 1..48 ASCII alphanumeric/underscore"
    );
    Ok(())
}
impl PluginManifest {
    pub fn validate(&self) -> Result<()> {
        validate_id(&self.id)?;
        ensure!(
            self.definition.name == format!("plugin__{}", self.id),
            "plugin tool name must be plugin__ID"
        );
        ensure!(
            self.definition.side_effect,
            "all plugins must be side effects"
        );
        ensure!(
            !self.definition.description.is_empty() && self.definition.description.len() <= 4096,
            "invalid description"
        );
        schema::validate(&self.definition.parameters)?;
        sandbox::validate_executable_path(&self.command.executable)?;
        ensure!(
            self.command.sha256.len() == 64
                && self
                    .command
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "expected lowercase executable SHA256"
        );
        ensure!(self.command.args.len() <= 64, "too many argv entries");
        let properties = self.definition.parameters["properties"]
            .as_object()
            .unwrap();
        for arg in &self.command.args {
            ensure!(
                arg.len() <= 8192 && !arg.contains('\0'),
                "invalid argv entry"
            );
            if arg.contains('{') || arg.contains('}') {
                let key = arg
                    .strip_prefix('{')
                    .and_then(|s| s.strip_suffix('}'))
                    .ok_or_else(|| anyhow::anyhow!("placeholder must occupy whole argv entry"))?;
                ensure!(properties.contains_key(key), "unknown argv placeholder");
                ensure!(
                    self.definition.parameters["required"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|v| v.as_str() == Some(key)),
                    "argv placeholder must be required"
                );
            }
        }
        ensure!(
            self.capabilities.read_roots.len() <= 16,
            "too many read roots"
        );
        for root in &self.capabilities.read_roots {
            sandbox::validate_root(root)?;
        }
        Ok(())
    }
    fn argv(&self, args: &Value) -> Result<Vec<String>> {
        schema::arguments(&self.definition.parameters, args)?;
        let mut argv = vec![self.command.executable.to_string_lossy().into_owned()];
        for a in &self.command.args {
            let item = if let Some(key) = a.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                match &args[key] {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    _ => bail!("only string/number substitutions"),
                }
            } else {
                a.clone()
            };
            ensure!(
                item.len() <= 8192 && !item.contains('\0'),
                "argument too long or NUL"
            );
            argv.push(item);
        }
        Ok(argv)
    }
}

/// The embedding runtime must supply a durable exact-call human approval gate.
/// There is deliberately no permissive implementation in this crate.
#[async_trait]
pub trait PluginApproval: Send + Sync {
    async fn authorize(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()>;
    fn configuration_fingerprint(&self) -> String;
}
pub struct PluginTools {
    registry: ToolRegistry,
    config: PluginConfig,
    plugins: BTreeMap<String, PluginProposal>,
    policy: Arc<Policy>,
    approval: Arc<dyn PluginApproval>,
    fingerprint: String,
}
impl PluginTools {
    /// Snapshot approved registry content only at runtime construction. No hot reload.
    pub fn new(
        registry: ToolRegistry,
        config: PluginConfig,
        policy: Arc<Policy>,
        approval: Arc<dyn PluginApproval>,
    ) -> Result<Self> {
        config.validate()?;
        let mut plugins = BTreeMap::new();
        if config.enabled {
            sandbox::check_available()?;
            for record in registry.enabled(&config)? {
                record.manifest.validate()?;
                for root in &record.manifest.capabilities.read_roots {
                    ensure!(
                        policy
                            .config()
                            .read_roots
                            .iter()
                            .any(|r| root.starts_with(r)),
                        "plugin read capability exceeds policy"
                    );
                }
                // Verify now, and repeat immediately before dispatch.
                sandbox::pinned_executable(&record.manifest.command)?;
                plugins.insert(record.manifest.definition.name.clone(), record);
            }
        }
        let fingerprint = hex::encode(Sha256::digest(serde_json::to_vec(&serde_json::json!({
            "config":config,"plugins":plugins,"proposal_tool":proposal::definition(),"policy":policy.config(),"approval":approval.configuration_fingerprint(),"sandbox":"landlock-seccomp-sealed-elf-v2"
        }))?));
        Ok(Self {
            registry,
            config,
            plugins,
            policy,
            approval,
            fingerprint,
        })
    }
    fn validate_call(
        &self,
        ctx: &ToolContext,
        call: &ToolCall,
    ) -> Result<(&PluginManifest, Vec<String>)> {
        ensure!(
            self.config.enabled && ctx.scope == self.config.scope,
            "plugin activation scope denied"
        );
        ensure!(
            !ctx.cancellation.is_cancelled(),
            "plugin call cancelled before dispatch"
        );
        let record = self
            .plugins
            .get(&call.name)
            .ok_or_else(|| anyhow::anyhow!("plugin not activated in this runtime"))?;
        let argv = record.manifest.argv(&call.arguments)?;
        // Never extend policy with model-generated vectors. The exact expanded
        // command must already be operator-allowlisted, including every value.
        self.policy.check_command(&argv)?;
        Ok((&record.manifest, argv))
    }
}
#[async_trait]
impl ToolExecutor for PluginTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions: Vec<_> = self
            .plugins
            .values()
            .map(|p| p.manifest.definition.clone())
            .collect();
        if self.config.enabled {
            definitions.push(proposal::definition());
        }
        definitions
    }
    fn configuration_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        if call.name == proposal::NAME {
            self.proposal_manifest(ctx, call)
                .map_err(|e| missy_core::ToolDispatchRefused {
                    reason: e.to_string(),
                })?;
            return self.approval.authorize(ctx, call).await;
        }
        self.validate_call(ctx, call)
            .map_err(|e| missy_core::ToolDispatchRefused {
                reason: e.to_string(),
            })?;
        self.approval.authorize(ctx, call).await
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        if call.name == proposal::NAME {
            let manifest =
                self.proposal_manifest(ctx, call)
                    .map_err(|e| missy_core::ToolDispatchRefused {
                        reason: e.to_string(),
                    })?;
            self.approval.authorize(ctx, call).await?;
            ensure!(
                !ctx.cancellation.is_cancelled(),
                "plugin proposal cancelled before write"
            );
            let pending = self.registry.propose(&ctx.scope, manifest)?;
            return Ok(serde_json::json!({
                "trust":"untrusted", "status":"pending_operator_review",
                "proposal":pending,"approved":false,"enabled":false,
                "requires_new_runtime":true,
                "notice":"Data-only proposal persisted. Operator must review this exact scope-bound hash, explicitly allow and enable the ID, then construct a new runtime. Nothing was compiled or executed."
            }));
        }
        let (manifest, argv) =
            self.validate_call(ctx, call)
                .map_err(|e| missy_core::ToolDispatchRefused {
                    reason: e.to_string(),
                })?;
        self.approval.authorize(ctx, call).await?;
        sandbox::execute(manifest, &argv, &self.config, ctx).await
    }
}
