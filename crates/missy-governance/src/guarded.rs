use crate::{
    ApprovalBinding, ApprovalRefusedError, ApprovalRegistry, ApprovalRequiredError, ApprovalStatus,
    digest, fingerprint,
};
use anyhow::{Result, ensure};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardConfig {
    pub sensitive_tool_names: BTreeSet<String>,
    pub context_fingerprint: String,
    pub approval_ttl_seconds: u64,
}
/// Annotation is an additional gate, never a permission grant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyAnnotation {
    DeniedUnknownTool,
    InnerPolicyRequired,
    OperatorApprovalAndInnerPolicyRequired,
}
pub struct GuardedTools {
    inner: Arc<dyn ToolExecutor>,
    registry: ApprovalRegistry,
    config: GuardConfig,
    fingerprint: String,
}
impl GuardedTools {
    pub fn new(
        inner: Arc<dyn ToolExecutor>,
        registry: ApprovalRegistry,
        config: GuardConfig,
    ) -> Result<Self> {
        ensure!(
            digest(&config.context_fingerprint),
            "guard context must be a SHA-256 fingerprint"
        );
        ensure!(
            (1..=86400).contains(&config.approval_ttl_seconds),
            "approval TTL must be 1..86400 seconds"
        );
        let names: BTreeSet<_> = inner.definitions().into_iter().map(|d| d.name).collect();
        ensure!(
            config.sensitive_tool_names.is_subset(&names),
            "sensitive tool name missing from executor catalog"
        );
        let fingerprint =
            fingerprint(&("missy-guard-v1", &config, inner.configuration_fingerprint()))?;
        Ok(Self {
            inner,
            registry,
            config,
            fingerprint,
        })
    }
    pub fn binding(&self, ctx: &ToolContext, call: &ToolCall) -> Result<ApprovalBinding> {
        ApprovalBinding::new(ctx, call, &self.fingerprint)
    }
    pub fn annotation(&self, tool_name: &str) -> PolicyAnnotation {
        if !self.inner.definitions().iter().any(|d| d.name == tool_name) {
            PolicyAnnotation::DeniedUnknownTool
        } else if self.config.sensitive_tool_names.contains(tool_name) {
            PolicyAnnotation::OperatorApprovalAndInnerPolicyRequired
        } else {
            PolicyAnnotation::InnerPolicyRequired
        }
    }
    fn check_known_and_cancelled(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        ensure!(
            !ctx.cancellation.is_cancelled(),
            "tool operation cancelled before dispatch"
        );
        ensure!(
            self.annotation(&call.name) != PolicyAnnotation::DeniedUnknownTool,
            "tool denied: unknown tool"
        );
        let current = fingerprint(&(
            "missy-guard-v1",
            &self.config,
            self.inner.configuration_fingerprint(),
        ))?;
        ensure!(
            current == self.fingerprint,
            "inner tool configuration changed; rebuild guarded executor"
        );
        Ok(())
    }
}
#[async_trait]
impl ToolExecutor for GuardedTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.inner.definitions().into_iter().map(|mut definition| {
            if self.config.sensitive_tool_names.contains(&definition.name) {
                definition.description.push_str(" [Policy: exact operator approval and inner executor policy required; approval does not override capability restrictions.]");
            }
            definition
        }).collect()
    }
    fn configuration_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.check_known_and_cancelled(ctx, call)?;
        self.inner.preflight(ctx, call).await?;
        if self.config.sensitive_tool_names.contains(&call.name) {
            let record = self
                .registry
                .request(&self.binding(ctx, call)?, self.config.approval_ttl_seconds)?;
            match record.status {
                ApprovalStatus::Approved => {}
                ApprovalStatus::Pending => {
                    return Err(ApprovalRequiredError {
                        request_id: record.id,
                    }
                    .into());
                }
                status => {
                    return Err(ApprovalRefusedError {
                        request_id: record.id,
                        status,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<serde_json::Value> {
        let predispatch: Result<()> = async {
            self.check_known_and_cancelled(ctx, call)?;
            self.inner.preflight(ctx, call).await?;
            if self.config.sensitive_tool_names.contains(&call.name) {
                let binding = self.binding(ctx, call)?;
                self.registry
                    .request(&binding, self.config.approval_ttl_seconds)?;
                self.registry.consume(&binding)?;
            }
            Ok(())
        }
        .await;
        // Do not retain arbitrary inner preflight diagnostics: they can contain
        // secrets. The type, not message parsing, proves non-dispatch to runtime.
        predispatch.map_err(|_| missy_core::ToolDispatchRefused {
            reason: "governance refused dispatch before inner execution".into(),
        })?;
        // Consumption is never undone, including when the inner operation fails.
        self.inner.execute(ctx, call).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decision;
    use missy_core::Scope;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Inner {
        calls: AtomicUsize,
        fail: bool,
        deny: bool,
    }
    #[async_trait]
    impl ToolExecutor for Inner {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "sensitive".into(),
                description: "test".into(),
                parameters: serde_json::json!({}),
                side_effect: true,
            }]
        }
        async fn preflight(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
            ensure!(!self.deny, "inner policy denied");
            Ok(())
        }
        async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<serde_json::Value> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            ensure!(!self.fail, "failed effect");
            Ok(serde_json::json!({"ok":true}))
        }
    }
    fn context() -> ToolContext {
        ToolContext {
            scope: Scope {
                tenant: "t".into(),
                user: "u".into(),
                channel: "c".into(),
                session: "s".into(),
            },
            turn_id: "turn".into(),
            cancellation: tokio_util::sync::CancellationToken::new(),
        }
    }
    fn config() -> GuardConfig {
        GuardConfig {
            sensitive_tool_names: BTreeSet::from(["sensitive".into()]),
            context_fingerprint: fingerprint(&"policy").unwrap(),
            approval_ttl_seconds: 60,
        }
    }
    #[tokio::test]
    async fn preflight_stable_pause_no_consume_execute_consumes_even_when_failed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let registry = ApprovalRegistry::open(dir.path().join("db"))?;
        let inner = Arc::new(Inner {
            calls: AtomicUsize::new(0),
            fail: true,
            deny: false,
        });
        let guard = GuardedTools::new(inner.clone(), registry.clone(), config())?;
        let ctx = context();
        let call = ToolCall {
            id: "call".into(),
            name: "sensitive".into(),
            arguments: serde_json::json!({"password":"private"}),
        };
        let error = guard.preflight(&ctx, &call).await.unwrap_err();
        let id = error
            .downcast_ref::<missy_core::ApprovalRequired>()
            .unwrap()
            .request_id
            .clone();
        assert_eq!(
            guard
                .preflight(&ctx, &call)
                .await
                .unwrap_err()
                .downcast_ref::<missy_core::ApprovalRequired>()
                .unwrap()
                .request_id,
            id
        );
        assert_eq!(inner.calls.load(Ordering::SeqCst), 0);
        let record = registry.get(&ctx.scope, &id)?.unwrap();
        registry.decide(
            &ctx.scope,
            &id,
            &record.binding_hash,
            Decision::Approve,
            "operator",
        )?;
        guard.preflight(&ctx, &call).await?;
        guard.preflight(&ctx, &call).await?;
        assert_eq!(
            registry.get(&ctx.scope, &id)?.unwrap().status,
            ApprovalStatus::Approved
        );
        let effect_error = guard.execute(&ctx, &call).await.unwrap_err();
        assert!(!effect_error.is::<missy_core::ToolDispatchRefused>());
        assert!(
            guard
                .execute(&ctx, &call)
                .await
                .unwrap_err()
                .is::<missy_core::ToolDispatchRefused>()
        );
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            registry.get(&ctx.scope, &id)?.unwrap().status,
            ApprovalStatus::Consumed
        );
        assert!(!serde_json::to_string(&registry.list(&ctx.scope, 100)?)?.contains("private"));
        Ok(())
    }
    #[tokio::test]
    async fn approvals_cannot_override_inner_policy_and_context_changes_require_new_decision()
    -> Result<()> {
        let dir = tempfile::tempdir()?;
        let registry = ApprovalRegistry::open(dir.path().join("db"))?;
        let inner = Arc::new(Inner {
            calls: AtomicUsize::new(0),
            fail: false,
            deny: true,
        });
        let guard = GuardedTools::new(inner.clone(), registry.clone(), config())?;
        let ctx = context();
        let call = ToolCall {
            id: "call".into(),
            name: "sensitive".into(),
            arguments: serde_json::json!({}),
        };
        let record = registry.request(&guard.binding(&ctx, &call)?, 60)?;
        registry.decide(
            &ctx.scope,
            &record.id,
            &record.binding_hash,
            Decision::Approve,
            "operator",
        )?;
        assert!(guard.preflight(&ctx, &call).await.is_err());
        assert!(
            guard
                .execute(&ctx, &call)
                .await
                .unwrap_err()
                .is::<missy_core::ToolDispatchRefused>()
        );
        assert_eq!(inner.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            registry.get(&ctx.scope, &record.id)?.unwrap().status,
            ApprovalStatus::Approved
        );
        let mut changed = config();
        changed.context_fingerprint = fingerprint(&"new persona")?;
        let other = GuardedTools::new(inner, registry, changed)?;
        assert_ne!(
            guard.binding(&ctx, &call)?.hash()?,
            other.binding(&ctx, &call)?.hash()?
        );
        Ok(())
    }
    #[tokio::test]
    async fn unknown_cancelled_or_changed_context_never_dispatches() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let registry = ApprovalRegistry::open(dir.path().join("db"))?;
        let inner = Arc::new(Inner {
            calls: AtomicUsize::new(0),
            fail: false,
            deny: false,
        });
        let guard = GuardedTools::new(inner.clone(), registry.clone(), config())?;
        let ctx = context();
        let call = ToolCall {
            id: "call".into(),
            name: "sensitive".into(),
            arguments: serde_json::json!({}),
        };
        assert_eq!(
            guard.annotation("unknown"),
            PolicyAnnotation::DeniedUnknownTool
        );
        assert_eq!(
            guard.annotation("sensitive"),
            PolicyAnnotation::OperatorApprovalAndInnerPolicyRequired
        );
        assert!(
            guard.definitions()[0]
                .description
                .contains("approval does not override")
        );
        let record = registry.request(&guard.binding(&ctx, &call)?, 60)?;
        registry.decide(
            &ctx.scope,
            &record.id,
            &record.binding_hash,
            Decision::Approve,
            "op",
        )?;
        let mut unknown = call.clone();
        unknown.name = "unknown".into();
        assert!(guard.preflight(&ctx, &unknown).await.is_err());
        assert!(
            guard
                .execute(&ctx, &unknown)
                .await
                .unwrap_err()
                .is::<missy_core::ToolDispatchRefused>()
        );
        let mut changed = config();
        changed.context_fingerprint = fingerprint(&"new policy")?;
        let other = GuardedTools::new(inner.clone(), registry.clone(), changed)?;
        assert!(
            other
                .preflight(&ctx, &call)
                .await
                .unwrap_err()
                .is::<missy_core::ApprovalRequired>()
        );
        assert!(
            other
                .execute(&ctx, &call)
                .await
                .unwrap_err()
                .is::<missy_core::ToolDispatchRefused>()
        );
        ctx.cancellation.cancel();
        assert!(
            guard
                .execute(&ctx, &call)
                .await
                .unwrap_err()
                .is::<missy_core::ToolDispatchRefused>()
        );
        assert_eq!(
            registry.get(&ctx.scope, &record.id)?.unwrap().status,
            ApprovalStatus::Approved
        );
        assert_eq!(inner.calls.load(Ordering::SeqCst), 0);
        Ok(())
    }
}
