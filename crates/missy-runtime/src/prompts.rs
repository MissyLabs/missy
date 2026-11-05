//! Exact-scope prompts are selected once at admission, not per provider call.
use crate::{Runtime, context::digest};
use anyhow::{Result, ensure};
use missy_core::Scope;
use serde::{Deserialize, Serialize};

/// Resolve exactly the supplied tenant/user/channel/session, never a wider scope.
/// Revisions are dynamic data, not part of the configuration fingerprint.
pub trait ScopedPromptResolver: Send + Sync {
    fn resolve(&self, scope: &Scope, base_prompt: &str) -> Result<String>;
    /// Stable adapter/registry identity, excluding live governance revisions.
    fn configuration_fingerprint(&self) -> String;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromptSnapshot {
    pub scope: Scope,
    pub text: String,
    pub sha256: String,
    pub context_hash: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct MemorySnapshot {
    pub scope: Scope,
    pub data: String,
    pub sha256: String,
}
impl MemorySnapshot {
    pub(crate) fn validate(&self, scope: &Scope) -> Result<()> {
        ensure!(
            self.scope == *scope && self.sha256 == digest(&self.data),
            "memory snapshot integrity mismatch"
        );
        Ok(())
    }
}
impl PromptSnapshot {
    pub(crate) fn validate(&self, scope: &Scope, context_hash: &str) -> Result<()> {
        ensure!(self.scope == *scope, "prompt snapshot scope mismatch");
        ensure!(
            self.context_hash == context_hash,
            "prompt snapshot configuration mismatch"
        );
        ensure!(
            !self.text.trim().is_empty() && self.sha256 == digest(&self.text),
            "prompt snapshot integrity mismatch"
        );
        Ok(())
    }
}
impl Runtime {
    pub(crate) fn admission_context(
        &self,
        scope: &Scope,
        prompt: &str,
    ) -> Result<serde_json::Value> {
        let snapshot = self.choose_prompt(scope)?;
        let data =
            crate::memory_context::snapshot(self.store(), scope, prompt, &self.inner.config)?;
        let memory = MemorySnapshot {
            scope: scope.clone(),
            sha256: digest(&data),
            data,
        };
        Ok(serde_json::json!({"prompt_snapshot":snapshot,"memory_snapshot":memory}))
    }
    pub(crate) fn admitted_memory(
        &self,
        scope: &Scope,
        turn_id: &str,
    ) -> Result<Option<MemorySnapshot>> {
        let metadata = self.store().admission_metadata(scope, turn_id)?;
        let snapshot = metadata
            .and_then(|v| v.get("memory_snapshot").cloned())
            .map(serde_json::from_value::<MemorySnapshot>)
            .transpose()?;
        if let Some(snapshot) = &snapshot {
            snapshot.validate(scope)?;
        }
        Ok(snapshot)
    }
    pub(crate) fn choose_prompt(&self, scope: &Scope) -> Result<PromptSnapshot> {
        self.check_configuration()?;
        scope.validate()?;
        let text = match &self.inner.prompt_resolver {
            Some(resolver) => resolver.resolve(scope, &self.inner.system)?,
            None => self.inner.system.clone(),
        };
        self.check_configuration()?;
        let snapshot = PromptSnapshot {
            scope: scope.clone(),
            sha256: digest(&text),
            text,
            context_hash: self.inner.context_hash.clone(),
        };
        snapshot.validate(scope, &self.inner.context_hash)?;
        ensure!(
            snapshot.text.len() <= self.inner.config.max_context_chars,
            "resolved system prompt exceeds context bound"
        );
        Ok(snapshot)
    }
    pub(crate) fn admitted_prompt(
        &self,
        scope: &Scope,
        turn_id: &str,
    ) -> Result<Option<PromptSnapshot>> {
        let metadata = self.store().admission_metadata(scope, turn_id)?;
        let snapshot = metadata
            .and_then(|v| v.get("prompt_snapshot").cloned())
            .map(serde_json::from_value::<PromptSnapshot>)
            .transpose()?;
        if let Some(snapshot) = &snapshot {
            snapshot.validate(scope, &self.inner.context_hash)?;
        }
        Ok(snapshot)
    }
}
