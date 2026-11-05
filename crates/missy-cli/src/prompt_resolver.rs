//! Every channel uses the same exact-scope, operator-approved prompt authority.
use anyhow::Result;
use missy_core::Scope;
use missy_runtime::ScopedPromptResolver;
use std::path::PathBuf;
pub struct GovernancePrompts {
    pub registry: missy_governance::ApprovalRegistry,
    pub identity: PathBuf,
}
impl ScopedPromptResolver for GovernancePrompts {
    fn resolve(&self, scope: &Scope, base_prompt: &str) -> Result<String> {
        let snapshot = self.registry.prompt_snapshot(scope)?;
        let mut prompt = snapshot
            .system_prompt
            .as_ref()
            .map_or_else(|| base_prompt.to_owned(), |r| r.content.clone());
        if let Some(persona) = &snapshot.persona {
            prompt.push_str("\n\nOperator-approved scoped persona:\n");
            prompt.push_str(&persona.content);
        }
        if snapshot.system_prompt.is_some() || snapshot.persona.is_some() {
            prompt.push_str("\n\n[Operator prompt revision fingerprint: ");
            prompt.push_str(&snapshot.prompt_hash);
            prompt.push(']');
        }
        Ok(prompt)
    }
    fn configuration_fingerprint(&self) -> String {
        missy_governance::fingerprint(&("operator-governance-prompts-v1", &self.identity))
            .expect("path serialization")
    }
}
