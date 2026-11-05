use crate::{
    MAX_ARGUMENT_BYTES, MAX_CONTENT_BYTES, MAX_OUTPUT_BYTES, MAX_QUERY_BYTES, MAX_RESULTS,
    PersonalMemory, TRUST,
};
use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolDispatchRefused, ToolExecutor};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Construction is an explicit cross-channel/cross-session PERSONAL namespace grant.
/// Runtime defaults must leave this executor absent; it does not enable itself.
pub struct PersonalTools {
    memory: Arc<PersonalMemory>,
}
impl PersonalTools {
    pub fn new(memory: Arc<PersonalMemory>) -> Self {
        Self { memory }
    }
    fn checked(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Args> {
        ctx.scope.validate()?;
        ensure!(!ctx.cancellation.is_cancelled(), "tool cancelled");
        ensure!(
            serde_json::to_vec(&call.arguments)?.len() <= MAX_ARGUMENT_BYTES,
            "tool arguments exceed byte limit"
        );
        let args = match call.name.as_str() {
            "personal_memory_put" => {
                let a: Put = serde_json::from_value(call.arguments.clone())?;
                crate::identifier(&a.id)?;
                crate::validate_content(&a.content)?;
                Args::Put(a)
            }
            "personal_memory_get" | "personal_memory_delete" => {
                let a: Id = serde_json::from_value(call.arguments.clone())?;
                crate::identifier(&a.id)?;
                Args::Id(a)
            }
            "personal_memory_search" => {
                let a: Search = serde_json::from_value(call.arguments.clone())?;
                crate::query_limit(&a.query, a.limit)?;
                Args::Search(a)
            }
            _ => bail!("unsupported personal memory tool"),
        };
        Ok(args)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Put {
    id: String,
    content: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Id {
    id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Search {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}
fn default_limit() -> usize {
    10
}
enum Args {
    Put(Put),
    Id(Id),
    Search(Search),
}

#[async_trait]
impl ToolExecutor for PersonalTools {
    fn configuration_fingerprint(&self) -> String {
        hex::encode(Sha256::digest(
            serde_json::to_vec(&json!({
                "memory_policy":self.memory.configuration_fingerprint(),"tools":self.definitions()
            }))
            .expect("serializable personal tool configuration"),
        ))
    }
    fn definitions(&self) -> Vec<ToolDefinition> {
        let id = json!({"type":"string","minLength":1,"maxLength":256});
        [
            ("personal_memory_put","Explicitly save untrusted PERSONAL memory for the authenticated tenant+user across channels and sessions.",json!({"id":id,"content":{"type":"string","minLength":1,"maxLength":MAX_CONTENT_BYTES}}),vec!["id","content"],true),
            ("personal_memory_get","Get untrusted PERSONAL memory for the authenticated tenant+user only. source_scope is the original write scope.",json!({"id":id}),vec!["id"],false),
            ("personal_memory_search","Search untrusted PERSONAL memory across the authenticated user's sessions/channels. Literal Unicode terms; no global fallback.",json!({"query":{"type":"string","maxLength":MAX_QUERY_BYTES},"limit":{"type":"integer","minimum":1,"maximum":MAX_RESULTS}}),vec!["query"],false),
            ("personal_memory_delete","Delete PERSONAL memory for the authenticated tenant+user only, across their sessions/channels.",json!({"id":id}),vec!["id"],true),
        ].into_iter().map(|(name,description,properties,required,side_effect)| ToolDefinition {
            name:name.into(),description:description.into(),parameters:json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),side_effect
        }).collect()
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.checked(ctx, call).map(|_| ()).map_err(|e| {
            ToolDispatchRefused {
                reason: e.to_string(),
            }
            .into()
        })
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        let args = self.checked(ctx, call).map_err(|e| ToolDispatchRefused {
            reason: e.to_string(),
        })?;
        let data = match args {
            Args::Put(a) => serde_json::to_value(self.memory.put(&ctx.scope, &a.id, &a.content)?)?,
            Args::Id(a) if call.name == "personal_memory_delete" => {
                json!({"deleted":self.memory.delete(&ctx.scope,&a.id)?})
            }
            Args::Id(a) => serde_json::to_value(self.memory.get(&ctx.scope, &a.id)?)?,
            Args::Search(a) => {
                serde_json::to_value(self.memory.search(&ctx.scope, &a.query, a.limit)?)?
            }
        };
        let framed = json!({"trust":TRUST,"namespace_kind":"personal","source":"explicit_user_memory","data":data});
        ensure!(
            serde_json::to_vec(&framed)?.len() <= MAX_OUTPUT_BYTES,
            "personal memory result exceeds byte limit"
        );
        Ok(framed)
    }
}
