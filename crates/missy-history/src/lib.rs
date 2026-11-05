//! Four-component exact-scope conversation recall. Receipts are immutable durable
//! snapshots; retrieving them never reexecutes tools. No global/personal fallback.
mod files;
mod retention;
mod sources;
mod summary;
use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use missy_core::{Scope, ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use missy_store::{Store, hash_text};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
pub use summary::{SummaryBackend, SummaryPolicy};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HistoryConfig {
    pub enabled: bool,
    /// Operator-owned immutable receipt database. Required when enabled.
    pub retention_path: Option<std::path::PathBuf>,
    pub max_window_turns: usize,
    pub max_results: usize,
    pub max_scan_bytes: usize,
    pub max_source_bytes: usize,
    pub max_retained_items: usize,
    pub max_retained_bytes: usize,
    pub max_expand_bytes: usize,
}
impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            retention_path: None,
            max_window_turns: 1000,
            max_results: 20,
            max_scan_bytes: 8 * 1024 * 1024,
            max_source_bytes: 1024 * 1024,
            max_retained_items: 1024,
            max_retained_bytes: 16 * 1024 * 1024,
            max_expand_bytes: 8192,
        }
    }
}
impl HistoryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.enabled || self.retention_path.is_some(),
            "enabled history requires retention_path"
        );
        ensure!(
            (1..=1000).contains(&self.max_window_turns),
            "history window must be 1..1000"
        );
        ensure!(
            (1..=20).contains(&self.max_results),
            "max_results must be 1..20"
        );
        ensure!(
            (1024..=16 * 1024 * 1024).contains(&self.max_scan_bytes),
            "invalid scan byte bound"
        );
        ensure!(
            (512..=1024 * 1024).contains(&self.max_source_bytes),
            "invalid source byte bound"
        );
        ensure!(
            (1..=4096).contains(&self.max_retained_items),
            "invalid retained item bound"
        );
        ensure!(
            (1024..=64 * 1024 * 1024).contains(&self.max_retained_bytes),
            "invalid retained byte bound"
        );
        ensure!(
            (1..=16384).contains(&self.max_expand_bytes),
            "invalid expansion byte bound"
        );
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Evidence {
    id: String,
    scope: Scope,
    turn_id: Option<String>,
    tool_id: Option<String>,
    source: String,
    created_at: i64,
    text: String,
    sha256: String,
}
impl Evidence {
    fn metadata(&self) -> Value {
        json!({"item_id":self.id,"output_id":self.id,"turn_id":self.turn_id,
            "tool_id":self.tool_id,"source":self.source,"created_at":self.created_at,
            "content_bytes":self.text.len(),"content_sha256":self.sha256,
            "trust":"untrusted_data","retention":"durable_immutable_snapshot"})
    }
}
pub struct HistoryTools {
    store: Arc<Store>,
    config: HistoryConfig,
    retained: Option<retention::Retained>,
    summary: Option<SummaryBackend>,
}
impl HistoryTools {
    pub fn new(store: Arc<Store>, config: HistoryConfig) -> Result<Self> {
        config.validate()?;
        let retained = if config.enabled {
            Some(retention::Retained::open(
                config.retention_path.as_ref().expect("validated path"),
            )?)
        } else {
            None
        };
        Ok(Self {
            store,
            config,
            retained,
            summary: None,
        })
    }
    pub fn with_summary_backend(mut self, backend: SummaryBackend) -> Result<Self> {
        backend.policy.validate()?;
        self.summary = Some(backend);
        Ok(self)
    }
    fn gate(&self, ctx: &ToolContext) -> Result<()> {
        ensure!(self.config.enabled, "history tools disabled");
        ctx.scope.validate()?;
        ensure!(!ctx.cancellation.is_cancelled(), "cancelled");
        Ok(())
    }
    fn evidence(&self, scope: &Scope, id: &str) -> Result<Evidence> {
        scope.validate()?;
        self.retained
            .as_ref()
            .ok_or_else(|| anyhow!("history tools disabled"))?
            .get(scope, id)
    }
    fn retain(&self, evidence: Evidence) -> Result<Evidence> {
        self.retained
            .as_ref()
            .ok_or_else(|| anyhow!("history tools disabled"))?
            .put(evidence, &self.config)
    }
    fn search(&self, ctx: &ToolContext, args: Search) -> Result<Value> {
        ensure!(
            !args.query.trim().is_empty() && args.query.len() <= 1024,
            "query must be 1..1024 bytes"
        );
        let limit = args.limit.unwrap_or(self.config.max_results);
        ensure!(
            (1..=self.config.max_results).contains(&limit),
            "invalid result limit"
        );
        let query = args.query.to_ascii_lowercase();
        let mut turns = self
            .store
            .list_turns(&ctx.scope, self.config.max_window_turns + 2)?;
        // Recall must not rank its own current query/tool-call checkpoint as
        // historical evidence. This also prevents self-generated false hits.
        turns.retain(|turn| turn.id != ctx.turn_id);
        let older_turns_omitted = turns.len() > self.config.max_window_turns;
        turns.truncate(self.config.max_window_turns);
        let window_turns = turns.len();
        let additional = sources::additional_sources(&self.store, &ctx.scope, &turns)?;
        let additional_source_count = additional.sources.len();
        let mut sources = Vec::new();
        let mut scanned_bytes = 0usize;
        let mut skipped_sources = 0usize;
        let mut scan_complete = true;
        let mut scanned_turns = 0;
        'turns: for turn in turns {
            ensure!(!ctx.cancellation.is_cancelled(), "cancelled");
            let mut parts = vec![("turn_prompt".to_owned(), None, turn.prompt)];
            if let Some(text) = turn.response {
                parts.push(("turn_response".into(), None, text));
            }
            if let Some(text) = turn.error {
                parts.push(("turn_error".into(), None, text));
            }
            for record in self.store.tools(&ctx.scope, &turn.id)? {
                if let Some(result) = record.result {
                    parts.push((
                        "tool_result_json".into(),
                        Some(record.call.id.clone()),
                        serde_json::to_string(&result)?,
                    ));
                }
                if let Some(error) = record.error {
                    parts.push(("tool_error".into(), Some(record.call.id), error));
                }
            }
            for (source, tool_id, text) in parts {
                if text.len() > self.config.max_source_bytes {
                    skipped_sources += 1;
                    continue;
                }
                if scanned_bytes.saturating_add(text.len()) > self.config.max_scan_bytes {
                    scan_complete = false;
                    break 'turns;
                }
                scanned_bytes += text.len();
                let folded = text.to_ascii_lowercase();
                if let Some(start) = folded.find(&query) {
                    let hits = folded.match_indices(&query).take(100).count();
                    let e = Evidence {
                        id: String::new(),
                        scope: ctx.scope.clone(),
                        turn_id: Some(turn.id.clone()),
                        tool_id,
                        source,
                        created_at: turn.created_at,
                        sha256: hash_text(&text),
                        text,
                    };
                    sources.push((hits, start, e));
                }
            }
            scanned_turns += 1;
        }
        let mut additional_sources_scanned = 0;
        for source in additional.sources {
            ensure!(!ctx.cancellation.is_cancelled(), "cancelled");
            if source.text.len() > self.config.max_source_bytes {
                skipped_sources += 1;
                continue;
            }
            if scanned_bytes.saturating_add(source.text.len()) > self.config.max_scan_bytes {
                scan_complete = false;
                break;
            }
            scanned_bytes += source.text.len();
            additional_sources_scanned += 1;
            let folded = source.text.to_ascii_lowercase();
            if let Some(start) = folded.find(&query) {
                let hits = folded.match_indices(&query).take(100).count();
                sources.push((
                    hits,
                    start,
                    Evidence {
                        id: String::new(),
                        scope: ctx.scope.clone(),
                        turn_id: source.turn_id,
                        tool_id: source.tool_id,
                        source: source.source,
                        created_at: source.created_at,
                        sha256: hash_text(&source.text),
                        text: source.text,
                    },
                ));
            }
        }
        sources.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.2.created_at.cmp(&a.2.created_at))
        });
        let matching_sources = sources.len();
        let mut results = Vec::new();
        for (score, hit, e) in sources.into_iter().take(limit) {
            let e = self.retain(e)?;
            let mut start = hit.saturating_sub(100);
            while !e.text.is_char_boundary(start) {
                start += 1;
            }
            let mut end = (start + 512.max(args.query.len() + 100)).min(e.text.len());
            while !e.text.is_char_boundary(end) {
                end -= 1;
            }
            let mut row = e.metadata();
            row["score"] = json!(score);
            row["start_byte"] = json!(start);
            row["end_byte"] = json!(end);
            row["match_start_byte"] = json!(hit);
            row["match_end_byte"] = json!(hit + args.query.len());
            row["text"] = json!(&e.text[start..end]);
            results.push(row);
        }
        Ok(
            json!({"results":results,"matching_sources_in_scan":matching_sources,
            "window":{"kind":"latest_exact_scope_turns","limit":self.config.max_window_turns,
                "turns_loaded":window_turns,"turns_scanned":scanned_turns,"older_turns_omitted":older_turns_omitted,
                "scan_complete":scan_complete,"scanned_bytes":scanned_bytes,"oversized_sources_skipped":skipped_sources,
                "full_corpus":false,"current_turn_excluded":true,"checkpoint_messages_included":true,"raw_message_table_included":true,
                "checkpoint_turns_loaded":additional.checkpoint_turns_loaded,
                "checkpoint_messages_loaded":additional.checkpoint_messages_loaded,
                "raw_messages_loaded":additional.raw_messages_loaded,"raw_message_limit":sources::RAW_MESSAGE_LIMIT,
                "raw_message_window_full_older_may_be_omitted":additional.raw_message_window_full,
                "additional_sources_loaded":additional_source_count,"additional_sources_scanned":additional_sources_scanned,
                "raw_message_turn_lineage":"unavailable_in_public_store_history_api",
                "policy_messages_excluded":additional.policy_messages_excluded},
            "query_semantics":"literal_phrase_ascii_case_insensitive_unicode_exact",
            "ranking":"bounded_occurrence_count_then_recency","trust":"untrusted_data"}),
        )
    }
    fn describe(&self, scope: &Scope, args: Items) -> Result<Value> {
        validate_items(&args.item_ids)?;
        let items = args
            .item_ids
            .iter()
            .map(|id| self.evidence(scope, id).map(|e| e.metadata()))
            .collect::<Result<Vec<_>>>()?;
        Ok(json!({"source_count":items.len(),"items":items,"trust":"untrusted_data"}))
    }
    fn expand(&self, scope: &Scope, args: Expand) -> Result<Value> {
        let e = self.evidence(scope, &args.item_id)?;
        ensure!(
            args.content_sha256 == e.sha256,
            "evidence checksum mismatch"
        );
        if let Some(id) = args.turn_id {
            ensure!(e.turn_id.as_deref() == Some(&id), "source turn mismatch");
        }
        if let Some(id) = args.tool_id {
            ensure!(e.tool_id.as_deref() == Some(&id), "source tool mismatch");
        }
        ensure!(
            args.start_byte <= args.end_byte && args.end_byte <= e.text.len(),
            "invalid byte interval"
        );
        ensure!(
            args.end_byte - args.start_byte <= self.config.max_expand_bytes,
            "expansion byte bound exceeded"
        );
        ensure!(
            e.text.is_char_boundary(args.start_byte) && e.text.is_char_boundary(args.end_byte),
            "byte interval splits UTF-8 codepoint"
        );
        let mut out = e.metadata();
        out["start_byte"] = json!(args.start_byte);
        out["end_byte"] = json!(args.end_byte);
        out["text"] = json!(&e.text[args.start_byte..args.end_byte]);
        out["truncated"] = json!(args.end_byte < e.text.len());
        out["next_start_byte"] = if args.end_byte < e.text.len() {
            json!(args.end_byte)
        } else {
            Value::Null
        };
        Ok(out)
    }
}
fn validate_items(ids: &[String]) -> Result<()> {
    ensure!(
        !ids.is_empty() && ids.len() <= 20,
        "item_ids must contain 1..20 receipt IDs"
    );
    ensure!(
        ids.iter().collect::<std::collections::HashSet<_>>().len() == ids.len(),
        "duplicate evidence IDs are not separate sources"
    );
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Search {
    query: String,
    limit: Option<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Items {
    item_ids: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expand {
    item_id: String,
    content_sha256: String,
    start_byte: usize,
    end_byte: usize,
    turn_id: Option<String>,
    tool_id: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Shunt {
    query: String,
    item_ids: Vec<String>,
}
#[async_trait]
impl ToolExecutor for HistoryTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        if !self.config.enabled {
            return vec![];
        }
        let string = json!({"type":"string"});
        let ids = json!({"type":"array","items":{"type":"string"},"minItems":1,"maxItems":20});
        let entries = vec![
            (
                "history_search",
                "Search bounded exact-scope conversation history. Literal phrase, no global fallback; not explicit memory records.",
                json!({"query":string,"limit":{"type":"integer","minimum":1,"maximum":self.config.max_results}}),
                vec!["query"],
                false,
            ),
            (
                "memory_describe",
                "Describe immutable retained history receipt sources and checksums.",
                json!({"item_ids":ids}),
                vec!["item_ids"],
                false,
            ),
            (
                "memory_expand",
                "Read exact UTF-8 byte interval from history receipt with checksum. Never reruns source tool.",
                json!({"item_id":string,"content_sha256":string,"start_byte":{"type":"integer","minimum":0},"end_byte":{"type":"integer","minimum":0},"turn_id":string,"tool_id":string}),
                vec!["item_id", "content_sha256", "start_byte", "end_byte"],
                false,
            ),
            (
                "context_shunt",
                "One separately budgeted no-tools summary completion over scoped evidence. Billable side effect; outer policy approval required.",
                json!({"query":string,"item_ids":ids}),
                vec!["query", "item_ids"],
                true,
            ),
        ];
        entries.into_iter().filter(|(name,_,_,_,_)| *name!="context_shunt" || self.summary.is_some()).map(|(name,description,properties,required,side_effect)|ToolDefinition {
            name:name.into(),description:description.into(),parameters:json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),side_effect
        }).collect()
    }
    fn configuration_fingerprint(&self) -> String {
        hash_text(&json!({"history":self.config,"summary":self.summary.as_ref().map(|s|s.fingerprint()),"scope":"tenant_user_channel_session_exact_v1"}).to_string())
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.gate(ctx)?;
        ensure!(
            serde_json::to_vec(&call.arguments)?.len() <= 8192,
            "arguments too large"
        );
        match call.name.as_str() {
            "history_search" => {
                let _: Search = serde_json::from_value(call.arguments.clone())?;
            }
            "memory_describe" => {
                let _: Items = serde_json::from_value(call.arguments.clone())?;
            }
            "memory_expand" => {
                let _: Expand = serde_json::from_value(call.arguments.clone())?;
            }
            "context_shunt" => {
                let _: Shunt = serde_json::from_value(call.arguments.clone())?;
                ensure!(
                    self.summary.is_some(),
                    "context_shunt summary backend disabled"
                );
            }
            _ => bail!("unknown history tool"),
        }
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.preflight(ctx, call).await?;
        match call.name.as_str() {
            "history_search" => self.search(ctx, serde_json::from_value(call.arguments.clone())?),
            "memory_describe" => {
                self.describe(&ctx.scope, serde_json::from_value(call.arguments.clone())?)
            }
            "memory_expand" => {
                self.expand(&ctx.scope, serde_json::from_value(call.arguments.clone())?)
            }
            "context_shunt" => {
                let args: Shunt = serde_json::from_value(call.arguments.clone())?;
                validate_items(&args.item_ids)?;
                ensure!(
                    !args.query.trim().is_empty() && args.query.len() <= 1024,
                    "invalid summary query"
                );
                let evidence = args
                    .item_ids
                    .iter()
                    .map(|id| self.evidence(&ctx.scope, id))
                    .collect::<Result<Vec<_>>>()?;
                self.summary
                    .as_ref()
                    .ok_or_else(|| anyhow!("context_shunt summary backend disabled"))?
                    .summarize(ctx, call, &args.query, evidence)
                    .await
            }
            _ => bail!("unknown history tool"),
        }
    }
}
#[cfg(test)]
mod tests;
