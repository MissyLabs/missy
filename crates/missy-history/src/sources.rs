//! Read-only supplementary history sources through public Store APIs only.
use anyhow::{Result, anyhow};
use missy_core::{Message, Scope};
use missy_store::{Store, Turn, hash_text};

pub(crate) const RAW_MESSAGE_LIMIT: usize = 10_000;

/// Turn lineage is deliberately absent when the public Store API does not expose
/// it. A raw message hash is source identity, NOT a guessed database row/turn ID.
pub(crate) struct HistorySource {
    pub turn_id: Option<String>,
    pub tool_id: Option<String>,
    pub source: String,
    pub created_at: i64,
    pub text: String,
}

pub(crate) struct AdditionalSources {
    pub sources: Vec<HistorySource>,
    pub checkpoint_turns_loaded: usize,
    pub checkpoint_messages_loaded: usize,
    pub raw_messages_loaded: usize,
    pub raw_message_window_full: bool,
    pub policy_messages_excluded: usize,
}

/// The supplied turns MUST come from the caller's exact-scope turn window.
/// Store rechecks scope on every call, including checkpoint event and transcript
/// reads. Checkpoint retrieval preserves checksum/context validation and does
/// not alter the continuation API or expose the private SQLite connection.
pub(crate) fn additional_sources(
    store: &Store,
    scope: &Scope,
    turns: &[Turn],
) -> Result<AdditionalSources> {
    scope.validate()?;
    let mut result = AdditionalSources {
        sources: Vec::new(),
        checkpoint_turns_loaded: 0,
        checkpoint_messages_loaded: 0,
        raw_messages_loaded: 0,
        raw_message_window_full: false,
        policy_messages_excluded: 0,
    };
    for turn in turns {
        let events = store.events(scope, &turn.id)?;
        let Some(event) = events.iter().rev().find(|event| event.kind == "checkpoint") else {
            continue;
        };
        let context_hash = event.payload["context_hash"]
            .as_str()
            .ok_or_else(|| anyhow!("checkpoint event missing context binding"))?;
        let Some(checkpoint) = store.checkpoint(scope, &turn.id, context_hash)? else {
            // A concurrent retention pass may have removed the source. No fake
            // transcript and no replay/reconstruction from provider calls.
            continue;
        };
        result.checkpoint_turns_loaded += 1;
        for (index, message) in checkpoint.transcript.into_iter().enumerate() {
            if policy_message(&message) {
                result.policy_messages_excluded += 1;
                continue;
            }
            result.checkpoint_messages_loaded += 1;
            append_message(
                &mut result.sources,
                &message,
                Some(&turn.id),
                turn.created_at,
                &format!("checkpoint_message:{index}"),
            )?;
        }
    }
    let messages = store.history(scope, RAW_MESSAGE_LIMIT)?;
    result.raw_messages_loaded = messages.len();
    result.raw_message_window_full = messages.len() == RAW_MESSAGE_LIMIT;
    // Newest first so an exhausted byte budget retains the recent end of the
    // raw history window, matching the turn search window's order.
    for message in messages.into_iter().rev() {
        if policy_message(&message) {
            result.policy_messages_excluded += 1;
            continue;
        }
        let message_hash = hash_text(&serde_json::to_string(&message)?);
        append_message(
            &mut result.sources,
            &message,
            None,
            0,
            &format!("history_message:{message_hash}"),
        )?;
    }
    Ok(result)
}

fn policy_message(message: &Message) -> bool {
    matches!(message.role.as_str(), "system" | "developer")
}

fn append_message(
    sources: &mut Vec<HistorySource>,
    message: &Message,
    turn_id: Option<&str>,
    created_at: i64,
    identity: &str,
) -> Result<()> {
    if !message.content.is_empty() {
        sources.push(HistorySource {
            turn_id: turn_id.map(str::to_owned),
            tool_id: message.tool_call_id.clone(),
            source: format!("{identity}:{}:content", message.role),
            created_at,
            text: message.content.clone(),
        });
    }
    for (index, call) in message.tool_calls.iter().enumerate() {
        sources.push(HistorySource {
            turn_id: turn_id.map(str::to_owned),
            tool_id: Some(call.id.clone()),
            source: format!("{identity}:{}:tool_call:{index}", message.role),
            created_at,
            text: serde_json::to_string(call)?,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "sources_tests.rs"]
mod tests;
