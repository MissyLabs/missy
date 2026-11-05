use super::*;
use missy_core::{ToolCall, Usage};
use missy_store::Checkpoint;
use serde_json::json;
use std::{os::unix::fs::PermissionsExt, time::Duration};

fn scope() -> Scope {
    Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "c".into(),
        session: "s".into(),
    }
}

#[test]
fn loads_checkpoint_only_transcript_and_message_only_history_with_honest_lineage() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let store = Store::open(dir.path().join("fixture.sqlite")).unwrap();
    let scope = scope();
    let turn = store.admit(&scope, "a", "ordinary prompt").unwrap().turn;
    let lease = store
        .acquire_lease(&scope, &turn.id, "fixture", Duration::from_secs(60))
        .unwrap();
    let system = "PRIVATE_POLICY_NOT_RECALLED";
    let checkpoint = Checkpoint {
        transcript: vec![
            Message::text("system", system),
            Message::text("assistant", "checkpoint-only 水"),
            Message {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "checkpoint-call".into(),
                    name: "fixture".into(),
                    arguments: json!({"needle":"checkpoint-call-only"}),
                }],
                tool_call_id: None,
            },
            Message::tool("checkpoint-call", "checkpoint-tool-output"),
        ],
        steps: 1,
        usage: Usage::default(),
        provider_context: json!({}),
        context_hash: "bound-fixture".into(),
        system_prompt_hash: hash_text(system),
    };
    store.save_checkpoint(&scope, &lease, &checkpoint).unwrap();
    store
        .append_message(
            &scope,
            &lease,
            &Message::text("user", "message-table-only é"),
        )
        .unwrap();
    store
        .append_message(
            &scope,
            &lease,
            &Message::tool("raw-call", "message-table-tool-only"),
        )
        .unwrap();
    // No finish: the checkpoint has never been published to the message table.
    let result = additional_sources(&store, &scope, std::slice::from_ref(&turn)).unwrap();
    assert_eq!(result.checkpoint_turns_loaded, 1);
    assert_eq!(result.checkpoint_messages_loaded, 3);
    assert_eq!(result.raw_messages_loaded, 2);
    assert!(!result.raw_message_window_full);
    assert_eq!(result.policy_messages_excluded, 1);
    let cp = result
        .sources
        .iter()
        .find(|s| s.text == "checkpoint-only 水")
        .unwrap();
    assert_eq!(cp.turn_id.as_deref(), Some(turn.id.as_str()));
    let raw = result
        .sources
        .iter()
        .find(|s| s.text == "message-table-only é")
        .unwrap();
    assert!(raw.turn_id.is_none());
    assert!(raw.source.starts_with("history_message:"));
    assert_eq!(
        result
            .sources
            .iter()
            .find(|s| s.text == "message-table-tool-only")
            .unwrap()
            .tool_id
            .as_deref(),
        Some("raw-call")
    );
    assert!(
        result
            .sources
            .iter()
            .any(|s| s.tool_id.as_deref() == Some("checkpoint-call")
                && s.text.contains("checkpoint-call-only"))
    );
    assert!(!result.sources.iter().any(|s| s.text.contains(system)));
    let rerun = additional_sources(&store, &scope, &[turn]).unwrap();
    assert_eq!(
        rerun.sources.iter().map(|s| &s.source).collect::<Vec<_>>(),
        result.sources.iter().map(|s| &s.source).collect::<Vec<_>>()
    );
}

#[test]
fn every_scope_component_blocks_checkpoint_and_message_table_reads() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let store = Store::open(dir.path().join("fixture.sqlite")).unwrap();
    let scope = scope();
    let turn = store.admit(&scope, "a", "scope secret").unwrap().turn;
    let lease = store
        .acquire_lease(&scope, &turn.id, "fixture", Duration::from_secs(60))
        .unwrap();
    store
        .append_message(&scope, &lease, &Message::text("user", "raw scope secret"))
        .unwrap();
    for component in 0..4 {
        let mut other = scope.clone();
        match component {
            0 => other.tenant.push('2'),
            1 => other.user.push('2'),
            2 => other.channel.push('2'),
            _ => other.session.push('2'),
        };
        assert!(additional_sources(&store, &other, std::slice::from_ref(&turn)).is_err());
        assert!(
            additional_sources(&store, &other, &[])
                .unwrap()
                .sources
                .is_empty()
        );
    }
}
