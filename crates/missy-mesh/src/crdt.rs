use crate::{
    Purpose, VerifiedEnvelope,
    envelope::{CLOCK_SKEW_SECONDS, validate_hex},
    json,
};
use anyhow::{Result, ensure};
use missy_core::Scope;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const MAX_CAUSAL_COUNTER: u32 = 1_000_000;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MemoryUpdate {
    pub key: String,
    pub value: Value,
    pub deleted: bool,
    pub timestamp: u64,
    /// Bounded same-second causal counter. A same identity/stamp conflict uses
    /// canonical value bytes as final tie break, never arrival order.
    pub counter: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub scope: Scope,
    pub peer_id: String,
    pub update: MemoryUpdate,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LwwMap {
    entries: BTreeMap<String, Entry>,
}
impl MemoryUpdate {
    fn validate(&self, now: u64) -> Result<()> {
        ensure!(
            !self.key.is_empty()
                && self.key.len() <= 256
                && !self.key.chars().any(char::is_control),
            "invalid CRDT key"
        );
        ensure!(
            self.timestamp > 0
                && self.timestamp <= now.saturating_add(CLOCK_SKEW_SECONDS)
                && self.counter <= MAX_CAUSAL_COUNTER,
            "invalid CRDT causal stamp"
        );
        ensure!(
            !self.deleted || self.value.is_null(),
            "tombstone must have null value"
        );
        ensure!(
            json::canonical(&self.value)?.len() <= 16 * 1024,
            "CRDT value too large"
        );
        Ok(())
    }
}
impl LwwMap {
    fn map_key(scope: &Scope, key: &str) -> Result<String> {
        Ok(serde_json::to_string(&(scope, key))?)
    }
    pub fn get(&self, scope: &Scope, key: &str) -> Option<&Value> {
        self.entries
            .get(&Self::map_key(scope, key).ok()?)
            .filter(|e| !e.update.deleted)
            .map(|e| &e.update.value)
    }
    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.values()
    }
    pub fn apply(&mut self, verified: &VerifiedEnvelope, now: u64) -> Result<bool> {
        let body = verified.body();
        ensure!(body.purpose == Purpose::MemoryUpdate, "not a memory update");
        let update: MemoryUpdate = serde_json::from_value(body.payload.clone())?;
        ensure!(
            update.timestamp == body.timestamp,
            "CRDT timestamp must equal signed envelope timestamp"
        );
        self.apply_entry(
            Entry {
                scope: body.scope.clone(),
                peer_id: body.sender.clone(),
                update,
            },
            now,
        )
    }
    /// Merge only trusted local snapshots. The network entry point requires
    /// a VerifiedEnvelope; it does not deserialize peer-provided snapshots.
    pub(crate) fn apply_entry(&mut self, entry: Entry, now: u64) -> Result<bool> {
        entry.scope.validate()?;
        validate_hex(&entry.peer_id, 64)?;
        entry.update.validate(now)?;
        let key = Self::map_key(&entry.scope, &entry.update.key)?;
        let dominates = match self.entries.get(&key) {
            Some(previous) => order(&entry)? > order(previous)?,
            None => {
                ensure!(
                    self.entries.len() < 4096,
                    "CRDT capacity exhausted; tombstones must not be silently evicted"
                );
                true
            }
        };
        if dominates {
            self.entries.insert(key, entry);
        }
        Ok(dominates)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(self.entries.len() <= 4096, "CRDT capacity exceeded");
        for (key, entry) in &self.entries {
            ensure!(
                key == &Self::map_key(&entry.scope, &entry.update.key)?,
                "invalid CRDT storage index"
            );
            entry.scope.validate()?;
            validate_hex(&entry.peer_id, 64)?;
            entry.update.validate(u64::MAX)?;
        }
        Ok(())
    }
}
fn order(e: &Entry) -> Result<(u64, &str, u32, bool, Vec<u8>)> {
    Ok((
        e.update.timestamp,
        &e.peer_id,
        e.update.counter,
        e.update.deleted,
        json::canonical(&e.update.value)?,
    ))
}
