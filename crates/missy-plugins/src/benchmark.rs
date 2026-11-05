//! Explicit read-only provider probes. No tools, recursive prompts or retries.
use anyhow::{Context, Result, ensure};
use missy_core::{Message, Provider, Usage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkConfig {
    pub ledger_path: PathBuf,
    pub prompt: String,
    pub iterations: u8,
    pub max_calls: u16,
    pub max_total_cost_microusd: u64,
    pub timeout_seconds: u64,
}
pub struct BenchmarkTarget {
    pub provider: Arc<dyn Provider>,
    pub max_output_tokens: u32,
    pub input_microusd_per_million: u64,
    pub output_microusd_per_million: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkObservation {
    pub provider: String,
    pub iteration: u8,
    pub elapsed_ms: u64,
    pub status: String,
    pub usage: Usage,
    pub output_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkReceipt {
    pub request_id: String,
    pub configuration_hash: String,
    pub reserved_microusd: u64,
    pub status: String,
    pub observations: Vec<BenchmarkObservation>,
    pub deduplicated: bool,
}
fn append(file: &mut File, receipt: &BenchmarkReceipt) -> Result<()> {
    serde_json::to_writer(&mut *file, receipt)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn open(path: &Path) -> Result<File> {
    ensure!(
        path.is_absolute(),
        "absolute private benchmark ledger required"
    );
    let parent = path.parent().context("benchmark ledger parent missing")?;
    ensure!(
        std::fs::canonicalize(parent)? == parent,
        "canonical benchmark ledger parent required"
    );
    let mut current = PathBuf::new();
    for part in parent.components() {
        current.push(part.as_os_str());
        ensure!(
            !std::fs::symlink_metadata(&current)?
                .file_type()
                .is_symlink(),
            "ledger ancestor symlink denied"
        );
    }
    let metadata = std::fs::symlink_metadata(parent)?;
    ensure!(
        metadata.is_dir()
            && metadata.uid() == unsafe { libc::geteuid() }
            && metadata.mode() & 0o077 == 0,
        "benchmark ledger parent must be owned private directory"
    );
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)?;
    let m = file.metadata()?;
    ensure!(
        m.is_file()
            && m.nlink() == 1
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o077 == 0,
        "unsafe benchmark ledger"
    );
    ensure!(
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "benchmark already active"
    );
    ensure!(
        m.len() <= 4 * 1024 * 1024,
        "benchmark ledger capacity exceeded"
    );
    Ok(file)
}
/// One explicit invocation, at most 24 calls. Reserve every planned call durably
/// before the first request; failed/unknown billing is never refunded or retried.
pub async fn benchmark_providers(
    targets: BTreeMap<String, BenchmarkTarget>,
    config: BenchmarkConfig,
    request_id: &str,
    cancel: CancellationToken,
) -> Result<BenchmarkReceipt> {
    ensure!(
        !request_id.is_empty()
            && request_id.len() <= 128
            && request_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-:.".contains(&b)),
        "bounded request identity required"
    );
    ensure!(
        !targets.is_empty()
            && targets.len() <= 8
            && (1..=3).contains(&config.iterations)
            && (1..=24).contains(&config.max_calls)
            && (1..=120).contains(&config.timeout_seconds)
            && config.max_total_cost_microusd > 0,
        "invalid benchmark bounds"
    );
    ensure!(
        !config.prompt.trim().is_empty() && config.prompt.len() <= 4096,
        "benchmark prompt must be 1..4096 bytes"
    );
    let calls = targets.len() * config.iterations as usize;
    ensure!(
        calls <= config.max_calls as usize,
        "benchmark call budget exceeded"
    );
    let identities = targets
        .iter()
        .map(|(name, t)| {
            (
                name,
                t.provider.configuration_fingerprint(),
                t.max_output_tokens,
                t.input_microusd_per_million,
                t.output_microusd_per_million,
            )
        })
        .collect::<Vec<_>>();
    let configuration_hash =
        hex::encode(Sha256::digest(serde_json::to_vec(&(&identities, &config))?));
    let mut reservation = 0u64;
    for target in targets.values() {
        ensure!(
            (1..=4096).contains(&target.max_output_tokens),
            "benchmark provider output bound required"
        );
        let input = (config.prompt.len() as u128 + 512) * target.input_microusd_per_million as u128;
        let output = target.max_output_tokens as u128 * target.output_microusd_per_million as u128;
        let call = u64::try_from((input + output).div_ceil(1_000_000))?;
        reservation = reservation
            .checked_add(
                call.checked_mul(config.iterations as u64)
                    .context("benchmark reservation overflow")?,
            )
            .context("benchmark reservation overflow")?;
    }
    let mut file = open(&config.ledger_path)?;
    let mut text = String::new();
    (&mut file)
        .take(4 * 1024 * 1024 + 1)
        .read_to_string(&mut text)?;
    ensure!(
        text.len() <= 4 * 1024 * 1024,
        "benchmark ledger exceeds bound"
    );
    let mut latest = BTreeMap::<String, BenchmarkReceipt>::new();
    for line in text.lines() {
        let receipt: BenchmarkReceipt =
            serde_json::from_str(line).context("benchmark ledger corrupt; no replay")?;
        latest.insert(receipt.request_id.clone(), receipt);
    }
    if let Some(existing) = latest.get(request_id) {
        ensure!(
            existing.configuration_hash == configuration_hash,
            "benchmark request payload/configuration conflict"
        );
        let mut receipt = existing.clone();
        receipt.deduplicated = true;
        return Ok(receipt);
    }
    let total = latest.values().try_fold(0u64, |n, r| {
        n.checked_add(r.reserved_microusd)
            .context("benchmark ledger total overflow")
    })?;
    ensure!(
        total
            .checked_add(reservation)
            .is_some_and(|n| n <= config.max_total_cost_microusd),
        "benchmark lifetime reservation budget exhausted"
    );
    let mut receipt = BenchmarkReceipt {
        request_id: request_id.into(),
        configuration_hash,
        reserved_microusd: reservation,
        status: "pending_outcome_unconfirmed".into(),
        observations: Vec::new(),
        deduplicated: false,
    };
    append(&mut file, &receipt)?;
    let messages = vec![
        Message::text(
            "system",
            "Answer the supplied read-only benchmark prompt. No tools or follow-up turns are available.",
        ),
        Message::text("user", &config.prompt),
    ];
    for (name, target) in targets {
        for iteration in 0..config.iterations {
            if cancel.is_cancelled() {
                receipt.status = "cancelled_pending_calls_not_dispatched".into();
                append(&mut file, &receipt)?;
                return Ok(receipt);
            }
            let start = tokio::time::Instant::now();
            let child = cancel.child_token();
            let result = tokio::select! {_=cancel.cancelled()=>Err(anyhow::anyhow!("benchmark cancelled")),result=tokio::time::timeout(Duration::from_secs(config.timeout_seconds),target.provider.complete(&messages,&[],child.clone()))=>result.context("benchmark provider timeout").and_then(|r|r)};
            child.cancel();
            let elapsed_ms = start.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let mut observation = BenchmarkObservation {
                provider: name.clone(),
                iteration,
                elapsed_ms,
                status: "outcome_or_cost_unconfirmed".into(),
                usage: Usage {
                    cost_known: false,
                    ..Usage::default()
                },
                output_bytes: 0,
            };
            if let Ok(completion) = result {
                observation.usage = completion.usage;
                observation.output_bytes = completion.message.content.len();
                if completion.message.role == "assistant"
                    && completion.message.tool_calls.is_empty()
                    && completion.message.tool_call_id.is_none()
                    && !completion.message.content.trim().is_empty()
                    && completion.message.content.len() <= 65536
                    && matches!(
                        completion.finish_reason.as_str(),
                        "stop" | "end_turn" | "completed"
                    )
                    && completion.usage.cost_known
                {
                    observation.status = "completed".into();
                }
            }
            let confirmed = observation.status == "completed";
            receipt.observations.push(observation);
            append(&mut file, &receipt)?;
            if !confirmed {
                receipt.status = "stopped_outcome_or_cost_unconfirmed".into();
                append(&mut file, &receipt)?;
                return Ok(receipt);
            }
        }
    }
    receipt.status = "completed".into();
    append(&mut file, &receipt)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use missy_core::{Completion, ToolDefinition};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Fixture(AtomicUsize, bool);
    #[async_trait]
    impl Provider for Fixture {
        fn name(&self) -> &str {
            "fixture"
        }
        async fn complete(
            &self,
            _: &[Message],
            tools: &[ToolDefinition],
            _: CancellationToken,
        ) -> Result<Completion> {
            assert!(tools.is_empty());
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(Completion {
                message: Message::text("assistant", "measured output"),
                finish_reason: "stop".into(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 3,
                    cost_microusd: 1,
                    cost_known: !self.1,
                },
            })
        }
    }
    fn config(path: PathBuf) -> BenchmarkConfig {
        BenchmarkConfig {
            ledger_path: path,
            prompt: "literal benchmark".into(),
            iterations: 1,
            max_calls: 2,
            max_total_cost_microusd: 100,
            timeout_seconds: 1,
        }
    }
    fn pool(provider: Arc<Fixture>) -> BTreeMap<String, BenchmarkTarget> {
        BTreeMap::from([(
            "fixture".into(),
            BenchmarkTarget {
                provider,
                max_output_tokens: 10,
                input_microusd_per_million: 1,
                output_microusd_per_million: 1,
            },
        )])
    }
    #[tokio::test]
    async fn actual_counts_durable_dedup_and_payload_conflict() {
        let root = tempfile::tempdir().unwrap();
        std::fs::set_permissions(
            root.path(),
            std::os::unix::fs::PermissionsExt::from_mode(0o700),
        )
        .unwrap();
        let c = config(root.path().join("benchmark.jsonl"));
        let p = Arc::new(Fixture(AtomicUsize::new(0), false));
        let first = benchmark_providers(pool(p.clone()), c.clone(), "r", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(first.status, "completed");
        let second = benchmark_providers(pool(p.clone()), c.clone(), "r", CancellationToken::new())
            .await
            .unwrap();
        assert!(second.deduplicated);
        assert_eq!(p.0.load(Ordering::SeqCst), 1);
        let mut changed = c;
        changed.prompt = "changed".into();
        assert!(
            benchmark_providers(pool(p), changed, "r", CancellationToken::new())
                .await
                .is_err()
        );
    }
    #[tokio::test]
    async fn unknown_billing_stops_and_never_replays() {
        let root = tempfile::tempdir().unwrap();
        std::fs::set_permissions(
            root.path(),
            std::os::unix::fs::PermissionsExt::from_mode(0o700),
        )
        .unwrap();
        let mut c = config(root.path().join("benchmark.jsonl"));
        c.iterations = 2;
        let p = Arc::new(Fixture(AtomicUsize::new(0), true));
        let receipt =
            benchmark_providers(pool(p.clone()), c.clone(), "r", CancellationToken::new())
                .await
                .unwrap();
        assert_eq!(receipt.status, "stopped_outcome_or_cost_unconfirmed");
        assert_eq!(p.0.load(Ordering::SeqCst), 1);
        benchmark_providers(pool(p.clone()), c, "r", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(p.0.load(Ordering::SeqCst), 1);
    }
}
