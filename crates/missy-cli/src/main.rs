//! Explicit, isolated Rust entrypoint. Never auto-imports legacy ~/.missy.
mod admin;
mod config_ops;
mod desktop_ocr;
mod extensions;
mod operator;
mod production;
mod prompt_resolver;
mod provider_ops;
mod screencast_adapter;
mod skills_ops;
mod voice;
mod watchdog;
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use missy_core::{Config, Scope};
use missy_providers::ProviderPool;
use missy_runtime::Runtime;
use missy_store::Store;
use missy_tools::BuiltinTools;
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
static TELEMETRY_HANDLES: std::sync::Mutex<Vec<missy_operations::TelemetryRuntime>> =
    std::sync::Mutex::new(Vec::new());
static RUNTIME_HANDLES: std::sync::Mutex<Vec<std::sync::Weak<Runtime>>> =
    std::sync::Mutex::new(Vec::new());
/// Dedicated operator-selected worker profiles do not participate in ambient
/// task/tier routing. Preserve budgets, pricing, weights and circuit settings.
fn explicit_profile(mut profile: missy_core::ProviderConfig) -> missy_core::ProviderConfig {
    profile.routing.tiers.clear();
    profile.routing.tasks.clear();
    profile
}

async fn shutdown_runtimes(bound: std::time::Duration) -> Result<()> {
    let runtimes: Vec<_> = RUNTIME_HANDLES
        .lock()
        .map_err(|_| anyhow::anyhow!("runtime lifecycle lock poisoned"))?
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .collect();
    let mut drains = tokio::task::JoinSet::new();
    for runtime in runtimes {
        drains.spawn(async move { runtime.shutdown(bound).await });
    }
    while let Some(result) = drains.join_next().await {
        result??;
    }
    Ok(())
}

#[derive(Parser)]
#[command(
    name = "missy-rs",
    version,
    about = "Missy, a native Rust agent runtime. Separate state; explicit capabilities."
)]
struct Cli {
    #[arg(long, global = true, default_value = "missy-rust.yaml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Operator {
        #[command(subcommand)]
        command: operator::OperatorCommand,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Admin {
        #[command(subcommand)]
        command: admin::AdminCommand,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Screencast {
        #[command(subcommand)]
        command: screencast_adapter::ScreencastCommand,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Skills {
        #[command(subcommand)]
        command: skills_ops::SkillsCommand,
    },
    Triggers {
        #[command(subcommand)]
        command: TriggerCommand,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Write a private, default-deny configuration without touching legacy Missy.
    Init,
    /// Validate the configuration and policy without connecting to providers.
    ConfigCheck,
    /// Explicit native provider profiles, credential checks and OAuth refresh.
    Providers {
        #[command(subcommand)]
        command: provider_ops::ProviderCommand,
    },
    /// Offline private snapshots and redacted change plans. Never reloads a service.
    Config {
        #[command(subcommand)]
        command: config_ops::ConfigCommand,
    },
    /// Copy-only offline legacy migration. Always inspect a plan before apply.
    Migrate {
        #[command(subcommand)]
        command: MigrationCommand,
    },
    /// Run one durable request. Reuse request-id to safely retrieve its result.
    Ask {
        prompt: String,
        #[command(flatten)]
        scope: ScopeArgs,
        #[arg(long)]
        request_id: Option<String>,
    },
    /// Interactive chat with a persistent scoped session.
    Run {
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Resume a checkpoint, retaining its original budgets and system lineage.
    Resume {
        turn_id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Retrieve a turn in this exact scope.
    Status {
        turn_id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Request cancellation without needing provider credentials or a new generation.
    Cancel {
        turn_id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Display persisted conversation history for this exact scope.
    History {
        #[command(flatten)]
        scope: ScopeArgs,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// List incomplete turns for this exact scope. Does not execute them.
    Recover {
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Authenticated loopback API and browser console.
    Serve,
    /// Unified native service supervisor. Separate from the Python gateway.
    Gateway {
        #[command(subcommand)]
        command: production::GatewayCommand,
    },
    /// Run the explicitly enabled Discord gateway with strict allowlists.
    Discord,
    /// Explicit consented Discord voice, sharing the authenticated text Gateway.
    DiscordVoice {
        guild: u64,
        channel: u64,
        actor: u64,
        #[arg(long)]
        approval: String,
    },
    /// Read-only authenticated Discord preflight. Does not connect a Gateway or send.
    DiscordProbe,
    /// Offline reconciliation. Never resends uncertain Discord effects.
    DiscordOutbox {
        #[command(subcommand)]
        command: OutboxCommand,
    },
    /// Register native guild slash commands with explicit operator confirmation.
    DiscordRegisterCommands {
        #[arg(long)]
        confirm: bool,
    },
    /// Edge audio service and explicit operator-managed device pairing.
    Voice {
        #[command(subcommand)]
        command: VoiceCommand,
    },
    /// Validate configuration, database, and optional provider readiness without inference.
    Doctor,
    /// Encrypt secrets locally. Values arrive on stdin, never in process arguments.
    Vault {
        #[command(subcommand)]
        command: VaultCommand,
    },
    /// Durable explicitly scoped scheduled requests. No implicit background daemon.
    Schedule {
        #[command(subcommand)]
        command: ScheduleCommand,
    },
    /// Authenticated local webhook intake and operational diagnostics.
    Operations,
    /// Run a bounded literal task DAG through ordinary durable turns.
    Workflow {
        path: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Mesh {
        #[command(subcommand)]
        command: MeshCommand,
    },
    /// Scoped local lexical retrieval and citation-preserving document indexing.
    Retrieval {
        #[command(subcommand)]
        command: RetrievalCommand,
    },
    /// Explicit tenant+user personal memory shared across that user's sessions.
    PersonalMemory {
        #[command(subcommand)]
        command: PersonalCommand,
    },
    /// Operator-reviewed source changes with immutable plan/HEAD pins and rollback.
    Evolve {
        #[command(subcommand)]
        command: EvolutionCommand,
    },
    /// Explicit bounded parallel agent DAG with durable root reservations.
    Agents {
        path: PathBuf,
        #[arg(long)]
        request_id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Hatch {
        answers: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Experiments {
        #[command(subcommand)]
        command: ExperimentCommand,
    },
    Plugins {
        #[command(subcommand)]
        command: PluginCommand,
    },
    /// Explicit read-only provider benchmark, with persistent no-replay reservations.
    Benchmark {
        path: PathBuf,
        #[arg(long)]
        request_id: String,
    },
    /// Inspect and decide exact durable tool approvals.
    Approvals {
        #[command(subcommand)]
        command: ApprovalCommand,
    },
    /// Scoped reviewed persona and system-prompt revision proposals.
    Persona {
        #[command(subcommand)]
        command: PersonaCommand,
    },
}
#[derive(Subcommand)]
enum TriggerCommand {
    /// Inspect exact child scope and uncertain/paused occurrences without execution.
    Receipts {
        #[arg(long)]
        name: String,
    },
    /// Explicit reset only after terminal turns and reconciliation of effects.
    Reset {
        #[arg(long)]
        name: String,
        #[arg(long)]
        confirm: bool,
    },
}
#[derive(Subcommand)]
enum MigrationCommand {
    Verify {
        destination: PathBuf,
    },
    Rollback {
        destination: PathBuf,
        retired: PathBuf,
        #[arg(long)]
        confirm: bool,
    },
    Plan {
        #[command(flatten)]
        inputs: MigrationArgs,
    },
    Apply {
        #[command(flatten)]
        inputs: MigrationArgs,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long)]
        confirm: bool,
    },
}
#[derive(Clone, Args)]
struct MigrationArgs {
    #[arg(long)]
    legacy_config: Option<PathBuf>,
    #[arg(long)]
    legacy_database: Option<PathBuf>,
    #[arg(long)]
    scope_mapping: Option<PathBuf>,
    #[arg(long)]
    backup_root: Option<PathBuf>,
}
impl MigrationArgs {
    fn input(self) -> missy_migrate::MigrationInput {
        missy_migrate::MigrationInput {
            config_path: self.legacy_config,
            database_path: self.legacy_database,
            mapping_path: self.scope_mapping,
            backup_root: self.backup_root,
            ..Default::default()
        }
    }
}
#[derive(Subcommand)]
enum MeshCommand {
    IdentityCreate { path: PathBuf },
    Status,
    Peers,
    Serve,
    PeerRemove { id: String },
    PeerAdd { path: PathBuf },
}
#[derive(Subcommand)]
enum RetrievalCommand {
    Ingest {
        id: String,
        path: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Query {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Get {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Delete {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
}
#[derive(Subcommand)]
enum PluginCommand {
    /// Propose private Rust source as data. Does not compile or execute it.
    SourcePropose {
        source: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    SourceShow {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    SourceReview {
        id: String,
        #[arg(long)]
        hash: String,
        #[arg(long)]
        approve: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Offline pinned compiler build. Result still requires separate binary review and enable.
    SourceBuild {
        id: String,
        #[arg(long)]
        hash: String,
        #[arg(long)]
        build_config: PathBuf,
        #[arg(long)]
        confirm: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Propose {
        manifest: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Review {
        id: String,
        #[arg(long)]
        hash: String,
        #[arg(long)]
        approve: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Enable {
        id: String,
        #[arg(long)]
        hash: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
}
#[derive(Subcommand)]
enum OutboxCommand {
    Inspect,
    Resolve {
        nonce: String,
        #[arg(long)]
        binding: String,
        #[arg(long)]
        operator: String,
        #[arg(long)]
        receipt: Option<String>,
        #[arg(long)]
        evidence: String,
        #[arg(long)]
        confirm: bool,
    },
}
#[derive(Subcommand)]
enum ExperimentCommand {
    Create {
        variants: PathBuf,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Show {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Assign {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Record {
        id: String,
        event_id: String,
        #[arg(long)]
        success: bool,
        #[arg(long)]
        refused: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Compare {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Promote {
        id: String,
        variant: String,
        #[arg(long)]
        experiment_hash: String,
        #[arg(long)]
        base_version: i64,
        #[arg(long)]
        context_hash: String,
        #[arg(long)]
        operator: String,
        #[arg(long)]
        confirm: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
}
#[derive(Subcommand)]
enum EvolutionCommand {
    Propose {
        repo: PathBuf,
        diff: PathBuf,
        title: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Show {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Review {
        id: String,
        #[arg(long)]
        plan_hash: String,
        #[arg(long)]
        operator: String,
        #[arg(long)]
        approve: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Apply {
        id: String,
        #[arg(long)]
        plan_hash: String,
        #[arg(long)]
        head: String,
        #[arg(long)]
        operator: String,
        #[arg(long)]
        confirm: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Rollback {
        id: String,
        #[arg(long)]
        plan_hash: String,
        #[arg(long)]
        head: String,
        #[arg(long)]
        operator: String,
        #[arg(long)]
        confirm: bool,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Audit {
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Recovery {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
}
#[derive(Subcommand)]
enum PersonalCommand {
    Put {
        id: String,
        content: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Get {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Delete {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
}
#[derive(Subcommand)]
enum ApprovalCommand {
    Review {
        id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    List {
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Approve {
        id: String,
        #[arg(long)]
        binding_hash: String,
        #[arg(long)]
        operator: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Deny {
        id: String,
        #[arg(long)]
        binding_hash: String,
        #[arg(long)]
        operator: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
}
#[derive(Subcommand)]
enum PersonaCommand {
    Show {
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Propose {
        kind: String,
        path: PathBuf,
        #[arg(long)]
        base_version: i64,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Proposals {
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Approve {
        id: String,
        #[arg(long)]
        proposal_hash: String,
        #[arg(long)]
        operator: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Deny {
        id: String,
        #[arg(long)]
        proposal_hash: String,
        #[arg(long)]
        operator: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
}
#[derive(Subcommand)]
enum VoiceCommand {
    Serve,
    Devices,
    Pair {
        friendly_name: String,
        #[arg(long, default_value = "")]
        room: String,
    },
    Approve {
        node_id: String,
        #[arg(long)]
        user: String,
        #[arg(long, default_value = "local")]
        tenant: String,
        #[arg(long, default_value = "safe-chat")]
        mode: String,
        #[arg(long)]
        token_out: PathBuf,
    },
    Mode {
        node_id: String,
        mode: String,
    },
    Revoke {
        node_id: String,
    },
}
#[derive(Subcommand)]
enum VaultCommand {
    Set { name: String },
    List,
    Delete { name: String },
}
#[derive(Subcommand)]
enum ScheduleCommand {
    AddText {
        prompt: String,
        expression: String,
        #[arg(long, default_value = "UTC")]
        timezone: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Next {
        expression: String,
        #[arg(long, default_value = "UTC")]
        timezone: String,
    },
    Add {
        prompt: String,
        #[arg(long)]
        every_seconds: u64,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    List {
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Pause {
        job_id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Resume {
        job_id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    Remove {
        job_id: String,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    RunDue {
        #[command(flatten)]
        scope: ScopeArgs,
        #[arg(long, default_value_t = 4)]
        workers: usize,
    },
}
#[derive(Clone, Args)]
struct ScopeArgs {
    #[arg(long, default_value = "local")]
    tenant: String,
    #[arg(long, default_value = "operator")]
    user: String,
    #[arg(long, default_value = "cli")]
    channel: String,
    #[arg(long, default_value = "default")]
    session: String,
}
impl ScopeArgs {
    fn scope(&self) -> Result<Scope> {
        let scope = Scope {
            tenant: self.tenant.clone(),
            user: self.user.clone(),
            channel: self.channel.clone(),
            session: self.session.clone(),
        };
        scope.validate()?;
        Ok(scope)
    }
}

fn load_config(path: &Path) -> Result<Config> {
    let meta =
        std::fs::symlink_metadata(path).context("configuration not found; run missy-rs init")?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 1_048_576 {
        bail!("configuration must be a regular non-symlink file <=1MiB");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.mode() & 0o022 != 0 || meta.uid() != unsafe { libc::geteuid() } {
            bail!("configuration must be owned and not group/world writable");
        }
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    }
    let file = options.open(path).context("could not read configuration")?;
    let opened = file.metadata()?;
    if !opened.is_file() || opened.len() > 1_048_576 {
        bail!("configuration file changed or exceeds limit");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.ino() != opened.ino() || meta.dev() != opened.dev() {
            bail!("configuration changed during open");
        }
    }
    use std::io::Read;
    let mut text = String::new();
    file.take(1_048_577)
        .read_to_string(&mut text)
        .context("configuration is not bounded UTF-8")?;
    if text.len() > 1_048_576 {
        bail!("configuration exceeds limit");
    }
    let mut cfg: Config = serde_yaml_ng::from_str(&text)
        .context("invalid Rust configuration; legacy Python YAML must be explicitly migrated")?;
    cfg.validate()?;
    let base = std::fs::canonicalize(path)?
        .parent()
        .context("config parent missing")?
        .to_path_buf();
    if cfg.state_dir.is_relative() {
        cfg.state_dir = base.join(&cfg.state_dir);
    }
    for provider in &mut cfg.providers {
        if let Some(oauth) = &mut provider.oauth
            && oauth.token_vault.is_relative()
        {
            oauth.token_vault = base.join(&oauth.token_vault);
        }
    }
    for root in cfg
        .policy
        .read_roots
        .iter_mut()
        .chain(cfg.policy.write_roots.iter_mut())
    {
        if root.is_relative() {
            *root = base.join(&*root);
        }
    }
    missy_policy::Policy::new(cfg.policy.clone())?;
    for path in [
        &mut cfg.extensions.integrations_config,
        &mut cfg.extensions.approvals_config,
        &mut cfg.extensions.cli_provider_config,
        &mut cfg.extensions.voice_config,
        &mut cfg.extensions.sandbox_config,
        &mut cfg.extensions.http_mcp_config,
        &mut cfg.extensions.desktop_config,
        &mut cfg.extensions.media_config,
        &mut cfg.extensions.embedding_config,
        &mut cfg.extensions.mesh_config,
        &mut cfg.extensions.operations_config,
        &mut cfg.extensions.capture_config,
        &mut cfg.extensions.specialty_config,
        &mut cfg.extensions.evolution_config,
        &mut cfg.extensions.discord_voice_config,
        &mut cfg.extensions.telemetry_config,
        &mut cfg.extensions.vision_config,
        &mut cfg.extensions.history_config,
        &mut cfg.extensions.discord_tools_config,
        &mut cfg.extensions.plugins_config,
        &mut cfg.extensions.audio_devices_config,
        &mut cfg.extensions.automation_config,
        &mut cfg.extensions.subagents_config,
        &mut cfg.extensions.workloads_config,
        &mut cfg.extensions.triggers_config,
        &mut cfg.extensions.watchdog_config,
        &mut cfg.extensions.screencast_config,
        &mut cfg.extensions.scene_generation_config,
        &mut cfg.extensions.evolution_repo,
    ]
    .into_iter()
    .flatten()
    {
        if path.is_relative() {
            *path = base.join(&*path);
        }
    }
    Ok(cfg)
}

fn load_extension<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    use std::io::Read;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .context("extension configuration unavailable")?;
    let m = file.metadata()?;
    if !m.is_file()
        || m.len() > 1_048_576
        || m.uid() != unsafe { libc::geteuid() }
        || m.mode() & 0o077 != 0
    {
        bail!("extension configuration must be an owned private regular file <=1MiB");
    }
    let mut text = zeroize::Zeroizing::new(String::new());
    file.take(1_048_577).read_to_string(&mut text)?;
    if text.len() > 1_048_576 {
        bail!("extension configuration exceeds limit");
    }
    serde_yaml_ng::from_str(&text).context("invalid extension configuration")
}

fn retrieval(cfg: &Config) -> Result<Arc<missy_retrieval::Retrieval>> {
    if !cfg.extensions.retrieval_enabled {
        bail!("retrieval is disabled; explicitly enable extensions.retrieval_enabled");
    }
    open_store(cfg)?;
    let path = cfg.state_dir.join("retrieval.sqlite3");
    let db = if let Some(config) = &cfg.extensions.embedding_config {
        missy_retrieval::Retrieval::open_with_embedder(
            path,
            Arc::new(missy_embedding::LocalHttpEmbedder::new(load_extension(
                config,
            )?)?),
        )?
    } else {
        missy_retrieval::Retrieval::open(path)?
    };
    Ok(Arc::new(db))
}

fn personal_memory(cfg: &Config) -> Result<Arc<missy_personal::PersonalMemory>> {
    if !cfg.extensions.personal_memory_enabled {
        bail!(
            "cross-session personal memory disabled; enable extensions.personal_memory_enabled explicitly"
        );
    }
    open_store(cfg)?;
    Ok(Arc::new(missy_personal::PersonalMemory::open(
        cfg.state_dir.join("personal.sqlite3"),
    )?))
}

fn initialize(path: &Path) -> Result<()> {
    let cfg = Config {
        state_dir: PathBuf::from("missy-rust-state"),
        ..Config::default()
    };
    let yaml = serde_yaml_ng::to_string(&cfg)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .context("refusing to overwrite existing configuration")?;
    file.write_all(yaml.as_bytes())?;
    file.sync_all()?;
    println!(
        "Created {}. All external capabilities remain denied. Configure a provider and exact allowed origin before asking.",
        path.display()
    );
    Ok(())
}

fn open_store(cfg: &Config) -> Result<Arc<Store>> {
    Ok(Arc::new(Store::open(
        cfg.state_dir.join("runtime.sqlite3"),
    )?))
}
async fn build_runtime(cfg: &Config) -> Result<Arc<Runtime>> {
    build_runtime_with_voice(cfg, None).await
}
async fn build_runtime_with_voice(
    cfg: &Config,
    voice_tools: Option<Arc<missy_discord_voice::DiscordVoiceSayTools>>,
) -> Result<Arc<Runtime>> {
    build_runtime_with_services(cfg, voice_tools, None).await
}
async fn build_runtime_with_services(
    cfg: &Config,
    voice_tools: Option<Arc<missy_discord_voice::DiscordVoiceSayTools>>,
    workloads: Option<Arc<missy_workloads::WorkloadTools>>,
) -> Result<Arc<Runtime>> {
    let configuration_fingerprint = production::fingerprint(cfg)?;
    if (cfg.extensions.automation_config.is_some() || cfg.extensions.subagents_config.is_some())
        && cfg.runtime.max_concurrent_turns < 2
    {
        bail!(
            "orchestration requires max_concurrent_turns >= 2 so a parent cannot hold the only child turn lease"
        );
    }
    if cfg.extensions.scene_generation_config.is_some()
        && (cfg.extensions.media_config.is_none() || cfg.extensions.integrations_config.is_none())
    {
        bail!("scene_generation_config requires media_config and integrations_config");
    }
    if cfg.extensions.embedding_config.is_some() && !cfg.extensions.retrieval_enabled {
        bail!("embedding_config requires retrieval_enabled");
    }
    let store = open_store(cfg)?;
    if let Some(path) = &cfg.extensions.cli_provider_config {
        let mut configured = serde_json::to_value(&cfg.extensions)?;
        let extensions = configured
            .as_object_mut()
            .context("extension map invalid")?;
        extensions.remove("cli_provider_config");
        extensions.remove("telemetry_config");
        if extensions
            .values()
            .any(|v| !v.is_null() && v != &serde_json::Value::Bool(false))
        {
            bail!(
                "CLI provider is explicitly no-tools; configured capability extensions would be ignored"
            );
        }
        let provider = missy_provider_process::CliProvider::new(load_extension(path)?)?;
        return make_runtime(
            cfg,
            store,
            Arc::new(provider),
            Arc::new(extensions::NoTools),
        );
    }
    let providers = ProviderPool::with_routing(
        cfg.providers.clone(),
        cfg.policy.clone(),
        cfg.routing.clone(),
    )?;
    if cfg.runtime.max_cost_microusd > 0 {
        providers.require_known_prices()?;
    }
    let builtins = BuiltinTools::new(cfg.policy.clone(), cfg.runtime.clone(), store.clone())?
        .with_mcp(cfg.mcp.clone())
        .await?;
    let mut executors: Vec<Arc<dyn missy_core::ToolExecutor>> = vec![Arc::new(builtins)];
    if cfg.extensions.personal_memory_enabled {
        executors.push(Arc::new(missy_personal::PersonalTools::new(
            personal_memory(cfg)?,
        )));
    }
    if cfg.extensions.retrieval_enabled {
        executors.push(Arc::new(extensions::RetrievalTools {
            db: retrieval(cfg)?,
            policy: missy_policy::Policy::new(cfg.policy.clone())?,
        }));
    }
    let mut integration_mutations = std::collections::BTreeSet::new();
    if let Some(path) = &cfg.extensions.discord_voice_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("Discord voice tools require exact durable approvals");
        }
        let tools = if let Some(tools) = voice_tools {
            tools
        } else {
            let app: voice::DiscordVoiceApplication = load_extension(path)?;
            Arc::new(missy_discord_voice::DiscordVoiceSayTools::pending(
                app.server,
                Arc::new(extensions::OuterGuardVerified),
            )?)
        };
        integration_mutations.extend(
            tools
                .definitions()
                .into_iter()
                .filter(|d| d.side_effect)
                .map(|d| d.name),
        );
        executors.push(tools);
    }
    let audio = if let Some(path) = &cfg.extensions.audio_devices_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("audio devices require exact durable approvals");
        }
        let mut audio = missy_voice::AudioDeviceTools::new(load_extension(path)?)?
            .with_approval(Arc::new(extensions::OuterGuardVerified));
        if let Some(path) = &cfg.extensions.voice_config {
            let voice: voice::VoiceApplication = load_extension(path)?;
            audio = audio
                .with_tts(Arc::new(missy_voice::CommandTts::new(voice.tts)?))
                .with_stt(Arc::new(missy_voice::CommandStt::new(voice.stt)?));
        }
        integration_mutations.extend(audio.definitions().into_iter().map(|d| d.name));
        let audio = Arc::new(audio);
        executors.push(audio.clone());
        Some(audio)
    } else {
        None
    };
    if let Some(path) = &cfg.extensions.history_config {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct SummaryApplication {
            provider: missy_core::ProviderConfig,
            policy: missy_history::SummaryPolicy,
        }
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct HistoryApplication {
            history: missy_history::HistoryConfig,
            summary: Option<SummaryApplication>,
        }
        use missy_core::ToolExecutor;
        let mut application: HistoryApplication = load_extension(path)?;
        application.history.retention_path = Some(cfg.state_dir.join("history-evidence.sqlite3"));
        let mut history = missy_history::HistoryTools::new(store.clone(), application.history)?;
        if let Some(mut summary) = application.summary {
            if cfg.extensions.approvals_config.is_none() {
                bail!("history summarization requires exact durable approvals");
            }
            summary.policy.ledger_path = cfg.state_dir.join("history-summary.jsonl");
            if summary.provider.max_output_tokens > summary.policy.max_output_tokens {
                bail!("summary provider output limit exceeds the explicit summary budget");
            }
            summary.policy.input_microusd_per_million = summary
                .provider
                .input_microusd_per_million
                .context("summary input pricing required")?;
            summary.policy.output_microusd_per_million = summary
                .provider
                .output_microusd_per_million
                .context("summary output pricing required")?;
            history = history.with_summary_backend(missy_history::SummaryBackend::new(
                Arc::new(ProviderPool::new(
                    vec![explicit_profile(summary.provider)],
                    cfg.policy.clone(),
                )?),
                summary.policy,
            )?)?;
        }
        let mutations: std::collections::BTreeSet<_> = history
            .definitions()
            .into_iter()
            .filter(|d| d.side_effect)
            .map(|d| d.name)
            .collect();
        if !mutations.is_empty() && cfg.extensions.approvals_config.is_none() {
            bail!("history context tools require exact durable approvals");
        }
        integration_mutations.extend(mutations);
        executors.push(Arc::new(history));
    }
    if cfg.policy.allow_file_delete {
        if cfg.extensions.approvals_config.is_none() {
            bail!("file deletion requires exact durable approvals");
        }
        integration_mutations.insert("file_delete".to_owned());
    }
    if let Some(path) = &cfg.extensions.discord_tools_config {
        use missy_core::ToolExecutor;
        let discord =
            missy_discord_tools::DiscordTools::new(cfg.policy.clone(), load_extension(path)?)?;
        let mutations: std::collections::BTreeSet<_> = discord
            .definitions()
            .into_iter()
            .filter(|d| d.side_effect)
            .map(|d| d.name)
            .collect();
        if !mutations.is_empty() && cfg.extensions.approvals_config.is_none() {
            bail!("Discord uploads require exact durable approvals");
        }
        integration_mutations.extend(mutations);
        executors.push(Arc::new(discord));
    }
    if let Some(path) = &cfg.extensions.plugins_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("native plugin execution requires exact durable approvals");
        }
        let registry = missy_plugins::ToolRegistry::open(cfg.state_dir.join("plugins"))?;
        let tools = missy_plugins::PluginTools::new(
            registry,
            load_extension(path)?,
            Arc::new(missy_policy::Policy::new(cfg.policy.clone())?),
            Arc::new(extensions::OuterGuardVerified),
        )?;
        integration_mutations.extend(tools.definitions().into_iter().map(|d| d.name));
        executors.push(Arc::new(tools));
    }
    if let Some(path) = &cfg.extensions.http_mcp_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("HTTP MCP tools require exact durable approvals configuration");
        }
        let configs: Vec<missy_mcp_http::HttpMcpConfig> = load_extension(path)?;
        let mcp = missy_mcp_http::HttpMcpTools::connect(
            missy_policy::PolicyHttpClient::new(cfg.policy.clone())?,
            configs,
        )
        .await?;
        integration_mutations.extend(mcp.definitions().into_iter().map(|d| d.name));
        executors.push(Arc::new(mcp));
    }
    let mut scene_integrations = None;
    if let Some(path) = &cfg.extensions.integrations_config {
        use missy_core::ToolExecutor;
        let mut integrations = missy_integrations::IntegrationTools::new(
            missy_policy::PolicyHttpClient::new(cfg.policy.clone())?,
            load_extension(path)?,
        )?;
        if let (Some(audio), Some(path)) = (&audio, &cfg.extensions.voice_config) {
            let voice: voice::VoiceApplication = load_extension(path)?;
            integrations = integrations.with_speech(
                Arc::new(missy_voice::CommandTts::new(voice.tts)?),
                audio.clone(),
            );
        }
        let mutations: std::collections::BTreeSet<_> = integrations
            .definitions()
            .into_iter()
            .filter(|d| d.side_effect)
            .map(|d| d.name)
            .collect();
        if !mutations.is_empty() {
            if cfg.extensions.approvals_config.is_none() {
                bail!("enabled integration mutations require durable approvals configuration");
            }
            integrations = integrations.with_approval(Arc::new(extensions::OuterGuardVerified));
        }
        integration_mutations.extend(mutations);
        if let Some(path) = &cfg.extensions.media_config {
            integrations = integrations.with_artifact_validator(Arc::new(
                extensions::NativeArtifactValidator(missy_media::MediaTools::new(
                    cfg.policy.clone(),
                    load_extension(path)?,
                )?),
            ));
        }
        let integrations = Arc::new(integrations);
        scene_integrations = Some(integrations.clone());
        executors.push(integrations);
    }
    if let Some(path) = &cfg.extensions.desktop_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("desktop capabilities require exact durable approval configuration");
        }
        let mut desktop = missy_desktop::DesktopTools::new(
            load_extension(path)?,
            missy_policy::PolicyHttpClient::new(cfg.policy.clone())?,
            Arc::new(extensions::OuterGuardVerified),
        )?;
        if let Some(path) = &cfg.extensions.vision_config {
            let mut vision: missy_vision::VisionConfig = load_extension(path)?;
            vision.ledger_path = cfg.state_dir.join("vision-budget.sqlite3");
            desktop = desktop.with_screen_interpreter(Arc::new(extensions::VisionInterpreter(
                Arc::new(missy_vision::VisionTools::new(cfg.policy.clone(), vision)?),
            )));
        }
        if let Some(path) = &cfg.extensions.specialty_config
            && let Some(ocr) = desktop_ocr::build(cfg.policy.clone(), load_extension(path)?)?
        {
            desktop = desktop.with_screen_ocr(ocr);
        }
        integration_mutations.extend(desktop.definitions().into_iter().map(|d| d.name));
        executors.push(Arc::new(desktop));
    }
    if let Some(path) = &cfg.extensions.media_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("media capabilities require exact durable approval configuration");
        }
        let mut media = missy_media::MediaTools::new(cfg.policy.clone(), load_extension(path)?)?;
        if let Some(path) = &cfg.extensions.scene_generation_config {
            let integrations = scene_integrations
                .clone()
                .context("scene generation requires integrations_config")?;
            media = media.with_scene_generator(Arc::new(extensions::SceneAdapter {
                integrations,
                grant: load_extension(path)?,
            }));
        }
        integration_mutations.extend(
            media
                .definitions()
                .into_iter()
                .filter(|d| d.side_effect)
                .map(|d| d.name),
        );
        executors.push(Arc::new(media));
    }
    if let Some(path) = &cfg.extensions.vision_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("vision provider tools require exact durable approvals");
        }
        let mut vision: missy_vision::VisionConfig = load_extension(path)?;
        vision.ledger_path = cfg.state_dir.join("vision-budget.sqlite3");
        let vision = missy_vision::VisionTools::new(cfg.policy.clone(), vision)?;
        integration_mutations.extend(vision.definitions().into_iter().map(|d| d.name));
        executors.push(Arc::new(vision));
    }
    if let Some(path) = &cfg.extensions.capture_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("live capture requires exact durable approval configuration");
        }
        let capture = missy_capture::CaptureTools::new(
            cfg.policy.clone(),
            load_extension(path)?,
            Arc::new(extensions::OuterGuardVerified),
        )?;
        integration_mutations.extend(capture.definitions().into_iter().map(|d| d.name));
        executors.push(Arc::new(capture));
    }
    if let Some(path) = &cfg.extensions.specialty_config {
        use missy_core::ToolExecutor;
        let specialty =
            missy_specialty::SpecialtyTools::new(cfg.policy.clone(), load_extension(path)?)?;
        let mutations: std::collections::BTreeSet<_> = specialty
            .definitions()
            .into_iter()
            .filter(|d| d.side_effect)
            .map(|d| d.name)
            .collect();
        if !mutations.is_empty() && cfg.extensions.approvals_config.is_none() {
            bail!("specialty mutations require exact durable approval configuration");
        }
        integration_mutations.extend(mutations);
        executors.push(Arc::new(specialty));
    }
    if let Some(path) = &cfg.extensions.workloads_config {
        use missy_core::ToolExecutor;
        if cfg.extensions.approvals_config.is_none() {
            bail!("workload dispatch requires exact durable approval configuration");
        }
        let workloads = match workloads {
            Some(workloads) => workloads,
            None => Arc::new(missy_workloads::WorkloadTools::open(
                cfg.policy.clone(),
                load_extension(path)?,
                cfg.state_dir.join("workloads"),
            )?),
        };
        integration_mutations.extend(
            workloads
                .definitions()
                .into_iter()
                .filter(|d| d.side_effect)
                .map(|d| d.name),
        );
        executors.push(workloads);
    }
    let mut tools: Arc<dyn missy_core::ToolExecutor> =
        Arc::new(extensions::CompositeTools::new(executors)?);
    if let Some(path) = &cfg.extensions.sandbox_config {
        tools = Arc::new(extensions::SandboxTools::new(
            tools,
            cfg.policy.clone(),
            load_extension(path)?,
        )?);
    }
    if let Some(repo) = &cfg.extensions.evolution_repo {
        if cfg.extensions.approvals_config.is_none() {
            bail!("evolution proposals require durable approval configuration");
        }
        let policy = load_extension(
            cfg.extensions
                .evolution_config
                .as_deref()
                .context("evolution_repo requires evolution_config")?,
        )?;
        tools = Arc::new(missy_evolution::EvolutionTools::new(
            tools,
            missy_evolution::EvolutionRegistry::open(cfg.state_dir.join("evolution.sqlite3"))?,
            policy,
            repo.clone(),
        )?);
        integration_mutations.insert("code_evolve".into());
    }
    let unguarded_tools = tools.clone();
    if let Some(path) = &cfg.extensions.approvals_config {
        let mut guard: missy_governance::GuardConfig = load_extension(path)?;
        if !integration_mutations.is_subset(&guard.sensitive_tool_names) {
            bail!("every integration mutation must be in sensitive_tool_names");
        }
        guard.context_fingerprint = configuration_fingerprint.clone();
        tools = Arc::new(missy_governance::GuardedTools::new(
            tools,
            missy_governance::ApprovalRegistry::open(cfg.state_dir.join("governance.sqlite3"))?,
            guard,
        )?);
    }
    let provider: Arc<dyn missy_core::Provider> = Arc::new(providers);
    let guarded_base_tools = tools.clone();
    if cfg.extensions.automation_config.is_some() || cfg.extensions.subagents_config.is_some() {
        tools = unguarded_tools;
    }
    if let Some(path) = &cfg.extensions.automation_config {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct AutomationApplication {
            enabled: bool,
        }
        let application: AutomationApplication = load_extension(path)?;
        if application.enabled {
            let base = make_runtime(
                cfg,
                store.clone(),
                provider.clone(),
                guarded_base_tools.clone(),
            )?;
            let scheduler =
                missy_automation::Scheduler::open(cfg.state_dir.join("scheduler.sqlite3"), base)?;
            tools = Arc::new(missy_automation::AutomationTools::new(tools, scheduler)?);
        }
    }
    if let Some(path) = &cfg.extensions.subagents_config {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct DelegationApplication {
            policy: missy_subagents::DelegationPolicy,
            limits: missy_subagents::Limits,
        }
        let application: DelegationApplication = load_extension(path)?;
        // Each operator-allowed identity resolves its own configured native
        // provider. Never alias distinct provider names to the fallback pool.
        let mut children = std::collections::BTreeMap::new();
        for name in application
            .policy
            .allowed_providers
            .iter()
            .chain(std::iter::once(&application.policy.provider))
        {
            if children.contains_key(name) {
                continue;
            }
            let selected = cfg
                .providers
                .iter()
                .find(|p| &p.name == name)
                .context("delegation provider must name an explicitly configured provider")?;
            let pool =
                ProviderPool::new(vec![explicit_profile(selected.clone())], cfg.policy.clone())?;
            if cfg.runtime.max_cost_microusd > 0 {
                pool.require_known_prices()?;
            }
            let child = make_runtime(
                cfg,
                store.clone(),
                Arc::new(pool),
                Arc::new(extensions::NoTools),
            )?;
            children.insert(name.clone(), child);
        }
        let pool = missy_subagents::AgentPool::new(children, application.limits)?;
        tools = Arc::new(missy_subagents::DelegationTools::new(
            tools,
            Arc::new(pool),
            Arc::new(missy_subagents::AgentRegistry::open(
                cfg.state_dir.join("agents.sqlite3"),
            )?),
            application.policy,
        )?);
    }
    // The base runtime above has guarded tools but no recursive orchestration.
    // The outer guard also protects newly composed automation/delegation mutations.
    if cfg.extensions.automation_config.is_some() || cfg.extensions.subagents_config.is_some() {
        let path = cfg
            .extensions
            .approvals_config
            .as_deref()
            .context("orchestration requires durable approvals")?;
        let mut guard: missy_governance::GuardConfig = load_extension(path)?;
        guard.sensitive_tool_names.extend(
            tools
                .definitions()
                .into_iter()
                .filter(|d| d.side_effect)
                .map(|d| d.name),
        );
        guard.context_fingerprint = configuration_fingerprint.clone();
        tools = Arc::new(missy_governance::GuardedTools::new(
            tools,
            missy_governance::ApprovalRegistry::open(cfg.state_dir.join("governance.sqlite3"))?,
            guard,
        )?);
    }
    if production::fingerprint(cfg)? != configuration_fingerprint {
        bail!(
            "extension configuration changed during composition; restart with stable private configuration"
        );
    }
    make_runtime(cfg, store, provider, tools)
}

fn make_runtime(
    cfg: &Config,
    store: Arc<Store>,
    provider: Arc<dyn missy_core::Provider>,
    tools: Arc<dyn missy_core::ToolExecutor>,
) -> Result<Arc<Runtime>> {
    let runtime = if let Some(path) = &cfg.extensions.telemetry_config {
        let telemetry = missy_operations::TelemetryRuntime::build_client(
            cfg.policy.clone(),
            load_extension(path)?,
        )?;
        let runtime = Runtime::new_observed(
            store,
            provider,
            tools,
            cfg.system_prompt.clone(),
            cfg.runtime.clone(),
            telemetry.observer(),
        )?;
        TELEMETRY_HANDLES
            .lock()
            .map_err(|_| anyhow::anyhow!("telemetry lifecycle lock poisoned"))?
            .push(telemetry);
        runtime
    } else {
        Runtime::new(
            store,
            provider,
            tools,
            cfg.system_prompt.clone(),
            cfg.runtime.clone(),
        )?
    };
    let identity = cfg.state_dir.join("governance.sqlite3");
    let runtime = runtime.with_prompt_resolver(Arc::new(prompt_resolver::GovernancePrompts {
        registry: missy_governance::ApprovalRegistry::open(&identity)?,
        identity,
    }))?;
    let runtime = Arc::new(runtime);
    RUNTIME_HANDLES
        .lock()
        .map_err(|_| anyhow::anyhow!("runtime lifecycle lock poisoned"))?
        .push(Arc::downgrade(&runtime));
    Ok(runtime)
}

fn build_safe_runtime(cfg: &Config) -> Result<Arc<Runtime>> {
    let provider: Arc<dyn missy_core::Provider> =
        if let Some(path) = &cfg.extensions.cli_provider_config {
            Arc::new(missy_provider_process::CliProvider::new(load_extension(
                path,
            )?)?)
        } else {
            let providers = ProviderPool::with_routing(
                cfg.providers.clone(),
                cfg.policy.clone(),
                cfg.routing.clone(),
            )?;
            if cfg.runtime.max_cost_microusd > 0 {
                providers.require_known_prices()?;
            }
            Arc::new(providers)
        };
    make_runtime(
        cfg,
        open_store(cfg)?,
        provider,
        Arc::new(extensions::NoTools),
    )
}

async fn build_scoped_runtime(cfg: &Config, scope: &Scope) -> Result<Arc<Runtime>> {
    scope.validate()?;
    build_runtime(cfg).await
}

#[tokio::main]
async fn main() {
    let result = execute(Cli::parse()).await;
    let _ = shutdown_runtimes(std::time::Duration::from_secs(5)).await;
    let telemetry = TELEMETRY_HANDLES
        .lock()
        .map(|mut handles| std::mem::take(&mut *handles))
        .unwrap_or_default();
    for handle in telemetry {
        if handle.shutdown().await.is_err() {
            eprintln!("missy-rs: telemetry flush failed; completed operations were not retried");
        }
    }
    if let Err(error) = result {
        // Never print debug chains, HTTP bodies, environment values or provider payloads.
        eprintln!("missy-rs: {error}");
        std::process::exit(1);
    }
}
async fn execute(cli: Cli) -> Result<()> {
    if let Command::Providers { command } = cli.command {
        return provider_ops::execute(&cli.config, command).await;
    }
    if let Command::Config { command } = cli.command {
        return config_ops::execute(&cli.config, command);
    }
    if matches!(cli.command, Command::Init) {
        return initialize(&cli.config);
    }
    if let Command::Migrate { command } = cli.command {
        match command {
            MigrationCommand::Verify { destination } => println!(
                "{}",
                serde_json::to_string_pretty(&missy_migrate::verify(destination)?)?
            ),
            MigrationCommand::Rollback {
                destination,
                retired,
                confirm,
            } => {
                missy_migrate::rollback(destination, retired, confirm)?;
                println!("Migration destination retired; source backups unchanged.");
            }
            MigrationCommand::Plan { inputs } => println!(
                "{}",
                serde_json::to_string_pretty(&missy_migrate::plan(inputs.input())?)?
            ),
            MigrationCommand::Apply {
                inputs,
                destination,
                confirm,
            } => {
                let plan = missy_migrate::plan(inputs.input())?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&missy_migrate::apply(
                        &plan,
                        destination,
                        confirm
                    )?)?
                );
            }
        }
        return Ok(());
    }
    let cfg = load_config(&cli.config)?;
    if cfg.discord.enabled {
        // Enabled but unstarted integrations must never disappear silently.
        // Discord has a dedicated explicit command once its transport is selected.
        if matches!(cli.command, Command::Serve) {
            bail!("Discord is enabled; use its dedicated transport command, not HTTP-only serve");
        }
    }
    match cli.command {
        Command::Init => unreachable!(),
        Command::Migrate { .. } => unreachable!(),
        Command::Providers { .. } => unreachable!(),
        Command::Config { .. } => unreachable!(),
        Command::Operator { command, scope } => {
            operator::execute(&cfg, command, scope.scope()?).await?
        }
        Command::Admin { command, scope } => {
            admin::execute(&cfg, &cli.config, command, scope.scope()?)?
        }
        Command::Screencast { command, scope } => {
            screencast_adapter::execute(&cfg, command, scope.scope()?).await?
        }
        Command::Skills { command } => skills_ops::execute(&cfg, command)?,
        Command::Triggers { command, scope } => {
            let scheduler = missy_automation::Scheduler::open(
                cfg.state_dir.join("scheduler.sqlite3"),
                build_runtime(&cfg).await?,
            )?;
            match command {
                TriggerCommand::Receipts { name } => println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &scheduler.trigger_receipts(&scope.scope()?, &name)?
                    )?
                ),
                TriggerCommand::Reset { name, confirm } => {
                    if !confirm {
                        bail!("trigger reset requires --confirm after occurrence reconciliation");
                    }
                    scheduler.reset_trigger(&scope.scope()?, &name)?;
                    println!("Trigger state reset; no turn executed.");
                }
            }
        }
        Command::Gateway { command } => production::execute(command, &cli.config, &cfg).await?,
        Command::ConfigCheck => println!(
            "Configuration version {} and explicit policy are valid.",
            cfg.version
        ),
        Command::Ask {
            prompt,
            scope,
            request_id,
        } => {
            let result = build_scoped_runtime(&cfg, &scope.scope()?)
                .await?
                .run(
                    scope.scope()?,
                    prompt,
                    request_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                )
                .await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Run { scope } => {
            let runtime = build_scoped_runtime(&cfg, &scope.scope()?).await?;
            let scope = scope.scope()?;
            println!(
                "Session {}. /quit exits. History and turns are durable.",
                scope.session
            );
            loop {
                print!("> ");
                io::stdout().flush()?;
                let mut line = String::new();
                if io::stdin().read_line(&mut line)? == 0 || line.trim() == "/quit" {
                    break;
                }
                if line.trim().is_empty() {
                    continue;
                }
                let result = runtime
                    .run(
                        scope.clone(),
                        line.trim_end().to_owned(),
                        uuid::Uuid::new_v4().to_string(),
                    )
                    .await?;
                println!(
                    "{}\n[turn {}: {}]",
                    result.content, result.turn_id, result.status
                );
            }
        }
        Command::Resume { turn_id, scope } => println!(
            "{}",
            serde_json::to_string_pretty(
                &build_scoped_runtime(&cfg, &scope.scope()?)
                    .await?
                    .resume(scope.scope()?, turn_id)
                    .await?
            )?
        ),
        Command::Status { turn_id, scope } => println!(
            "{}",
            serde_json::to_string_pretty(&open_store(&cfg)?.get_turn(&scope.scope()?, &turn_id)?)?
        ),
        Command::History { scope, limit } => println!(
            "{}",
            serde_json::to_string_pretty(
                &open_store(&cfg)?.history(&scope.scope()?, limit.min(1000))?
            )?
        ),
        Command::Recover { scope } => println!(
            "{}",
            serde_json::to_string_pretty(&{
                let store = open_store(&cfg)?;
                let scope = scope.scope()?;
                store.recover(&scope)?;
                store
                    .list_turns(&scope, 1000)?
                    .into_iter()
                    .filter(|turn| {
                        matches!(
                            turn.status,
                            missy_store::TurnStatus::Admitted
                                | missy_store::TurnStatus::Interrupted
                        )
                    })
                    .collect::<Vec<_>>()
            })?
        ),
        Command::Cancel { turn_id, scope } => {
            open_store(&cfg)?.request_cancel(&scope.scope()?, &turn_id)?;
            println!(
                "{}",
                serde_json::json!({"turn_id":turn_id,"cancellation_requested":true})
            );
        }
        Command::Serve => missy_gateway::serve(build_runtime(&cfg).await?, cfg.api.clone()).await?,
        Command::Discord => {
            if !cfg.discord.enabled {
                bail!("Discord service requires discord.enabled");
            }
            production::execute(production::GatewayCommand::Start, &cli.config, &cfg).await?;
        }
        Command::DiscordVoice {
            guild,
            channel,
            actor,
            approval,
        } => {
            let application: voice::DiscordVoiceApplication = load_extension(
                cfg.extensions
                    .discord_voice_config
                    .as_deref()
                    .context("extensions.discord_voice_config required")?,
            )?;
            application.server.validate()?;
            if approval != missy_discord_voice::DiscordVoiceService::join_approval(guild, channel) {
                bail!("exact operator voice join approval required");
            }
            if !application.server.bindings.iter().any(|b| {
                b.guild_id == guild && b.channel_id == channel && b.user_id == actor && b.consent
            }) {
                bail!("actor has no explicit consented voice binding");
            }
            production::start_discord_voice(
                &cli.config,
                &cfg,
                voice::DiscordVoiceJoin {
                    guild,
                    channel,
                    actor,
                    approval,
                },
            )
            .await?;
        }
        Command::DiscordProbe => println!(
            "{}",
            serde_json::to_string_pretty(
                &missy_gateway::probe_discord(cfg.discord.clone(), cfg.policy.clone()).await?
            )?
        ),
        Command::DiscordOutbox { command } => {
            let dir = cfg.state_dir.join("discord");
            match command {
                OutboxCommand::Inspect => println!(
                    "{}",
                    serde_json::to_string_pretty(&missy_gateway::discord_outbox_inspect(&dir)?)?
                ),
                OutboxCommand::Resolve {
                    nonce,
                    binding,
                    operator,
                    receipt,
                    evidence,
                    confirm,
                } => {
                    missy_gateway::discord_outbox_resolve(
                        &dir,
                        &nonce,
                        &binding,
                        &operator,
                        receipt.as_deref(),
                        &evidence,
                        confirm,
                    )?;
                    println!(
                        "{}",
                        serde_json::json!({"reconciled":true,"replayed":false})
                    );
                }
            }
        }
        Command::Voice { command } => {
            // Pairing commands do not start a listener or inference runtime.
            open_store(&cfg)?;
            let registry = Arc::new(missy_voice::DeviceRegistry::open(
                cfg.state_dir.join("voice-devices.json"),
            )?);
            match command {
                VoiceCommand::Devices => {
                    println!("{}", serde_json::to_string_pretty(&registry.list()?)?)
                }
                VoiceCommand::Pair {
                    friendly_name,
                    room,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &registry.request_pairing(&friendly_name, &room)?
                    )?
                ),
                VoiceCommand::Approve {
                    node_id,
                    user,
                    tenant,
                    mode,
                    token_out,
                } => {
                    use std::os::unix::fs::OpenOptionsExt;
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o600)
                        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                        .open(token_out)
                        .context("token output must be a new private file")?;
                    let token = zeroize::Zeroizing::new(registry.approve(
                        &node_id,
                        Scope {
                            tenant,
                            user,
                            channel: "voice".into(),
                            session: node_id.clone(),
                        },
                        voice::mode(&mode)?,
                    )?);
                    file.write_all(token.as_bytes())?;
                    file.sync_all()?;
                    println!("Approved {node_id}; pairing token written privately, not printed.");
                }
                VoiceCommand::Mode { node_id, mode } => {
                    registry.set_mode(&node_id, voice::mode(&mode)?)?
                }
                VoiceCommand::Revoke { node_id } => registry.revoke(&node_id)?,
                VoiceCommand::Serve => {
                    if cfg.extensions.voice_config.is_none() {
                        bail!("extensions.voice_config required");
                    }
                    drop(registry);
                    production::execute(production::GatewayCommand::Start, &cli.config, &cfg)
                        .await?;
                }
            }
        }
        Command::Retrieval { command } => {
            let db = retrieval(&cfg)?;
            match command {
                RetrievalCommand::Ingest { id, path, scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(&db.ingest_file(
                        &scope.scope()?,
                        &id,
                        &path,
                        &missy_policy::Policy::new(cfg.policy.clone())?
                    )?)?
                ),
                RetrievalCommand::Query {
                    query,
                    limit,
                    scope,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(&db.search(&scope.scope()?, &query, limit)?)?
                ),
                RetrievalCommand::Get { id, scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(&db.get_document(&scope.scope()?, &id)?)?
                ),
                RetrievalCommand::Delete { id, scope } => println!(
                    "{}",
                    serde_json::json!({"deleted":db.drop_document(&scope.scope()?,&id)?})
                ),
            }
        }
        Command::Approvals { command } => {
            open_store(&cfg)?;
            let registry =
                missy_governance::ApprovalRegistry::open(cfg.state_dir.join("governance.sqlite3"))?;
            match command {
                ApprovalCommand::Review { id, scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.review(&scope.scope()?, &id, None)?)?
                ),
                ApprovalCommand::List { scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.list(&scope.scope()?, 100)?)?
                ),
                ApprovalCommand::Approve {
                    id,
                    binding_hash,
                    operator,
                    scope,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.decide(
                        &scope.scope()?,
                        &id,
                        &binding_hash,
                        missy_governance::Decision::Approve,
                        &operator
                    )?)?
                ),
                ApprovalCommand::Deny {
                    id,
                    binding_hash,
                    operator,
                    scope,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.decide(
                        &scope.scope()?,
                        &id,
                        &binding_hash,
                        missy_governance::Decision::Deny,
                        &operator
                    )?)?
                ),
            }
        }
        Command::Persona { command } => {
            use sha2::Digest;
            open_store(&cfg)?;
            let registry =
                missy_governance::ApprovalRegistry::open(cfg.state_dir.join("governance.sqlite3"))?;
            let context = hex::encode(sha2::Sha256::digest(serde_json::to_vec(&cfg)?));
            match command {
                PersonaCommand::Show { scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.prompt_snapshot(&scope.scope()?)?)?
                ),
                PersonaCommand::Proposals { scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &registry.document_proposals(&scope.scope()?, 100)?
                    )?
                ),
                PersonaCommand::Propose {
                    kind,
                    path,
                    base_version,
                    scope,
                } => {
                    let kind = match kind.as_str() {
                        "persona" => missy_governance::DocumentKind::Persona,
                        "system-prompt" => missy_governance::DocumentKind::SystemPrompt,
                        "patch" => missy_governance::DocumentKind::Patch,
                        _ => bail!("kind must be persona, system-prompt or patch"),
                    };
                    let content = String::from_utf8(
                        missy_policy::Policy::new(cfg.policy.clone())?.read_file(path)?,
                    )?;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&registry.propose_document(
                            &scope.scope()?,
                            kind,
                            base_version,
                            &content,
                            &context
                        )?)?
                    );
                }
                PersonaCommand::Approve {
                    id,
                    proposal_hash,
                    operator,
                    scope,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.decide_document(
                        &scope.scope()?,
                        &id,
                        &proposal_hash,
                        &context,
                        missy_governance::Decision::Approve,
                        &operator
                    )?)?
                ),
                PersonaCommand::Deny {
                    id,
                    proposal_hash,
                    operator,
                    scope,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.decide_document(
                        &scope.scope()?,
                        &id,
                        &proposal_hash,
                        &context,
                        missy_governance::Decision::Deny,
                        &operator
                    )?)?
                ),
            }
        }
        Command::Vault { command } => {
            // Store establishes the private state directory without touching legacy state.
            let _store = open_store(&cfg)?;
            let vault = missy_security::Vault::open(cfg.state_dir.join("vault"))?;
            match command {
                VaultCommand::Set { name } => {
                    use std::io::Read;
                    let mut value = zeroize::Zeroizing::new(Vec::new());
                    io::stdin().take(65537).read_to_end(&mut value)?;
                    if value.len() > 65536 {
                        bail!("secret exceeds 64KiB");
                    }
                    vault.set(&name, &value)?;
                    value.fill(0);
                    println!("Stored encrypted secret {name}.");
                }
                VaultCommand::List => println!("{}", serde_json::to_string_pretty(&vault.list()?)?),
                VaultCommand::Delete { name } => {
                    println!("{}", serde_json::json!({"deleted":vault.delete(&name)?}))
                }
            }
        }
        Command::Schedule { command } => {
            // Calendar preview is pure and must not require inference credentials,
            // open a database, connect MCP, or instantiate effect-capable tools.
            if let ScheduleCommand::Next {
                expression,
                timezone,
            } = &command
            {
                let now = chrono::Utc::now();
                let spec = missy_schedule_text::ScheduleSpec::parse_at(expression, timezone, now)?;
                println!(
                    "{}",
                    serde_json::json!({"spec":spec,"next":spec.next_after(now)?})
                );
                return Ok(());
            }
            let scheduler = missy_automation::Scheduler::open(
                cfg.state_dir.join("scheduler.sqlite3"),
                build_runtime(&cfg).await?,
            )?;
            use missy_automation::NewJob;
            match command {
                ScheduleCommand::AddText {
                    prompt,
                    expression,
                    timezone,
                    scope,
                } => {
                    let now = chrono::Utc::now();
                    let spec =
                        missy_schedule_text::ScheduleSpec::parse_at(&expression, &timezone, now)?;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&scheduler.add_spec(
                            &scope.scope()?,
                            prompt,
                            spec,
                            now
                        )?)?
                    );
                }
                ScheduleCommand::Next { .. } => {
                    unreachable!("pure preview returned before runtime construction")
                }
                ScheduleCommand::Add {
                    prompt,
                    every_seconds,
                    scope,
                } => println!(
                    "{}",
                    serde_json::to_string_pretty(&scheduler.add(
                        &scope.scope()?,
                        NewJob {
                            prompt,
                            interval_seconds: every_seconds,
                            first_due: chrono::Utc::now()
                        }
                    )?)?
                ),
                ScheduleCommand::List { scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(&scheduler.list(&scope.scope()?)?)?
                ),
                ScheduleCommand::Pause { job_id, scope } => {
                    scheduler.pause(&scope.scope()?, &job_id, true)?;
                    println!("Paused {job_id}.");
                }
                ScheduleCommand::Resume { job_id, scope } => {
                    scheduler.pause(&scope.scope()?, &job_id, false)?;
                    println!("Resumed {job_id}.");
                }
                ScheduleCommand::Remove { job_id, scope } => {
                    scheduler.remove(&scope.scope()?, &job_id)?;
                    println!("Removed {job_id}.");
                }
                ScheduleCommand::RunDue { scope, workers } => println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &scheduler
                            .run_due(
                                &scope.scope()?,
                                workers,
                                tokio_util::sync::CancellationToken::new()
                            )
                            .await?
                    )?
                ),
            }
        }
        Command::Mesh { command } => match command {
            MeshCommand::IdentityCreate { path } => {
                let identity = missy_mesh::Identity::create(&path)?;
                println!(
                    "{}",
                    serde_json::json!({"peer_id":identity.peer_id(),"public_key":identity.public_key()})
                );
            }
            command => {
                let config = missy_mesh::MeshConfig::load(
                    cfg.extensions
                        .mesh_config
                        .as_deref()
                        .context("extensions.mesh_config required")?,
                )?;
                let node = missy_mesh::MeshNode::open(config)?;
                match command {
                    MeshCommand::Status => {
                        println!("{}", serde_json::to_string_pretty(&node.status()?)?)
                    }
                    MeshCommand::Peers => {
                        println!("{}", serde_json::to_string_pretty(&node.peers()?)?)
                    }
                    MeshCommand::PeerAdd { path } => {
                        let peer: missy_mesh::Peer = load_extension(&path)?;
                        println!("{}", node.peer_add(peer)?);
                    }
                    MeshCommand::PeerRemove { id } => println!("{}", node.peer_remove(&id)?),
                    MeshCommand::Serve => {
                        let runtime = Arc::new(missy_mesh::MeshRuntime::try_from_runtime(
                            build_runtime(&cfg).await?,
                        )?);
                        node.serve_with_runtime(runtime).await?;
                    }
                    MeshCommand::IdentityCreate { .. } => unreachable!(),
                }
            }
        },
        Command::Benchmark { path, request_id } => {
            let mut benchmark: missy_plugins::BenchmarkConfig = load_extension(&path)?;
            open_store(&cfg)?;
            benchmark.ledger_path = cfg.state_dir.join("provider-benchmark.jsonl");
            let mut targets = std::collections::BTreeMap::new();
            for provider in &cfg.providers {
                let input = provider
                    .input_microusd_per_million
                    .context("benchmark input price required")?;
                let output = provider
                    .output_microusd_per_million
                    .context("benchmark output price required")?;
                targets.insert(
                    provider.name.clone(),
                    missy_plugins::BenchmarkTarget {
                        provider: Arc::new(ProviderPool::new(
                            vec![explicit_profile(provider.clone())],
                            cfg.policy.clone(),
                        )?),
                        max_output_tokens: provider.max_output_tokens,
                        input_microusd_per_million: input,
                        output_microusd_per_million: output,
                    },
                );
            }
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &missy_plugins::benchmark_providers(
                        targets,
                        benchmark,
                        &request_id,
                        tokio_util::sync::CancellationToken::new()
                    )
                    .await?
                )?
            );
        }
        Command::Plugins { command } => {
            open_store(&cfg)?;
            let registry = missy_plugins::ToolRegistry::open(cfg.state_dir.join("plugins"))?;
            match command {
                PluginCommand::SourcePropose { source, scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &registry.propose_rust_source(&scope.scope()?, load_extension(&source)?)?
                    )?
                ),
                PluginCommand::SourceShow { id, scope } => println!(
                    "{}",
                    serde_json::to_string_pretty(&registry.rust_source(&scope.scope()?, &id)?)?
                ),
                PluginCommand::SourceReview {
                    id,
                    hash,
                    approve,
                    scope,
                } => {
                    registry.review_rust_source(&scope.scope()?, &id, &hash, approve)?;
                    println!(
                        "{}",
                        serde_json::json!({"reviewed":approve,"compiled":false,"enabled":false})
                    );
                }
                PluginCommand::SourceBuild {
                    id,
                    hash,
                    build_config,
                    confirm,
                    scope,
                } => {
                    let build = load_extension(&build_config)?;
                    let scope = scope.scope()?;
                    let stop = tokio_util::sync::CancellationToken::new();
                    let signal = production::shutdown_signal();
                    tokio::pin!(signal);
                    let compile = registry.build_rust_source(
                        &scope,
                        &id,
                        &hash,
                        &build,
                        confirm,
                        stop.clone(),
                    );
                    tokio::pin!(compile);
                    let result = tokio::select! {
                        result=&mut compile=>result,
                        _=&mut signal=> {stop.cancel();compile.await}
                    }?;
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
                PluginCommand::Propose { manifest, scope } => {
                    let manifest = load_extension(&manifest)?;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(
                            &registry.propose(&scope.scope()?, manifest)?
                        )?
                    );
                }
                PluginCommand::Review {
                    id,
                    hash,
                    approve,
                    scope,
                } => {
                    registry.review(&scope.scope()?, &id, &hash, approve)?;
                    println!(
                        "{}",
                        serde_json::json!({"reviewed":true,"approved":approve})
                    );
                }
                PluginCommand::Enable { id, hash, scope } => {
                    let config = load_extension(
                        cfg.extensions
                            .plugins_config
                            .as_deref()
                            .context("extensions.plugins_config required")?,
                    )?;
                    registry.enable(&scope.scope()?, &id, &hash, &config)?;
                    println!(
                        "{}",
                        serde_json::json!({"enabled":true,"requires_new_runtime":true})
                    );
                }
            }
        }
        Command::Hatch { answers, scope } => {
            let answers: missy_evolution::HatchingAnswers = load_extension(&answers)?;
            let scope = scope.scope()?;
            open_store(&cfg)?;
            let governance =
                missy_governance::ApprovalRegistry::open(cfg.state_dir.join("governance.sqlite3"))?;
            use sha2::Digest;
            let context = hex::encode(sha2::Sha256::digest(serde_json::to_vec(&cfg)?));
            println!(
                "{}",
                serde_json::to_string_pretty(&missy_evolution::propose_hatching(
                    &scope,
                    answers,
                    &governance,
                    &context
                )?)?
            );
        }
        Command::Experiments { command } => {
            open_store(&cfg)?;
            let registry = missy_evolution::ExperimentRegistry::new(
                missy_evolution::EvolutionRegistry::open(cfg.state_dir.join("evolution.sqlite3"))?,
            );
            let value = match command {
                ExperimentCommand::Create { variants, scope } => {
                    let variants = load_extension(&variants)?;
                    serde_json::to_value(registry.create(&scope.scope()?, variants)?)?
                }
                ExperimentCommand::Show { id, scope } => {
                    serde_json::to_value(registry.experiment(&scope.scope()?, &id)?)?
                }
                ExperimentCommand::Assign { id, scope } => {
                    serde_json::to_value(registry.assign(&scope.scope()?, &id)?)?
                }
                ExperimentCommand::Record {
                    id,
                    event_id,
                    success,
                    refused,
                    scope,
                } => {
                    registry.record_outcome(&scope.scope()?, &id, &event_id, success, refused)?;
                    serde_json::json!({"recorded":true,"trusted_for_promotion":false})
                }
                ExperimentCommand::Compare { id, scope } => {
                    serde_json::to_value(registry.compare(&scope.scope()?, &id)?)?
                }
                ExperimentCommand::Promote {
                    id,
                    variant,
                    experiment_hash,
                    base_version,
                    context_hash,
                    operator,
                    confirm,
                    scope,
                } => {
                    let governance = missy_governance::ApprovalRegistry::open(
                        cfg.state_dir.join("governance.sqlite3"),
                    )?;
                    serde_json::to_value(registry.propose_promotion(
                        &scope.scope()?,
                        &id,
                        &experiment_hash,
                        &variant,
                        &governance,
                        base_version,
                        &context_hash,
                        &operator,
                        confirm,
                    )?)?
                }
            };
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        Command::Agents {
            path,
            request_id,
            scope,
        } => {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct AgentApplication {
                limits: missy_subagents::Limits,
                plan: missy_subagents::DagSpec,
            }
            let application: AgentApplication = load_extension(&path)?;
            let scope = scope.scope()?;
            if cfg.extensions.cli_provider_config.is_some() {
                bail!(
                    "agent pools require explicitly configured native provider entries, not ambient CLI credentials"
                );
            }
            let mut runtimes = std::collections::BTreeMap::new();
            for provider in &cfg.providers {
                let mut selected = cfg.clone();
                selected.providers = vec![explicit_profile(provider.clone())];
                selected.routing = Default::default();
                runtimes.insert(
                    provider.name.clone(),
                    build_scoped_runtime(&selected, &scope).await?,
                );
            }
            open_store(&cfg)?;
            let registry =
                missy_subagents::AgentRegistry::open(cfg.state_dir.join("agents.sqlite3"))?;
            let pool = missy_subagents::AgentPool::new(runtimes, application.limits)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &pool
                        .run(
                            &registry,
                            scope,
                            &request_id,
                            application.plan,
                            tokio_util::sync::CancellationToken::new()
                        )
                        .await?
                )?
            );
        }
        Command::Evolve { command } => {
            let policy: missy_evolution::EvolutionPolicy = load_extension(
                cfg.extensions
                    .evolution_config
                    .as_deref()
                    .context("extensions.evolution_config required")?,
            )?;
            open_store(&cfg)?;
            let registry =
                missy_evolution::EvolutionRegistry::open(cfg.state_dir.join("evolution.sqlite3"))?;
            let value = match command {
                EvolutionCommand::Propose {
                    repo,
                    diff,
                    title,
                    scope,
                } => {
                    let read = missy_policy::Policy::new(cfg.policy.clone())?.read_file(&diff)?;
                    let text = std::str::from_utf8(&read).context("patch is not UTF-8")?;
                    serde_json::to_value(registry.propose(
                        &scope.scope()?,
                        &policy,
                        repo,
                        text,
                        &title,
                    )?)?
                }
                EvolutionCommand::Show { id, scope } => {
                    serde_json::to_value(registry.proposal(&scope.scope()?, &id)?)?
                }
                EvolutionCommand::Review {
                    id,
                    plan_hash,
                    operator,
                    approve,
                    scope,
                } => serde_json::to_value(registry.review(
                    &scope.scope()?,
                    &id,
                    &plan_hash,
                    &operator,
                    approve,
                )?)?,
                EvolutionCommand::Apply {
                    id,
                    plan_hash,
                    head,
                    operator,
                    confirm,
                    scope,
                } => serde_json::to_value(registry.apply(
                    &scope.scope()?,
                    &id,
                    &plan_hash,
                    &head,
                    &policy,
                    &operator,
                    confirm,
                )?)?,
                EvolutionCommand::Rollback {
                    id,
                    plan_hash,
                    head,
                    operator,
                    confirm,
                    scope,
                } => serde_json::to_value(registry.rollback(
                    &scope.scope()?,
                    &id,
                    &plan_hash,
                    &head,
                    &policy,
                    &operator,
                    confirm,
                )?)?,
                EvolutionCommand::Audit { scope } => {
                    serde_json::to_value(registry.audit(&scope.scope()?)?)?
                }
                EvolutionCommand::Recovery { id, scope } => {
                    serde_json::to_value(registry.recovery_journal(&scope.scope()?, &id)?)?
                }
            };
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        Command::PersonalMemory { command } => {
            let memory = personal_memory(&cfg)?;
            let value = match command {
                PersonalCommand::Put { id, content, scope } => {
                    serde_json::to_value(memory.put(&scope.scope()?, &id, &content)?)?
                }
                PersonalCommand::Get { id, scope } => {
                    serde_json::to_value(memory.get(&scope.scope()?, &id)?)?
                }
                PersonalCommand::Search {
                    query,
                    limit,
                    scope,
                } => serde_json::to_value(memory.search(&scope.scope()?, &query, limit)?)?,
                PersonalCommand::Delete { id, scope } => {
                    serde_json::json!({"deleted":memory.delete(&scope.scope()?,&id)?})
                }
            };
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        Command::Workflow { path, scope } => {
            let workflow: missy_operations::Workflow = load_extension(&path)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &workflow
                        .execute(
                            build_runtime(&cfg).await?,
                            scope.scope()?,
                            missy_operations::WorkflowBounds::default()
                        )
                        .await?
                )?
            );
        }
        Command::Operations => {
            let mut operations: missy_operations::OperationsConfig = load_extension(
                cfg.extensions
                    .operations_config
                    .as_deref()
                    .context("extensions.operations_config required")?,
            )?;
            open_store(&cfg)?;
            operations.replay_db = cfg.state_dir.join("operations.sqlite3");
            missy_operations::serve(build_runtime(&cfg).await?, operations).await?;
        }
        Command::DiscordRegisterCommands { confirm } => println!(
            "{}",
            serde_json::to_string_pretty(
                &missy_gateway::register_discord_commands(
                    cfg.discord.clone(),
                    cfg.policy.clone(),
                    confirm
                )
                .await?
            )?
        ),
        Command::Doctor => {
            open_store(&cfg)?;
            let prices_known = cfg.providers.iter().all(|p| {
                p.input_microusd_per_million.is_some() && p.output_microusd_per_million.is_some()
            });
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"config":"valid","database":"opened","providers_configured":cfg.providers.len(),"provider_prices_known":prices_known,"network_origins":cfg.policy.allowed_origins.len(),"shell_enabled":cfg.policy.shell_enabled,"mcp_enabled":cfg.policy.mcp_enabled,"discord_requested":cfg.discord.enabled,"inference_tested":false,"legacy_state_used":false})
                )?
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn init_does_not_overwrite() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("config.yaml");
        initialize(&path).unwrap();
        assert!(initialize(&path).is_err());
        let c = load_config(&path).unwrap();
        assert_eq!(c.state_dir, temp.path().join("missy-rust-state"));
        assert!(c.policy.allowed_origins.is_empty());
    }
    #[test]
    fn legacy_unknown_keys_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("config.yaml");
        std::fs::write(&path, "network:\n  unrestricted: true\n").unwrap();
        assert!(load_config(&path).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn symlink_config_denied() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("config.yaml");
        initialize(&path).unwrap();
        let link = temp.path().join("link.yaml");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(load_config(&link).is_err());
    }
}
