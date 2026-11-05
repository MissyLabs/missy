//! Explicit, isolated Rust entrypoint. Never auto-imports legacy ~/.missy.
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
    /// Write a private, default-deny configuration without touching legacy Missy.
    Init,
    /// Validate the configuration and policy without connecting to providers.
    ConfigCheck,
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
    /// Run the explicitly enabled Discord gateway with strict allowlists.
    Discord,
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
}
#[derive(Subcommand)]
enum VaultCommand {
    Set { name: String },
    List,
    Delete { name: String },
}
#[derive(Subcommand)]
enum ScheduleCommand {
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
    Ok(cfg)
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
    let store = open_store(cfg)?;
    let providers = ProviderPool::new(cfg.providers.clone(), cfg.policy.clone())?;
    if cfg.runtime.max_cost_microusd > 0 {
        providers.require_known_prices()?;
    }
    let tools = BuiltinTools::new(cfg.policy.clone(), cfg.runtime.clone(), store.clone())?
        .with_mcp(cfg.mcp.clone())
        .await?;
    Ok(Arc::new(Runtime::new(
        store,
        Arc::new(providers),
        Arc::new(tools),
        cfg.system_prompt.clone(),
        cfg.runtime.clone(),
    )?))
}

#[tokio::main]
async fn main() {
    if let Err(error) = execute(Cli::parse()).await {
        // Never print debug chains, HTTP bodies, environment values or provider payloads.
        eprintln!("missy-rs: {error}");
        std::process::exit(1);
    }
}
async fn execute(cli: Cli) -> Result<()> {
    if matches!(cli.command, Command::Init) {
        return initialize(&cli.config);
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
        Command::ConfigCheck => println!(
            "Configuration version {} and explicit policy are valid.",
            cfg.version
        ),
        Command::Ask {
            prompt,
            scope,
            request_id,
        } => {
            let result = build_runtime(&cfg)
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
            let runtime = build_runtime(&cfg).await?;
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
                &build_runtime(&cfg)
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
            missy_gateway::serve_discord(
                build_runtime(&cfg).await?,
                cfg.discord.clone(),
                cfg.policy.clone(),
            )
            .await?
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
            let scheduler = missy_automation::Scheduler::open(
                cfg.state_dir.join("scheduler.sqlite3"),
                build_runtime(&cfg).await?,
            )?;
            use missy_automation::NewJob;
            match command {
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
