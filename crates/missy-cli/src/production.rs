//! One owner for long-lived services, startup evidence and bounded shutdown.
//! No service-management commands, implicit migration or ambient credentials.
use anyhow::{Context, Result, bail, ensure};
use clap::Subcommand;
use missy_core::Config;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::{net::TcpListener, task::JoinSet};
use tokio_util::sync::CancellationToken;

#[derive(Subcommand)]
pub enum GatewayCommand {
    /// Start every configured native listener and worker under one lifecycle.
    Start,
    /// Read private startup evidence and verify the process still holds its lock.
    Status,
    /// Validate config syntax and detect changes. Changes always require restart.
    CheckReload,
}

#[derive(Serialize, Deserialize)]
struct Status {
    version: u32,
    pid: u32,
    instance: String,
    phase: String,
    ready: bool,
    updated_at: i64,
    config_fingerprint: String,
    reload: String,
    subsystems: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    failure: Option<String>,
    #[serde(default)]
    startup_stage: String,
}

struct Owner {
    _lock: File,
    dir: PathBuf,
    status: Status,
}
impl Owner {
    fn acquire(cfg: &Config) -> Result<Self> {
        // Store establishes the private owned state directory, not ~/.missy.
        crate::open_store(cfg)?;
        let dir = cfg.state_dir.join("run");
        if !dir.exists() {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
        }
        check_directory(&dir)?;
        let lock = lock_file(&dir)?;
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another native gateway owns this state directory"
        );
        Ok(Self {
            _lock: lock,
            dir,
            status: Status {
                version: 1,
                pid: std::process::id(),
                instance: uuid::Uuid::new_v4().to_string(),
                phase: "starting".into(),
                ready: false,
                updated_at: 0,
                config_fingerprint: String::new(),
                reload: "restart_required_for_changes".into(),
                subsystems: BTreeMap::new(),
                failure: None,
                startup_stage: "configuration_fingerprint".into(),
            },
        })
    }
    fn stage(&mut self, name: &str) -> Result<()> {
        self.status.startup_stage = name.into();
        self.write()
    }
    fn write(&mut self) -> Result<()> {
        self.status.updated_at = chrono::Utc::now().timestamp();
        let temporary = self.dir.join(format!(".status-{}", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(&temporary)?;
            file.write_all(&serde_json::to_vec_pretty(&self.status)?)?;
            file.sync_all()?;
            std::fs::rename(&temporary, self.dir.join("gateway_status.json"))?;
            File::open(&self.dir)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        result
    }
}
fn check_directory(dir: &Path) -> Result<()> {
    let m = std::fs::symlink_metadata(dir)?;
    ensure!(
        m.is_dir()
            && !m.file_type().is_symlink()
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o077 == 0,
        "gateway run directory must be owned, private and not a symlink"
    );
    Ok(())
}
fn lock_file(dir: &Path) -> Result<File> {
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(dir.join("gateway.lock"))?;
    let m = f.metadata()?;
    ensure!(
        m.is_file() && m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "gateway lock must be an owned private regular file"
    );
    Ok(f)
}
fn read_status(cfg: &Config) -> Result<(Status, bool)> {
    let dir = cfg.state_dir.join("run");
    check_directory(&dir)?;
    let lock = lock_file(&dir)?;
    let running = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0;
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(dir.join("gateway_status.json"))
        .context("no native gateway startup status")?;
    let m = file.metadata()?;
    ensure!(
        m.is_file()
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o077 == 0
            && m.len() <= 65_536,
        "gateway status must be an owned private bounded regular file"
    );
    let mut bytes = vec![];
    (&mut file).take(65_537).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 65_536, "gateway status exceeds limit");
    Ok((serde_json::from_slice(&bytes)?, running))
}

/// Hash semantic config and each referenced extension. Never publish paths, contents or env values.
pub(crate) fn fingerprint(cfg: &Config) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(serde_json::to_vec(cfg)?);
    let extensions = serde_json::to_value(&cfg.extensions)?;
    for (key, value) in extensions.as_object().context("extension map invalid")? {
        if key.ends_with("_config")
            && let Some(path) = value.as_str()
        {
            let parsed: serde_json::Value = crate::load_extension(Path::new(path))?;
            hash.update(key.as_bytes());
            hash.update(serde_json::to_vec(&parsed)?);
        }
    }
    Ok(hex::encode(hash.finalize()))
}
pub async fn execute(command: GatewayCommand, path: &Path, cfg: &Config) -> Result<()> {
    match command {
        GatewayCommand::Start => start(path, cfg, None).await,
        GatewayCommand::Status => {
            let (mut status, running) = read_status(cfg)?;
            let fresh = chrono::Utc::now()
                .timestamp()
                .saturating_sub(status.updated_at)
                <= 20;
            status.ready &= running && fresh;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"running":running,"fresh":fresh,"status":status})
                )?
            );
            ensure!(
                status.ready,
                "native gateway is not ready; inspect subsystem startup status"
            );
            Ok(())
        }
        GatewayCommand::CheckReload => {
            let current = fingerprint(cfg)?;
            let (status, running) = read_status(cfg)?;
            let unchanged = status.config_fingerprint == current;
            println!(
                "{}",
                serde_json::json!({"valid":true,"running":running,"restart_required":!unchanged,"hot_reload_supported":false})
            );
            ensure!(
                unchanged,
                "validated configuration changed; explicit gateway restart required"
            );
            Ok(())
        }
    }
}

type Services = JoinSet<(&'static str, Result<()>)>;
struct AbortOnDrop(tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TriggerEntry {
    scope: missy_core::Scope,
    config: missy_automation::TriggerConfig,
    #[serde(default)]
    source: Option<missy_automation::ObservationSource>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TriggerApplication {
    poll_seconds: u64,
    #[serde(default)]
    entries: Vec<TriggerEntry>,
    #[serde(default)]
    maintenance: Vec<MaintenanceEntry>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MaintenanceEntry {
    scope: missy_core::Scope,
    config: missy_automation::MaintenanceConfig,
}
async fn http_service(
    listener: TcpListener,
    router: axum::Router,
    stop: CancellationToken,
) -> Result<()> {
    axum::serve(listener, router)
        .with_graceful_shutdown(stop.cancelled_owned())
        .await?;
    Ok(())
}
/// Standalone voice shares production startup, cancellation, readiness and drain.
pub async fn start_discord_voice(
    path: &Path,
    cfg: &Config,
    join: crate::voice::DiscordVoiceJoin,
) -> Result<()> {
    start(path, cfg, Some(join)).await
}
pub async fn shutdown_signal() -> Result<()> {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {result=tokio::signal::ctrl_c()=>result?,_=term.recv()=>{}}
    Ok(())
}
async fn start(
    path: &Path,
    cfg: &Config,
    voice_join: Option<crate::voice::DiscordVoiceJoin>,
) -> Result<()> {
    let mut owner = Owner::acquire(cfg)?;
    for (name, enabled) in [
        ("http", cfg.gateway.http_enabled),
        ("discord", cfg.discord.enabled),
        ("voice", cfg.extensions.voice_config.is_some()),
        ("operations", cfg.extensions.operations_config.is_some()),
        ("scheduler", cfg.gateway.scheduler.enabled),
        ("mesh", cfg.extensions.mesh_config.is_some()),
        (
            "discord_voice",
            cfg.extensions.discord_voice_config.is_some(),
        ),
        ("triggers", cfg.extensions.triggers_config.is_some()),
        ("watchdog", cfg.extensions.watchdog_config.is_some()),
        ("workloads", cfg.extensions.workloads_config.is_some()),
        ("screencast", cfg.extensions.screencast_config.is_some()),
    ] {
        owner.status.subsystems.insert(
            name.into(),
            if enabled { "starting" } else { "disabled" }.into(),
        );
    }
    owner.write()?;
    let stop = CancellationToken::new();
    let signal_stop = stop.clone();
    let signal_task = tokio::spawn(async move {
        let _ = shutdown_signal().await;
        signal_stop.cancel();
    });
    let _signal_guard = AbortOnDrop(signal_task);
    let mut services = Services::new();
    let preparation = async {
        owner.status.config_fingerprint = fingerprint(cfg)?;
        prepare(cfg, &mut owner, &mut services, stop.clone(), voice_join).await
    };
    let startup = tokio::select! {
        biased;
        _=stop.cancelled()=>None,
        result=tokio::time::timeout(Duration::from_secs(cfg.gateway.startup_timeout_seconds),preparation)=>Some(result),
    };
    let (runtime, safe) = match startup {
        Some(Ok(Ok(runtimes))) => runtimes,
        failure => {
            owner.status.failure = Some(
                match &failure {
                    None => "startup_cancelled_by_signal",
                    Some(Err(_)) => "startup_deadline_exceeded",
                    Some(Ok(Err(_))) => "startup_configuration_dependency_or_bind_failed",
                    _ => "startup_failed",
                }
                .into(),
            );
            stop.cancel();
            let bound = Duration::from_secs(cfg.gateway.shutdown_timeout_seconds);
            let _ = tokio::time::timeout(bound, async {
                let _ = crate::shutdown_runtimes(bound).await;
                while services.join_next().await.is_some() {}
            })
            .await;
            services.abort_all();
            owner.status.phase = "failed".into();
            for state in owner.status.subsystems.values_mut() {
                if state == "starting" {
                    *state = "startup_failed".into();
                }
            }
            owner.write()?;
            bail!(
                "native gateway startup failed ({}); inspect subsystem state, private configuration, credential availability and listener bindings",
                owner.status.failure.as_deref().unwrap_or("unknown")
            );
        }
    };
    owner.status.phase = "running".into();
    owner.status.ready = true;
    owner.write()?;
    notify("READY=1\nSTATUS=Native gateway ready");
    let mut hup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())?;
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    let mut failed = false;
    loop {
        tokio::select! {
            biased;
            _ = stop.cancelled() => break,
            _ = hup.recv() => { check_changes(path, &mut owner); if owner.write().is_err() {failed=true;break;} },
            _ = tick.tick() => { check_changes(path, &mut owner); if owner.write().is_err() {failed=true;break;} },
            ended = services.join_next() => {
                if stop.is_cancelled() {break;}
                let name = ended.and_then(|r| r.ok()).map(|(name,_)|name).unwrap_or("supervisor");
                owner.status.subsystems.insert(name.into(), "unexpected_exit".into()); failed = true; break;
            }
        }
    }
    owner.status.phase = "stopping".into();
    owner.status.ready = false;
    if owner.write().is_err() {
        failed = true;
    }
    notify("STOPPING=1\nSTATUS=Native gateway draining");
    stop.cancel();
    let bound = Duration::from_secs(cfg.gateway.shutdown_timeout_seconds);
    let drain = async {
        runtime.shutdown(bound).await?;
        if let Some(safe) = safe {
            safe.shutdown(bound).await?;
        }
        crate::shutdown_runtimes(bound).await?;
        while let Some(result) = services.join_next().await {
            if !matches!(result, Ok((_, Ok(())))) {
                failed = true;
            }
        }
        Ok::<_, anyhow::Error>(())
    };
    if !matches!(tokio::time::timeout(bound, drain).await, Ok(Ok(()))) {
        services.abort_all();
        failed = true;
        owner.status.subsystems.insert(
            "shutdown".into(),
            "deadline_or_drain_failure_interrupted_work_not_replayed".into(),
        );
    }
    owner.status.phase = if failed { "failed" } else { "stopped" }.into();
    owner.write()?;
    ensure!(
        !failed,
        "native gateway stopped with failure; inspect private subsystem status"
    );
    Ok(())
}
fn check_changes(path: &Path, owner: &mut Owner) {
    owner.status.reload = match crate::load_config(path).and_then(|c| fingerprint(&c)) {
        Ok(hash) if hash == owner.status.config_fingerprint => "unchanged",
        Ok(_) => "restart_required",
        Err(_) => "invalid_config_restart_refused",
    }
    .into();
    // Readiness reflects the configuration actually requested, not just a live PID.
    owner.status.ready = owner.status.reload == "unchanged";
}

async fn prepare(
    cfg: &Config,
    owner: &mut Owner,
    services: &mut Services,
    stop: CancellationToken,
    voice_join: Option<crate::voice::DiscordVoiceJoin>,
) -> Result<(
    Arc<missy_runtime::Runtime>,
    Option<Arc<missy_runtime::Runtime>>,
)> {
    ensure!(
        cfg.extensions.discord_voice_config.is_none() || cfg.discord.enabled,
        "Discord voice requires the shared Discord gateway enabled"
    );
    ensure!(
        cfg.gateway.http_enabled
            || cfg.discord.enabled
            || cfg.extensions.voice_config.is_some()
            || cfg.extensions.operations_config.is_some()
            || cfg.gateway.scheduler.enabled
            || cfg.extensions.mesh_config.is_some()
            || cfg.extensions.triggers_config.is_some()
            || cfg.extensions.watchdog_config.is_some()
            || cfg.extensions.workloads_config.is_some()
            || cfg.extensions.screencast_config.is_some(),
        "no native gateway services enabled"
    );
    owner
        .status
        .subsystems
        .insert("runtime".into(), "starting".into());
    owner.write()?;
    let mut voice_application: Option<crate::voice::DiscordVoiceApplication> = cfg
        .extensions
        .discord_voice_config
        .as_deref()
        .map(crate::load_extension)
        .transpose()?;
    if let Some(join) = voice_join {
        voice_application
            .as_mut()
            .context("Discord voice configuration required")?
            .joins = vec![join];
    }
    let voice_tools = voice_application
        .as_ref()
        .map(|app| {
            missy_discord_voice::DiscordVoiceSayTools::pending(
                app.server.clone(),
                Arc::new(crate::extensions::OuterGuardVerified),
            )
            .map(Arc::new)
        })
        .transpose()?;
    owner.stage("workloads_configuration")?;
    let workloads = cfg
        .extensions
        .workloads_config
        .as_ref()
        .map(|path| {
            missy_workloads::WorkloadTools::open(
                cfg.policy.clone(),
                crate::load_extension(path)?,
                cfg.state_dir.join("workloads"),
            )
            .map(Arc::new)
        })
        .transpose()?;
    owner.stage("runtime_extensions")?;
    let runtime =
        crate::build_runtime_with_services(cfg, voice_tools.clone(), workloads.clone()).await?;
    owner
        .status
        .subsystems
        .insert("runtime".into(), "ready".into());
    let mut safe = None;
    if cfg.gateway.http_enabled {
        owner.stage("http")?;
        let router = missy_gateway::router(missy_gateway::ApiState::new(
            runtime.clone(),
            cfg.api.clone(),
        )?);
        let listener = TcpListener::bind(&cfg.api.bind).await?;
        let cancel = stop.clone();
        services.spawn(async move { ("http", http_service(listener, router, cancel).await) });
        owner
            .status
            .subsystems
            .insert("http".into(), "ready".into());
    }
    if let Some(path) = &cfg.extensions.operations_config {
        owner.stage("operations")?;
        let mut config: missy_operations::OperationsConfig = crate::load_extension(path)?;
        config.replay_db = cfg.state_dir.join("operations.sqlite3");
        let bind = config.bind;
        let router = missy_operations::build_router(runtime.clone(), config)?;
        let listener = TcpListener::bind(bind).await?;
        let cancel = stop.clone();
        services.spawn(async move { ("operations", http_service(listener, router, cancel).await) });
        owner
            .status
            .subsystems
            .insert("operations".into(), "ready".into());
    }
    if let Some(path) = &cfg.extensions.voice_config {
        owner.stage("voice")?;
        let config: crate::voice::VoiceApplication = crate::load_extension(path)?;
        config.server.validate()?;
        let listener = TcpListener::bind(config.server.bind).await?;
        let safe_runtime = crate::build_safe_runtime(cfg)?;
        let server = Arc::new(missy_voice::VoiceServer::new(
            config.server,
            Arc::new(missy_voice::DeviceRegistry::open(
                cfg.state_dir.join("voice-devices.json"),
            )?),
            Arc::new(missy_voice::CommandStt::new(config.stt)?),
            Arc::new(missy_voice::CommandTts::new(config.tts)?),
            Arc::new(crate::voice::RuntimeVoice {
                full: runtime.clone(),
                safe: safe_runtime.clone(),
            }),
        )?);
        safe = Some(safe_runtime);
        let cancel = stop.clone();
        services.spawn(async move { ("voice", server.serve(listener, cancel).await) });
        owner
            .status
            .subsystems
            .insert("voice".into(), "ready".into());
    }
    if cfg.gateway.scheduler.enabled {
        owner.stage("scheduler")?;
        let scheduler = missy_automation::Scheduler::open(
            cfg.state_dir.join("scheduler.sqlite3"),
            runtime.clone(),
        )?;
        let config = cfg.gateway.scheduler.clone();
        let cancel = stop.clone();
        services.spawn(async move {
            (
                "scheduler",
                scheduler_worker(scheduler, config, cancel).await,
            )
        });
        owner
            .status
            .subsystems
            .insert("scheduler".into(), "ready".into());
    }
    if let Some(path) = &cfg.extensions.mesh_config {
        // Below setup remains per-subsystem; shared cancellation is supervisory.
        owner.stage("mesh")?;
        let config = missy_mesh::MeshConfig::load(path)?;
        config.validate()?;
        let listener = TcpListener::bind(config.listen).await?;
        let node = missy_mesh::MeshNode::open(config)?;
        let mesh_runtime = Arc::new(missy_mesh::MeshRuntime::try_from_runtime(runtime.clone())?);
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        let router = node.router(tx)?;
        let cancel = stop.clone();
        services.spawn(async move {
            let result = axum::serve(
                listener,
                router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .with_graceful_shutdown(cancel.cancelled_owned())
            .await
            .map_err(Into::into);
            ("mesh_listener", result)
        });
        let cancel = stop.clone();
        services.spawn(async move {
            let result = async {
                loop {
                    let envelope=tokio::select! { biased; _=cancel.cancelled()=>return Ok(()), envelope=rx.recv()=>envelope };
                    let Some(envelope)=envelope else {return Ok(());};
                    node.dispatch(envelope,&mesh_runtime).await?;
                }
            }.await;
            ("mesh_worker",result)
        });
        owner
            .status
            .subsystems
            .insert("mesh".into(), "ready".into());
    }
    if let Some(prepared) = crate::screencast_adapter::prepare(cfg)? {
        owner.stage("screencast")?;
        let running = prepared.start().await?;
        let cancel = stop.clone();
        services.spawn(async move { ("screencast", running.run(cancel).await) });
        owner
            .status
            .subsystems
            .insert("screencast".into(), "ready".into());
    }
    if let Some(path) = &cfg.extensions.triggers_config {
        owner.stage("triggers")?;
        let application: TriggerApplication = crate::load_extension(path)?;
        ensure!(
            (1..=3600).contains(&application.poll_seconds) && application.entries.len() <= 128,
            "invalid trigger worker limits"
        );
        let mut names = std::collections::HashSet::new();
        ensure!(
            application.maintenance.len() <= 128,
            "too many maintenance scopes"
        );
        let mut maintenance_scopes = std::collections::HashSet::new();
        for entry in &application.maintenance {
            entry.scope.validate()?;
            entry.config.validate()?;
            ensure!(
                maintenance_scopes.insert(entry.scope.clone()),
                "duplicate maintenance scope"
            );
        }
        for entry in &application.entries {
            entry.scope.validate()?;
            entry.config.validate()?;
            ensure!(
                names.insert((entry.scope.clone(), entry.config.name.clone())),
                "duplicate trigger scope/name"
            );
            if let Some(source) = &entry.source {
                missy_automation::validate_source(source, &entry.config)?;
            } else {
                ensure!(
                    matches!(
                        entry.config.kind,
                        missy_automation::TriggerKind::Heartbeat
                            | missy_automation::TriggerKind::Schedule
                    ),
                    "trigger requires an explicit trusted observation source"
                );
            }
        }
        let scheduler = missy_automation::Scheduler::open(
            cfg.state_dir.join("scheduler.sqlite3"),
            runtime.clone(),
        )?;
        let store = runtime.store().clone();
        let cancel = stop.clone();
        services.spawn(async move {
            (
                "triggers",
                trigger_worker(scheduler, store, application, cancel).await,
            )
        });
        owner
            .status
            .subsystems
            .insert("triggers".into(), "ready".into());
    }
    if let Some(workloads) = workloads {
        owner.stage("workloads")?;
        let cancel = stop.clone();
        services.spawn(async move {
            let result = async {
                let mut tick = tokio::time::interval(Duration::from_secs(5));
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    tokio::select! { _=cancel.cancelled()=>return Ok(()), _=tick.tick()=>{} }
                    workloads
                        .run_due(chrono::Utc::now(), cancel.clone(), 16)
                        .await?;
                }
            }
            .await;
            ("workloads", result)
        });
        owner
            .status
            .subsystems
            .insert("workloads".into(), "ready".into());
    }
    if let Some(path) = &cfg.extensions.watchdog_config {
        owner.stage("watchdog")?;
        let config: crate::watchdog::WatchdogConfig = crate::load_extension(path)?;
        crate::watchdog::validate(&config, &cfg.policy)?;
        if config.enabled {
            let (policy, state, worker, stop) = (
                cfg.policy.clone(),
                cfg.state_dir.clone(),
                runtime.clone(),
                stop.child_token(),
            );
            services.spawn(async move {
                (
                    "watchdog",
                    crate::watchdog::run(config, policy, state, worker, stop).await,
                )
            });
            owner
                .status
                .subsystems
                .insert("watchdog".into(), "ready".into());
        } else {
            owner
                .status
                .subsystems
                .insert("watchdog".into(), "disabled".into());
        }
    }
    if cfg.discord.enabled {
        owner.stage("discord")?;
        use std::os::unix::fs::DirBuilderExt;
        let dir = cfg.state_dir.join("discord");
        if !dir.exists() {
            std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
        }
        let (ready, wait) = tokio::sync::oneshot::channel();
        let (runtime, config, policy, cancel) = (
            runtime.clone(),
            cfg.discord.clone(),
            cfg.policy.clone(),
            stop.clone(),
        );
        if let Some(application) = voice_application {
            ensure!(
                application.joins.len() <= 32,
                "too many automatic consented voice joins"
            );
            let safe_runtime = if let Some(safe) = &safe {
                safe.clone()
            } else {
                crate::build_safe_runtime(cfg)?
            };
            safe = Some(safe_runtime.clone());
            let (gateway, hook) = missy_gateway::discord_voice_bridge();
            let service = Arc::new(missy_discord_voice::DiscordVoiceService::new(
                application.server,
                Arc::new(missy_policy::PolicyHttpClient::new(cfg.policy.clone())?),
                gateway,
                Arc::new(missy_voice::CommandStt::new(application.stt)?),
                Arc::new(missy_voice::CommandTts::new(application.tts)?),
                Arc::new(crate::voice::RuntimeVoice {
                    full: runtime.clone(),
                    safe: safe_runtime,
                }),
            )?);
            voice_tools
                .context("voice tool catalog missing")?
                .attach(service.clone())?;
            let voice_stop = stop.clone();
            let lifecycle = service.clone();
            services.spawn(async move {
                voice_stop.cancelled().await;
                lifecycle.shutdown().await;
                ("discord_voice", Ok(()))
            });
            services.spawn(async move {
                (
                    "discord",
                    missy_gateway::serve_discord_with_voice_supervised(
                        runtime, config, policy, dir, hook, cancel, ready,
                    )
                    .await,
                )
            });
            wait.await
                .context("Discord exited before authenticated readiness")?;
            for join in application.joins {
                service
                    .join_cancellable(
                        join.guild,
                        join.channel,
                        join.actor,
                        &join.approval,
                        stop.child_token(),
                    )
                    .await?;
            }
            owner
                .status
                .subsystems
                .insert("discord_voice".into(), "ready".into());
        } else {
            services.spawn(async move {
                (
                    "discord",
                    missy_gateway::serve_discord_supervised(
                        runtime, config, policy, dir, cancel, ready,
                    )
                    .await,
                )
            });
            wait.await
                .context("Discord exited before authenticated readiness")?;
        }
        owner
            .status
            .subsystems
            .insert("discord".into(), "ready".into());
    }
    ensure!(
        !services.is_empty(),
        "no enabled native services remain after validating extensions"
    );
    owner.write()?;
    Ok((runtime, safe))
}
async fn scheduler_worker(
    scheduler: missy_automation::Scheduler,
    config: missy_core::SchedulerWorkerConfig,
    stop: CancellationToken,
) -> Result<()> {
    let mut tick = tokio::time::interval(Duration::from_secs(config.poll_seconds));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! { _ = stop.cancelled() => return Ok(()), _ = tick.tick() => {} }
        for scope in &config.scopes {
            if stop.is_cancelled() {
                return Ok(());
            }
            scheduler
                .run_due(scope, config.workers, stop.clone())
                .await?;
        }
    }
}
async fn trigger_worker(
    scheduler: missy_automation::Scheduler,
    store: Arc<missy_store::Store>,
    application: TriggerApplication,
    stop: CancellationToken,
) -> Result<()> {
    let mut tick = tokio::time::interval(Duration::from_secs(application.poll_seconds));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! { _=stop.cancelled()=>return Ok(()), _=tick.tick()=>{} }
        for entry in &application.entries {
            if stop.is_cancelled() {
                return Ok(());
            }
            let now = chrono::Utc::now();
            let observation = if let Some(source) = &entry.source {
                missy_automation::observe(source, &entry.config, &store, &entry.scope, now)?
            } else {
                missy_automation::TriggerObservation::default()
            };
            scheduler
                .fire_trigger(&entry.scope, &entry.config, &observation, now)
                .await?;
        }
        for entry in &application.maintenance {
            if stop.is_cancelled() {
                return Ok(());
            }
            scheduler.run_maintenance(&entry.scope, &entry.config, chrono::Utc::now())?;
        }
    }
}
/// sd_notify without a libsystemd runtime dependency; filesystem and abstract sockets.
fn notify(message: &str) {
    if let Some(path) = std::env::var_os("NOTIFY_SOCKET")
        && let Ok(socket) = std::os::unix::net::UnixDatagram::unbound()
    {
        use std::os::{linux::net::SocketAddrExt, unix::ffi::OsStrExt};
        let bytes = path.as_bytes();
        if let Some(name) = bytes.strip_prefix(b"@") {
            if let Ok(address) = std::os::unix::net::SocketAddr::from_abstract_name(name) {
                let _ = socket.send_to_addr(message.as_bytes(), &address);
            }
        } else {
            let _ = socket.send_to(message.as_bytes(), Path::new(&path));
        }
    }
}
