//! Opt-in, chat-only native CLI/ACP transport. No Python, shell command construction,
//! credential import, native tools, or implicit executable/PATH discovery.
//!
//! Use with a NoTools executor. These adapters deliberately reject tool schemas
//! and tool history instead of outsourcing Missy's policy to a CLI agent.
#![cfg(unix)]

mod acp;
mod process;
mod protocol;

use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_core::{Completion, Message, Provider, ToolDefinition};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CliProviderKind {
    ClaudeCli,
    /// Recognized for migration, but unavailable: read-only is not no-tools.
    CodexCli,
    /// Direct ACP stdio, restricted to the audited Claude adapter/version.
    #[serde(alias = "claude_acp")]
    Acpx,
}

/// Trusted operator configuration, never supplied by a model or a chat request.
/// Executable scripts are allowed only because this is an explicit executable
/// trust decision. The adapter does not invoke a shell or accept extra argv/env.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CliProviderConfig {
    pub name: String,
    pub kind: CliProviderKind,
    pub executable: PathBuf,
    /// Explicit operator-owned HOME for the CLI's own login. Never read by Missy.
    pub auth_home: PathBuf,
    pub model: String,
    #[serde(default)]
    pub approved_executable: bool,
    #[serde(default)]
    pub acp_adapter: Option<AcpAdapterPin>,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
    #[serde(default = "default_stdout")]
    pub max_stdout_bytes: usize,
    #[serde(default = "default_stderr")]
    pub max_stderr_bytes: usize,
    #[serde(default = "default_input")]
    pub max_input_bytes: usize,
    #[serde(default)]
    pub input_microusd_per_million: Option<u64>,
    #[serde(default)]
    pub output_microusd_per_million: Option<u64>,
}
fn default_timeout() -> u64 {
    120
}
fn default_stdout() -> usize {
    1024 * 1024
}
fn default_stderr() -> usize {
    64 * 1024
}
fn default_input() -> usize {
    1024 * 1024
}

pub struct CliProvider {
    config: CliProviderConfig,
}

/// Operator-provisioned local entrypoint. No npx, downloads or PATH lookup.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcpAdapterPin {
    pub entrypoint: PathBuf,
    pub sha256: String,
}

pub fn create_provider(config: CliProviderConfig) -> Result<Arc<dyn Provider>> {
    Ok(Arc::new(CliProvider::new(config)?))
}

impl CliProvider {
    pub fn new(mut config: CliProviderConfig) -> Result<Self> {
        if config.kind == CliProviderKind::CodexCli {
            bail!(
                "codex_cli disabled: read-only sandbox still permits native tool execution; a verified no-tools protocol is required"
            );
        }
        validate_config(&mut config)?;
        Ok(Self { config })
    }

    /// Display argv as data, not a shell command. Run manually in the intended
    /// operator environment. This method does not initiate OAuth or open a URL.
    pub fn manual_login_argv(kind: CliProviderKind, executable: &Path) -> Vec<String> {
        let mut args = vec![executable.to_string_lossy().into_owned()];
        match kind {
            CliProviderKind::ClaudeCli => args.extend(["auth".into(), "login".into()]),
            CliProviderKind::CodexCli => args.push("login".into()),
            CliProviderKind::Acpx => return Vec::new(),
        }
        args
    }

    /// Non-inference status probe only. Never returns raw stdout/stderr, tokens,
    /// account metadata, or authentication URLs. No credential files are read.
    /// Codex may be probed even though its completion adapter is disabled.
    pub async fn probe_auth(
        mut config: CliProviderConfig,
        cancellation: CancellationToken,
    ) -> Result<AuthStatus> {
        if config.kind == CliProviderKind::Acpx {
            bail!(
                "ACP has no non-inference authentication status command; provision the pinned adapter manually"
            );
        }
        validate_config(&mut config)?;
        config.timeout_seconds = config.timeout_seconds.min(20);
        let cwd = tempfile::Builder::new()
            .prefix("missy-cli-status-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir()
            .map_err(|_| anyhow::anyhow!("cannot create private CLI status directory"))?;
        let args: Vec<String> = match config.kind {
            CliProviderKind::ClaudeCli => vec!["auth".into(), "status".into()],
            CliProviderKind::CodexCli => vec!["login".into(), "status".into()],
            CliProviderKind::Acpx => unreachable!("rejected before spawning"),
        };
        let output = process::run(&config, &args, cwd.path(), &[], cancellation).await?;
        if !output.success {
            return Ok(AuthStatus::NotAuthenticated);
        }
        if config.kind == CliProviderKind::CodexCli {
            return Ok(AuthStatus::Authenticated);
        }
        let value: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|_| anyhow::anyhow!("CLI status returned invalid protocol"))?;
        match value.get("loggedIn").and_then(serde_json::Value::as_bool) {
            Some(true) => Ok(AuthStatus::Authenticated),
            Some(false) => Ok(AuthStatus::NotAuthenticated),
            None => bail!("CLI status returned invalid protocol"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthStatus {
    Authenticated,
    NotAuthenticated,
}

fn validate_config(config: &mut CliProviderConfig) -> Result<()> {
    if !config.approved_executable {
        bail!("CLI executable requires explicit operator approval");
    }
    if config.name.is_empty()
        || config.name.len() > 128
        || config.model.is_empty()
        || config.model.len() > 200
        || !config
            .model
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
    {
        bail!("invalid CLI provider name or model");
    }
    if !(1..=3600).contains(&config.timeout_seconds)
        || !(1..=16 * 1024 * 1024).contains(&config.max_stdout_bytes)
        || !(1..=1024 * 1024).contains(&config.max_stderr_bytes)
        || !(1..=16 * 1024 * 1024).contains(&config.max_input_bytes)
    {
        bail!("CLI limits are outside supported bounds");
    }
    if !config.executable.is_absolute() || !config.auth_home.is_absolute() {
        bail!("CLI executable and auth_home must be exact absolute paths");
    }
    config.executable = fs::canonicalize(&config.executable)
        .map_err(|_| anyhow::anyhow!("CLI executable is unavailable"))?;
    let metadata = fs::metadata(&config.executable)
        .map_err(|_| anyhow::anyhow!("CLI executable is unavailable"))?;
    if !metadata.is_file() || metadata.mode() & 0o111 == 0 || !trusted_metadata(&metadata) {
        bail!("CLI executable must be an operator-owned executable, not group/world-writable");
    }
    for parent in config.executable.ancestors().skip(1) {
        let meta = fs::metadata(parent)
            .map_err(|_| anyhow::anyhow!("CLI executable parent is unavailable"))?;
        if !(trusted_metadata(&meta) || meta.uid() == 0 && meta.mode() & 0o1000 != 0) {
            bail!("CLI executable has an untrusted writable parent");
        }
    }
    config.auth_home = fs::canonicalize(&config.auth_home)
        .map_err(|_| anyhow::anyhow!("CLI auth_home is unavailable; provision it manually"))?;
    let home = fs::metadata(&config.auth_home)
        .map_err(|_| anyhow::anyhow!("CLI auth_home is unavailable"))?;
    if !home.is_dir() || !trusted_metadata(&home) {
        bail!("CLI auth_home must be operator-owned and not group/world-writable");
    }
    if config.kind == CliProviderKind::Acpx {
        let mut magic = [0; 4];
        fs::File::open(&config.executable)
            .and_then(|mut file| file.read_exact(&mut magic))
            .map_err(|_| anyhow::anyhow!("ACP runtime executable unavailable"))?;
        if magic != *b"\x7fELF" {
            bail!("ACP requires an exact native ELF runtime executable, not a script or launcher");
        }
        let pin = config.acp_adapter.as_mut().ok_or_else(|| {
            anyhow::anyhow!("ACP requires an operator-pinned local adapter entrypoint")
        })?;
        if !pin.entrypoint.is_absolute()
            || pin.sha256.len() != 64
            || !pin.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            bail!("ACP requires an absolute adapter entrypoint and SHA-256");
        }
        pin.entrypoint = fs::canonicalize(&pin.entrypoint)
            .map_err(|_| anyhow::anyhow!("ACP adapter is unavailable"))?;
        for path in pin.entrypoint.ancestors() {
            let meta =
                fs::metadata(path).map_err(|_| anyhow::anyhow!("ACP adapter path unavailable"))?;
            if !(trusted_metadata(&meta) || meta.uid() == 0 && meta.mode() & 0o1000 != 0) {
                bail!("ACP adapter has an untrusted writable path");
            }
        }
        verify_adapter(pin)?;
    } else if config.acp_adapter.is_some() {
        bail!("ACP adapter pin is only valid for acpx");
    }
    Ok(())
}

fn verify_adapter(pin: &AcpAdapterPin) -> Result<()> {
    let metadata =
        fs::metadata(&pin.entrypoint).map_err(|_| anyhow::anyhow!("ACP adapter unavailable"))?;
    if !metadata.is_file() || metadata.len() > 16 * 1024 * 1024 {
        bail!("ACP adapter must be a bounded regular file");
    }
    let bytes =
        fs::read(&pin.entrypoint).map_err(|_| anyhow::anyhow!("ACP adapter unavailable"))?;
    if !hex::encode(Sha256::digest(bytes)).eq_ignore_ascii_case(&pin.sha256) {
        bail!("ACP adapter SHA-256 mismatch");
    }
    Ok(())
}
fn trusted_metadata(meta: &fs::Metadata) -> bool {
    // SAFETY: geteuid has no memory or process side effects.
    (meta.uid() == 0 || meta.uid() == unsafe { libc::geteuid() }) && meta.mode() & 0o022 == 0
}

#[async_trait]
impl Provider for CliProvider {
    fn name(&self) -> &str {
        &self.config.name
    }
    fn configuration_fingerprint(&self) -> String {
        // Config contains paths and rates, never credentials. Return only a digest.
        hex::encode(Sha256::digest(
            serde_json::to_vec(&self.config).expect("serializable CLI configuration"),
        ))
    }
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        cancellation: CancellationToken,
    ) -> Result<Completion> {
        if !tools.is_empty() {
            bail!("CLI providers are chat-only; configure a NoTools runtime");
        }
        let (system, prompt) = protocol::prompt(messages, self.config.max_input_bytes)?;
        if cancellation.is_cancelled() {
            bail!("CLI completion cancelled");
        }
        let cwd = tempfile::Builder::new()
            .prefix("missy-cli-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir()
            .map_err(|_| anyhow::anyhow!("cannot create private CLI directory"))?;
        if self.config.kind == CliProviderKind::Acpx {
            verify_adapter(self.config.acp_adapter.as_ref().expect("validated ACP pin"))?;
            return acp::complete(&self.config, cwd.path(), system, prompt, cancellation).await;
        }
        let system_path = cwd.path().join("system.txt");
        let mcp_path = cwd.path().join("mcp.json");
        fs::write(&system_path, system)
            .map_err(|_| anyhow::anyhow!("cannot write private CLI system prompt"))?;
        fs::write(&mcp_path, b"{\"mcpServers\":{}}")
            .map_err(|_| anyhow::anyhow!("cannot write private empty MCP configuration"))?;
        let args = claude_args(&self.config.model, &system_path, &mcp_path);
        let output = process::run(
            &self.config,
            &args,
            cwd.path(),
            prompt.as_bytes(),
            cancellation,
        )
        .await?;
        if !output.success {
            bail!("CLI completion failed; subprocess diagnostics withheld");
        }
        protocol::parse_claude(&output.stdout, &self.config)
    }
}

fn claude_args(model: &str, system: &Path, mcp: &Path) -> Vec<String> {
    [
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--tools",
        "",
        "--disallowedTools",
        "mcp__*",
        "--strict-mcp-config",
        "--mcp-config",
        mcp.to_str().expect("UTF-8 temp path"),
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--no-chrome",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--max-turns",
        "1",
        "--settings",
        "{\"switchModelsOnFlag\":false,\"disableAllHooks\":true}",
        "--system-prompt-file",
        system.to_str().expect("UTF-8 temp path"),
        "--model",
        model,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}
