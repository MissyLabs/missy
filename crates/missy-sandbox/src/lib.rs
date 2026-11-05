//! Opt-in Linux child-only Landlock execution and hardened container argv construction.
//! No public function restricts the calling thread. See README.md for threat limits.
#![cfg(target_os = "linux")]

mod container;
mod landlock;
mod process;
use anyhow::{Result, bail};
pub use container::{ContainerCommand, ContainerConfig, build_container_command};
pub use landlock::landlock_abi;
use missy_core::{PolicyConfig, ToolContext};
use missy_policy::Policy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SandboxConfig {
    pub read_roots: Vec<PathBuf>,
    pub write_roots: Vec<PathBuf>,
    /// TCP bind/connect denial requires ABI >= 4. This does NOT cover UDP/Unix.
    pub deny_tcp: bool,
    pub timeout_ms: u64,
    /// Combined raw stdout/stderr byte cap; overflow kills the process group.
    pub max_output_bytes: usize,
}
impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            read_roots: vec![],
            write_roots: vec![],
            deny_tcp: true,
            timeout_ms: 30_000,
            max_output_bytes: 64 * 1024,
        }
    }
}
impl SandboxConfig {
    pub fn validate(&self, policy: &PolicyConfig) -> Result<()> {
        if self.timeout_ms == 0
            || self.timeout_ms > 3_600_000
            || self.max_output_bytes == 0
            || self.max_output_bytes > 16 * 1024 * 1024
        {
            bail!("sandbox timeout/output limit outside hard bounds");
        }
        for (roots, approved) in [
            (&self.read_roots, &policy.read_roots),
            (&self.write_roots, &policy.write_roots),
        ] {
            for root in roots {
                validate_path(root)?;
                if !approved
                    .iter()
                    .any(|allowed| validate_path(allowed).is_ok() && root.starts_with(allowed))
                {
                    bail!("sandbox filesystem root is outside corresponding policy roots");
                }
            }
        }
        Ok(())
    }
}
pub(crate) fn validate_path(path: &Path) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    if !path.is_absolute()
        || path.as_os_str().as_bytes().contains(&0)
        || path
            .as_os_str()
            .as_bytes()
            .split(|b| *b == b'/')
            .any(|c| c == b"." || c == b"..")
        || path
            .components()
            .any(|c| !matches!(c, Component::RootDir | Component::Normal(_)))
    {
        bail!("sandbox paths must be absolute without dot components or NUL");
    }
    Ok(())
}
/// One-shot command execution, NOT persistent MCP transport/session support.
pub struct SandboxedCommand {
    policy: Policy,
    config: SandboxConfig,
}
impl SandboxedCommand {
    pub fn new(policy: PolicyConfig, config: SandboxConfig) -> Result<Self> {
        config.validate(&policy)?;
        landlock::require_abi(landlock_abi()?, config.deny_tcp)?;
        Ok(Self {
            policy: Policy::new(policy)?,
            config,
        })
    }
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }
    pub fn check_command(&self, argv: &[String]) -> Result<()> {
        self.policy.check_command(argv)
    }
    pub fn check_mcp_command(&self, argv: &[String]) -> Result<()> {
        self.policy.check_mcp_command(argv)
    }
    pub async fn execute(&self, argv: &[String], ctx: &ToolContext) -> Result<Value> {
        self.check_command(argv)?;
        self.run(argv, ctx).await
    }
    pub async fn execute_mcp(&self, argv: &[String], ctx: &ToolContext) -> Result<Value> {
        self.check_mcp_command(argv)?;
        self.run(argv, ctx).await
    }
    async fn run(&self, argv: &[String], ctx: &ToolContext) -> Result<Value> {
        if ctx.cancellation.is_cancelled() {
            bail!("sandbox command cancelled before spawn");
        }
        process::execute(
            argv,
            &self.config,
            landlock::Prepared::new(&self.config)?,
            ctx,
        )
        .await
    }
}
