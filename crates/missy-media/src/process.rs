//! Generic native-command adapter. Only this crate's generated argv reaches it.
//! missy-sandbox owns a separate process group, cancellation, bounded combined
//! output, timeout, synchronous drop cleanup/reap, and child-only Landlock.
use crate::{ExecutablePin, MediaConfig};
use anyhow::{Result, ensure};
use missy_core::{PolicyConfig, ToolContext};
use missy_sandbox::{SandboxConfig, SandboxedCommand};
use serde_json::Value;
use std::path::Path;

pub(crate) async fn run(
    config: &MediaConfig,
    executable: &ExecutablePin,
    args: Vec<String>,
    staging: &Path,
    ctx: &ToolContext,
) -> Result<Value> {
    executable.verify()?;
    let mut argv = vec![executable.path.to_string_lossy().into_owned()];
    argv.extend(args);
    let mut reads = config.runtime_read_roots.clone();
    reads.push(staging.to_owned());
    // Independent internal capability, not inherited shell permission. The exact
    // full argv is generated from typed validated operations and copied inputs.
    // Caller tool arguments can never supply flags, filters or executable paths.
    let policy = PolicyConfig {
        shell_enabled: true,
        allowed_commands: vec![argv.clone()],
        read_roots: reads.clone(),
        write_roots: vec![staging.to_owned()],
        ..Default::default()
    };
    let sandbox = SandboxedCommand::new(
        policy,
        SandboxConfig {
            read_roots: reads,
            write_roots: vec![staging.to_owned()],
            deny_tcp: true,
            timeout_ms: config.timeout_ms,
            max_output_bytes: 128 * 1024,
        },
    )?;
    // Keep this explicit check: exact argv policy is an execution requirement,
    // not merely a builder assertion. execute independently checks it again.
    sandbox.check_command(&argv)?;
    let result = sandbox.execute(&argv, ctx).await.map_err(|error| {
        // A caller must reconcile an interrupted turn through its durable
        // journal, never blindly repeat a mutation after losing its future.
        if error.to_string().contains("timed out") || error.to_string().contains("cancelled") {
            anyhow::anyhow!("media outcome unknown; no automatic retry: {error}")
        } else {
            error
        }
    })?;
    ensure!(
        result["success"] == true,
        "native media command failed: {}",
        result["stderr"]
    );
    Ok(result)
}
