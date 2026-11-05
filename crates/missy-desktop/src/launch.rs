//! Explicit fixed application/package bindings. No PATH, shell, inherited
//! display/session environment, arbitrary argv or automatic privilege escalation.
use super::*;
use anyhow::ensure;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    os::unix::{fs::MetadataExt, io::AsRawFd},
    path::PathBuf,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesktopExecutablePin {
    pub path: PathBuf,
    pub sha256: String,
}
impl DesktopExecutablePin {
    fn open(&self) -> Result<std::fs::File> {
        ensure!(
            self.path.is_absolute() && self.path.canonicalize()? == self.path,
            "launch executable must be canonical"
        );
        let mut f = std::fs::File::open(&self.path)?;
        let m = f.metadata()?;
        ensure!(
            m.is_file()
                && m.mode() & 0o022 == 0
                && m.mode() & 0o111 != 0
                && m.len() <= 128 * 1024 * 1024,
            "unsafe executable"
        );
        let mut hash = Sha256::new();
        let mut buf = [0; 65536];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
        ensure!(
            hex::encode(hash.finalize()) == self.sha256,
            "executable pin mismatch"
        );
        Ok(f)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesktopCommand {
    pub executable: DesktopExecutablePin,
    pub args: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesktopLaunchConfig {
    pub isolated_session: bool,
    pub applications: BTreeMap<String, DesktopCommand>,
    /// Each key is one EXACT package/version; argv is fully operator supplied,
    /// e.g. /usr/bin/apt-get [install,-y,--,name=version]. Disabled when empty.
    pub packages: BTreeMap<String, DesktopCommand>,
    /// Explicit environment only. No ambient DISPLAY/D-Bus or credentials.
    pub environment: BTreeMap<String, String>,
    pub working_directory: PathBuf,
    pub max_children: usize,
}
impl DesktopLaunchConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.isolated_session && (1..=16).contains(&self.max_children),
            "launch requires isolated session and 1..16 children"
        );
        ensure!(
            self.working_directory.is_dir()
                && self.working_directory.canonicalize()? == self.working_directory,
            "invalid launch directory"
        );
        ensure!(
            self.applications.len() <= 64 && self.packages.len() <= 64,
            "too many command bindings"
        );
        for (name, cmd) in self.applications.iter().chain(&self.packages) {
            ensure!(
                !name.is_empty()
                    && name.len() <= 128
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-+=:~".contains(&b)),
                "invalid binding name"
            );
            ensure!(
                cmd.args.len() <= 64
                    && cmd
                        .args
                        .iter()
                        .all(|s| s.len() <= 8192 && !s.contains('\0')),
                "invalid fixed argv"
            );
            cmd.executable.open()?;
        }
        for (k, v) in &self.environment {
            ensure!(
                matches!(
                    k.as_str(),
                    "DISPLAY"
                        | "XAUTHORITY"
                        | "DBUS_SESSION_BUS_ADDRESS"
                        | "XDG_RUNTIME_DIR"
                        | "HOME"
                        | "LANG"
                        | "PATH"
                        | "DEBIAN_FRONTEND"
                ) && v.len() <= 4096
                    && !v.contains('\0'),
                "launch environment key/value denied"
            );
        }
        Ok(())
    }
}
#[derive(Default)]
pub(crate) struct LaunchState {
    children: tokio::sync::Mutex<Vec<tokio::process::Child>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct App {
    app: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Package {
    package: String,
}
fn selected<'a>(
    c: &'a DesktopLaunchConfig,
    call: &ToolCall,
) -> Result<(&'a DesktopCommand, String)> {
    let (map, key) = if call.name == "install_software_confirmed" {
        (&c.packages, parse::<Package>(call)?.package)
    } else {
        (&c.applications, parse::<App>(call)?.app)
    };
    let command = map
        .get(&key)
        .ok_or_else(|| anyhow::anyhow!("exact command binding not configured"))?;
    Ok((command, key))
}
pub(crate) fn validate_call(c: &DesktopLaunchConfig, call: &ToolCall) -> Result<()> {
    selected(c, call)?.0.executable.open()?;
    Ok(())
}
pub(crate) fn definitions(c: &DesktopLaunchConfig) -> Vec<ToolDefinition> {
    let mut out = Vec::new();
    for name in [
        "desktop_launch_app",
        "x11_launch",
        "install_software_confirmed",
    ] {
        let install = name == "install_software_confirmed";
        let map = if install {
            &c.packages
        } else {
            &c.applications
        };
        if map.is_empty() {
            continue;
        }
        let field = if install { "package" } else { "app" };
        let mut props = serde_json::Map::new();
        props.insert(
            field.into(),
            json!({"type":"string","enum":map.keys().collect::<Vec<_>>()}),
        );
        out.push(ToolDefinition {name:name.into(),description:if install {"Always approval-gated exact operator-bound package installation command. No arbitrary package, repository, flags, shell or automatic sudo. System mutation; timeout/cancellation requires reconciliation, never automatic retry."} else {"Approval-gated launch of exact pinned operator-bound application argv in an explicitly isolated session. No arbitrary args or ambient DISPLAY. Children are bounded, supervised and killed on shutdown."}.into(),parameters:json!({"type":"object","additionalProperties":false,"properties":props,"required":[field]}),side_effect:true});
    }
    out
}
pub(crate) async fn execute(
    c: &DesktopLaunchConfig,
    state: &LaunchState,
    call: &ToolCall,
) -> Result<Value> {
    let (binding, key) = selected(c, call)?;
    let pin = binding.executable.open()?;
    let mut children = state.children.lock().await;
    children.retain_mut(|child| matches!(child.try_wait(), Ok(None)));
    ensure!(
        children.len() < c.max_children,
        "supervised application capacity reached"
    );
    let mut command = tokio::process::Command::new(format!("/proc/self/fd/{}", pin.as_raw_fd()));
    command
        .args(&binding.args)
        .env_clear()
        .envs(&c.environment)
        .current_dir(&c.working_directory)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    // fexec-style descriptor path pins the verified inode through exec. No shell.
    let mut child = command.spawn()?;
    let pid = child.id();
    if call.name == "install_software_confirmed" {
        drop(children);
        let status = child.wait().await?;
        ensure!(
            status.success(),
            "installation command failed; reconcile package state before retry"
        );
        Ok(json!({"package":key,"success":true,"pid":pid,"validation_required":true}))
    } else {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if let Some(status) = child.try_wait()? {
            ensure!(status.success(), "application launch exited unsuccessfully");
            return Ok(json!({"app":key,"pid":pid,"exited":true}));
        }
        children.push(child);
        Ok(json!({"app":key,"pid":pid,"supervised":true}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn fixed_fixture_launch_pin_and_argument_denial() {
        let dir = tempfile::tempdir().unwrap();
        let path = std::fs::canonicalize("/usr/bin/true").unwrap();
        let pin = DesktopExecutablePin {
            sha256: hex::encode(Sha256::digest(std::fs::read(&path).unwrap())),
            path,
        };
        let cfg = DesktopLaunchConfig {
            isolated_session: true,
            applications: BTreeMap::from([(
                "fixture".into(),
                DesktopCommand {
                    executable: pin.clone(),
                    args: vec![],
                },
            )]),
            packages: BTreeMap::new(),
            environment: BTreeMap::new(),
            working_directory: dir.path().to_owned(),
            max_children: 1,
        };
        cfg.validate().unwrap();
        let call = ToolCall {
            id: "fixture".into(),
            name: "desktop_launch_app".into(),
            arguments: json!({"app":"fixture"}),
        };
        assert_eq!(
            execute(&cfg, &LaunchState::default(), &call).await.unwrap()["exited"],
            true
        );
        let mut bad = call.clone();
        bad.arguments["args"] = json!(["--arbitrary"]);
        assert!(validate_call(&cfg, &bad).is_err());
        bad = call.clone();
        bad.arguments["app"] = json!("not-configured");
        assert!(validate_call(&cfg, &bad).is_err());
        let mut cfg = cfg;
        cfg.applications
            .get_mut("fixture")
            .unwrap()
            .executable
            .sha256 = "00".repeat(32);
        assert!(cfg.validate().is_err());
    }
}
