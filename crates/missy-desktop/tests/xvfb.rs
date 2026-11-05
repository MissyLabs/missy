//! This test starts ONLY a new authenticated Xserver. It never reads DISPLAY,
//! XAUTHORITY or an existing session and sends no XTest input.
use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_desktop::{DesktopApproval, DesktopConfig, DesktopTools, X11Config};
use missy_policy::PolicyHttpClient;
use serde_json::json;
use std::{
    collections::BTreeSet,
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use x11rb::{
    connection::Connection,
    protocol::xproto::*,
    rust_connection::{DefaultStream, RustConnection},
    wrapper::ConnectionExt as _,
};

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Approval(AtomicUsize);
#[async_trait]
impl DesktopApproval for Approval {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "isolated-test".into()
    }
}
#[tokio::test]
async fn authenticated_isolated_xvfb_enumerates_and_screenshots_only_allowed_window() -> Result<()>
{
    use std::os::unix::fs::PermissionsExt;
    if !std::path::Path::new("/usr/bin/Xvfb").exists() {
        bail!("Xvfb unavailable: isolated acceptance not run");
    }
    let temp = tempfile::tempdir()?;
    let cookie = [42u8; 16];
    let cookie_path = temp.path().join("cookie");
    std::fs::write(&cookie_path, cookie)?;
    std::fs::set_permissions(&cookie_path, std::fs::Permissions::from_mode(0o600))?;
    // Xauth wildcard record for this fresh server only. Never load user auth.
    let authority = temp.path().join("authority");
    let mut file = std::fs::File::create(&authority)?;
    file.write_all(&65535u16.to_be_bytes())?;
    for value in [&b""[..], &b""[..], &b"MIT-MAGIC-COOKIE-1"[..], &cookie[..]] {
        file.write_all(&(value.len() as u16).to_be_bytes())?;
        file.write_all(value)?;
    }
    std::fs::set_permissions(&authority, std::fs::Permissions::from_mode(0o600))?;
    let child = Command::new("/usr/bin/Xvfb")
        .env_clear()
        .args([
            "-displayfd",
            "1",
            "-screen",
            "0",
            "128x96x24",
            "-nolisten",
            "tcp",
            "-auth",
        ])
        .arg(&authority)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut guard = ChildGuard(child);
    let stdout = guard.0.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stdout).read_line(&mut line).map(|_| line);
        let _ = tx.send(result);
    });
    let display: u16 = rx
        .recv_timeout(std::time::Duration::from_secs(10))??
        .trim()
        .parse()?;
    let socket = std::path::PathBuf::from(format!("/tmp/.X11-unix/X{display}"));
    let (stream, _) =
        DefaultStream::from_unix_stream(std::os::unix::net::UnixStream::connect(&socket)?)?;
    let c = RustConnection::connect_to_stream_with_auth_info(
        stream,
        0,
        b"MIT-MAGIC-COOKIE-1".to_vec(),
        cookie.to_vec(),
    )?;
    let root = c.setup().roots[0].root;
    let window = c.generate_id()?;
    c.create_window(
        c.setup().roots[0].root_depth,
        window,
        root,
        0,
        0,
        32,
        24,
        0,
        WindowClass::INPUT_OUTPUT,
        0,
        &CreateWindowAux::new().background_pixel(0x336699),
    )?
    .check()?;
    c.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"Isolated Missy Test",
    )?
    .check()?;
    c.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_CLASS,
        AtomEnum::STRING,
        b"missy\0MissyTest\0",
    )?
    .check()?;
    c.map_window(window)?.check()?;
    c.flush()?;
    let config = DesktopConfig {
        x11: Some(X11Config {
            socket_path: socket,
            screen: 0,
            cookie_path,
            allowed_classes: BTreeSet::from(["MissyTest".into()]),
            allowed_titles: BTreeSet::from(["Isolated Missy Test".into()]),
            isolated_session: true,
        }),
        enable_screenshots: true,
        approve_all: false,
        ..Default::default()
    };
    let approval = Arc::new(Approval(AtomicUsize::new(0)));
    let tools = DesktopTools::new(
        config,
        PolicyHttpClient::new(PolicyConfig::default())?,
        approval.clone(),
    )?;
    let ctx = ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "test".into(),
        cancellation: Default::default(),
    };
    let windows = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "list".into(),
                name: "desktop_windows".into(),
                arguments: json!({}),
            },
        )
        .await?;
    assert_eq!(windows["data"]["windows"][0]["window"], window);
    assert_eq!(windows["trust"], "untrusted");
    let image = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "image".into(),
                name: "desktop_screenshot".into(),
                arguments: json!({"window":window}),
            },
        )
        .await?;
    assert_eq!(
        image["data"]["artifact"]["media_type"],
        "application/x-x11-zpixmap"
    );
    assert_eq!(image["data"]["window"]["width"], 32);
    assert!(image["data"]["artifact"]["data"].as_str().unwrap().len() > 100);
    assert_eq!(approval.0.load(Ordering::SeqCst), 1);
    c.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"Not allowed",
    )?
    .check()?;
    let call = ToolCall {
        id: "deny".into(),
        name: "desktop_screenshot".into(),
        arguments: json!({"window":window}),
    };
    assert!(
        tools
            .execute(&ctx, &call)
            .await
            .unwrap_err()
            .to_string()
            .contains("denied")
    );
    let root_call = ToolCall {
        id: "root".into(),
        name: "desktop_screenshot".into(),
        arguments: json!({"window":root}),
    };
    assert!(tools.execute(&ctx, &root_call).await.is_err());
    Ok(())
}
