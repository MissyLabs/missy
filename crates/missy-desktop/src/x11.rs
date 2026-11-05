use crate::parse;
use anyhow::{Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use missy_core::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet, io::Read, os::unix::net::UnixStream, path::PathBuf, sync::mpsc,
    time::Duration,
};
use x11rb::{
    connection::Connection,
    protocol::{xproto::*, xtest::ConnectionExt as _},
    rust_connection::{DefaultStream, RustConnection},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct X11Config {
    /// Explicit local socket, e.g. /tmp/.X11-unix/X90. DISPLAY is never consulted.
    pub socket_path: PathBuf,
    pub screen: usize,
    /// Explicit file containing exactly 16 raw MIT-MAGIC-COOKIE-1 bytes, not an
    /// Xauthority file. File permissions must exclude group/other access.
    pub cookie_path: PathBuf,
    pub allowed_classes: BTreeSet<String>,
    pub allowed_titles: BTreeSet<String>,
    /// Required: X11 cannot enforce per-window input or screenshot isolation.
    pub isolated_session: bool,
}
impl X11Config {
    pub fn validate(&self) -> Result<()> {
        if !self.isolated_session
            || !self.socket_path.is_absolute()
            || !self.cookie_path.is_absolute()
            || self.allowed_classes.is_empty()
            || self.allowed_titles.is_empty()
            || self
                .allowed_classes
                .iter()
                .chain(self.allowed_titles.iter())
                .any(|s| s.is_empty() || s.len() > 1024)
        {
            bail!(
                "X11 requires explicit isolated socket, cookie file and nonempty exact class/title allowlists"
            );
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WindowArgs {
    pub window: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputArgs {
    pub window: u32,
    pub kind: String,
    pub code: u8,
    pub pressed: bool,
}
impl InputArgs {
    pub fn validate(&self) -> Result<()> {
        if self.window == 0
            || !matches!(self.kind.as_str(), "key" | "button")
            || (self.kind == "key" && self.code < 8)
            || (self.kind == "button" && !(1..=5).contains(&self.code))
        {
            bail!("invalid XTest action");
        }
        Ok(())
    }
}

fn connection(config: &X11Config) -> Result<(RustConnection, mpsc::Sender<()>)> {
    use std::os::unix::fs::PermissionsExt;
    let mut file = std::fs::File::open(&config.cookie_path)?;
    if file.metadata()?.permissions().mode() & 0o077 != 0 {
        bail!("X11 cookie permissions must exclude group and other");
    }
    let mut cookie = Vec::new();
    file.by_ref().take(17).read_to_end(&mut cookie)?;
    if cookie.len() != 16 {
        bail!("X11 cookie must contain exactly 16 raw bytes");
    }
    let socket = socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None)?;
    socket.connect_timeout(
        &socket2::SockAddr::unix(&config.socket_path)?,
        Duration::from_secs(2),
    )?;
    let socket: UnixStream = socket.into();
    let deadline_socket = socket.try_clone()?;
    // x11rb's nonblocking poll may otherwise outlive cancellation indefinitely.
    // Closing this connection after five seconds bounds even malformed servers.
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        if matches!(
            rx.recv_timeout(Duration::from_secs(5)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            let _ = deadline_socket.shutdown(std::net::Shutdown::Both);
        }
    });
    let (stream, _) = DefaultStream::from_unix_stream(socket)?;
    let connection = RustConnection::connect_to_stream_with_auth_info(
        stream,
        config.screen,
        b"MIT-MAGIC-COOKIE-1".to_vec(),
        cookie,
    )?;
    Ok((connection, tx))
}
fn property(c: &RustConnection, w: u32, atom: Atom, ty: Atom) -> Result<Vec<u8>> {
    let reply = c.get_property(false, w, atom, ty, 0, 512)?.reply()?;
    if reply.bytes_after != 0 {
        bail!("X11 property exceeds limit");
    }
    Ok(reply.value)
}
fn metadata(c: &RustConnection, config: &X11Config, w: u32) -> Result<Value> {
    let net_name = c.intern_atom(false, b"_NET_WM_NAME")?.reply()?.atom;
    let utf8 = c.intern_atom(false, b"UTF8_STRING")?.reply()?.atom;
    let mut title = property(c, w, net_name, utf8)?;
    if title.is_empty() {
        title = property(c, w, AtomEnum::WM_NAME.into(), AtomEnum::STRING.into())?;
    }
    let title = String::from_utf8(title)?;
    let class = property(c, w, AtomEnum::WM_CLASS.into(), AtomEnum::STRING.into())?;
    let class = class
        .split(|b| *b == 0)
        .rfind(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing WM_CLASS"))?;
    let class = std::str::from_utf8(class)?;
    if !config.allowed_classes.contains(class) || !config.allowed_titles.contains(&title) {
        bail!("X11 window denied");
    }
    let geo = c.get_geometry(w)?.reply()?;
    Ok(
        json!({"window":w,"title":title,"class":class,"x":geo.x,"y":geo.y,"width":geo.width,"height":geo.height,"depth":geo.depth}),
    )
}
fn allowed(c: &RustConnection, config: &X11Config, w: u32) -> Result<Value> {
    let root = c
        .setup()
        .roots
        .get(config.screen)
        .ok_or_else(|| anyhow::anyhow!("X11 screen missing"))?
        .root;
    if w == 0 || w == root {
        bail!("root screenshots/input forbidden");
    }
    // Require an actual top-level client, not a model-chosen arbitrary drawable.
    let tree = c.query_tree(root)?.reply()?;
    let atom = c.intern_atom(false, b"_NET_CLIENT_LIST")?.reply()?.atom;
    let clients = c
        .get_property(false, root, atom, AtomEnum::WINDOW, 0, 4096)?
        .reply()?;
    let listed = clients
        .value32()
        .is_some_and(|mut windows| windows.any(|v| v == w));
    if !tree.children.contains(&w) && !listed {
        bail!("window not an enumerated top-level client");
    }
    metadata(c, config, w)
}
pub(crate) fn execute(
    config: &X11Config,
    call: &ToolCall,
    limit: usize,
    cancellation: &tokio_util::sync::CancellationToken,
) -> Result<Value> {
    if cancellation.is_cancelled() {
        bail!("X11 operation cancelled before connection");
    }
    let (c, _deadline) = connection(config)?;
    let screen = c
        .setup()
        .roots
        .get(config.screen)
        .ok_or_else(|| anyhow::anyhow!("screen not found"))?;
    match call.name.as_str() {
        "desktop_windows" => {
            let atom = c.intern_atom(false, b"_NET_CLIENT_LIST")?.reply()?.atom;
            let clients = c
                .get_property(false, screen.root, atom, AtomEnum::WINDOW, 0, 4096)?
                .reply()?;
            if clients.bytes_after != 0 {
                bail!("too many X11 clients");
            }
            let mut candidates = c.query_tree(screen.root)?.reply()?.children;
            if let Some(ids) = clients.value32() {
                candidates.extend(ids);
            }
            candidates.sort_unstable();
            candidates.dedup();
            if candidates.len() > 4096 {
                bail!("too many X11 children");
            }
            Ok(
                json!({"windows":candidates.into_iter().filter_map(|w| metadata(&c,config,w).ok()).take(256).collect::<Vec<_>>() }),
            )
        }
        "desktop_screenshot" => {
            let a: WindowArgs = parse(call)?;
            let meta = allowed(&c, config, a.window)?;
            let g = c.get_geometry(a.window)?.reply()?;
            if g.width as usize * g.height as usize * 4 > limit / 2 {
                bail!("X11 image exceeds byte budget");
            }
            if cancellation.is_cancelled() {
                bail!("X11 screenshot cancelled");
            }
            let image = c
                .get_image(
                    ImageFormat::Z_PIXMAP,
                    a.window,
                    0,
                    0,
                    g.width,
                    g.height,
                    u32::MAX,
                )?
                .reply()?;
            if image.data.len() > limit / 2 {
                bail!("X11 image exceeds byte budget");
            }
            let format = c
                .setup()
                .pixmap_formats
                .iter()
                .find(|f| f.depth == image.depth)
                .ok_or_else(|| anyhow::anyhow!("missing pixel format"))?;
            let visual = screen
                .allowed_depths
                .iter()
                .flat_map(|d| &d.visuals)
                .find(|v| v.visual_id == image.visual);
            Ok(
                json!({"window":meta,"artifact":{"media_type":"application/x-x11-zpixmap","encoding":"base64","data":STANDARD.encode(image.data),"depth":image.depth,"bits_per_pixel":format.bits_per_pixel,"scanline_pad":format.scanline_pad,"image_byte_order":u8::from(c.setup().image_byte_order),"red_mask":visual.map(|v|v.red_mask),"green_mask":visual.map(|v|v.green_mask),"blue_mask":visual.map(|v|v.blue_mask)}}),
            )
        }
        "desktop_input" => {
            let a: InputArgs = parse(call)?;
            a.validate()?;
            allowed(&c, config, a.window)?;
            if c.get_input_focus()?.reply()?.focus != a.window {
                bail!("input denied: window is not already focused");
            }
            if a.kind == "key" && (a.code < c.setup().min_keycode || a.code > c.setup().max_keycode)
            {
                bail!("keycode outside server range");
            }
            if a.kind == "button" && c.query_pointer(screen.root)?.reply()?.child != a.window {
                bail!("pointer is outside allowed window");
            }
            let type_ = match (a.kind.as_str(), a.pressed) {
                ("key", true) => KEY_PRESS_EVENT,
                ("key", false) => KEY_RELEASE_EVENT,
                (_, true) => BUTTON_PRESS_EVENT,
                (_, false) => BUTTON_RELEASE_EVENT,
            };
            if cancellation.is_cancelled() {
                bail!("X11 input cancelled before dispatch");
            }
            c.xtest_fake_input(type_, a.code, 0, screen.root, 0, 0, 0)?
                .check()?;
            c.flush()?;
            Ok(json!({"dispatched":true}))
        }
        _ => bail!("unknown X11 operation"),
    }
}
