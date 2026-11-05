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
pub(crate) struct ReadScreenArgs {
    pub window: u32,
    pub question: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputArgs {
    pub window: u32,
    pub kind: String,
    pub code: Option<u8>,
    pub pressed: Option<bool>,
    pub x: Option<i16>,
    pub y: Option<i16>,
    pub to_x: Option<i16>,
    pub to_y: Option<i16>,
    pub text: Option<String>,
}
impl InputArgs {
    pub fn validate(&self) -> Result<()> {
        if self.window == 0 {
            bail!("invalid XTest window");
        }
        let coords =
            self.x.is_some() || self.y.is_some() || self.to_x.is_some() || self.to_y.is_some();
        match self.kind.as_str() {
            "key" | "button" => {
                let code = self.code.ok_or_else(|| anyhow::anyhow!("code required"))?;
                if self.pressed.is_none()
                    || coords
                    || self.text.is_some()
                    || (self.kind == "key" && code < 8)
                    || (self.kind == "button" && !(1..=5).contains(&code))
                {
                    bail!("invalid XTest key/button");
                }
            }
            "move" | "drag" => {
                if self.x.is_none_or(|v| v < 0)
                    || self.y.is_none_or(|v| v < 0)
                    || self.pressed.is_some()
                    || self.text.is_some()
                {
                    bail!("window-relative x/y required");
                }
                if self.kind == "move" {
                    if self.code.is_some() || self.to_x.is_some() || self.to_y.is_some() {
                        bail!("invalid move arguments");
                    }
                } else if self.to_x.is_none_or(|v| v < 0)
                    || self.to_y.is_none_or(|v| v < 0)
                    || self.code.is_none_or(|v| !(1..=3).contains(&v))
                {
                    bail!("drag requires bounded endpoint and button 1..3");
                }
            }
            "type_text" => {
                if coords
                    || self.code.is_some()
                    || self.pressed.is_some()
                    || self.text.as_ref().is_none_or(|v| {
                        v.is_empty() || v.len() > 256 || !v.bytes().all(|b| (32..=126).contains(&b))
                    })
                {
                    bail!("type_text requires 1..256 printable ASCII bytes only");
                }
            }
            _ => bail!("invalid XTest action"),
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
/// Handshake only; no enumeration, capture, focus query, or input.
pub(crate) fn probe(config: &X11Config) -> Result<()> {
    let (c, _deadline) = connection(config)?;
    if c.setup().roots.get(config.screen).is_none() {
        bail!("configured X11 screen missing");
    }
    Ok(())
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
        "desktop_focus_window" => {
            let a: WindowArgs = parse(call)?;
            c.grab_server()?.check()?;
            let result = (|| -> Result<Value> {
                let meta = allowed(&c, config, a.window)?;
                if cancellation.is_cancelled() {
                    bail!("focus cancelled");
                }
                c.configure_window(
                    a.window,
                    &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
                )?
                .check()?;
                c.set_input_focus(InputFocus::PARENT, a.window, x11rb::CURRENT_TIME)?
                    .check()?;
                c.flush()?;
                if c.get_input_focus()?.reply()?.focus != a.window {
                    bail!("window manager refused focus");
                }
                Ok(json!({"focused":true,"window":meta}))
            })();
            c.ungrab_server()?.check()?;
            c.flush()?;
            result
        }
        "desktop_screenshot" | "x11_read_screen" => {
            let window_call = ToolCall {
                arguments: json!({"window":call.arguments["window"]}),
                ..call.clone()
            };
            let a: WindowArgs = parse(&window_call)?;
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
            if call.name == "x11_read_screen" {
                let v = visual.ok_or_else(|| anyhow::anyhow!("screen visual missing"))?;
                let png = encode_png(
                    &image.data,
                    g.width as usize,
                    g.height as usize,
                    format.bits_per_pixel,
                    format.scanline_pad,
                    u8::from(c.setup().image_byte_order),
                    [v.red_mask, v.green_mask, v.blue_mask],
                )?;
                return Ok(
                    json!({"window":meta,"artifact":{"media_type":"image/png","encoding":"base64","data":STANDARD.encode(png)}}),
                );
            }
            Ok(
                json!({"window":meta,"artifact":{"media_type":"application/x-x11-zpixmap","encoding":"base64","data":STANDARD.encode(image.data),"depth":image.depth,"bits_per_pixel":format.bits_per_pixel,"scanline_pad":format.scanline_pad,"image_byte_order":u8::from(c.setup().image_byte_order),"red_mask":visual.map(|v|v.red_mask),"green_mask":visual.map(|v|v.green_mask),"blue_mask":visual.map(|v|v.blue_mask)}}),
            )
        }
        "desktop_input" => {
            let a: InputArgs = parse(call)?;
            a.validate()?;
            // Keep focus, geometry, allowlist and events in one server transaction.
            // This is bounded by the connection watchdog and never changes focus.
            c.grab_server()?.check()?;
            let result = native_input(&c, config, &a, screen.root, cancellation);
            c.ungrab_server()?.check()?;
            c.flush()?;
            result?;
            Ok(json!({"dispatched":true}))
        }
        _ => bail!("unknown X11 operation"),
    }
}

fn encode_png(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: u8,
    pad: u8,
    order: u8,
    masks: [u32; 3],
) -> Result<Vec<u8>> {
    if !matches!(bpp, 16 | 24 | 32)
        || !matches!(pad, 8 | 16 | 32)
        || width == 0
        || height == 0
        || width * height > 4_000_000
    {
        bail!("unsupported X11 pixel format");
    }
    let stride = (width * bpp as usize).div_ceil(pad as usize) * (pad as usize / 8);
    if data.len() != stride * height || masks.contains(&0) {
        bail!("invalid X11 image/masks");
    }
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let bytes = &data[y * stride + x * (bpp as usize / 8)..][..bpp as usize / 8];
            let mut value = 0u32;
            for (i, b) in bytes.iter().enumerate() {
                let shift = if order == 0 { i } else { bytes.len() - 1 - i };
                value |= (*b as u32) << (shift * 8);
            }
            for mask in masks {
                let shift = mask.trailing_zeros();
                let max = mask >> shift;
                pixels.push((((value & mask) >> shift) as u64 * 255 / max as u64) as u8);
            }
        }
    }
    crate::screen::reject_blank_rgb(&pixels)?;
    let image = image::RgbImage::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("invalid image dimensions"))?;
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image).write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

fn native_input(
    c: &RustConnection,
    config: &X11Config,
    a: &InputArgs,
    root: u32,
    cancellation: &tokio_util::sync::CancellationToken,
) -> Result<()> {
    allowed(c, config, a.window)?;
    let mut focus = c.get_input_focus()?.reply()?.focus;
    for _ in 0..32 {
        if focus == a.window {
            break;
        }
        if focus == 0 || focus == root || focus == 1 {
            bail!("input denied: window is not already focused");
        }
        focus = c.query_tree(focus)?.reply()?.parent;
    }
    if focus != a.window {
        bail!("focused descendant depth exceeded");
    }
    if cancellation.is_cancelled() {
        bail!("X11 input cancelled before dispatch");
    }
    let send = |event, code, x, y| -> Result<()> {
        c.xtest_fake_input(event, code, 0, root, x, y, 0)?.check()?;
        Ok(())
    };
    if a.kind == "key" || a.kind == "button" {
        let code = a.code.expect("validated");
        if a.kind == "key" && (code < c.setup().min_keycode || code > c.setup().max_keycode) {
            bail!("keycode outside server range");
        }
        if a.kind == "button" && !pointer_inside(c, a.window)? {
            bail!("pointer outside allowed window");
        }
        return send(
            match (a.kind.as_str(), a.pressed.expect("validated")) {
                ("key", true) => KEY_PRESS_EVENT,
                ("key", false) => KEY_RELEASE_EVENT,
                (_, true) => BUTTON_PRESS_EVENT,
                (_, false) => BUTTON_RELEASE_EVENT,
            },
            code,
            0,
            0,
        );
    }
    if a.kind == "type_text" {
        // Never modify the server keymap. Reject active modifiers/locks/buttons
        // and unsupported layouts rather than synthesizing surprising shortcuts.
        if u16::from(c.query_pointer(a.window)?.reply()?.mask) != 0 {
            bail!("typing requires no active modifiers, locks or buttons");
        }
        if c.query_keymap()?.reply()?.keys.iter().any(|v| *v != 0) {
            bail!("typing requires no physically held keys");
        }
        let min = c.setup().min_keycode;
        let map = c
            .get_keyboard_mapping(min, c.setup().max_keycode - min + 1)?
            .reply()?;
        let width = map.keysyms_per_keycode as usize;
        if width < 2 {
            bail!("typing requires explicit Latin unshifted/shifted keymap");
        }
        let key = |sym: u32| -> Option<(u8, bool)> {
            map.keysyms.chunks(width).enumerate().find_map(|(i, row)| {
                if row[0] == sym {
                    Some((min + i as u8, false))
                } else if row[1] == sym {
                    Some((min + i as u8, true))
                } else {
                    None
                }
            })
        };
        let shift = key(0xffe1)
            .ok_or_else(|| anyhow::anyhow!("Shift_L unavailable"))?
            .0;
        let keys: Vec<_> = a
            .text
            .as_ref()
            .expect("validated")
            .bytes()
            .map(|b| {
                key(b as u32)
                    .ok_or_else(|| anyhow::anyhow!("character unavailable in current keymap"))
            })
            .collect::<Result<_>>()?;
        for (code, shifted) in keys {
            if cancellation.is_cancelled() {
                bail!("typing cancelled between complete keystrokes");
            }
            if shifted {
                send(KEY_PRESS_EVENT, shift, 0, 0)?;
            }
            let press = send(KEY_PRESS_EVENT, code, 0, 0);
            let release = send(KEY_RELEASE_EVENT, code, 0, 0);
            let unshift = if shifted {
                send(KEY_RELEASE_EVENT, shift, 0, 0)
            } else {
                Ok(())
            };
            press?;
            release?;
            unshift?;
        }
        return Ok(());
    }
    let g = c.get_geometry(a.window)?.reply()?;
    if a.kind == "drag" && u16::from(c.query_pointer(a.window)?.reply()?.mask) != 0 {
        bail!("drag requires no active modifiers, locks or buttons");
    }
    let translated = |x: i16, y: i16| -> Result<(i16, i16)> {
        if x < 0 || y < 0 || x as u16 >= g.width || y as u16 >= g.height {
            bail!("coordinate outside window");
        }
        let t = c.translate_coordinates(a.window, root, x, y)?.reply()?;
        if !t.same_screen {
            bail!("cross-screen input denied");
        }
        let screen = &c.setup().roots[config.screen];
        if t.dst_x < 0
            || t.dst_y < 0
            || t.dst_x as u16 >= screen.width_in_pixels
            || t.dst_y as u16 >= screen.height_in_pixels
        {
            bail!("window coordinate is outside configured screen");
        }
        Ok((t.dst_x, t.dst_y))
    };
    let (x, y) = translated(a.x.expect("validated"), a.y.expect("validated"))?;
    let end = if a.kind == "drag" {
        Some(translated(
            a.to_x.expect("validated"),
            a.to_y.expect("validated"),
        )?)
    } else {
        None
    };
    send(MOTION_NOTIFY_EVENT, 0, x, y)?;
    if !pointer_inside(c, a.window)? {
        bail!("pointer target obscured or outside allowed window");
    }
    if let Some((to_x, to_y)) = end {
        let button = a.code.expect("validated");
        // Check endpoint visibility before pressing, then restore the start.
        send(MOTION_NOTIFY_EVENT, 0, to_x, to_y)?;
        if !pointer_inside(c, a.window)? {
            bail!("drag endpoint obscured");
        }
        send(MOTION_NOTIFY_EVENT, 0, x, y)?;
        send(BUTTON_PRESS_EVENT, button, 0, 0)?;
        let motion = send(MOTION_NOTIFY_EVENT, 0, to_x, to_y);
        let release = send(BUTTON_RELEASE_EVENT, button, 0, 0);
        motion?;
        release?;
    }
    Ok(())
}
fn pointer_inside(c: &RustConnection, w: u32) -> Result<bool> {
    let p = c.query_pointer(w)?.reply()?;
    let g = c.get_geometry(w)?.reply()?;
    // Translate root pointer back into the selected client and require the
    // root's top child to be this client or an ancestor (WM frame).
    if !p.same_screen
        || p.win_x < 0
        || p.win_y < 0
        || p.win_x as u16 >= g.width
        || p.win_y as u16 >= g.height
    {
        return Ok(false);
    }
    let t = c
        .translate_coordinates(g.root, w, p.root_x, p.root_y)?
        .reply()?;
    if t.dst_x != p.win_x || t.dst_y != p.win_y {
        return Ok(false);
    }
    let top = c.query_pointer(g.root)?.reply()?.child;
    let mut current = w;
    for _ in 0..32 {
        if current == top {
            return Ok(true);
        }
        let parent = c.query_tree(current)?.reply()?.parent;
        if parent == g.root || parent == 0 {
            break;
        }
        current = parent;
    }
    Ok(false)
}
