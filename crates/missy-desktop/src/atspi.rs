//! Typed AT-SPI protocol. Connect only to an explicitly supplied accessibility
//! bus; never discovers org.a11y.Bus through the user's session bus.
use crate::parse;
use anyhow::{Result, bail};
use missy_core::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeSet, VecDeque};
use zbus::OwnedObjectPath;

type ObjectRef = (String, OwnedObjectPath);

#[zbus::proxy(interface = "org.a11y.atspi.Accessible")]
trait Accessible {
    #[zbus(property)]
    fn name(&self) -> zbus::Result<String>;
    fn get_role(&self) -> zbus::Result<u32>;
    fn get_role_name(&self) -> zbus::Result<String>;
    fn get_children(&self) -> zbus::Result<Vec<ObjectRef>>;
}
#[zbus::proxy(interface = "org.a11y.atspi.Action")]
trait Action {
    #[zbus(property, name = "NActions")]
    fn n_actions(&self) -> zbus::Result<i32>;
    fn get_name(&self, index: i32) -> zbus::Result<String>;
    fn do_action(&self, index: i32) -> zbus::Result<bool>;
}
#[zbus::proxy(interface = "org.a11y.atspi.Text")]
trait Text {
    fn get_text(&self, start_offset: i32, end_offset: i32) -> zbus::Result<String>;
}
#[zbus::proxy(interface = "org.a11y.atspi.EditableText")]
trait EditableText {
    fn set_text_contents(&self, new_contents: &str) -> zbus::Result<bool>;
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextArgs {
    application: String,
    window: String,
    path: String,
    start: Option<i32>,
    end: Option<i32>,
    value: Option<String>,
}
pub(crate) fn validate_text_call(call: &ToolCall) -> Result<()> {
    let a: TextArgs = parse(call)?;
    ActionArgs {
        application: a.application,
        window: a.window,
        path: a.path,
        action: 0,
    }
    .validate()?;
    match call.name.as_str() {
        "desktop_accessibility_get_text" if a.value.is_none() => match (a.start, a.end) {
            (Some(start), Some(end)) if start >= 0 && end >= start && end <= 4096 => {}
            _ => bail!("text offsets require 0 <= start <= end <= 4096"),
        },
        "desktop_accessibility_set_value" if a.start.is_none() && a.end.is_none() => {
            if a.value
                .as_ref()
                .is_none_or(|v| v.len() > 8192 || v.contains('\0'))
            {
                bail!("bounded value required");
            }
        }
        _ => bail!("invalid typed text operation"),
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtspiConfig {
    /// Accessibility bus address, not a session-bus discovery hint.
    pub bus_address: String,
    pub allowed_applications: BTreeSet<String>,
    pub allowed_windows: BTreeSet<String>,
    /// AT-SPI exposes an entire session; isolate this bus from the real desktop.
    pub isolated_session: bool,
}
impl AtspiConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.isolated_session
            || !self.bus_address.starts_with("unix:path=/")
            || self.bus_address.contains([';', '\0', '\n'])
            || self.bus_address.len() > 4096
            || self.allowed_applications.is_empty()
            || self.allowed_windows.is_empty()
            || self
                .allowed_applications
                .iter()
                .chain(&self.allowed_windows)
                .any(|s| s.is_empty() || s.len() > 1024)
        {
            bail!(
                "AT-SPI requires explicit isolated Unix bus and exact application/window allowlists"
            );
        }
        Ok(())
    }
}
fn default_nodes() -> usize {
    128
}
fn default_depth() -> usize {
    4
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TreeArgs {
    application: String,
    window: String,
    #[serde(default = "default_nodes")]
    max_nodes: usize,
    #[serde(default = "default_depth")]
    max_depth: usize,
}
impl TreeArgs {
    pub fn validate(&self) -> Result<()> {
        if self.application.is_empty()
            || self.application.len() > 1024
            || self.window.is_empty()
            || self.window.len() > 1024
            || !(1..=256).contains(&self.max_nodes)
            || !(1..=8).contains(&self.max_depth)
        {
            bail!("AT-SPI tree bounds invalid");
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActionArgs {
    application: String,
    window: String,
    path: String,
    action: i32,
}
impl ActionArgs {
    pub fn validate(&self) -> Result<()> {
        TreeArgs {
            application: self.application.clone(),
            window: self.window.clone(),
            max_nodes: 256,
            max_depth: 8,
        }
        .validate()?;
        if self.path.len() > 1024
            || !self.path.starts_with("/org/a11y/atspi/accessible/")
            || !(0..32).contains(&self.action)
        {
            bail!("invalid AT-SPI action target");
        }
        OwnedObjectPath::try_from(self.path.as_str())?;
        Ok(())
    }
}
async fn accessible<'a>(
    c: &'a zbus::Connection,
    dest: &'a str,
    path: &'a str,
) -> Result<AccessibleProxy<'a>> {
    Ok(AccessibleProxy::builder(c)
        .destination(dest)
        .path(path)
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await?)
}
async fn action<'a>(
    c: &'a zbus::Connection,
    dest: &'a str,
    path: &'a str,
) -> Result<ActionProxy<'a>> {
    Ok(ActionProxy::builder(c)
        .destination(dest)
        .path(path)
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await?)
}
fn bounded(s: String) -> Result<String> {
    if s.len() > 4096 {
        bail!("AT-SPI string exceeds limit");
    }
    Ok(s)
}
async fn children(p: &AccessibleProxy<'_>) -> Result<Vec<ObjectRef>> {
    let children = p.get_children().await?;
    if children.len() > 256 {
        bail!("AT-SPI child count exceeds limit");
    }
    Ok(children)
}
fn same_application(dest: &str, child: &ObjectRef) -> Result<()> {
    if child.0 != dest || !child.1.as_str().starts_with("/org/a11y/atspi/accessible/") {
        bail!("AT-SPI cross-application reference denied");
    }
    Ok(())
}
async fn select(c: &zbus::Connection, app: &str, window: &str) -> Result<ObjectRef> {
    let registry = accessible(
        c,
        "org.a11y.atspi.Registry",
        "/org/a11y/atspi/accessible/root",
    )
    .await?;
    let mut matches = Vec::new();
    for application in children(&registry).await? {
        // Pin a unique bus owner, not a rebindable well-known destination.
        if !application.0.starts_with(':') {
            continue;
        }
        let application_proxy = accessible(c, &application.0, application.1.as_str()).await?;
        if bounded(application_proxy.name().await?)? != app {
            continue;
        }
        for candidate in children(&application_proxy).await? {
            same_application(&application.0, &candidate)?;
            let p = accessible(c, &candidate.0, candidate.1.as_str()).await?;
            if bounded(p.name().await?)? == window {
                matches.push(candidate);
            }
        }
    }
    if matches.len() != 1 {
        bail!("AT-SPI application/window missing or ambiguous");
    }
    Ok(matches.remove(0))
}
pub(crate) async fn execute(config: &AtspiConfig, call: &ToolCall) -> Result<Value> {
    if matches!(
        call.name.as_str(),
        "desktop_accessibility_get_text" | "desktop_accessibility_set_value"
    ) {
        validate_text_call(call)?;
        let a: TextArgs = parse(call)?;
        if !config.allowed_applications.contains(&a.application)
            || !config.allowed_windows.contains(&a.window)
        {
            bail!("AT-SPI application/window denied");
        }
        let c = zbus::connection::Builder::address(config.bus_address.as_str())
            .build()
            .await?;
        let root = select(&c, &a.application, &a.window).await?;
        // Re-select the exact app/window and prove descendant membership twice.
        // Pin the unique bus owner throughout; no arbitrary object destinations.
        for _ in 0..2 {
            if select(&c, &a.application, &a.window).await? != root {
                bail!("AT-SPI selection changed");
            }
            if !descendant(&c, &root, &a.path).await? {
                bail!("AT-SPI text target not in bounded allowlisted window tree");
            }
        }
        return if call.name == "desktop_accessibility_get_text" {
            let p = TextProxy::builder(&c)
                .destination(root.0.as_str())
                .path(a.path.as_str())
                .build()
                .await?;
            Ok(
                json!({"text":bounded(p.get_text(a.start.expect("validated"),a.end.expect("validated")).await?)?}),
            )
        } else {
            let p = EditableTextProxy::builder(&c)
                .destination(root.0.as_str())
                .path(a.path.as_str())
                .build()
                .await?;
            Ok(json!({"set":p.set_text_contents(a.value.as_deref().expect("validated")).await?}))
        };
    }
    let (tree_args, action_args) = if call.name == "desktop_accessibility_action" {
        let a: ActionArgs = parse(call)?;
        a.validate()?;
        (
            TreeArgs {
                application: a.application.clone(),
                window: a.window.clone(),
                max_nodes: 256,
                max_depth: 8,
            },
            Some(a),
        )
    } else {
        let a: TreeArgs = parse(call)?;
        a.validate()?;
        (a, None)
    };
    if !config.allowed_applications.contains(&tree_args.application)
        || !config.allowed_windows.contains(&tree_args.window)
    {
        bail!("AT-SPI application/window denied");
    }
    let c = zbus::connection::Builder::address(config.bus_address.as_str())
        .build()
        .await?;
    let root = select(&c, &tree_args.application, &tree_args.window).await?;
    let selected_root = root.clone();
    let dest = root.0.clone();
    let mut queue = VecDeque::from([(root, 0usize)]);
    let mut seen = BTreeSet::new();
    let mut nodes = Vec::new();
    let mut truncated = false;
    while let Some((object, depth)) = queue.pop_front() {
        same_application(&dest, &object)?;
        if !seen.insert(object.1.to_string()) {
            continue;
        }
        if seen.len() > tree_args.max_nodes {
            truncated = true;
            break;
        }
        let p = accessible(&c, &dest, object.1.as_str()).await?;
        if action_args
            .as_ref()
            .is_some_and(|a| a.path == object.1.as_str())
        {
            // Re-resolve the scoped root immediately before mutation.
            if select(&c, &tree_args.application, &tree_args.window).await? != selected_root
                || !descendant(&c, &selected_root, object.1.as_str()).await?
            {
                bail!("AT-SPI selection changed");
            }
            let a = action(&c, &dest, object.1.as_str()).await?;
            let index = action_args.as_ref().expect("checked").action;
            let n = a.n_actions().await?;
            if !(0..=32).contains(&n) || index >= n {
                bail!("AT-SPI action index denied");
            }
            return Ok(json!({"performed":a.do_action(index).await?}));
        }
        let actions = match action(&c, &dest, object.1.as_str())
            .await?
            .n_actions()
            .await
        {
            Ok(n) if (0..=32).contains(&n) => {
                let a = action(&c, &dest, object.1.as_str()).await?;
                let mut names = Vec::new();
                for index in 0..n {
                    names.push(bounded(a.get_name(index).await?)?);
                }
                names
            }
            Ok(_) => bail!("AT-SPI action count exceeds limit"),
            Err(_) => Vec::new(),
        };
        nodes.push(json!({"path":object.1.as_str(),"depth":depth,"name":bounded(p.name().await?)?,"role":p.get_role().await?,"role_name":bounded(p.get_role_name().await?)?,"actions":actions}));
        if depth < tree_args.max_depth {
            for child in children(&p).await? {
                same_application(&dest, &child)?;
                if queue.len() >= 256 {
                    truncated = true;
                    break;
                }
                queue.push_back((child, depth + 1));
            }
        } else if !children(&p).await?.is_empty() {
            truncated = true;
        }
    }
    if action_args.is_some() {
        bail!("AT-SPI action target not in bounded allowlisted window tree");
    }
    Ok(
        json!({"application":tree_args.application,"window":tree_args.window,"nodes":nodes,"truncated":truncated}),
    )
}

async fn descendant(c: &zbus::Connection, root: &ObjectRef, target: &str) -> Result<bool> {
    let mut queue = VecDeque::from([(root.clone(), 0usize)]);
    let mut seen = BTreeSet::new();
    while let Some((object, depth)) = queue.pop_front() {
        same_application(&root.0, &object)?;
        if !seen.insert(object.1.to_string()) {
            continue;
        }
        if seen.len() > 256 {
            bail!("AT-SPI descendant budget exceeded");
        }
        if object.1.as_str() == target {
            return Ok(true);
        }
        if depth < 8 {
            let p = accessible(c, &root.0, object.1.as_str()).await?;
            for child in children(&p).await? {
                same_application(&root.0, &child)?;
                if queue.len() >= 256 {
                    bail!("AT-SPI descendant queue exceeded");
                }
                queue.push_back((child, depth + 1));
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader},
        process::{Child, Command, Stdio},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    struct ChildGuard(Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    struct Node {
        name: String,
        children: Vec<ObjectRef>,
    }
    #[zbus::interface(name = "org.a11y.atspi.Accessible")]
    impl Node {
        #[zbus(property)]
        fn name(&self) -> String {
            self.name.clone()
        }
        fn get_role(&self) -> u32 {
            42
        }
        fn get_role_name(&self) -> String {
            "mock".into()
        }
        fn get_children(&self) -> Vec<ObjectRef> {
            self.children.clone()
        }
    }
    struct MockAction(Arc<AtomicUsize>);
    struct MockText(Arc<std::sync::Mutex<String>>);
    #[zbus::interface(name = "org.a11y.atspi.Text")]
    impl MockText {
        fn get_text(&self, start_offset: i32, end_offset: i32) -> String {
            self.0
                .lock()
                .unwrap()
                .chars()
                .skip(start_offset as usize)
                .take((end_offset - start_offset) as usize)
                .collect()
        }
    }
    struct MockEditable(Arc<std::sync::Mutex<String>>);
    #[zbus::interface(name = "org.a11y.atspi.EditableText")]
    impl MockEditable {
        fn set_text_contents(&self, new_contents: &str) -> bool {
            *self.0.lock().unwrap() = new_contents.into();
            true
        }
    }
    #[zbus::interface(name = "org.a11y.atspi.Action")]
    impl MockAction {
        #[zbus(property, name = "NActions")]
        fn n_actions(&self) -> i32 {
            1
        }
        fn get_name(&self, index: i32) -> String {
            assert_eq!(index, 0);
            "press".into()
        }
        fn do_action(&self, index: i32) -> bool {
            assert_eq!(index, 0);
            self.0.fetch_add(1, Ordering::SeqCst);
            true
        }
    }
    fn object(dest: &str, path: &str) -> ObjectRef {
        (dest.into(), OwnedObjectPath::try_from(path).unwrap())
    }
    #[tokio::test]
    async fn typed_private_bus_tree_and_descendant_scoped_action() -> Result<()> {
        let child = Command::new("/usr/bin/dbus-daemon")
            .env_clear()
            .args(["--session", "--nofork", "--nopidfile", "--print-address=1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut guard = ChildGuard(child);
        let stdout = guard.0.stdout.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut s = String::new();
            let result = BufReader::new(stdout).read_line(&mut s).map(|_| s);
            let _ = tx.send(result);
        });
        let address = rx
            .recv_timeout(std::time::Duration::from_secs(5))??
            .trim()
            .to_string();
        let server = zbus::connection::Builder::address(address.as_str())
            .name("org.a11y.atspi.Registry")
            .build()
            .await?;
        let dest = server.unique_name().unwrap().to_string();
        let root = "/org/a11y/atspi/accessible/root";
        let app = "/org/a11y/atspi/accessible/app";
        let window = "/org/a11y/atspi/accessible/window";
        let button = "/org/a11y/atspi/accessible/button";
        let outside = "/org/a11y/atspi/accessible/outside";
        for (path, name, children) in [
            (root, "root", vec![object(&dest, app)]),
            (app, "Allowed App", vec![object(&dest, window)]),
            (window, "Allowed Window", vec![object(&dest, button)]),
            (button, "Button", vec![]),
            (outside, "Out of Scope", vec![]),
        ] {
            server
                .object_server()
                .at(
                    path,
                    Node {
                        name: name.into(),
                        children,
                    },
                )
                .await?;
        }
        let calls = Arc::new(AtomicUsize::new(0));
        server
            .object_server()
            .at(button, MockAction(calls.clone()))
            .await?;
        server
            .object_server()
            .at(outside, MockAction(calls.clone()))
            .await?;
        let text = Arc::new(std::sync::Mutex::new("initial text".into()));
        for path in [button, outside] {
            server
                .object_server()
                .at(path, MockText(text.clone()))
                .await?;
            server
                .object_server()
                .at(path, MockEditable(text.clone()))
                .await?;
        }
        let config = AtspiConfig {
            bus_address: address,
            allowed_applications: BTreeSet::from(["Allowed App".into()]),
            allowed_windows: BTreeSet::from(["Allowed Window".into()]),
            isolated_session: true,
        };
        config.validate()?;
        let call = ToolCall {
            id: "1".into(),
            name: "desktop_accessibility_tree".into(),
            arguments: json!({"application":"Allowed App","window":"Allowed Window"}),
        };
        let tree = execute(&config, &call).await?;
        assert_eq!(tree["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(tree["nodes"][1]["actions"][0], "press");
        let call = ToolCall {
            id: "2".into(),
            name: "desktop_accessibility_action".into(),
            arguments: json!({"application":"Allowed App","window":"Allowed Window","path":button,"action":0}),
        };
        assert_eq!(execute(&config, &call).await?["performed"], true);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let call = ToolCall {
            id: "3".into(),
            name: "desktop_accessibility_action".into(),
            arguments: json!({"application":"Allowed App","window":"Allowed Window","path":outside,"action":0}),
        };
        assert!(
            execute(&config, &call)
                .await
                .unwrap_err()
                .to_string()
                .contains("not in bounded")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let call = ToolCall {
            id: "4".into(),
            name: "desktop_accessibility_tree".into(),
            arguments: json!({"application":"Denied","window":"Allowed Window"}),
        };
        assert!(
            execute(&config, &call)
                .await
                .unwrap_err()
                .to_string()
                .contains("denied")
        );
        let mut read = ToolCall {
            id: "text".into(),
            name: "desktop_accessibility_get_text".into(),
            arguments: json!({"application":"Allowed App","window":"Allowed Window","path":button,"start":0,"end":7}),
        };
        assert_eq!(execute(&config, &read).await?["text"], "initial");
        let mut set = ToolCall {
            id: "set".into(),
            name: "desktop_accessibility_set_value".into(),
            arguments: json!({"application":"Allowed App","window":"Allowed Window","path":button,"value":"edited"}),
        };
        assert_eq!(execute(&config, &set).await?["set"], true);
        assert_eq!(*text.lock().unwrap(), "edited");
        read.arguments["path"] = json!(outside);
        set.arguments["path"] = json!(outside);
        for denied in [&read, &set] {
            assert!(
                execute(&config, denied)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("not in bounded")
            );
        }
        assert_eq!(*text.lock().unwrap(), "edited");
        assert!(server.object_server().remove::<Node, _>(window).await?);
        server
            .object_server()
            .at(
                window,
                Node {
                    name: "Allowed Window".into(),
                    children: vec![],
                },
            )
            .await?;
        set.arguments["path"] = json!(button);
        assert!(
            execute(&config, &set)
                .await
                .unwrap_err()
                .to_string()
                .contains("not in bounded")
        );
        assert_eq!(*text.lock().unwrap(), "edited");
        Ok(())
    }
}
