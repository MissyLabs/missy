//! Opt-in native desktop capabilities. Never discovers the operator's display,
//! accessibility bus, browser endpoint, or camera from the process environment.
//! Every returned value, including image bytes, is untrusted external content.
mod atspi;
mod browser;
mod launch;
mod screen;
mod status;
mod x11;
pub use launch::{DesktopCommand, DesktopExecutablePin, DesktopLaunchConfig};
pub use screen::{OcrAnchor, OcrAvailability, ScreenOcr, ScreenOcrResult, parse_tesseract_tsv};
pub use status::{DesktopBackendStatus, DesktopStatusReport, EndpointHealth};

use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolDispatchRefused, ToolExecutor};
use missy_policy::PolicyHttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, sync::Arc, time::Duration};

pub use atspi::AtspiConfig;
pub use browser::BrowserConfig;
pub use x11::X11Config;

/// Integrate with the parent's durable, scope-bound approval system. An allow
/// implementation must attest approval of this exact call in this exact scope.
/// This crate deliberately has no built-in "allow all" implementation.
#[async_trait]
pub trait DesktopApproval: Send + Sync {
    async fn authorize(&self, context: &ToolContext, call: &ToolCall) -> Result<()>;
    /// Bind approval policy into the executor's configuration identity.
    fn configuration_fingerprint(&self) -> String;
}
/// Called after capture approval. Implementations retain governed disclosure /
/// billing approval and must not promote visual observations to verified facts.
#[async_trait]
pub trait ScreenInterpreter: Send + Sync {
    async fn interpret(&self, ctx: &ToolContext, png: Vec<u8>, question: &str) -> Result<Value>;
    fn configuration_fingerprint(&self) -> String;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopConfig {
    pub x11: Option<X11Config>,
    pub atspi: Option<AtspiConfig>,
    pub browser: Option<BrowserConfig>,
    pub launch: Option<DesktopLaunchConfig>,
    /// Extra operator opt-in, independent of per-call approval.
    pub enable_screenshots: bool,
    /// Enables XTest, AT-SPI actions, browser click/fill/navigation.
    pub enable_input: bool,
    /// Optional stronger gate: approval for enumeration and text reads as well.
    pub approve_all: bool,
    pub max_output_bytes: usize,
    pub timeout_seconds: u64,
}
impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            x11: None,
            atspi: None,
            browser: None,
            launch: None,
            enable_screenshots: false,
            enable_input: false,
            approve_all: true,
            max_output_bytes: 1024 * 1024,
            timeout_seconds: 15,
        }
    }
}

pub struct DesktopTools {
    config: DesktopConfig,
    http: PolicyHttpClient,
    approval: Arc<dyn DesktopApproval>,
    x11_slots: Arc<tokio::sync::Semaphore>,
    launch_state: launch::LaunchState,
    screen_interpreter: Option<Arc<dyn ScreenInterpreter>>,
    screen_ocr: Option<Arc<dyn ScreenOcr>>,
}
impl DesktopTools {
    /// Validates only. No desktop connection or discovery occurs here.
    pub fn new(
        config: DesktopConfig,
        http: PolicyHttpClient,
        approval: Arc<dyn DesktopApproval>,
    ) -> Result<Self> {
        if !(1024..=8 * 1024 * 1024).contains(&config.max_output_bytes)
            || !(1..=60).contains(&config.timeout_seconds)
        {
            bail!("desktop output/timeout bounds invalid");
        }
        if let Some(c) = &config.x11 {
            c.validate()?;
        }
        if let Some(c) = &config.atspi {
            c.validate()?;
        }
        if let Some(c) = &config.browser {
            c.validate()?;
        }
        if let Some(c) = &config.launch {
            c.validate()?;
        }
        Ok(Self {
            config,
            http,
            approval,
            x11_slots: Arc::new(tokio::sync::Semaphore::new(4)),
            launch_state: Default::default(),
            screen_interpreter: None,
            screen_ocr: None,
        })
    }
    pub fn with_screen_interpreter(mut self, interpreter: Arc<dyn ScreenInterpreter>) -> Self {
        self.screen_interpreter = Some(interpreter);
        self
    }
    pub fn with_screen_ocr(mut self, ocr: Arc<dyn ScreenOcr>) -> Self {
        self.screen_ocr = Some(ocr);
        self
    }
    fn validate_call(&self, call: &ToolCall) -> Result<()> {
        let names: BTreeSet<_> = self.definitions().into_iter().map(|d| d.name).collect();
        if !names.contains(&call.name) {
            bail!("desktop capability disabled or unknown");
        }
        if screenshot(&call.name) && !self.config.enable_screenshots {
            bail!("screenshots disabled");
        }
        if input(&call.name) && !self.config.enable_input {
            bail!("desktop input disabled");
        }
        // Validate every argument before requesting approval or opening a connection.
        match call.name.as_str() {
            "x11_read_screen" => {
                let a: x11::ReadScreenArgs = parse(call)?;
                if a.window == 0 || a.question.trim().is_empty() || a.question.len() > 4096 {
                    bail!("invalid read-screen arguments");
                }
            }
            "desktop_launch_app" | "x11_launch" | "install_software_confirmed" => {
                launch::validate_call(self.config.launch.as_ref().expect("catalog checked"), call)?
            }
            "desktop_windows" | "desktop_status" => {
                parse::<Empty>(call)?;
            }
            "desktop_screenshot" | "desktop_focus_window" => {
                parse::<x11::WindowArgs>(call)?;
            }
            "desktop_input" => {
                parse::<x11::InputArgs>(call)?.validate()?;
            }
            "desktop_accessibility_tree" => {
                parse::<atspi::TreeArgs>(call)?.validate()?;
            }
            "desktop_accessibility_action" => {
                parse::<atspi::ActionArgs>(call)?.validate()?;
            }
            "desktop_accessibility_get_text" | "desktop_accessibility_set_value" => {
                atspi::validate_text_call(call)?;
            }
            _ => browser::validate_call(
                call,
                self.config.browser.as_ref().expect("catalog checked"),
            )?,
        }
        Ok(())
    }
    async fn dispatch(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        let value = match call.name.as_str() {
            "desktop_status" => serde_json::to_value(self.status_report().await?)?,
            "desktop_launch_app" | "x11_launch" | "install_software_confirmed" => {
                launch::execute(
                    self.config.launch.as_ref().expect("validated"),
                    &self.launch_state,
                    call,
                )
                .await?
            }
            "desktop_windows"
            | "desktop_screenshot"
            | "desktop_input"
            | "desktop_focus_window"
            | "x11_read_screen" => {
                let config = self.config.x11.clone().expect("validated");
                let call = call.clone();
                let limit = self.config.max_output_bytes;
                let slot = self.x11_slots.clone().acquire_owned().await?;
                let cancellation = tokio_util::sync::CancellationToken::new();
                let _guard = cancellation.clone().drop_guard();
                tokio::task::spawn_blocking(move || {
                    let _slot = slot;
                    x11::execute(&config, &call, limit, &cancellation)
                })
                .await??
            }
            "desktop_accessibility_tree"
            | "desktop_accessibility_action"
            | "desktop_accessibility_get_text"
            | "desktop_accessibility_set_value" => {
                atspi::execute(self.config.atspi.as_ref().expect("validated"), call).await?
            }
            _ => {
                browser::execute(
                    self.config.browser.as_ref().expect("validated"),
                    &self.http,
                    call,
                    self.config.max_output_bytes,
                )
                .await?
            }
        };
        let value = if call.name == "x11_read_screen" {
            use base64::Engine;
            let png = base64::engine::general_purpose::STANDARD.decode(
                value["artifact"]["data"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("screen PNG missing"))?,
            )?;
            let a: x11::ReadScreenArgs = parse(call)?;
            let width = value["window"]["width"].as_u64().expect("X11 dimensions") as u32;
            let height = value["window"]["height"].as_u64().expect("X11 dimensions") as u32;
            let ocr = if let Some(backend) = &self.screen_ocr {
                match backend.recognize_tsv(ctx, png.clone()).await {
                    Ok(tsv) => parse_tesseract_tsv(&tsv, width, height)
                        .unwrap_or_else(|_| ScreenOcrResult::absent(width, height, true)),
                    Err(_) => ScreenOcrResult::absent(width, height, true),
                }
            } else {
                ScreenOcrResult::absent(width, height, false)
            };
            if ctx.cancellation.is_cancelled() {
                bail!("screen read cancelled before vision interpretation");
            }
            let analysis = if let Some(interpreter) = &self.screen_interpreter {
                Some(interpreter.interpret(ctx, png, &a.question).await?)
            } else {
                None
            };
            json!({"window":value["window"],"analysis":analysis,"ocr":ocr,"trust":"untrusted_visual_observation"})
        } else {
            value
        };
        let value = json!({"trust":"untrusted", "data":value});
        if serde_json::to_vec(&value)?.len() > self.config.max_output_bytes {
            bail!("desktop output exceeds configured byte limit");
        }
        Ok(value)
    }
    async fn status_report(&self) -> Result<DesktopStatusReport> {
        let mut x11 =
            DesktopBackendStatus::configured(self.config.x11.as_ref().map(|c| c.isolated_session));
        if let Some(config) = self.config.x11.clone() {
            let slot = self.x11_slots.clone().acquire_owned().await?;
            let healthy = tokio::task::spawn_blocking(move || {
                let _slot = slot;
                x11::probe(&config).is_ok()
            })
            .await?;
            x11.endpoint_health = if healthy {
                EndpointHealth::Reachable
            } else {
                EndpointHealth::Unavailable
            };
            x11.diagnostic = (!healthy).then(|| "configured_x11_endpoint_unavailable".into());
        }
        Ok(DesktopStatusReport {
            x11,
            accessibility: DesktopBackendStatus::configured(
                self.config.atspi.as_ref().map(|c| c.isolated_session),
            ),
            browser: DesktopBackendStatus::configured(
                self.config
                    .browser
                    .as_ref()
                    .map(|c| c.isolated_browser_network),
            ),
            launch_configured: self.config.launch.is_some(),
            screenshots_enabled: self.config.enable_screenshots,
            input_enabled: self.config.enable_input,
            vision_configured: self.screen_interpreter.is_some(),
            ocr_configured: self.screen_ocr.is_some(),
            read_screen_enabled: self.config.x11.is_some()
                && self.config.enable_screenshots
                && (self.screen_interpreter.is_some() || self.screen_ocr.is_some()),
            approval_required: true,
            ambient_discovery: false,
            input_dispatched: false,
        })
    }
}

fn screenshot(name: &str) -> bool {
    matches!(
        name,
        "desktop_screenshot" | "browser_screenshot" | "x11_read_screen"
    )
}
fn input(name: &str) -> bool {
    matches!(
        name,
        "desktop_input"
            | "desktop_focus_window"
            | "desktop_launch_app"
            | "x11_launch"
            | "install_software_confirmed"
            | "desktop_accessibility_action"
            | "desktop_accessibility_set_value"
            | "browser_close"
            | "browser_navigate"
            | "browser_click"
            | "browser_fill"
            | "browser_fill_many"
    )
}
pub(crate) fn parse<T: serde::de::DeserializeOwned>(call: &ToolCall) -> Result<T> {
    Ok(serde_json::from_value(call.arguments.clone())?)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}

#[async_trait]
impl ToolExecutor for DesktopTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = Vec::new();
        if self.config.enable_input
            && let Some(c) = &self.config.launch
        {
            definitions.extend(launch::definitions(c));
        }
        let mut add = |name: &str,
                       description: &str,
                       properties: Value,
                       required: &[&str],
                       side_effect| {
            definitions.push(ToolDefinition { name: name.into(), description: description.into(),
                parameters: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}), side_effect });
        };
        if self.config.x11.is_some()
            || self.config.atspi.is_some()
            || self.config.browser.is_some()
            || self.config.launch.is_some()
        {
            add(
                "desktop_status",
                "Approval-gated configured capabilities/isolation diagnostics. Probes only the configured isolated X11 endpoint; no ambient discovery, capture, enumeration or input. Isolation is operator-attested.",
                json!({}),
                &[],
                true,
            );
        }
        if self.config.x11.is_some() {
            if self.config.enable_screenshots
                && (self.screen_interpreter.is_some() || self.screen_ocr.is_some())
            {
                add(
                    "x11_read_screen",
                    "Capture an approved isolated allowlisted X11 window; reject blank frames before OCR/vision. Returns governed analysis and/or actual native-window-pixel OCR word anchors from an explicit backend, never inferred coordinates.",
                    json!({"window":{"type":"integer","minimum":1},"question":{"type":"string","minLength":1,"maxLength":4096}}),
                    &["window", "question"],
                    true,
                );
            }
            if self.config.enable_input {
                add(
                    "desktop_focus_window",
                    "Approval-gated native X11 focus and raise of an exact allowlisted top-level window in the explicitly isolated session. Never ambient DISPLAY.",
                    json!({"window":{"type":"integer","minimum":1}}),
                    &["window"],
                    true,
                );
            }
            add(
                "desktop_windows",
                "List explicitly allowlisted X11 windows; untrusted metadata.",
                json!({}),
                &[],
                false,
            );
            if self.config.enable_screenshots {
                add(
                    "desktop_screenshot",
                    "Approval-gated allowlisted-window raw X11 image, base64 plus pixel metadata.",
                    json!({"window":{"type":"integer","minimum":1}}),
                    &["window"],
                    true,
                );
            }
            if self.config.enable_input {
                add(
                    "desktop_input",
                    "Approval-gated native XTest key/button/move/drag/type_text on an already-focused allowlisted window. Coordinates are window-relative. Printable ASCII text only; no keymap modification or focus stealing.",
                    json!({"window":{"type":"integer"},"kind":{"enum":["key","button","move","drag","type_text"]},"code":{"type":"integer"},"pressed":{"type":"boolean"},"x":{"type":"integer","minimum":0},"y":{"type":"integer","minimum":0},"to_x":{"type":"integer","minimum":0},"to_y":{"type":"integer","minimum":0},"text":{"type":"string","maxLength":256}}),
                    &["window", "kind"],
                    true,
                );
            }
        }
        if self.config.atspi.is_some() {
            let selection = json!({"application":{"type":"string"},"window":{"type":"string"},"max_nodes":{"type":"integer","minimum":1,"maximum":256},"max_depth":{"type":"integer","minimum":1,"maximum":8}});
            add(
                "desktop_accessibility_tree",
                "Read bounded AT-SPI roles, names and action names in exact allowlisted application/window.",
                selection,
                &["application", "window"],
                false,
            );
            if self.config.enable_input {
                add(
                    "desktop_accessibility_action",
                    "Approval-gated AT-SPI action by index inside an allowlisted application/window and descendant path.",
                    json!({"application":{"type":"string"},"window":{"type":"string"},"path":{"type":"string"},"action":{"type":"integer","minimum":0,"maximum":31}}),
                    &["application", "window", "path", "action"],
                    true,
                );
                add(
                    "desktop_accessibility_set_value",
                    "Approval-gated typed AT-SPI EditableText.SetTextContents on an exact scoped descendant. No arbitrary interface calls.",
                    json!({"application":{"type":"string"},"window":{"type":"string"},"path":{"type":"string"},"value":{"type":"string","maxLength":8192}}),
                    &["application", "window", "path", "value"],
                    true,
                );
            }
            add(
                "desktop_accessibility_get_text",
                "Read bounded typed AT-SPI Text.GetText on an exact scoped descendant.",
                json!({"application":{"type":"string"},"window":{"type":"string"},"path":{"type":"string"},"start":{"type":"integer","minimum":0},"end":{"type":"integer","minimum":0,"maximum":4096}}),
                &["application", "window", "path", "start", "end"],
                false,
            );
        }
        if self.config.browser.is_some() {
            add(
                "browser_extract",
                "Typed bounded text, links, table or form-metadata extraction on the isolated pinned page. No JavaScript input or field secrets.",
                json!({"selector":{"type":"string"},"mode":{"enum":["text","links","table","form"]}}),
                &["selector", "mode"],
                false,
            );
            if self.config.enable_input {
                add(
                    "browser_fill_many",
                    "Approval-gated multi-field form fill and optional submit in one renderer turn. Values are data, never scripts.",
                    json!({"fields":{"type":"array","minItems":1,"maxItems":32,"items":{"type":"object","additionalProperties":false,"properties":{"selector":{"type":"string"},"value":{"type":"string","maxLength":8192}},"required":["selector","value"]}},"submit_selector":{"type":"string"}}),
                    &["fields"],
                    true,
                );
            }
            add(
                "browser_get_url",
                "Read URL of the exact configured isolated page.",
                json!({}),
                &[],
                false,
            );
            add(
                "browser_wait",
                "Wait at most 15 seconds for selector visible/hidden on the pinned page. Each poll checks origin in the same renderer turn.",
                json!({"selector":{"type":"string"},"state":{"enum":["visible","hidden"]},"timeout_ms":{"type":"integer","minimum":1,"maximum":15000}}),
                &["selector", "state", "timeout_ms"],
                false,
            );
            add(
                "browser_evaluate",
                "Safe fixed read expression IDs only. Arbitrary JavaScript and legacy evaluate functions are unsupported.",
                json!({"expression_id":{"enum":["url","title","ready_state"]}}),
                &["expression_id"],
                false,
            );
            add(
                "browser_read",
                "Read bounded text using a fixed CDP script. No arbitrary JavaScript.",
                json!({}),
                &[],
                false,
            );
            if self.config.enable_screenshots {
                add(
                    "browser_screenshot",
                    "Approval-gated bounded PNG screenshot of the configured isolated browser page.",
                    json!({}),
                    &[],
                    true,
                );
            }
            if self.config.enable_input {
                add(
                    "browser_navigate",
                    "Approval-gated navigation to an exact allowlisted HTTP(S) origin. Browser network isolation required.",
                    json!({"url":{"type":"string"}}),
                    &["url"],
                    true,
                );
                add(
                    "browser_close",
                    "Approval-gated Page.close for the exact configured target only. Never kills a browser process.",
                    json!({}),
                    &[],
                    true,
                );
                add(
                    "browser_click",
                    "Approval-gated bounded CSS-selector click using a fixed script.",
                    json!({"selector":{"type":"string"}}),
                    &["selector"],
                    true,
                );
                add(
                    "browser_fill",
                    "Approval-gated input/textarea fill using a fixed script.",
                    json!({"selector":{"type":"string"},"value":{"type":"string"}}),
                    &["selector", "value"],
                    true,
                );
            }
        }
        definitions
    }
    fn configuration_fingerprint(&self) -> String {
        serde_json::to_string(&json!({"desktop":self.config,"approval":self.approval.configuration_fingerprint(),"network":self.http.config(),"screen_interpreter":self.screen_interpreter.as_ref().map(|i|i.configuration_fingerprint()),"screen_ocr":self.screen_ocr.as_ref().map(|i|i.configuration_fingerprint())})).expect("serializable config")
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        if ctx.cancellation.is_cancelled() {
            bail!(ToolDispatchRefused {
                reason: "desktop request cancelled".into()
            });
        }
        self.validate_call(call).map_err(|e| {
            anyhow::anyhow!(ToolDispatchRefused {
                reason: e.to_string()
            })
        })?;
        if self.config.approve_all
            || call.name == "desktop_status"
            || screenshot(&call.name)
            || input(&call.name)
        {
            self.approval.authorize(ctx, call).await?;
        }
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        // Recheck even when a caller skips preflight. Approval must be durable and
        // idempotent, not a consume-once token invalidated by the first check.
        self.preflight(ctx, call).await?;
        tokio::select! {
            biased;
            _ = ctx.cancellation.cancelled() => bail!("desktop operation cancelled; effects may already have occurred"),
            result = tokio::time::timeout(Duration::from_secs(self.config.timeout_seconds), self.dispatch(ctx,call)) => result.map_err(|_| anyhow::anyhow!("desktop operation timed out; effects may already have occurred"))?,
        }
    }
}
