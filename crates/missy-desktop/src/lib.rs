//! Opt-in native desktop capabilities. Never discovers the operator's display,
//! accessibility bus, browser endpoint, or camera from the process environment.
//! Every returned value, including image bytes, is untrusted external content.
mod atspi;
mod browser;
mod x11;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopConfig {
    pub x11: Option<X11Config>,
    pub atspi: Option<AtspiConfig>,
    pub browser: Option<BrowserConfig>,
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
        Ok(Self {
            config,
            http,
            approval,
            x11_slots: Arc::new(tokio::sync::Semaphore::new(4)),
        })
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
            "desktop_windows" => {
                parse::<Empty>(call)?;
            }
            "desktop_screenshot" => {
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
            _ => browser::validate_call(
                call,
                self.config.browser.as_ref().expect("catalog checked"),
            )?,
        }
        Ok(())
    }
    async fn dispatch(&self, call: &ToolCall) -> Result<Value> {
        let value = match call.name.as_str() {
            "desktop_windows" | "desktop_screenshot" | "desktop_input" => {
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
            "desktop_accessibility_tree" | "desktop_accessibility_action" => {
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
        let value = json!({"trust":"untrusted", "data":value});
        if serde_json::to_vec(&value)?.len() > self.config.max_output_bytes {
            bail!("desktop output exceeds configured byte limit");
        }
        Ok(value)
    }
}

fn screenshot(name: &str) -> bool {
    matches!(name, "desktop_screenshot" | "browser_screenshot")
}
fn input(name: &str) -> bool {
    matches!(
        name,
        "desktop_input"
            | "desktop_accessibility_action"
            | "browser_navigate"
            | "browser_click"
            | "browser_fill"
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
        let mut add = |name: &str,
                       description: &str,
                       properties: Value,
                       required: &[&str],
                       side_effect| {
            definitions.push(ToolDefinition { name: name.into(), description: description.into(),
                parameters: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}), side_effect });
        };
        if self.config.x11.is_some() {
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
                    "Approval-gated XTest key/button on an already-focused allowlisted window. No focus stealing.",
                    json!({"window":{"type":"integer"},"kind":{"enum":["key","button"]},"code":{"type":"integer"},"pressed":{"type":"boolean"}}),
                    &["window", "kind", "code", "pressed"],
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
            }
        }
        if self.config.browser.is_some() {
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
        serde_json::to_string(&json!({"desktop":self.config,"approval":self.approval.configuration_fingerprint(),"network":self.http.config()})).expect("serializable config")
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
        if self.config.approve_all || screenshot(&call.name) || input(&call.name) {
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
            result = tokio::time::timeout(Duration::from_secs(self.config.timeout_seconds), self.dispatch(call)) => result.map_err(|_| anyhow::anyhow!("desktop operation timed out; effects may already have occurred"))?,
        }
    }
}
