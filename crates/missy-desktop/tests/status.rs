//! No live session touched. Every path is scoped to a new temporary directory.
use anyhow::Result;
use async_trait::async_trait;
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_desktop::{DesktopApproval, DesktopConfig, DesktopTools, ScreenOcr, X11Config};
use missy_policy::PolicyHttpClient;
use serde_json::json;
use std::{collections::BTreeSet, sync::Arc};
struct Allow;
#[async_trait]
impl DesktopApproval for Allow {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "isolated-fixture-approval".into()
    }
}
struct Ocr(&'static str);
#[async_trait]
impl ScreenOcr for Ocr {
    async fn recognize_tsv(&self, _: &ToolContext, _: Vec<u8>) -> Result<String> {
        panic!("must not capture during status")
    }
    fn configuration_fingerprint(&self) -> String {
        self.0.into()
    }
}
#[tokio::test]
async fn configured_endpoint_failure_is_typed_bounded_and_no_discovery() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let config = DesktopConfig {
        x11: Some(X11Config {
            socket_path: dir.path().join("absent-socket"),
            cookie_path: dir.path().join("absent-cookie"),
            screen: 0,
            allowed_classes: BTreeSet::from(["Fixture".into()]),
            allowed_titles: BTreeSet::from(["Fixture".into()]),
            isolated_session: true,
        }),
        approve_all: false,
        enable_screenshots: true,
        ..Default::default()
    };
    let base = || {
        DesktopTools::new(
            config.clone(),
            PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
            Arc::new(Allow),
        )
        .unwrap()
    };
    let tools = base().with_screen_ocr(Arc::new(Ocr("one")));
    assert_ne!(
        tools.configuration_fingerprint(),
        base()
            .with_screen_ocr(Arc::new(Ocr("two")))
            .configuration_fingerprint()
    );
    assert!(
        tools
            .definitions()
            .iter()
            .any(|d| d.name == "x11_read_screen")
    );
    let ctx = ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "fixture".into(),
        cancellation: Default::default(),
    };
    let out = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "s".into(),
                name: "desktop_status".into(),
                arguments: json!({}),
            },
        )
        .await?;
    assert_eq!(out["data"]["x11"]["endpoint_health"], "unavailable");
    assert_eq!(
        out["data"]["x11"]["diagnostic"],
        "configured_x11_endpoint_unavailable"
    );
    assert_eq!(
        out["data"]["accessibility"]["endpoint_health"],
        "not_configured"
    );
    assert_eq!(out["data"]["ocr_configured"], true);
    assert_eq!(out["data"]["vision_configured"], false);
    assert_eq!(out["data"]["read_screen_enabled"], true);
    assert_eq!(out["data"]["ambient_discovery"], false);
    assert_eq!(out["data"]["input_dispatched"], false);
    assert!(!out.to_string().contains(dir.path().to_str().unwrap()));
    Ok(())
}
