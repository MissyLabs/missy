use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_desktop::{BrowserConfig, DesktopApproval, DesktopConfig, DesktopTools};
use missy_policy::PolicyHttpClient;
use serde_json::json;
use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

struct Deny(AtomicUsize);
#[async_trait]
impl DesktopApproval for Deny {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        bail!("test approval denied")
    }
    fn configuration_fingerprint(&self) -> String {
        "test-deny".into()
    }
}
fn context() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "turn".into(),
        cancellation: Default::default(),
    }
}
fn browser_config() -> DesktopConfig {
    DesktopConfig {
        browser: Some(BrowserConfig {
            endpoint: "http://127.0.0.1:9".into(),
            target_id: "test".into(),
            allowed_origins: BTreeSet::from(["https://example.org".into()]),
            isolated_browser_network: true,
        }),
        ..Default::default()
    }
}
#[tokio::test]
async fn disabled_default_and_no_implicit_connections() {
    let tools = DesktopTools::new(
        Default::default(),
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        Arc::new(Deny(AtomicUsize::new(0))),
    )
    .unwrap();
    assert!(tools.definitions().is_empty());
    let call = ToolCall {
        id: "1".into(),
        name: "desktop_windows".into(),
        arguments: json!({}),
    };
    assert!(
        tools
            .execute(&context(), &call)
            .await
            .unwrap_err()
            .to_string()
            .contains("disabled")
    );
}
#[tokio::test]
async fn privacy_and_mutation_always_gated_without_network() {
    let approval = Arc::new(Deny(AtomicUsize::new(0)));
    let config = DesktopConfig {
        enable_input: true,
        enable_screenshots: true,
        approve_all: false,
        ..browser_config()
    };
    let tools = DesktopTools::new(
        config,
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        approval.clone(),
    )
    .unwrap();
    for (name, args) in [
        ("browser_screenshot", json!({})),
        ("browser_click", json!({"selector":"button"})),
        ("browser_fill", json!({"selector":"input","value":"x"})),
        ("browser_navigate", json!({"url":"https://example.org/"})),
    ] {
        let call = ToolCall {
            id: "1".into(),
            name: name.into(),
            arguments: args,
        };
        assert!(
            tools
                .execute(&context(), &call)
                .await
                .unwrap_err()
                .to_string()
                .contains("approval denied")
        );
        assert!(
            tools
                .definitions()
                .iter()
                .find(|d| d.name == name)
                .unwrap()
                .side_effect
        );
    }
    assert_eq!(approval.0.load(Ordering::SeqCst), 4);
}
#[tokio::test]
async fn unknown_fields_and_urls_fail_before_approval() {
    let approval = Arc::new(Deny(AtomicUsize::new(0)));
    let config = DesktopConfig {
        enable_input: true,
        ..browser_config()
    };
    let tools = DesktopTools::new(
        config,
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        approval.clone(),
    )
    .unwrap();
    for (name, args) in [
        ("browser_read", json!({"expression":"arbitrary"})),
        ("browser_navigate", json!({"url":"file:///etc/passwd"})),
        (
            "browser_navigate",
            json!({"url":"https://example.org.evil.invalid"}),
        ),
        (
            "browser_fill",
            json!({"selector":"x","value":"x","evaluate":"evil"}),
        ),
    ] {
        let call = ToolCall {
            id: "1".into(),
            name: name.into(),
            arguments: args,
        };
        assert!(tools.execute(&context(), &call).await.is_err());
    }
    assert_eq!(approval.0.load(Ordering::SeqCst), 0);
}
#[test]
fn endpoints_and_network_boundary_must_be_explicit() {
    let mut c = browser_config();
    c.browser.as_mut().unwrap().isolated_browser_network = false;
    assert!(
        DesktopTools::new(
            c,
            PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
            Arc::new(Deny(AtomicUsize::new(0)))
        )
        .is_err()
    );
    for endpoint in [
        "http://user:secret@localhost:9222",
        "http://localhost:9222/path",
        "http://localhost:9222/?secret=x",
    ] {
        let mut c = browser_config();
        c.browser.as_mut().unwrap().endpoint = endpoint.into();
        assert!(
            DesktopTools::new(
                c,
                PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
                Arc::new(Deny(AtomicUsize::new(0)))
            )
            .is_err()
        );
    }
}
