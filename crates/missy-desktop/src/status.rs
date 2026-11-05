use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EndpointHealth {
    NotConfigured,
    NotProbed,
    Reachable,
    Unavailable,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesktopBackendStatus {
    pub configured: bool,
    /// Operator attestation, NOT independent isolation verification.
    pub isolation_attested: bool,
    pub endpoint_health: EndpointHealth,
    pub diagnostic: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesktopStatusReport {
    pub x11: DesktopBackendStatus,
    pub accessibility: DesktopBackendStatus,
    pub browser: DesktopBackendStatus,
    pub launch_configured: bool,
    pub screenshots_enabled: bool,
    pub input_enabled: bool,
    pub vision_configured: bool,
    pub ocr_configured: bool,
    pub read_screen_enabled: bool,
    pub approval_required: bool,
    pub ambient_discovery: bool,
    pub input_dispatched: bool,
}
impl DesktopBackendStatus {
    pub(crate) fn configured(isolated: Option<bool>) -> Self {
        Self {
            configured: isolated.is_some(),
            isolation_attested: isolated.unwrap_or(false),
            endpoint_health: if isolated.is_some() {
                EndpointHealth::NotProbed
            } else {
                EndpointHealth::NotConfigured
            },
            diagnostic: None,
        }
    }
}
