use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use url::Url;

/// Operator configuration only. Never accept this object as model tool arguments.
/// Disabled by default; endpoints have no localhost or host-discovery defaults.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntegrationConfig {
    pub nomad: Option<NomadConfig>,
    pub incus: Option<IncusConfig>,
    pub comfyui: Option<ComfyConfig>,
    pub obs: Option<ObsConfig>,
    pub vtube: Option<VtubeConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConnectionConfig {
    pub enabled: bool,
    pub endpoint: String,
    /// Environment variable NAME, not the credential. Resolved at construction.
    pub token_env: Option<String>,
    pub timeout_seconds: u64,
    pub max_output_bytes: usize,
    /// Independent of per-call trusted approval, which is also mandatory.
    pub allow_mutations: bool,
}
impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: String::new(),
            token_env: None,
            timeout_seconds: 10,
            max_output_bytes: 32_000,
            allow_mutations: false,
        }
    }
}
impl ConnectionConfig {
    pub(crate) fn validate(&self, websocket: bool) -> Result<()> {
        if !(1..=120).contains(&self.timeout_seconds)
            || !(128..=262_144).contains(&self.max_output_bytes)
        {
            bail!("integration bounds invalid");
        }
        let u = Url::parse(&self.endpoint)
            .map_err(|_| anyhow::anyhow!("invalid integration endpoint"))?;
        let schemes = if websocket {
            &["ws", "wss"][..]
        } else {
            &["http", "https"][..]
        };
        if !schemes.contains(&u.scheme())
            || u.host_str().is_none()
            || !u.username().is_empty()
            || u.password().is_some()
            || u.query().is_some()
            || u.fragment().is_some()
            || u.path() != "/"
            || self.endpoint.chars().any(char::is_control)
        {
            bail!(
                "integration requires an exact origin without credentials, path, query or fragment"
            );
        }
        if let Some(env) = &self.token_env
            && (env.is_empty()
                || env.len() > 128
                || !env
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                || env.as_bytes()[0].is_ascii_digit())
        {
            bail!("invalid credential environment name");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NomadConfig {
    pub connection: ConnectionConfig,
    pub namespace: String,
    /// All visible job IDs, not a prefix. List results are filtered to this set.
    pub jobs: BTreeSet<String>,
    /// Fully operator-authored Nomad Job objects keyed by exact allowed ID.
    /// Model callers select a name; they cannot supply task drivers/commands/mounts.
    pub deployments: BTreeMap<String, Value>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IncusConfig {
    pub connection: ConnectionConfig,
    pub project: String,
    pub instances: BTreeSet<String>,
    pub actions: BTreeSet<InstanceAction>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstanceAction {
    Start,
    Stop,
    Restart,
    Freeze,
    Unfreeze,
}
impl InstanceAction {
    pub(crate) fn wire(&self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Stop => "stop",
            Self::Restart => "restart",
            Self::Freeze => "freeze",
            Self::Unfreeze => "unfreeze",
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComfyConfig {
    pub connection: ConnectionConfig,
    pub workflows: BTreeMap<String, Workflow>,
    pub allowed_classes: BTreeSet<String>,
    /// Hard ceiling on in-memory submitted-job ownership records. No eviction/replay.
    pub max_tracked_jobs: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workflow {
    pub graph: Value,
    /// Optional single CLIPTextEncode node whose text may be replaced by a prompt.
    /// All other node classes, links and parameters remain operator-authored.
    pub prompt_node: Option<String>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObsConfig {
    pub connection: ConnectionConfig,
    pub scenes: BTreeSet<String>,
    /// Exact scene:item-id pairs, e.g. Studio:12, not arbitrary source names.
    pub scene_items: BTreeSet<String>,
    pub actions: BTreeSet<ObsAction>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObsAction {
    SwitchScene,
    SetVisibility,
    StartRecording,
    StopRecording,
    StartStreaming,
    StopStreaming,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VtubeConfig {
    pub connection: ConnectionConfig,
    pub plugin_name: String,
    pub plugin_developer: String,
    pub models: BTreeSet<String>,
    pub hotkeys: BTreeSet<String>,
    pub parameters: BTreeMap<String, ParameterRange>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterRange {
    pub min: f64,
    pub max: f64,
}

pub(crate) fn identifier(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 128
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    {
        bail!("invalid exact resource identifier");
    }
    Ok(())
}
pub(crate) fn label(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > 256 || s.chars().any(char::is_control) {
        bail!("invalid resource label");
    }
    Ok(())
}
