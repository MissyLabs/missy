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
    /// Exact node IDs visible to discovery and placement. No fleet-wide fallback.
    pub nodes: BTreeSet<String>,
    /// Named, reviewed scale requests. Callers cannot supply a new count or group.
    pub scales: BTreeMap<String, NomadScale>,
    /// Protected operator ownership and optimistic index pins for SERVICE jobs only.
    pub services: BTreeMap<String, NomadService>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NomadService {
    pub owner: String,
    pub expected_modify_index: u64,
    /// Additional allocations per group reserved during rolling replacement.
    pub rollout_overlap: u32,
    pub allow_stop: bool,
    pub allow_restart: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NomadScale {
    pub job: String,
    pub group: String,
    pub count: u32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IncusConfig {
    pub connection: ConnectionConfig,
    pub project: String,
    pub instances: BTreeSet<String>,
    pub actions: BTreeSet<InstanceAction>,
    pub resources: BTreeSet<IncusResource>,
    /// Exact typed action + resource + operator-authored payload grants.
    pub operations: BTreeMap<String, IncusOperation>,
    /// Remote paths are guest paths, not host filesystem roots. Exact grants only.
    pub file_reads: BTreeMap<String, IncusFile>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IncusResource {
    Instance { name: String },
    Image { fingerprint: String },
    Network { name: String },
    StoragePool { name: String },
    StorageVolume { pool: String, name: String },
    Profile { name: String },
    Project { name: String },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IncusFile {
    pub instance: String,
    pub path: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum IncusOperation {
    Create {
        resource: IncusResource,
        definition: Value,
    },
    Update {
        resource: IncusResource,
        definition: Value,
    },
    Delete {
        resource: IncusResource,
    },
    Exec {
        instance: String,
        argv: Vec<String>,
    },
    FileWrite {
        file: IncusFile,
        content: String,
    },
    SnapshotCreate {
        instance: String,
        snapshot: String,
    },
    SnapshotRestore {
        instance: String,
        snapshot: String,
    },
    SnapshotDelete {
        instance: String,
        snapshot: String,
    },
    DeviceSet {
        instance: String,
        device: String,
        definition: BTreeMap<String, String>,
    },
    DeviceDelete {
        instance: String,
        device: String,
    },
    Copy {
        instance: String,
        target: String,
    },
    Move {
        instance: String,
        target: String,
    },
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
    /// Opt-in download ceiling, 1..4 MiB. Zero disables artifact downloads.
    pub max_artifact_bytes: usize,
    /// Named host read paths and LoadImage node bindings, authored by the operator.
    pub conditioning: BTreeMap<String, ImageConditioning>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageConditioning {
    pub workflow: String,
    pub node: String,
    pub path: std::path::PathBuf,
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
    /// Exact text input names allowed for SetInputSettings(text only).
    pub text_sources: BTreeSet<String>,
    pub actions: BTreeSet<ObsAction>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObsAction {
    SetSourceText,
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
