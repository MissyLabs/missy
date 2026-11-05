//! Opt-in native integrations. All HTTP and WS through PolicyHttpClient.
mod comfy;
mod scene;
pub use scene::{SceneGenerationGrant, SceneImageRequest};
mod config;
mod desktop_ws;
mod incus_ops;
mod infra;
mod speech;
mod transport;
use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
pub use comfy::ArtifactMetadata;
use comfy::Comfy;
pub use config::*;
use desktop_ws::{Obs, Vtube};
use infra::{Incus, Nomad};
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use missy_policy::PolicyHttpClient;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use transport::Connection;

/// Trusted embedding application boundary, never a model-provided confirm flag.
/// Verify the exact scope, turn, call ID, name and arguments when granting approval.
/// The host runtime must journal side_effect tools and never retry uncertain effects.
#[async_trait]
pub trait ApprovalGate: Send + Sync {
    async fn approve(&self, ctx: &ToolContext, call: &ToolCall) -> Result<bool>;
}
struct DenyMutations;
#[async_trait]
pub trait ArtifactValidator: Send + Sync {
    fn configuration_fingerprint(&self) -> String;
    async fn validate_mp4(&self, ctx: &ToolContext, bytes: &[u8]) -> Result<Value>;
}
#[async_trait]
impl ApprovalGate for DenyMutations {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> Result<bool> {
        Ok(false)
    }
}

pub struct IntegrationTools {
    nomad: Option<Nomad>,
    incus: Option<Incus>,
    comfy: Option<Comfy>,
    obs: Option<Obs>,
    vtube: Option<Vtube>,
    speech: Option<(
        Arc<dyn missy_voice::Tts>,
        Arc<dyn missy_voice::SpeechOutput>,
    )>,
    approval: Arc<dyn ApprovalGate>,
    fingerprint: String,
    artifact_validator: Option<Arc<dyn ArtifactValidator>>,
}
impl IntegrationTools {
    /// Resolves named credential environment vars once, validates config, no network.
    /// Default approval denies every mutation, even if allow_mutations=true.
    pub fn new(http: PolicyHttpClient, config: IntegrationConfig) -> Result<Self> {
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(
                "missy-integrations-v1",
                &config,
                http.config()
            ))?)
        );
        let nomad = config
            .nomad
            .filter(|c| c.connection.enabled)
            .map(|cfg| -> Result<_> {
                Nomad::validate(&cfg)?;
                Ok(Nomad::new(
                    Connection::new(&http, cfg.connection.clone(), false)?,
                    cfg,
                ))
            })
            .transpose()?;
        let incus = config
            .incus
            .filter(|c| c.connection.enabled)
            .map(|cfg| -> Result<_> {
                Incus::validate(&cfg)?;
                Incus::validate_operations(&cfg)?;
                Ok(Incus {
                    conn: Connection::new(&http, cfg.connection.clone(), false)?,
                    cfg,
                })
            })
            .transpose()?;
        let comfy = config
            .comfyui
            .filter(|c| c.connection.enabled)
            .map(|cfg| -> Result<_> {
                Comfy::validate(&cfg)?;
                Ok(Comfy {
                    conn: Connection::new(&http, cfg.connection.clone(), false)?,
                    cfg,
                    jobs: Mutex::new(HashMap::new()),
                })
            })
            .transpose()?;
        let obs = config
            .obs
            .filter(|c| c.connection.enabled)
            .map(|cfg| -> Result<_> {
                Obs::validate(&cfg)?;
                Ok(Obs {
                    conn: Connection::new(&http, cfg.connection.clone(), true)?,
                    cfg,
                })
            })
            .transpose()?;
        let vtube = config
            .vtube
            .filter(|c| c.connection.enabled)
            .map(|cfg| -> Result<_> {
                Vtube::validate(&cfg)?;
                Ok(Vtube {
                    conn: Connection::new(&http, cfg.connection.clone(), true)?,
                    cfg,
                })
            })
            .transpose()?;
        Ok(Self {
            nomad,
            incus,
            comfy,
            obs,
            vtube,
            speech: None,
            approval: Arc::new(DenyMutations),
            fingerprint,
            artifact_validator: None,
        })
    }
    /// Privileged host setup, not a tool argument or dynamic configuration operation.
    pub fn with_approval(mut self, approval: Arc<dyn ApprovalGate>) -> Self {
        self.approval = approval;
        self
    }
    /// Full decoder required before external MP4 publication.
    pub fn with_artifact_validator(mut self, validator: Arc<dyn ArtifactValidator>) -> Self {
        self.fingerprint = format!(
            "{:x}",
            Sha256::digest(format!(
                "{}:{}",
                self.fingerprint,
                validator.configuration_fingerprint()
            ))
        );
        self.artifact_validator = Some(validator);
        self
    }
    fn connection(&self, name: &str) -> Option<&Connection> {
        // Output adapters enforce durable approval for synchronized speech.
        match name {
            "nomad_read" | "nomad_mutate" => self.nomad.as_ref().map(|s| &s.conn),
            "incus_read" | "incus_mutate" | "incus_discover" | "incus_file_read"
            | "incus_operation" => self.incus.as_ref().map(|s| &s.conn),
            "comfyui_history" | "comfyui_submit" | "comfyui_artifact" => {
                self.comfy.as_ref().map(|s| &s.conn)
            }
            "obs_read" | "obs_mutate" => self.obs.as_ref().map(|s| &s.conn),
            "vtube_read" | "vtube_mutate" | "vtube_speak" => self.vtube.as_ref().map(|s| &s.conn),
            _ => None,
        }
    }
    // Pure validation, repeated by adapters as defense in depth. No approval
    // consumption or network occurs here; refusals are known non-execution.
    fn validate_call(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        ctx.scope.validate()?;
        let conn = self
            .connection(&call.name)
            .ok_or_else(|| anyhow!("integration tool disabled or unknown"))?;
        if !call.arguments.is_object() || serde_json::to_vec(&call.arguments)?.len() > 16_384 {
            bail!("integration argument bound exceeded");
        }
        if mutation(&call.name) && !conn.cfg.allow_mutations {
            bail!("integration mutations disabled by operator");
        }
        match call.name.as_str() {
            "vtube_speak" => {
                let a: speech::Speak = parse(call)?;
                self.vtube.as_ref().unwrap().validate_speech(&a)?;
                let (_, output) = self
                    .speech
                    .as_ref()
                    .ok_or_else(|| anyhow!("speech output not configured"))?;
                output.validate_output(&a.sink, a.volume_pct)?;
            }
            "nomad_read" => {
                let cfg = &self.nomad.as_ref().unwrap().cfg;
                match parse(call)? {
                    infra::NomadRead::Status { job }
                    | infra::NomadRead::Allocations { job }
                    | infra::NomadRead::Results { job }
                    | infra::NomadRead::Placement { job } => {
                        if !cfg.jobs.contains(&job) {
                            bail!("Nomad job denied");
                        }
                    }
                    infra::NomadRead::Node { node } => {
                        if !cfg.nodes.contains(&node) {
                            bail!("Nomad node denied");
                        }
                    }
                    infra::NomadRead::List | infra::NomadRead::Nodes => {}
                }
            }
            "nomad_mutate" => {
                let cfg = &self.nomad.as_ref().unwrap().cfg;
                match parse(call)? {
                    infra::NomadMutate::Deploy { job, .. }
                    | infra::NomadMutate::Plan { job }
                    | infra::NomadMutate::Restart { job, .. }
                    | infra::NomadMutate::PlanRestart { job }
                    | infra::NomadMutate::Stop { job, .. }
                    | infra::NomadMutate::PlanStop { job } => {
                        if !cfg.jobs.contains(&job)
                            || !cfg.deployments.contains_key(&job)
                            || !cfg.services.contains_key(&job)
                        {
                            bail!("Nomad deployment denied");
                        }
                    }
                    infra::NomadMutate::Scale { grant, .. }
                    | infra::NomadMutate::PlanScale { grant } => {
                        if !cfg
                            .scales
                            .get(&grant)
                            .is_some_and(|s| cfg.services.contains_key(&s.job))
                        {
                            bail!("Nomad scale denied");
                        }
                    }
                }
            }
            "incus_discover" => self
                .incus
                .as_ref()
                .unwrap()
                .validate_discover(&parse(call)?)?,
            "incus_file_read" => {
                let a: incus_ops::Grant = parse(call)?;
                if !self
                    .incus
                    .as_ref()
                    .unwrap()
                    .cfg
                    .file_reads
                    .contains_key(&a.grant)
                {
                    bail!("Incus file grant denied");
                }
            }
            "incus_operation" => {
                let a: incus_ops::Grant = parse(call)?;
                if !self
                    .incus
                    .as_ref()
                    .unwrap()
                    .cfg
                    .operations
                    .contains_key(&a.grant)
                {
                    bail!("Incus operation grant denied");
                }
            }
            "comfyui_artifact" => self
                .comfy
                .as_ref()
                .unwrap()
                .validate_download(&ctx.scope, &parse(call)?)?,
            "incus_read" => {
                if let infra::IncusRead::Status { instance } = parse(call)?
                    && !self
                        .incus
                        .as_ref()
                        .unwrap()
                        .cfg
                        .instances
                        .contains(&instance)
                {
                    bail!("Incus instance denied");
                }
            }
            "incus_mutate" => {
                let a: infra::IncusMutate = parse(call)?;
                let cfg = &self.incus.as_ref().unwrap().cfg;
                if !cfg.instances.contains(&a.instance) || !cfg.actions.contains(&a.action) {
                    bail!("Incus lifecycle action denied");
                }
            }
            "comfyui_submit" => {
                let a: comfy::Submit = parse(call)?;
                let cfg = &self.comfy.as_ref().unwrap().cfg;
                let w = cfg
                    .workflows
                    .get(&a.workflow)
                    .ok_or_else(|| anyhow!("ComfyUI workflow not approved"))?;
                if let Some(name) = &a.conditioning
                    && cfg
                        .conditioning
                        .get(name)
                        .is_none_or(|input| input.workflow != a.workflow)
                {
                    bail!("ComfyUI conditioning grant denied");
                }
                if let Some(prompt) = a.prompt
                    && (w.prompt_node.is_none()
                        || prompt.is_empty()
                        || prompt.len() > 8192
                        || prompt.contains('\0'))
                {
                    bail!("ComfyUI prompt binding denied");
                }
            }
            "comfyui_history" => {
                let a: comfy::History = parse(call)?;
                identifier(&a.prompt_id)?;
                if self
                    .comfy
                    .as_ref()
                    .unwrap()
                    .jobs
                    .lock()
                    .map_err(|_| anyhow!("ComfyUI registry unavailable"))?
                    .get(&a.prompt_id)
                    != Some(&ctx.scope)
                {
                    bail!("ComfyUI prompt is not owned by this scope");
                }
            }
            "obs_read" => {
                let _: desktop_ws::ObsRead = parse(call)?;
            }
            "obs_mutate" => {
                let cfg = &self.obs.as_ref().unwrap().cfg;
                let action = match parse(call)? {
                    desktop_ws::ObsMutate::SetSourceText { source, text } => {
                        if !cfg.text_sources.contains(&source)
                            || text.len() > 8192
                            || text.contains('\0')
                        {
                            bail!("OBS text source/content denied");
                        }
                        ObsAction::SetSourceText
                    }
                    desktop_ws::ObsMutate::SwitchScene { scene } => {
                        if !cfg.scenes.contains(&scene) {
                            bail!("OBS scene denied");
                        }
                        ObsAction::SwitchScene
                    }
                    desktop_ws::ObsMutate::SetVisibility { scene, item_id, .. } => {
                        if !cfg.scenes.contains(&scene)
                            || !cfg.scene_items.contains(&format!("{scene}:{item_id}"))
                        {
                            bail!("OBS scene item denied");
                        }
                        ObsAction::SetVisibility
                    }
                    desktop_ws::ObsMutate::StartRecording => ObsAction::StartRecording,
                    desktop_ws::ObsMutate::StopRecording => ObsAction::StopRecording,
                    desktop_ws::ObsMutate::StartStreaming => ObsAction::StartStreaming,
                    desktop_ws::ObsMutate::StopStreaming => ObsAction::StopStreaming,
                };
                if !cfg.actions.contains(&action) {
                    bail!("OBS mutation action denied");
                }
            }
            "vtube_read" => {
                let _: desktop_ws::VtubeRead = parse(call)?;
            }
            "vtube_mutate" => {
                let cfg = &self.vtube.as_ref().unwrap().cfg;
                match parse(call)? {
                    desktop_ws::VtubeMutate::LoadModel { model } => {
                        if !cfg.models.contains(&model) {
                            bail!("VTube Studio model denied");
                        }
                    }
                    desktop_ws::VtubeMutate::TriggerHotkey { hotkey } => {
                        if !cfg.hotkeys.contains(&hotkey) {
                            bail!("VTube Studio hotkey denied");
                        }
                    }
                    desktop_ws::VtubeMutate::SetParameter { parameter, value } => {
                        let r = cfg
                            .parameters
                            .get(&parameter)
                            .ok_or_else(|| anyhow!("VTube Studio parameter denied"))?;
                        if !value.is_finite() || value < r.min || value > r.max {
                            bail!("VTube Studio parameter value outside approved range");
                        }
                    }
                }
            }
            _ => bail!("integration tool disabled or unknown"),
        }
        Ok(())
    }
    async fn dispatch(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        match call.name.as_str() {
            "vtube_speak" => {
                let (tts, output) = self
                    .speech
                    .as_ref()
                    .ok_or_else(|| anyhow!("speech not configured"))?;
                self.vtube
                    .as_ref()
                    .unwrap()
                    .speak(ctx, call, parse(call)?, tts.as_ref(), output.as_ref())
                    .await
            }
            "nomad_read" => self.nomad.as_ref().unwrap().read(parse(call)?).await,
            "nomad_mutate" => self.nomad.as_ref().unwrap().mutate(parse(call)?).await,
            "incus_read" => self.incus.as_ref().unwrap().read(parse(call)?).await,
            "incus_mutate" => self.incus.as_ref().unwrap().mutate(parse(call)?).await,
            "incus_discover" => self.incus.as_ref().unwrap().discover(parse(call)?).await,
            "incus_file_read" => self.incus.as_ref().unwrap().file_read(parse(call)?).await,
            "incus_operation" => self.incus.as_ref().unwrap().operation(parse(call)?).await,
            "comfyui_artifact" => {
                self.comfy
                    .as_ref()
                    .unwrap()
                    .download(ctx, parse(call)?, self.artifact_validator.as_deref())
                    .await
            }
            "comfyui_submit" => {
                self.comfy
                    .as_ref()
                    .unwrap()
                    .submit(&ctx.scope, parse(call)?)
                    .await
            }
            "comfyui_history" => {
                self.comfy
                    .as_ref()
                    .unwrap()
                    .history(&ctx.scope, parse(call)?)
                    .await
            }
            "obs_read" => self.obs.as_ref().unwrap().read(parse(call)?).await,
            "obs_mutate" => self.obs.as_ref().unwrap().mutate(parse(call)?).await,
            "vtube_read" => self.vtube.as_ref().unwrap().read(parse(call)?).await,
            "vtube_mutate" => self.vtube.as_ref().unwrap().mutate(parse(call)?).await,
            _ => bail!("integration tool disabled or unknown"),
        }
    }
}
fn parse<T: DeserializeOwned>(call: &ToolCall) -> Result<T> {
    let schema = schemas()
        .into_iter()
        .find(|(name, _, _)| *name == call.name)
        .ok_or_else(|| anyhow!("unknown integration schema"))?
        .2;
    if !validate_schema(&call.arguments, &schema) {
        bail!("invalid strict integration arguments");
    }
    serde_json::from_value(call.arguments.clone())
        .map_err(|_| anyhow!("invalid strict integration arguments"))
}
// Serde's internally-tagged unit variants accept unknown fields even with
// deny_unknown_fields. Enforce the advertised exact property set independently.
fn validate_schema(value: &Value, schema: &Value) -> bool {
    if let Some(options) = schema["oneOf"].as_array() {
        return options.iter().filter(|s| validate_schema(value, s)).count() == 1;
    }
    if let Some(c) = schema.get("const")
        && c != value
    {
        return false;
    }
    if let Some(options) = schema["enum"].as_array()
        && !options.contains(value)
    {
        return false;
    }
    match schema["type"].as_str() {
        Some("object") => {
            let Some(obj) = value.as_object() else {
                return false;
            };
            let Some(props) = schema["properties"].as_object() else {
                return false;
            };
            if obj.keys().any(|k| !props.contains_key(k)) {
                return false;
            }
            if schema["required"].as_array().is_some_and(|required| {
                required
                    .iter()
                    .any(|k| k.as_str().is_none_or(|k| !obj.contains_key(k)))
            }) {
                return false;
            }
            obj.iter().all(|(k, v)| validate_schema(v, &props[k]))
        }
        Some("string") => value.as_str().is_some_and(|s| {
            s.len() >= schema["minLength"].as_u64().unwrap_or(0) as usize
                && s.len() <= schema["maxLength"].as_u64().unwrap_or(u64::MAX) as usize
        }),
        Some("boolean") => value.is_boolean(),
        Some("number") => value.as_f64().is_some_and(|v| v.is_finite()),
        Some("integer") => value
            .as_u64()
            .is_some_and(|v| v >= schema["minimum"].as_u64().unwrap_or(0)),
        _ => false,
    }
}
fn mutation(name: &str) -> bool {
    name.ends_with("_mutate")
        || matches!(
            name,
            "comfyui_submit" | "comfyui_artifact" | "incus_operation" | "vtube_speak"
        )
}
#[async_trait]
impl ToolExecutor for IntegrationTools {
    fn configuration_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.validate_call(ctx, call).map_err(|e| {
            missy_core::ToolDispatchRefused {
                reason: e.to_string(),
            }
            .into()
        })
    }
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = vec![];
        for (name, description, parameters) in schemas() {
            if name == "vtube_speak" && self.speech.is_none() {
                continue;
            }
            if let Some(conn) = self.connection(name)
                && (!mutation(name) || conn.cfg.allow_mutations)
            {
                defs.push(ToolDefinition {
                    name: name.into(),
                    description: description.into(),
                    parameters,
                    side_effect: mutation(name),
                });
            }
        }
        defs
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.preflight(ctx, call).await?;
        ctx.scope.validate()?;
        let conn = self
            .connection(&call.name)
            .ok_or_else(|| anyhow!("integration tool disabled or unknown"))?;
        if !call.arguments.is_object() || serde_json::to_vec(&call.arguments)?.len() > 16_384 {
            bail!("integration argument bound exceeded");
        }
        let future = async {
            if mutation(&call.name) && call.name != "vtube_speak" {
                if !conn.cfg.allow_mutations {
                    return Err(missy_core::ToolDispatchRefused {
                        reason: "integration mutations disabled by operator".into(),
                    }
                    .into());
                }
                if !self
                    .approval
                    .approve(ctx, call)
                    .await
                    .map_err(|_| anyhow!("integration approval failed"))?
                {
                    return Err(missy_core::ToolDispatchRefused {
                        reason: "integration requires trusted per-call approval".into(),
                    }
                    .into());
                }
            }
            let mut data = self.dispatch(ctx, call).await?;
            for c in [
                self.nomad.as_ref().map(|x| &x.conn),
                self.incus.as_ref().map(|x| &x.conn),
                self.comfy.as_ref().map(|x| &x.conn),
                self.obs.as_ref().map(|x| &x.conn),
                self.vtube.as_ref().map(|x| &x.conn),
            ]
            .into_iter()
            .flatten()
            {
                c.redact(&mut data);
            }
            let wrapped = json!({"trust":"untrusted","data":data});
            if serde_json::to_vec(&wrapped)?.len() > conn.cfg.max_output_bytes {
                bail!("integration result exceeds output bound");
            }
            Ok(wrapped)
        };
        // Speech owns cancellation and a bounded mouth reset. An outer select
        // would drop its cleanup future and leave stale avatar state.
        if call.name == "vtube_speak" {
            return future.await;
        }
        tokio::select! {
            biased;
            _ = ctx.cancellation.cancelled() => bail!("integration cancelled; mutations may have uncertain outcome, do not replay"),
            result = tokio::time::timeout(conn.duration(),future) => result.map_err(|_| anyhow!("integration deadline exceeded; mutations may have uncertain outcome, do not replay"))?,
        }
    }
}

fn object(fields: &[(&str, Value)], required: &[&str]) -> Value {
    json!({"type":"object","additionalProperties":false,"properties":fields.iter().map(|(k,v)| ((*k).to_owned(),v.clone())).collect::<serde_json::Map<_,_>>(),"required":required})
}
fn variant(action: &str, fields: &[(&str, Value)]) -> Value {
    let mut fields = fields.to_vec();
    fields.push(("action", json!({"type":"string","const":action})));
    let required: Vec<_> = fields.iter().map(|(k, _)| *k).collect();
    object(&fields, &required)
}
fn choice(options: Vec<Value>) -> Value {
    json!({"oneOf":options})
}
fn schemas() -> Vec<(&'static str, &'static str, Value)> {
    let s = json!({"type":"string","minLength":1,"maxLength":256});
    vec![
        (
            "nomad_read",
            "List or inspect exact operator-allowed Nomad jobs.",
            choice(vec![
                variant("list", &[]),
                variant("status", &[("job", s.clone())]),
                variant("nodes", &[]),
                variant("node", &[("node", s.clone())]),
                variant("placement", &[("job", s.clone())]),
                variant("allocations", &[("job", s.clone())]),
                variant("results", &[("job", s.clone())]),
            ]),
        ),
        (
            "nomad_mutate",
            "Plan and submit owned pinned SERVICE lifecycle changes with one-use reviewed plans and optimistic CAS. Requires human approval.",
            choice(vec![
                variant("deploy", &[("job", s.clone()), ("plan", s.clone())]),
                variant("stop", &[("job", s.clone()), ("plan", s.clone())]),
                variant("plan_stop", &[("job", s.clone())]),
                variant("plan", &[("job", s.clone())]),
                variant("scale", &[("grant", s.clone()), ("plan", s.clone())]),
                variant("plan_scale", &[("grant", s.clone())]),
                variant("plan_restart", &[("job", s.clone())]),
                variant("restart", &[("job", s.clone()), ("plan", s.clone())]),
            ]),
        ),
        (
            "incus_read",
            "List allowed Incus instances or inspect instance state.",
            choice(vec![
                variant("list", &[]),
                variant("status", &[("instance", s.clone())]),
            ]),
        ),
        (
            "incus_mutate",
            "Non-forced lifecycle action on an exactly allowed instance. Requires human approval.",
            object(
                &[
                    ("instance", s.clone()),
                    (
                        "action",
                        json!({"type":"string","enum":["start","stop","restart","freeze","unfreeze"]}),
                    ),
                ],
                &["instance", "action"],
            ),
        ),
        (
            "comfyui_submit",
            "Submit a named operator-approved workflow. Optional text binds only to the configured CLIP prompt node. Requires human approval.",
            object(
                &[
                    ("workflow", s.clone()),
                    ("conditioning", s.clone()),
                    (
                        "prompt",
                        json!({"type":"string","minLength":1,"maxLength":8192}),
                    ),
                ],
                &["workflow"],
            ),
        ),
        (
            "incus_discover",
            "Discover or inspect exact operator-allowed typed resources.",
            choice(vec![
                variant("list", &[("kind", s.clone())]),
                variant("inspect", &[("resource", s.clone())]),
                variant("snapshots", &[("instance", s.clone())]),
            ]),
        ),
        (
            "incus_file_read",
            "Read an exact operator-approved guest text file. No arbitrary guest paths.",
            object(&[("grant", s.clone())], &["grant"]),
        ),
        (
            "incus_operation",
            "Execute a named typed operator-pinned Incus operation. Human approval required. Caller cannot supply argv, bodies or routes.",
            object(&[("grant", s.clone())], &["grant"]),
        ),
        (
            "comfyui_artifact",
            "Download an exact owned-history artifact descriptor to a policy write path with human approval. Configured ceiling at most 4 MiB; images decoded, MP4 container structure checked.",
            object(
                &[
                    ("prompt_id", s.clone()),
                    ("node", s.clone()),
                    ("kind", s.clone()),
                    ("output_index", json!({"type":"integer","minimum":0})),
                    (
                        "output_path",
                        json!({"type":"string","minLength":1,"maxLength":4096}),
                    ),
                ],
                &["prompt_id", "node", "kind", "output_index", "output_path"],
            ),
        ),
        (
            "comfyui_history",
            "Read scope-owned prompt history. Validated artifact metadata only, no downloads.",
            object(&[("prompt_id", s.clone())], &["prompt_id"]),
        ),
        (
            "obs_read",
            "Read OBS streaming/recording status or allowed scene metadata.",
            choice(vec![variant("status", &[]), variant("list_scenes", &[])]),
        ),
        (
            "obs_mutate",
            "Allowed scene/item/recording/broadcast action. Requires trusted human approval for the exact call.",
            choice(vec![
                variant("switch_scene", &[("scene", s.clone())]),
                variant(
                    "set_source_text",
                    &[
                        ("source", s.clone()),
                        ("text", json!({"type":"string","maxLength":8192})),
                    ],
                ),
                variant(
                    "set_visibility",
                    &[
                        ("scene", s.clone()),
                        ("item_id", json!({"type":"integer","minimum":0})),
                        ("visible", json!({"type":"boolean"})),
                    ],
                ),
                variant("start_recording", &[]),
                variant("stop_recording", &[]),
                variant("start_streaming", &[]),
                variant("stop_streaming", &[]),
            ]),
        ),
        (
            "vtube_speak",
            "Approved routed speech synchronized with a bounded avatar mouth parameter.",
            object(
                &[
                    (
                        "text",
                        json!({"type":"string","minLength":1,"maxLength":8192}),
                    ),
                    ("sink", s.clone()),
                    ("volume_pct", json!({"type":"integer","minimum":0})),
                    ("parameter", s.clone()),
                ],
                &["text", "sink", "volume_pct", "parameter"],
            ),
        ),
        (
            "vtube_read",
            "Read allowed VTube Studio model/hotkey metadata using a pre-provisioned token. No authorization popup.",
            choice(vec![
                variant("status", &[]),
                variant("list_models", &[]),
                variant("list_hotkeys", &[]),
            ]),
        ),
        (
            "vtube_mutate",
            "Load an allowed model, trigger allowed hotkey or set bounded allowed parameter. Requires human approval.",
            choice(vec![
                variant("load_model", &[("model", s.clone())]),
                variant("trigger_hotkey", &[("hotkey", s.clone())]),
                variant(
                    "set_parameter",
                    &[("parameter", s), ("value", json!({"type":"number"}))],
                ),
            ]),
        ),
    ]
}
