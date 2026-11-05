//! Typed Incus API adapters. Grants are operator-authored, never caller routes/bodies.
use crate::{
    config::*,
    infra::{Incus, select},
};
use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Grant {
    pub grant: String,
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Discover {
    List { kind: String },
    Inspect { resource: String },
    Snapshots { instance: String },
}

impl IncusResource {
    pub fn key(&self) -> String {
        match self {
            Self::Instance { name } => format!("instance/{name}"),
            Self::Image { fingerprint } => format!("image/{fingerprint}"),
            Self::Network { name } => format!("network/{name}"),
            Self::StoragePool { name } => format!("storage_pool/{name}"),
            Self::StorageVolume { pool, name } => format!("storage_volume/{pool}/{name}"),
            Self::Profile { name } => format!("profile/{name}"),
            Self::Project { name } => format!("project/{name}"),
        }
    }
    fn parts(&self) -> (&'static str, String, String) {
        match self {
            Self::Instance { name } => ("instance", "/1.0/instances".into(), name.clone()),
            Self::Image { fingerprint } => ("image", "/1.0/images".into(), fingerprint.clone()),
            Self::Network { name } => ("network", "/1.0/networks".into(), name.clone()),
            Self::StoragePool { name } => {
                ("storage_pool", "/1.0/storage-pools".into(), name.clone())
            }
            Self::StorageVolume { pool, name } => (
                "storage_volume",
                format!("/1.0/storage-pools/{pool}/volumes/custom"),
                name.clone(),
            ),
            Self::Profile { name } => ("profile", "/1.0/profiles".into(), name.clone()),
            Self::Project { name } => ("project", "/1.0/projects".into(), name.clone()),
        }
    }
    fn validate(&self) -> Result<()> {
        let (_, _, name) = self.parts();
        identifier(&name)?;
        if let Self::StorageVolume { pool, .. } = self {
            identifier(pool)?;
        }
        if let Self::Image { fingerprint } = self
            && (fingerprint.len() != 64 || !fingerprint.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            bail!("image resource requires full fingerprint");
        }
        Ok(())
    }
}
fn guest_path(path: &str) -> Result<()> {
    if !path.starts_with('/')
        || path.len() > 4096
        || path.contains('\\')
        || path.chars().any(char::is_control)
        || path
            .split('/')
            .skip(1)
            .any(|p| p.is_empty() || p == "." || p == "..")
    {
        bail!("guest path must be exact absolute file path without dot components");
    }
    Ok(())
}
fn encoded(path: &str) -> String {
    url::form_urlencoded::byte_serialize(path.as_bytes()).collect()
}

impl Incus {
    pub(crate) fn validate_operations(cfg: &IncusConfig) -> Result<()> {
        for resource in &cfg.resources {
            resource.validate()?;
            if let IncusResource::Instance { name } = resource
                && !cfg.instances.contains(name)
            {
                bail!("instance resource must also be allowed instance");
            }
            if let IncusResource::Project { name } = resource
                && name != &cfg.project
            {
                bail!("project resource must match configured project");
            }
        }
        let check_instance = |instance: &str| -> Result<()> {
            identifier(instance)?;
            if !cfg.instances.contains(instance) {
                bail!("Incus operation instance denied");
            }
            Ok(())
        };
        let check_file = |f: &IncusFile| -> Result<()> {
            check_instance(&f.instance)?;
            guest_path(&f.path)
        };
        for (name, file) in &cfg.file_reads {
            identifier(name)?;
            check_file(file)?;
        }
        for (name, op) in &cfg.operations {
            identifier(name)?;
            match op {
                IncusOperation::Create {
                    resource,
                    definition,
                }
                | IncusOperation::Update {
                    resource,
                    definition,
                } => {
                    if !cfg.resources.contains(resource) {
                        bail!("Incus resource grant denied");
                    }
                    Self::validate_definition(
                        resource,
                        definition,
                        matches!(op, IncusOperation::Create { .. }),
                    )?;
                    if let Some(fingerprint) = definition["source"]["fingerprint"].as_str()
                        && !cfg.resources.contains(&IncusResource::Image {
                            fingerprint: fingerprint.into(),
                        })
                    {
                        bail!("launch image must also be resource-allowed");
                    }
                }
                IncusOperation::Delete { resource } => {
                    if !cfg.resources.contains(resource) {
                        bail!("Incus resource grant denied");
                    }
                }
                IncusOperation::Exec { instance, argv } => {
                    check_instance(instance)?;
                    if argv.is_empty()
                        || argv.len() > 128
                        || !argv[0].starts_with('/')
                        || argv.iter().any(|a| a.contains('\0'))
                        || argv.iter().map(String::len).sum::<usize>() > 16384
                    {
                        bail!("Incus exec requires pinned absolute argv within bounds");
                    }
                }
                IncusOperation::FileWrite { file, content } => {
                    check_file(file)?;
                    if content.len() > cfg.connection.max_output_bytes {
                        bail!("Incus file content exceeds bound");
                    }
                }
                IncusOperation::SnapshotCreate { instance, snapshot }
                | IncusOperation::SnapshotRestore { instance, snapshot }
                | IncusOperation::SnapshotDelete { instance, snapshot } => {
                    check_instance(instance)?;
                    identifier(snapshot)?;
                }
                IncusOperation::DeviceSet {
                    instance,
                    device,
                    definition,
                } => {
                    check_instance(instance)?;
                    identifier(device)?;
                    if definition.is_empty()
                        || definition.len() > 64
                        || !definition.contains_key("type")
                        || serde_json::to_vec(definition)?.len() > 16384
                    {
                        bail!("invalid pinned device");
                    }
                }
                IncusOperation::DeviceDelete { instance, device } => {
                    check_instance(instance)?;
                    identifier(device)?;
                }
                IncusOperation::Copy { instance, target }
                | IncusOperation::Move { instance, target } => {
                    check_instance(instance)?;
                    check_instance(target)?;
                    if instance == target {
                        bail!("Incus source and destination must differ");
                    }
                }
            }
        }
        Ok(())
    }
    fn validate_definition(resource: &IncusResource, body: &Value, create: bool) -> Result<()> {
        let obj = body
            .as_object()
            .ok_or_else(|| anyhow!("Incus definition must be object"))?;
        if serde_json::to_vec(body)?.len() > 262144 {
            bail!("Incus definition exceeds bound");
        }
        let fields: &[&str] = match resource {
            IncusResource::Instance { .. } => &[
                "name",
                "architecture",
                "config",
                "devices",
                "ephemeral",
                "profiles",
                "source",
                "type",
                "description",
                "restore",
                "stateful",
            ],
            IncusResource::Image { .. } => &["auto_update", "properties", "public", "expires_at"],
            IncusResource::Network { .. } => &["name", "type", "config", "description"],
            IncusResource::StoragePool { .. } => &["name", "driver", "config", "description"],
            IncusResource::StorageVolume { .. } => &[
                "name",
                "type",
                "content_type",
                "config",
                "description",
                "source",
            ],
            IncusResource::Profile { .. } => &["name", "config", "devices", "description"],
            IncusResource::Project { .. } => &["name", "config", "description"],
        };
        if obj.keys().any(|k| !fields.contains(&k.as_str())) {
            bail!("unknown Incus definition field");
        }
        let (_, _, name) = resource.parts();
        if (create && !matches!(resource, IncusResource::Image { .. }) && body["name"] != name)
            || body.get("name").is_some_and(|n| n != &name)
        {
            bail!("Incus definition name must equal exact resource");
        }
        if create && matches!(resource, IncusResource::Image { .. }) {
            bail!("image import not enabled; use pinned image update/delete");
        }
        if create && matches!(resource, IncusResource::Instance { .. }) {
            if !["container", "virtual-machine"].contains(&body["type"].as_str().unwrap_or(""))
                || body["source"]["type"] != "image"
            {
                bail!("instance creation requires explicit type and pinned local image");
            }
            let source = body["source"]
                .as_object()
                .ok_or_else(|| anyhow!("invalid image source"))?;
            if source
                .keys()
                .any(|k| !["type", "fingerprint"].contains(&k.as_str()))
                || !body["source"]["fingerprint"]
                    .as_str()
                    .is_some_and(|f| f.len() == 64 && f.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                bail!("instance source must be exact local image fingerprint; no remote fetch");
            }
        }
        Ok(())
    }
    pub(crate) fn validate_discover(&self, a: &Discover) -> Result<()> {
        match a {
            Discover::List { kind } => {
                if ![
                    "instance",
                    "image",
                    "network",
                    "storage_pool",
                    "profile",
                    "project",
                ]
                .contains(&kind.as_str())
                {
                    bail!("unsupported resource list kind");
                }
            }
            Discover::Inspect { resource } => {
                if !self.cfg.resources.iter().any(|r| r.key() == *resource) {
                    bail!("Incus resource denied");
                }
            }
            Discover::Snapshots { instance } => self.check(instance)?,
        }
        Ok(())
    }
    pub(crate) async fn discover(&self, a: Discover) -> Result<Value> {
        self.validate_discover(&a)?;
        match a {
            Discover::Snapshots { instance } => {
                let r = self
                    .query(
                        "GET",
                        &format!(
                            "/1.0/instances/{instance}/snapshots?project={}&recursion=1",
                            self.cfg.project
                        ),
                        None,
                    )
                    .await?;
                let rows = r["metadata"]
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid Incus snapshots"))?;
                Ok(Value::Array(
                    rows.iter()
                        .map(|v| select(v, &["name", "created_at", "expires_at", "stateful"]))
                        .collect(),
                ))
            }
            Discover::Inspect { resource } => {
                let r = self
                    .cfg
                    .resources
                    .iter()
                    .find(|r| r.key() == resource)
                    .unwrap();
                let (_, base, name) = r.parts();
                let v = self
                    .query(
                        "GET",
                        &format!("{base}/{name}?project={}", self.cfg.project),
                        None,
                    )
                    .await?;
                Ok(v["metadata"].clone())
            }
            Discover::List { kind } => {
                let base = match kind.as_str() {
                    "instance" => "instances",
                    "image" => "images",
                    "network" => "networks",
                    "storage_pool" => "storage-pools",
                    "profile" => "profiles",
                    "project" => "projects",
                    _ => unreachable!(),
                };
                let v = self
                    .query(
                        "GET",
                        &format!("/1.0/{base}?project={}&recursion=1", self.cfg.project),
                        None,
                    )
                    .await?;
                let rows = v["metadata"]
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid Incus resources"))?;
                Ok(Value::Array(
                    rows.iter()
                        .filter(|v| {
                            self.cfg.resources.iter().any(|r| {
                                let (k, _, name) = r.parts();
                                k == kind
                                    && (v["name"] == name
                                        || (kind == "image" && v["fingerprint"] == name))
                            })
                        })
                        .map(|v| {
                            select(
                                v,
                                &[
                                    "name",
                                    "fingerprint",
                                    "description",
                                    "type",
                                    "status",
                                    "driver",
                                    "size",
                                    "created_at",
                                ],
                            )
                        })
                        .collect(),
                ))
            }
        }
    }
    pub(crate) async fn file_read(&self, a: Grant) -> Result<Value> {
        let f = self
            .cfg
            .file_reads
            .get(&a.grant)
            .ok_or_else(|| anyhow!("Incus file grant denied"))?;
        let data = self
            .conn
            .bytes(
                "GET",
                &format!(
                    "/1.0/instances/{}/files?project={}&path={}",
                    f.instance,
                    self.cfg.project,
                    encoded(&f.path)
                ),
                None,
                self.cfg.connection.max_output_bytes / 2,
            )
            .await?;
        let text = String::from_utf8(data).map_err(|_| anyhow!("Incus file is not UTF-8 text"))?;
        Ok(json!({"instance":f.instance,"path":f.path,"content":text}))
    }
    pub(crate) async fn operation(&self, a: Grant) -> Result<Value> {
        let op = self
            .cfg
            .operations
            .get(&a.grant)
            .ok_or_else(|| anyhow!("Incus operation grant denied"))?;
        let p = &self.cfg.project;
        let (method, path, body) = match op {
            IncusOperation::Create {
                resource,
                definition,
            } => {
                let (_, base, _) = resource.parts();
                (
                    "POST",
                    format!("{base}?project={p}"),
                    Some(definition.clone()),
                )
            }
            IncusOperation::Update {
                resource,
                definition,
            } => {
                let (_, base, name) = resource.parts();
                (
                    "PUT",
                    format!("{base}/{name}?project={p}"),
                    Some(definition.clone()),
                )
            }
            IncusOperation::Delete { resource } => {
                let (_, base, name) = resource.parts();
                ("DELETE", format!("{base}/{name}?project={p}"), None)
            }
            IncusOperation::Exec { instance, argv } => (
                "POST",
                format!("/1.0/instances/{instance}/exec?project={p}"),
                Some(
                    json!({"command":argv,"interactive":false,"wait-for-websocket":false,"record-output":true,"environment":{}}),
                ),
            ),
            IncusOperation::FileWrite { file, content } => {
                let r = self
                    .conn
                    .bytes(
                        "POST",
                        &format!(
                            "/1.0/instances/{}/files?project={p}&path={}",
                            file.instance,
                            encoded(&file.path)
                        ),
                        Some(content.as_bytes().to_vec()),
                        self.cfg.connection.max_output_bytes,
                    )
                    .await?;
                let envelope: Value = serde_json::from_slice(&r)
                    .map_err(|_| anyhow!("invalid Incus file response"))?;
                if envelope["type"] != "sync"
                    || envelope["error_code"].as_u64().is_some_and(|n| n != 0)
                {
                    bail!("Incus file write rejected");
                }
                return Ok(
                    json!({"written":true,"instance":file.instance,"path":file.path,"bytes":content.len()}),
                );
            }
            IncusOperation::SnapshotCreate { instance, snapshot } => (
                "POST",
                format!("/1.0/instances/{instance}/snapshots?project={p}"),
                Some(json!({"name":snapshot,"stateful":false})),
            ),
            IncusOperation::SnapshotRestore { instance, snapshot } => (
                "PUT",
                format!("/1.0/instances/{instance}?project={p}"),
                Some(json!({"restore":snapshot,"stateful":false})),
            ),
            IncusOperation::SnapshotDelete { instance, snapshot } => (
                "DELETE",
                format!("/1.0/instances/{instance}/snapshots/{snapshot}?project={p}"),
                None,
            ),
            IncusOperation::DeviceSet {
                instance, device, ..
            }
            | IncusOperation::DeviceDelete { instance, device } => {
                let v = self
                    .query(
                        "GET",
                        &format!("/1.0/instances/{instance}?project={p}"),
                        None,
                    )
                    .await?;
                let mut devices = v["metadata"]["devices"]
                    .as_object()
                    .ok_or_else(|| anyhow!("invalid Incus devices"))?
                    .clone();
                if let IncusOperation::DeviceSet { definition, .. } = op {
                    devices.insert(device.clone(), serde_json::to_value(definition)?);
                } else {
                    devices.remove(device);
                }
                (
                    "PATCH",
                    format!("/1.0/instances/{instance}?project={p}"),
                    Some(json!({"devices":devices})),
                )
            }
            IncusOperation::Copy { instance, target } => (
                "POST",
                format!("/1.0/instances?project={p}"),
                Some(
                    json!({"name":target,"source":{"type":"copy","source":instance},"instance_only":false}),
                ),
            ),
            IncusOperation::Move { instance, target } => (
                "POST",
                format!("/1.0/instances/{instance}?project={p}"),
                Some(json!({"name":target,"migration":false})),
            ),
        };
        let r = self.query(method, &path, body).await?;
        if matches!(op, IncusOperation::Exec { .. }) && r["type"] == "async" {
            let operation = r["operation"]
                .as_str()
                .and_then(|s| s.strip_prefix("/1.0/operations/"))
                .ok_or_else(|| anyhow!("invalid Incus operation ID"))?;
            uuid::Uuid::parse_str(operation).map_err(|_| anyhow!("invalid Incus operation ID"))?;
            let done = self
                .query(
                    "GET",
                    &format!("/1.0/operations/{operation}/wait?timeout=10&project={p}"),
                    None,
                )
                .await?;
            let mut result = select(&done["metadata"], &["id", "status", "status_code", "err"]);
            result["exit_code"] = done["metadata"]["metadata"]["return"].clone();
            if let IncusOperation::Exec { instance, .. } = op {
                for (fd, stream, suffix) in [("1", "stdout", "stdout"), ("2", "stderr", "stderr")] {
                    if let Some(descriptor) = done["metadata"]["metadata"]["output"][fd].as_str() {
                        let prefix = format!("/1.0/instances/{instance}/logs/exec_");
                        let log_id = descriptor
                            .strip_prefix(&prefix)
                            .and_then(|s| s.strip_suffix(&format!(".{suffix}")))
                            .ok_or_else(|| anyhow!("Incus exec output descriptor mismatch"))?;
                        uuid::Uuid::parse_str(log_id)
                            .map_err(|_| anyhow!("Incus exec output descriptor mismatch"))?;
                        let expected = format!("{prefix}{log_id}.{suffix}");
                        let data = self
                            .conn
                            .bytes(
                                "GET",
                                &format!("{expected}?project={p}"),
                                None,
                                self.cfg.connection.max_output_bytes / 4,
                            )
                            .await?;
                        result[stream] = Value::String(
                            String::from_utf8(data)
                                .map_err(|_| anyhow!("Incus exec output is not UTF-8"))?,
                        );
                    }
                }
            }
            return Ok(result);
        }
        Ok(select(
            &r,
            &["type", "status", "status_code", "operation", "metadata"],
        ))
    }
}
