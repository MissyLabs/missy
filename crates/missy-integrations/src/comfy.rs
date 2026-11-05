use crate::{config::*, transport::Connection};
use anyhow::{Result, anyhow, bail};
use missy_core::Scope;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashMap, sync::Mutex};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Submit {
    pub workflow: String,
    pub prompt: Option<String>,
    pub conditioning: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct History {
    pub prompt_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Download {
    pub prompt_id: String,
    pub node: String,
    pub kind: String,
    pub output_index: usize,
    pub output_path: String,
}
#[derive(Debug, Serialize)]
pub struct ArtifactMetadata {
    pub node: String,
    pub kind: String,
    pub output_index: usize,
    pub filename: String,
    pub subfolder: String,
    pub storage_type: String,
}
pub(crate) struct Comfy {
    pub conn: Connection,
    pub cfg: ComfyConfig,
    pub jobs: Mutex<HashMap<String, Scope>>,
}
impl Comfy {
    pub fn validate(cfg: &ComfyConfig) -> Result<()> {
        if cfg.max_artifact_bytes > missy_policy::MAX_HTTP_RESPONSE_BYTES {
            bail!("ComfyUI artifact ceiling exceeds policy HTTP 4 MiB hard limit");
        }
        for (name, input) in &cfg.conditioning {
            identifier(name)?;
            if !input.path.is_absolute()
                || input.path.components().any(|c| {
                    matches!(
                        c,
                        std::path::Component::ParentDir | std::path::Component::CurDir
                    )
                })
            {
                bail!("conditioning path must be absolute without dot components");
            }
            let workflow = cfg
                .workflows
                .get(&input.workflow)
                .ok_or_else(|| anyhow!("conditioning workflow missing"))?;
            if workflow.graph[&input.node]["class_type"] != "LoadImage"
                || !workflow.graph[&input.node]["inputs"]["image"].is_string()
            {
                bail!("conditioning binding must target LoadImage.image");
            }
        }
        if cfg.max_tracked_jobs == 0 || cfg.max_tracked_jobs > 10_000 {
            bail!("ComfyUI max_tracked_jobs must be 1..10000");
        }
        for (name, workflow) in &cfg.workflows {
            identifier(name)?;
            let graph = workflow
                .graph
                .as_object()
                .ok_or_else(|| anyhow!("workflow must be an API graph object"))?;
            if graph.is_empty() || graph.len() > 256 || serde_json::to_vec(graph)?.len() > 262_144 {
                bail!("workflow graph exceeds bounds");
            }
            for (node_id, node) in graph {
                identifier(node_id)?;
                let obj = node
                    .as_object()
                    .ok_or_else(|| anyhow!("invalid workflow node"))?;
                if obj
                    .keys()
                    .any(|k| !["class_type", "inputs", "_meta"].contains(&k.as_str()))
                {
                    bail!("unknown workflow node field");
                }
                let class = node["class_type"]
                    .as_str()
                    .ok_or_else(|| anyhow!("workflow class missing"))?;
                if !cfg.allowed_classes.contains(class) {
                    bail!("workflow node class not explicitly allowed");
                }
                let inputs = node["inputs"]
                    .as_object()
                    .ok_or_else(|| anyhow!("workflow node inputs missing"))?;
                for input in inputs.values() {
                    if let Some(link) = input.as_array() {
                        if link.len() != 2
                            || !link[0].as_str().is_some_and(|id| graph.contains_key(id))
                            || link[1].as_u64().is_none_or(|n| n >= 256)
                        {
                            bail!("invalid workflow graph link");
                        }
                    } else if !(input.is_string() || input.is_number() || input.is_boolean()) {
                        bail!("workflow inputs must be scalar values or validated links");
                    }
                }
            }
            if let Some(id) = &workflow.prompt_node
                && (workflow.graph[id]["class_type"] != "CLIPTextEncode"
                    || !workflow.graph[id]["inputs"]["text"].is_string())
            {
                bail!("prompt binding must target CLIPTextEncode.text");
            }
            // Detect cycles, including indirect cycles. Trusted graphs still need structural validation.
            fn visit(
                id: &str,
                graph: &serde_json::Map<String, Value>,
                active: &mut Vec<String>,
                done: &mut Vec<String>,
            ) -> Result<()> {
                if active.iter().any(|s| s == id) {
                    bail!("workflow graph contains a cycle");
                }
                if done.iter().any(|s| s == id) {
                    return Ok(());
                }
                active.push(id.into());
                for input in graph[id]["inputs"].as_object().unwrap().values() {
                    if let Some(link) = input.as_array() {
                        visit(link[0].as_str().unwrap(), graph, active, done)?;
                    }
                }
                active.pop();
                done.push(id.into());
                Ok(())
            }
            let mut done = vec![];
            for id in graph.keys() {
                visit(id, graph, &mut vec![], &mut done)?;
            }
        }
        Ok(())
    }
    pub async fn submit(&self, scope: &Scope, a: Submit) -> Result<Value> {
        let workflow = self
            .cfg
            .workflows
            .get(&a.workflow)
            .ok_or_else(|| anyhow!("ComfyUI workflow not approved"))?;
        let mut graph = workflow.graph.clone();
        if let Some(name) = &a.conditioning {
            let input = self
                .cfg
                .conditioning
                .get(name)
                .ok_or_else(|| anyhow!("conditioning grant denied"))?;
            if input.workflow != a.workflow {
                bail!("conditioning workflow mismatch");
            }
            let filename = self.upload_conditioning(input).await?;
            graph[&input.node]["inputs"]["image"] = Value::String(filename);
        }
        if let Some(prompt) = a.prompt {
            if prompt.is_empty() || prompt.len() > 8192 || prompt.contains('\0') {
                bail!("prompt exceeds bounds");
            }
            let id = workflow
                .prompt_node
                .as_ref()
                .ok_or_else(|| anyhow!("workflow has no approved prompt input"))?;
            graph[id]["inputs"]["text"] = Value::String(prompt);
        }
        // Reserve capacity before sending. Ambiguous network outcomes consume a reservation:
        // never automatically replay an uncertain generation or evict ownership records.
        let reservation = format!("reserved-{}", uuid::Uuid::new_v4());
        {
            let mut jobs = self
                .jobs
                .lock()
                .map_err(|_| anyhow!("ComfyUI job registry unavailable"))?;
            if jobs.len() >= self.cfg.max_tracked_jobs {
                bail!("ComfyUI job registry full");
            }
            jobs.insert(reservation.clone(), scope.clone());
        }
        let r = self
            .conn
            .json(
                "POST",
                "/prompt",
                Some(json!({"prompt":graph,"client_id":uuid::Uuid::new_v4().to_string()})),
                false,
            )
            .await?;
        if r.get("node_errors")
            .is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|m| !m.is_empty()))
        {
            bail!("ComfyUI rejected workflow nodes");
        }
        let id = r["prompt_id"]
            .as_str()
            .ok_or_else(|| anyhow!("ComfyUI prompt ID missing"))?;
        identifier(id)?;
        let mut jobs = self
            .jobs
            .lock()
            .map_err(|_| anyhow!("ComfyUI job registry unavailable"))?;
        if jobs.contains_key(id) {
            bail!("ComfyUI prompt ID collision");
        }
        jobs.remove(&reservation);
        jobs.insert(id.into(), scope.clone());
        Ok(json!({"prompt_id":id,"workflow":a.workflow,"queued":true}))
    }
    pub async fn history(&self, scope: &Scope, a: History) -> Result<Value> {
        identifier(&a.prompt_id)?;
        {
            let jobs = self
                .jobs
                .lock()
                .map_err(|_| anyhow!("ComfyUI job registry unavailable"))?;
            if jobs.get(&a.prompt_id) != Some(scope) {
                bail!("ComfyUI prompt is not owned by this scope");
            }
        }
        let r = self
            .conn
            .json("GET", &format!("/history/{}", a.prompt_id), None, false)
            .await?;
        if !r.is_object() {
            bail!("invalid ComfyUI history envelope");
        }
        let Some(entry) = r.get(&a.prompt_id) else {
            return Ok(json!({"prompt_id":a.prompt_id,"completed":false,"artifacts":[]}));
        };
        if !entry.is_object() || !entry["status"].is_object() || !entry["outputs"].is_object() {
            bail!("invalid ComfyUI history entry");
        }
        let mut artifacts = vec![];
        if let Some(outputs) = entry["outputs"].as_object() {
            for (node, output) in outputs {
                identifier(node)?;
                for kind in ["images", "gifs", "videos", "audio"] {
                    if let Some(items) = output.get(kind) {
                        let items = items
                            .as_array()
                            .ok_or_else(|| anyhow!("invalid ComfyUI output list"))?;
                        for (output_index, item) in items.iter().enumerate() {
                            if artifacts.len() >= 64 {
                                bail!("ComfyUI artifact count exceeds bound");
                            }
                            let filename = item["filename"]
                                .as_str()
                                .ok_or_else(|| anyhow!("invalid artifact filename"))?;
                            let subfolder = item["subfolder"]
                                .as_str()
                                .ok_or_else(|| anyhow!("artifact subfolder missing"))?;
                            let storage_type = item["type"]
                                .as_str()
                                .ok_or_else(|| anyhow!("artifact storage type missing"))?;
                            safe_component(filename)?;
                            if !subfolder.is_empty() {
                                for part in subfolder.split('/') {
                                    safe_component(part)?;
                                }
                            }
                            if !["output", "temp"].contains(&storage_type) {
                                bail!("artifact storage type denied");
                            }
                            artifacts.push(ArtifactMetadata {
                                node: node.clone(),
                                kind: kind.into(),
                                output_index,
                                filename: filename.into(),
                                subfolder: subfolder.into(),
                                storage_type: storage_type.into(),
                            });
                        }
                    }
                }
            }
        }
        Ok(
            json!({"prompt_id":a.prompt_id,"completed":entry["status"]["completed"].as_bool().unwrap_or(false),
            "status":entry["status"]["status_str"],"artifacts":artifacts,
            "downloaded":false}),
        )
    }
}
impl Comfy {
    async fn upload_conditioning(&self, input: &ImageConditioning) -> Result<String> {
        let policy = missy_policy::Policy::new(self.conn.http.config().clone())?;
        let bytes = policy.read_file(&input.path)?;
        if bytes.len() > missy_policy::MAX_HTTP_REQUEST_BYTES - 4096 {
            bail!("conditioning image exceeds policy upload limit");
        }
        let mut reader = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .map_err(|_| anyhow!("conditioning image format invalid"))?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(64 * 1024 * 1024);
        reader.limits(limits);
        let decoded = reader
            .decode()
            .map_err(|_| anyhow!("conditioning image failed decode or limits"))?;
        // Upload a freshly encoded PNG, not opaque source bytes/metadata.
        let mut png = std::io::Cursor::new(Vec::new());
        decoded
            .write_to(&mut png, image::ImageFormat::Png)
            .map_err(|_| anyhow!("conditioning PNG encoding failed"))?;
        let png = png.into_inner();
        if png.len() > missy_policy::MAX_HTTP_REQUEST_BYTES - 4096 {
            bail!("conditioning PNG exceeds policy upload limit");
        }
        let filename = format!("missy-{}.png", uuid::Uuid::new_v4());
        let boundary = format!("missy{}", uuid::Uuid::new_v4().simple());
        let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\nfalse\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").into_bytes();
        body.extend_from_slice(&png);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let mut headers = vec![(
            "Content-Type".into(),
            format!("multipart/form-data; boundary={boundary}"),
        )];
        if let Some(secret) = self.conn.credential() {
            headers.push(("Authorization".into(), format!("Bearer {secret}")));
        }
        let r = self
            .conn
            .http
            .request("POST", &self.conn.url("/upload/image"), headers, Some(body))
            .await
            .map_err(|_| anyhow!("conditioning upload denied or transport failed"))?;
        if !(200..300).contains(&r.status) || r.body.len() > self.cfg.connection.max_output_bytes {
            bail!("conditioning upload rejected");
        }
        let descriptor: Value = serde_json::from_slice(&r.body)
            .map_err(|_| anyhow!("invalid conditioning upload descriptor"))?;
        if descriptor["name"] != filename
            || descriptor["subfolder"] != ""
            || descriptor["type"] != "input"
        {
            bail!("conditioning upload descriptor mismatch");
        }
        Ok(filename)
    }
    pub(crate) fn validate_download(&self, scope: &Scope, a: &Download) -> Result<()> {
        identifier(&a.prompt_id)?;
        identifier(&a.node)?;
        if self.cfg.max_artifact_bytes == 0 {
            bail!("ComfyUI artifact downloads disabled");
        }
        if !["images", "gifs", "videos"].contains(&a.kind.as_str()) || a.output_index >= 64 {
            bail!("ComfyUI artifact selector denied");
        }
        if self
            .jobs
            .lock()
            .map_err(|_| anyhow!("ComfyUI registry unavailable"))?
            .get(&a.prompt_id)
            != Some(scope)
        {
            bail!("ComfyUI prompt is not owned by this scope");
        }
        let path = std::path::Path::new(&a.output_path);
        if !path.is_absolute()
            || a.output_path.split('/').any(|p| p == "." || p == "..")
            || a.output_path.contains('\0')
        {
            bail!("ComfyUI output path must be absolute without dot components");
        }
        if !self
            .conn
            .http
            .config()
            .write_roots
            .iter()
            .any(|r| path.starts_with(r))
        {
            bail!("ComfyUI output path outside policy write roots");
        }
        Ok(())
    }
    pub(crate) async fn download(
        &self,
        ctx: &missy_core::ToolContext,
        a: Download,
        validator: Option<&dyn crate::ArtifactValidator>,
    ) -> Result<Value> {
        let scope = &ctx.scope;
        self.validate_download(scope, &a)?;
        let history = self
            .history(
                scope,
                History {
                    prompt_id: a.prompt_id.clone(),
                },
            )
            .await?;
        if history["completed"] != true {
            bail!("ComfyUI generation not completed");
        }
        let artifact = history["artifacts"]
            .as_array()
            .and_then(|rows| {
                rows.iter().find(|v| {
                    v["node"] == a.node
                        && v["kind"] == a.kind
                        && v["output_index"].as_u64() == Some(a.output_index as u64)
                })
            })
            .ok_or_else(|| anyhow!("ComfyUI exact artifact descriptor missing"))?;
        let mut url = url::Url::parse(&self.conn.url("/view"))?;
        for (key, field) in [
            ("filename", "filename"),
            ("subfolder", "subfolder"),
            ("type", "storage_type"),
        ] {
            url.query_pairs_mut().append_pair(
                key,
                artifact[field]
                    .as_str()
                    .ok_or_else(|| anyhow!("invalid artifact descriptor"))?,
            );
        }
        let path = format!("/view?{}", url.query().unwrap());
        let bytes = self
            .conn
            .bytes("GET", &path, None, self.cfg.max_artifact_bytes)
            .await?;
        // Decode images under explicit allocation and dimension bounds, not extension sniffing.
        // A valid container header is not a decoded video.
        let media = if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
            validate_mp4(&bytes)?;
            let validator = validator
                .ok_or_else(|| anyhow!("external MP4 requires configured native media decoder"))?;
            let metadata = validator.validate_mp4(ctx, &bytes).await?;
            json!({"format":"mp4","validation":"full_bounded_decode","decoded":true,"metadata":metadata})
        } else {
            let format = image::guess_format(&bytes)
                .map_err(|_| anyhow!("artifact is not a supported media file"))?;
            if ![
                image::ImageFormat::Png,
                image::ImageFormat::Jpeg,
                image::ImageFormat::WebP,
                image::ImageFormat::Gif,
            ]
            .contains(&format)
            {
                bail!("artifact media format denied");
            }
            let mut reader = image::ImageReader::with_format(std::io::Cursor::new(&bytes), format);
            let mut limits = image::Limits::default();
            limits.max_image_width = Some(8192);
            limits.max_image_height = Some(8192);
            limits.max_alloc = Some(64 * 1024 * 1024);
            reader.limits(limits);
            let decoded = reader
                .decode()
                .map_err(|_| anyhow!("artifact image decode failed or exceeded limits"))?;
            json!({"format":format!("{format:?}"),"width":decoded.width(),"height":decoded.height(),"validation":"decoded_image","decoded":true})
        };
        let policy = missy_policy::Policy::new(self.conn.http.config().clone())?;
        policy.write_file(&a.output_path, &bytes)?;
        use sha2::{Digest, Sha256};
        Ok(
            json!({"prompt_id":a.prompt_id,"artifact":artifact,"output_path":a.output_path,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes)),"media":media,"downloaded":true}),
        )
    }
}
fn validate_mp4(data: &[u8]) -> Result<()> {
    let mut offset = 0usize;
    let mut boxes = 0;
    let mut ftyp = false;
    let mut moov = false;
    let mut mdat = false;
    while offset < data.len() {
        boxes += 1;
        if boxes > 4096 || data.len() - offset < 8 {
            bail!("invalid MP4 box header");
        }
        let mut size = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        let kind = &data[offset + 4..offset + 8];
        let mut header = 8;
        if size == 1 {
            if data.len() - offset < 16 {
                bail!("invalid MP4 extended header");
            }
            size = usize::try_from(u64::from_be_bytes(
                data[offset + 8..offset + 16].try_into().unwrap(),
            ))
            .map_err(|_| anyhow!("MP4 box overflow"))?;
            header = 16;
        } else if size == 0 {
            size = data.len() - offset;
        }
        if size < header || size > data.len() - offset {
            bail!("invalid MP4 box size");
        }
        match kind {
            b"ftyp" => {
                if size < 16 {
                    bail!("invalid MP4 file type");
                }
                ftyp = true;
            }
            b"moov" => moov = size > header,
            b"mdat" => mdat = size > header,
            _ => {}
        }
        offset += size;
    }
    if !ftyp || !moov || !mdat {
        bail!("MP4 missing required boxes");
    }
    Ok(())
}
fn safe_component(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 256
        || s == "."
        || s == ".."
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._- ".contains(&b))
    {
        bail!("unsafe ComfyUI artifact path component");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::validate_mp4;
    #[test]
    fn mp4_structure_bounds_are_not_codec_validation() {
        let mut data = vec![];
        for (kind, content) in [
            (b"ftyp", b"isom0000".as_slice()),
            (b"moov", b"track".as_slice()),
            (b"mdat", b"frames".as_slice()),
        ] {
            data.extend_from_slice(&((content.len() + 8) as u32).to_be_bytes());
            data.extend_from_slice(kind);
            data.extend_from_slice(content);
        }
        assert!(validate_mp4(&data).is_ok());
        assert!(validate_mp4(&data[..data.len() - 1]).is_err());
        assert!(validate_mp4(&data[..16]).is_err());
        data[0..4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(validate_mp4(&data).is_err());
    }
}
