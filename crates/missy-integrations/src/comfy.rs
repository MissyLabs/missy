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
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct History {
    pub prompt_id: String,
}
#[derive(Debug, Serialize)]
pub struct ArtifactMetadata {
    pub node: String,
    pub kind: String,
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
                        for item in items {
                            if artifacts.len() >= 64 {
                                bail!("ComfyUI artifact count exceeds bound");
                            }
                            let filename = item["filename"]
                                .as_str()
                                .ok_or_else(|| anyhow!("invalid artifact filename"))?;
                            let subfolder = item["subfolder"].as_str().unwrap_or("");
                            let storage_type = item["type"].as_str().unwrap_or("output");
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
