//! Model-facing data-only proposal tool. No approval or activation authority.
use crate::{PluginManifest, PluginTools};
use anyhow::{Result, ensure};
use missy_core::{ToolCall, ToolContext, ToolDefinition};
use serde::Deserialize;
use serde_json::{Value, json};

pub(crate) const NAME: &str = "self_create_tool";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    manifest: PluginManifest,
}

pub(crate) fn definition() -> ToolDefinition {
    let scalar = json!({
        "type":"object", "additionalProperties":false,
        "properties":{
            "type":{"type":"string","enum":["string","integer","number"]},
            "description":{"type":"string","maxLength":4096},
            "enum":{"type":"array","maxItems":64,"items":{"anyOf":[{"type":"string"},{"type":"number"}]}},
            "minLength":{"type":"integer","minimum":0,"maximum":8192},
            "maxLength":{"type":"integer","minimum":0,"maximum":8192},
            "minimum":{"type":"number"},"maximum":{"type":"number"}
        }, "required":["type"]
    });
    ToolDefinition {
        name: NAME.into(),
        description: "Propose a data-only native plugin manifest for exact-hash human review. Creates a pending proposal only: never approves, enables, compiles, installs code, executes a program, or changes the running catalog. Only operator-provided SHA256-pinned native ELF is supported. Execution after separate operator activation still requires exact argv policy and per-call human approval.".into(),
        parameters: json!({
            "type":"object","additionalProperties":false,"required":["manifest"],
            "properties":{"manifest":{
                "type":"object","additionalProperties":false,
                "required":["id","definition","command","capabilities"],
                "properties":{
                    "id":{"type":"string","minLength":1,"maxLength":48},
                    "definition":{
                        "type":"object","additionalProperties":false,
                        "required":["name","description","parameters","side_effect"],
                        "properties":{
                            "name":{"type":"string","maxLength":56},
                            "description":{"type":"string","maxLength":4096},
                            "side_effect":{"type":"boolean","enum":[true]},
                            "parameters":{
                                "type":"object","additionalProperties":false,
                                "required":["type","properties","required","additionalProperties"],
                                "properties":{
                                    "type":{"type":"string","enum":["object"]},
                                    "additionalProperties":{"type":"boolean","enum":[false]},
                                    "properties":{"type":"object","maxProperties":32,"additionalProperties":scalar},
                                    "required":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":48}}
                                }
                            }
                        }
                    },
                    "command":{
                        "type":"object","additionalProperties":false,
                        "required":["executable","sha256","args"],
                        "properties":{
                            "executable":{"type":"string","maxLength":4096},
                            "sha256":{"type":"string","minLength":64,"maxLength":64},
                            "args":{"type":"array","maxItems":64,"items":{"type":"string","maxLength":8192}}
                        }
                    },
                    "capabilities":{
                        "type":"object","additionalProperties":false,"required":["read_roots"],
                        "properties":{"read_roots":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":4096}}}
                    }
                }
            }}
        }),
        side_effect: true,
    }
}

impl PluginTools {
    pub(crate) fn proposal_manifest(
        &self,
        ctx: &ToolContext,
        call: &ToolCall,
    ) -> Result<PluginManifest> {
        ensure!(
            self.config.enabled && ctx.scope == self.config.scope,
            "plugin proposal scope denied"
        );
        ctx.scope.validate()?;
        ensure!(
            !ctx.cancellation.is_cancelled(),
            "plugin proposal cancelled before dispatch"
        );
        ensure!(call.name == NAME, "unsupported proposal tool");
        ensure!(
            !call.id.is_empty() && call.id.len() <= 256,
            "bounded original tool call id required"
        );
        ensure!(
            serde_json::to_vec(&call.arguments)?.len() <= 65_536,
            "proposal arguments exceed 64KiB"
        );
        // ToolDefinition is shared protocol without deny_unknown_fields, so
        // enforce its closed shape before typed deserialization.
        let definition = call
            .arguments
            .pointer("/manifest/definition")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("manifest definition required"))?;
        ensure!(definition.keys().all(|k|["name","description","parameters","side_effect"].contains(&k.as_str())), "unknown tool definition field");
        let args: Arguments = serde_json::from_value(call.arguments.clone())?;
        args.manifest.validate()?;
        for root in &args.manifest.capabilities.read_roots {
            ensure!(
                self.policy
                    .config()
                    .read_roots
                    .iter()
                    .any(|r| root.starts_with(r)),
                "proposed read capability exceeds policy"
            );
        }
        Ok(args.manifest)
    }
}
