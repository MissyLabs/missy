use crate::{Scheduler, TypedPlan};
use anyhow::{Result, ensure};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
/// Install before outer policy/approval guards. The scheduler Runtime must
/// omit execute_plan to prohibit recursive execution and session deadlock.
pub struct AutomationTools {
    inner: Arc<dyn ToolExecutor>,
    scheduler: Scheduler,
}
impl AutomationTools {
    pub fn new(inner: Arc<dyn ToolExecutor>, scheduler: Scheduler) -> Result<Self> {
        ensure!(
            !inner
                .definitions()
                .iter()
                .any(|d| AUTOMATION_TOOL_NAMES.contains(&d.name.as_str())),
            "automation tool name collision"
        );
        ensure!(
            !scheduler
                .runtime
                .tool_definitions()
                .iter()
                .any(|d| ["execute_plan", "delegate_task"].contains(&d.name.as_str())),
            "plan runtime cannot expose recursive execution/delegation"
        );
        Ok(Self { inner, scheduler })
    }
}
pub const AUTOMATION_TOOL_NAMES: &[&str] = &[
    "schedule_task",
    "list_schedules",
    "update_schedule",
    "delete_schedule",
    "execute_plan",
];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Add {
    prompt: String,
    spec: missy_schedule_text::ScheduleSpec,
    #[serde(default)]
    after: Option<DateTime<Utc>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Update {
    id: String,
    paused: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Id {
    id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Execute {
    operation_id: String,
    plan: TypedPlan,
}
fn plan_schema() -> Value {
    let condition = json!({"type":"object","properties":{"node":{"type":"string"},"pointer":{"type":"string"},"negate":{"type":"boolean"},"test":{"type":"object","properties":{"kind":{"enum":["exists","equals","contains","not_empty","is_number"]},"value":{}},"required":["kind"],"additionalProperties":false}},"required":["test"],"additionalProperties":false});
    json!({"type":"object","properties":{"nodes":{"type":"array","minItems":1,"maxItems":128,"items":{"type":"object","properties":{
        "id":{"type":"string","minLength":1,"maxLength":128},"tool":{"type":"string","minLength":1,"maxLength":128},
        "arguments":{"type":"object","description":"Literal JSON tool arguments. Typed reference at any value: {\"$ref\":{\"node\":\"upstream_id\",\"pointer\":\"/field\"}}; no string interpolation."},
        "depends_on":{"type":"array","maxItems":128,"items":{"type":"string"}},"preconditions":{"type":"array","maxItems":32,"items":condition},"postconditions":{"type":"array","maxItems":32,"items":condition}
    },"required":["id","tool","arguments"],"additionalProperties":false}}},"required":["nodes"],"additionalProperties":false})
}
fn schedule_schema() -> Value {
    json!({"type":"object","properties":{"timezone":{"type":"string","description":"IANA timezone, e.g. America/New_York or UTC"},"schedule":{"oneOf":[
        {"type":"object","properties":{"kind":{"const":"once"},"at":{"type":"string","format":"date-time"}},"required":["kind","at"],"additionalProperties":false},
        {"type":"object","properties":{"kind":{"const":"interval"},"seconds":{"type":"integer","minimum":1,"maximum":31622400},"anchor":{"type":"string","format":"date-time"}},"required":["kind","seconds","anchor"],"additionalProperties":false},
        {"type":"object","properties":{"kind":{"const":"cron"},"expression":{"type":"string","maxLength":256}},"required":["kind","expression"],"additionalProperties":false},
        {"type":"object","properties":{"kind":{"const":"daily"},"hour":{"type":"integer","minimum":0,"maximum":23},"minute":{"type":"integer","minimum":0,"maximum":59}},"required":["kind","hour","minute"],"additionalProperties":false},
        {"type":"object","properties":{"kind":{"const":"weekly"},"weekday":{"type":"integer","minimum":0,"maximum":6},"hour":{"type":"integer","minimum":0,"maximum":23},"minute":{"type":"integer","minimum":0,"maximum":59}},"required":["kind","weekday","hour","minute"],"additionalProperties":false}
    ]}},"required":["timezone","schedule"],"additionalProperties":false})
}
#[async_trait]
impl ToolExecutor for AutomationTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut d = self.inner.definitions();
        for (name, description, properties, required) in [
            (
                "schedule_task",
                "Schedule a scoped bounded prompt with a typed time specification. No extra tool permissions.",
                json!({"prompt":{"type":"string","maxLength":65536},"spec":schedule_schema(),"after":{"type":"string","format":"date-time"}}),
                vec!["prompt", "spec"],
            ),
            (
                "list_schedules",
                "List this authenticated scope's schedules.",
                json!({}),
                vec![],
            ),
            (
                "update_schedule",
                "Pause or resume this scope's schedule.",
                json!({"id":{"type":"string"},"paused":{"type":"boolean"}}),
                vec!["id", "paused"],
            ),
            (
                "delete_schedule",
                "Remove this scope's schedule, preserving audit records.",
                json!({"id":{"type":"string"}}),
                vec!["id"],
            ),
            (
                "execute_plan",
                "Execute a typed tool DAG using JSON $ref objects and typed conditions. Effects retain policy/approval checks. Reuse operation_id for retries; never replace IDs after uncertainty.",
                json!({"operation_id":{"type":"string","minLength":1,"maxLength":128},"plan":plan_schema()}),
                vec!["operation_id", "plan"],
            ),
        ] {
            d.push(ToolDefinition{name:name.into(),description:description.into(),parameters:json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),side_effect:name!="list_schedules"});
        }
        d
    }
    fn configuration_fingerprint(&self) -> String {
        format!(
            "automation-tools-v1:{}:{}",
            self.inner.configuration_fingerprint(),
            self.scheduler
                .runtime
                .configuration_fingerprint()
                .unwrap_or_else(|_| "invalid".into())
        )
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        if AUTOMATION_TOOL_NAMES.contains(&call.name.as_str()) {
            ctx.scope.validate()?;
            Ok(())
        } else {
            self.inner.preflight(ctx, call).await
        }
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        ctx.scope.validate()?;
        match call.name.as_str() {
            "schedule_task" => {
                let a: Add = serde_json::from_value(call.arguments.clone())?;
                Ok(serde_json::to_value(self.scheduler.add_spec(
                    &ctx.scope,
                    a.prompt,
                    a.spec,
                    a.after.unwrap_or_else(Utc::now),
                )?)?)
            }
            "list_schedules" => {
                ensure!(call.arguments == json!({}), "no list arguments accepted");
                Ok(serde_json::to_value(self.scheduler.list(&ctx.scope)?)?)
            }
            "update_schedule" => {
                let a: Update = serde_json::from_value(call.arguments.clone())?;
                self.scheduler.pause(&ctx.scope, &a.id, a.paused)?;
                Ok(json!({"updated":true}))
            }
            "delete_schedule" => {
                let a: Id = serde_json::from_value(call.arguments.clone())?;
                self.scheduler.remove(&ctx.scope, &a.id)?;
                Ok(json!({"removed":true}))
            }
            "execute_plan" => {
                let a: Execute = serde_json::from_value(call.arguments.clone())?;
                ensure!(
                    !a.operation_id.is_empty() && a.operation_id.len() <= 128,
                    "bounded operation ID required"
                );
                // Child session avoids trying to re-enter the active parent lease.
                let scope = crate::model_plan_scope(&ctx.scope, &a.operation_id)?;
                Ok(serde_json::to_value(
                    self.scheduler
                        .execute_plan(&scope, &a.operation_id, a.plan)
                        .await?,
                )?)
            }
            _ => self.inner.execute(ctx, call).await,
        }
    }
}
