use crate::{Plan, PlanNode, Scheduler};
use anyhow::{Context, Result, ensure};
use missy_core::{Scope, ToolCall};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

struct CancelPlanNode {
    runtime: std::sync::Arc<missy_runtime::Runtime>,
    scope: Scope,
    request_id: String,
    armed: bool,
}
impl Drop for CancelPlanNode {
    fn drop(&mut self) {
        if self.armed
            && let Ok(turns) = self.runtime.store().list_turns(&self.scope, 1000)
            && let Some(turn) = turns.iter().find(|t| t.request_id == self.request_id)
        {
            let _ = self.runtime.cancel(&self.scope, &turn.id);
        }
    }
}

pub(crate) fn fingerprint(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
/// Derive the same child scope used by the model adapter for operator inspect,
/// original-turn resume and reconciliation. Never accepts a model scope claim.
pub fn model_plan_scope(parent: &Scope, operation_id: &str) -> Result<Scope> {
    parent.validate()?;
    ensure!(
        !operation_id.is_empty() && operation_id.len() <= 128,
        "bounded operation ID required"
    );
    let mut scope = parent.clone();
    scope.session = format!("plan:{}", &fingerprint(&(parent, operation_id))?[..64]);
    Ok(scope)
}
/// References are {"$ref":{"node":"source","pointer":"/data/0"}}.
/// Values retain their type; strings are never templates or programs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedPlan {
    pub nodes: Vec<TypedNode>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedNode {
    pub id: String,
    pub tool: String,
    pub arguments: Value,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub preconditions: Vec<Condition>,
    #[serde(default)]
    pub postconditions: Vec<Condition>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    /// None means this node; valid only for postconditions.
    #[serde(default)]
    pub node: Option<String>,
    #[serde(default)]
    pub pointer: String,
    pub test: ConditionTest,
    #[serde(default)]
    pub negate: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ConditionTest {
    Exists,
    Equals(Value),
    Contains(String),
    NotEmpty,
    IsNumber,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    node: String,
    #[serde(default)]
    pointer: String,
}
fn reference(v: &Value) -> Result<Option<Reference>> {
    if let Some(obj) = v.as_object().filter(|o| o.contains_key("$ref")) {
        ensure!(obj.len() == 1, "$ref cannot have sibling fields");
        let r: Reference = serde_json::from_value(obj["$ref"].clone())?;
        ensure!(
            r.pointer.is_empty() || r.pointer.starts_with('/'),
            "JSON pointer required"
        );
        return Ok(Some(r));
    }
    Ok(None)
}
fn references(v: &Value, deps: &mut BTreeSet<String>, depth: usize) -> Result<()> {
    ensure!(depth <= 32, "reference nesting limit exceeded");
    if let Some(r) = reference(v)? {
        deps.insert(r.node);
    } else {
        match v {
            Value::Array(a) => {
                for v in a {
                    references(v, deps, depth + 1)?;
                }
            }
            Value::Object(o) => {
                for v in o.values() {
                    references(v, deps, depth + 1)?;
                }
            }
            _ => (),
        }
    }
    Ok(())
}
fn resolve(v: &Value, outputs: &BTreeMap<String, Value>) -> Result<Value> {
    if let Some(r) = reference(v)? {
        return outputs
            .get(&r.node)
            .and_then(|o| o.pointer(&r.pointer))
            .cloned()
            .context("reference output or JSON pointer unavailable");
    }
    Ok(match v {
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|v| resolve(v, outputs))
                .collect::<Result<_>>()?,
        ),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| Ok((k.clone(), resolve(v, outputs)?)))
                .collect::<Result<_>>()?,
        ),
        _ => v.clone(),
    })
}
fn check(c: &Condition, own: &str, outputs: &BTreeMap<String, Value>) -> bool {
    let value = outputs
        .get(c.node.as_deref().unwrap_or(own))
        .and_then(|v| v.pointer(&c.pointer));
    let passed = match &c.test {
        ConditionTest::Exists => value.is_some(),
        ConditionTest::Equals(expected) => value == Some(expected),
        ConditionTest::Contains(needle) => value
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains(needle)),
        ConditionTest::NotEmpty => value.is_some_and(|v| match v {
            Value::Null => false,
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
            _ => true,
        }),
        ConditionTest::IsNumber => value.is_some_and(Value::is_number),
    };
    passed != c.negate
}
impl TypedPlan {
    pub fn parse_json(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() <= 1_048_576, "plan exceeds byte bound");
        Ok(serde_json::from_slice(bytes)?)
    }
    pub fn compile(&self, scope: &Scope) -> Result<Vec<String>> {
        let mut nodes = Vec::new();
        for n in &self.nodes {
            ensure!(
                n.preconditions.len() <= 32 && n.postconditions.len() <= 32,
                "condition bound exceeded"
            );
            ensure!(n.depends_on.len() <= 128, "dependency bound exceeded");
            let mut deps: BTreeSet<String> = n.depends_on.iter().cloned().collect();
            references(&n.arguments, &mut deps, 0)?;
            for (pre, conditions) in [(true, &n.preconditions), (false, &n.postconditions)] {
                for c in conditions {
                    ensure!(
                        c.pointer.is_empty() || c.pointer.starts_with('/'),
                        "JSON pointer required"
                    );
                    match &c.node {
                        Some(id) if id != &n.id => {
                            deps.insert(id.clone());
                        }
                        _ => ensure!(!pre, "preconditions must target an upstream node"),
                    }
                }
            }
            nodes.push(PlanNode {
                id: n.id.clone(),
                dependencies: deps.into_iter().collect(),
                tool_calls: vec![ToolCall {
                    id: n.id.clone(),
                    name: n.tool.clone(),
                    arguments: n.arguments.clone(),
                }],
            });
        }
        ensure!(
            serde_json::to_vec(self)?.len() <= 1_048_576,
            "plan exceeds byte bound"
        );
        Ok(Plan {
            scope: scope.clone(),
            nodes,
        }
        .compile(scope)?
        .nodes
        .into_iter()
        .map(|n| n.id)
        .collect())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeReceipt {
    pub status: String,
    pub operation_id: String,
    pub turn_id: Option<String>,
    pub output: Option<Value>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanReceipt {
    pub request_id: String,
    pub status: String,
    pub untrusted: bool,
    pub nodes: BTreeMap<String, NodeReceipt>,
}
impl Scheduler {
    /// Immutable scoped request, fail closed on coordinator loss. Runtime is
    /// the only effect dispatcher, including approval gates and uncertainty.
    pub async fn execute_plan(
        &self,
        scope: &Scope,
        request_id: &str,
        plan: TypedPlan,
    ) -> Result<PlanReceipt> {
        self.execute_plan_inner(scope, request_id, plan, false)
            .await
    }
    /// Explicit operator reconciliation after resuming an interrupted exact-tool
    /// Runtime turn under its original ID. Never resumes or replays the turn.
    pub async fn reconcile_plan(
        &self,
        scope: &Scope,
        request_id: &str,
        plan: TypedPlan,
    ) -> Result<PlanReceipt> {
        self.execute_plan_inner(scope, request_id, plan, true).await
    }
    async fn execute_plan_inner(
        &self,
        scope: &Scope,
        request_id: &str,
        plan: TypedPlan,
        reconcile: bool,
    ) -> Result<PlanReceipt> {
        ensure!(
            !request_id.is_empty() && request_id.len() <= 256,
            "bounded request ID required"
        );
        let order = plan.compile(scope)?;
        let key = serde_json::to_string(scope)?;
        let binding = fingerprint(&(&plan, self.runtime.configuration_fingerprint()?))?;
        let root = fingerprint(&(scope, request_id))?;
        let mut receipt = PlanReceipt {
            request_id: request_id.into(),
            status: "running".into(),
            untrusted: true,
            nodes: BTreeMap::new(),
        };
        {
            let mut db = self
                .db
                .lock()
                .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let old: Option<(String, String)> = tx
                .query_row(
                    "SELECT binding,receipt FROM automation_plans WHERE scope=?1 AND request_id=?2",
                    params![key, request_id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if let Some((b, r)) = old {
                ensure!(
                    b == binding,
                    "plan request ID reused with changed plan or runtime"
                );
                receipt = serde_json::from_str(&r)?;
                if !reconcile || receipt.status == "completed" {
                    return Ok(receipt);
                }
                ensure!(
                    ["interrupted", "admitted"].contains(&receipt.status.as_str()),
                    "only known interrupted/admitted plans can be reconciled; active/failed/uncertain plans require investigation"
                );
                for node in receipt
                    .nodes
                    .values()
                    .filter(|n| !["completed", "blocked", "planned"].contains(&n.status.as_str()))
                {
                    let turn = node
                        .turn_id
                        .as_ref()
                        .context("interrupted node missing turn identity")?;
                    ensure!(
                        serde_json::to_value(self.runtime.status(scope, turn)?.status)?.as_str()
                            == Some("completed"),
                        "explicitly resume original Runtime turn before reconciling plan"
                    );
                }
                receipt.status = "running".into();
                tx.execute(
                    "UPDATE automation_plans SET receipt=?1 WHERE scope=?2 AND request_id=?3",
                    params![serde_json::to_string(&receipt)?, key, request_id],
                )?;
            } else {
                ensure!(!reconcile, "plan not found in scope");
                for id in &order {
                    receipt.nodes.insert(
                        id.clone(),
                        NodeReceipt {
                            status: "planned".into(),
                            operation_id: format!("plan:{}", fingerprint(&(&root, id))?),
                            turn_id: None,
                            output: None,
                            error: None,
                        },
                    );
                }
                tx.execute("INSERT INTO automation_plans(scope,request_id,binding,receipt) VALUES(?1,?2,?3,?4)",params![key,request_id,binding,serde_json::to_string(&receipt)?])?;
            }
            tx.commit()?;
        }
        let mut outputs = BTreeMap::new();
        for id in order {
            if receipt.nodes[&id].status == "completed" {
                outputs.insert(
                    id.clone(),
                    receipt.nodes[&id]
                        .output
                        .clone()
                        .context("completed node missing output")?,
                );
                continue;
            }
            let node = plan
                .nodes
                .iter()
                .find(|n| n.id == id)
                .expect("validated node");
            let args = resolve(&node.arguments, &outputs);
            if !node.preconditions.iter().all(|c| check(c, &id, &outputs)) || args.is_err() {
                let r = receipt.nodes.get_mut(&id).unwrap();
                r.status = "blocked".into();
                r.error = Some("precondition_or_reference_failed".into());
                receipt.status = "failed".into();
                break;
            }
            let r = receipt.nodes.get_mut(&id).unwrap();
            r.status = "running".into();
            let operation = r.operation_id.clone();
            self.save_plan(&key, &receipt)?;
            let mut active = CancelPlanNode {
                runtime: self.runtime.clone(),
                scope: scope.clone(),
                request_id: operation.clone(),
                armed: true,
            };
            let result = self
                .runtime
                .run_tools(
                    scope.clone(),
                    vec![ToolCall {
                        id: id.clone(),
                        name: node.tool.clone(),
                        arguments: args?,
                    }],
                    operation,
                )
                .await;
            active.armed = false;
            let r = receipt.nodes.get_mut(&id).unwrap();
            match result {
                Ok(result) => {
                    r.turn_id = Some(result.turn_id);
                    r.status = result.status.clone();
                    if result.status == "completed" {
                        let envelope: Value = serde_json::from_str(&result.content)
                            .context("runtime exact-tool result must be JSON")?;
                        // Runtime journals bounded text envelopes. Truncation
                        // cannot satisfy typed refs or postconditions safely.
                        if envelope.get("truncated").and_then(Value::as_bool) == Some(true) {
                            r.status = "failed".into();
                            r.error = Some("tool_output_truncated".into());
                            receipt.status = "failed".into();
                            break;
                        }
                        let output = if envelope.get("untrusted") == Some(&Value::Bool(true)) {
                            envelope
                                .get("data")
                                .and_then(Value::as_str)
                                .map(serde_json::from_str::<Value>)
                                .transpose()?
                                .unwrap_or(envelope)
                        } else {
                            envelope
                        };
                        r.output = Some(output.clone());
                        outputs.insert(id.clone(), output);
                        if !node.postconditions.iter().all(|c| check(c, &id, &outputs)) {
                            r.status = "failed".into();
                            r.error = Some("postcondition_failed".into());
                        }
                    }
                }
                Err(_) => {
                    r.status = "outcome_unknown".into();
                    r.error = Some("runtime_failed_reconciliation_required".into());
                }
            }
            if r.status != "completed" {
                receipt.status = r.status.clone();
                break;
            }
            self.save_plan(&key, &receipt)?;
        }
        if receipt.nodes.values().all(|n| n.status == "completed") {
            receipt.status = "completed".into();
        }
        for n in receipt.nodes.values_mut().filter(|n| n.status == "planned") {
            n.status = "blocked".into();
        }
        self.save_plan(&key, &receipt)?;
        Ok(receipt)
    }
    fn save_plan(&self, key: &str, receipt: &PlanReceipt) -> Result<()> {
        self.db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?
            .execute(
                "UPDATE automation_plans SET receipt=?1 WHERE scope=?2 AND request_id=?3",
                params![serde_json::to_string(receipt)?, key, receipt.request_id],
            )?;
        Ok(())
    }
    pub fn plan_receipt(&self, scope: &Scope, request_id: &str) -> Result<Option<PlanReceipt>> {
        scope.validate()?;
        let raw: Option<String> = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?
            .query_row(
                "SELECT receipt FROM automation_plans WHERE scope=?1 AND request_id=?2",
                params![serde_json::to_string(scope)?, request_id],
                |r| r.get(0),
            )
            .optional()?;
        raw.map(|s| Ok(serde_json::from_str(&s)?)).transpose()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn scope() -> Scope {
        Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        }
    }
    #[test]
    fn references_are_typed_nested_and_fail_closed() {
        let o = BTreeMap::from([("a".into(), json!({"x":[42]}))]);
        assert_eq!(
            resolve(&json!({"v":[{"$ref":{"node":"a","pointer":"/x/0"}}]}), &o).unwrap(),
            json!({"v":[42]})
        );
        assert!(resolve(&json!({"$ref":{"node":"a","pointer":"/missing"}}), &o).is_err());
        assert!(resolve(&json!({"$ref":{"node":"a"},"extra":1}), &o).is_err());
        assert_eq!(
            resolve(&json!("${a.output}"), &o).unwrap(),
            json!("${a.output}")
        );
    }
    #[test]
    fn refs_add_edges_and_cycles_and_own_preconditions_fail() {
        let mut p:TypedPlan=serde_json::from_value(json!({"nodes":[{"id":"b","tool":"test","arguments":{"n":{"$ref":{"node":"a"}}}},{"id":"a","tool":"test","arguments":{}}]})).unwrap();
        assert_eq!(p.compile(&scope()).unwrap(), vec!["a", "b"]);
        p.nodes[1].depends_on.push("b".into());
        assert!(p.compile(&scope()).is_err());
        p.nodes[1].depends_on.clear();
        p.nodes[1].preconditions.push(Condition {
            node: None,
            pointer: "".into(),
            test: ConditionTest::Exists,
            negate: false,
        });
        assert!(p.compile(&scope()).is_err());
    }
    #[test]
    fn conditions_do_not_coerce_or_equate_missing_with_null() {
        let o = BTreeMap::from([("a".into(), json!({"x":42}))]);
        let mut c = Condition {
            node: Some("a".into()),
            pointer: "/x".into(),
            test: ConditionTest::IsNumber,
            negate: false,
        };
        assert!(check(&c, "b", &o));
        c.test = ConditionTest::Equals(json!("42"));
        assert!(!check(&c, "b", &o));
        c.pointer = "/missing".into();
        c.test = ConditionTest::Equals(Value::Null);
        assert!(!check(&c, "b", &o));
    }
}
