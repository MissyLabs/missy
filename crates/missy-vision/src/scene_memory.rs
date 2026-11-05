//! Scoped durable caller assertions. No implicit promotion of model output to
//! authoritative scene state. Observation records are append-only in this API.
use super::*;
use anyhow::ensure;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    action: String,
    #[serde(default)]
    task_id: String,
    #[serde(default = "general")]
    task_type: String,
    #[serde(default)]
    observation: String,
    #[serde(default)]
    state_updates: String,
}
fn general() -> String {
    "general".into()
}
pub(super) fn is_session(c: &ToolCall) -> bool {
    c.name == "vision_scene" && c.arguments.get("action").is_some()
}
pub(super) fn validate(c: &ToolCall) -> Result<Request> {
    let r: Request = serde_json::from_value(c.arguments.clone())?;
    ensure!(
        matches!(
            r.action.as_str(),
            "create" | "add_observation" | "update_state" | "summarize" | "close"
        ),
        "unknown scene action"
    );
    ensure!(
        r.task_id.len() <= 128
            && r.task_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
        "invalid task id"
    );
    ensure!(
        matches!(
            r.task_type.as_str(),
            "general" | "puzzle" | "painting" | "inspection"
        ),
        "invalid task type"
    );
    ensure!(
        r.observation.len() <= 8192 && r.state_updates.len() <= 16384,
        "scene input too large"
    );
    if r.action == "add_observation" {
        ensure!(!r.observation.trim().is_empty(), "observation required");
    }
    if r.action == "update_state" {
        let v: Value = serde_json::from_str(&r.state_updates)?;
        ensure!(
            v.is_object() && v.as_object().unwrap().len() <= 64,
            "state updates must be bounded object"
        );
    }
    Ok(r)
}
pub(super) fn schema() -> Value {
    json!({"type":"object","additionalProperties":false,"properties":{"action":{"enum":["create","add_observation","update_state","summarize","close"]},"task_id":{"type":"string","maxLength":128},"task_type":{"enum":["general","puzzle","painting","inspection"]},"observation":{"type":"string","maxLength":8192},"state_updates":{"type":"string","maxLength":16384},"imagepath":{"type":"string"},"max_objects":{"type":"integer","minimum":1,"maximum":100}},"oneOf":[{"required":["action"],"not":{"anyOf":[{"required":["imagepath"]},{"required":["max_objects"]}]}},{"required":["imagepath","max_objects"],"not":{"required":["action"]}}]})
}
pub(super) fn execute(c: &VisionConfig, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
    let r = validate(call)?;
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "scene operation cancelled"
    );
    let scope = serde_json::to_string(&ctx.scope)?;
    let mut db = ledger_connection(c)?;
    db.execute_batch("CREATE TABLE IF NOT EXISTS vision_scene_sessions(scope TEXT NOT NULL,id TEXT NOT NULL,kind TEXT NOT NULL,state TEXT NOT NULL DEFAULT '{}',closed INTEGER NOT NULL DEFAULT 0,updated INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(scope,id)); CREATE TABLE IF NOT EXISTS vision_scene_observations(scope TEXT NOT NULL,session TEXT NOT NULL,seq INTEGER NOT NULL,text TEXT NOT NULL,turn TEXT NOT NULL,PRIMARY KEY(scope,session,seq));")?;
    let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if r.action == "create" {
        let count: i64 = tx.query_row(
            "SELECT COUNT(*) FROM vision_scene_sessions WHERE scope=?1",
            [&scope],
            |row| row.get(0),
        )?;
        ensure!(count < 1000, "scene session capacity reached");
        let id = if r.task_id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            r.task_id
        };
        tx.execute(
            "INSERT INTO vision_scene_sessions(scope,id,kind,updated) VALUES(?1,?2,?3,unixepoch())",
            params![scope, id, r.task_type],
        )?;
        tx.commit()?;
        return Ok(
            json!({"action":"created","task_id":id,"task_type":r.task_type,"untrusted":true}),
        );
    }
    let selected: Option<(String, String, String, i64)> = if r.task_id.is_empty() {
        tx.query_row("SELECT id,kind,state,closed FROM vision_scene_sessions WHERE scope=?1 AND closed=0 ORDER BY updated DESC,rowid DESC LIMIT 1",[&scope],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).optional()?
    } else {
        tx.query_row(
            "SELECT id,kind,state,closed FROM vision_scene_sessions WHERE scope=?1 AND id=?2",
            params![scope, r.task_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?
    };
    let Some((id, kind, state, closed)) = selected else {
        if r.action == "summarize" && r.task_id.is_empty() {
            let mut stmt=tx.prepare("SELECT id,kind,closed FROM vision_scene_sessions WHERE scope=?1 ORDER BY updated DESC,rowid DESC LIMIT 100")?;
            let rows=stmt.query_map([&scope],|row|Ok(json!({"task_id":row.get::<_,String>(0)?,"task_type":row.get::<_,String>(1)?,"closed":row.get::<_,i64>(2)?!=0})))?.collect::<rusqlite::Result<Vec<_>>>()?;
            return Ok(json!({"sessions":rows,"untrusted":true}));
        }
        bail!("scene session not found in this exact scope");
    };
    ensure!(
        closed == 0 || r.action == "summarize",
        "scene session is closed"
    );
    let mut state: Value = serde_json::from_str(&state)?;
    match r.action.as_str() {
        "add_observation" => {
            let count: i64 = tx.query_row(
                "SELECT COUNT(*) FROM vision_scene_observations WHERE scope=?1 AND session=?2",
                params![scope, id],
                |row| row.get(0),
            )?;
            ensure!(count < 256, "scene observation cap reached");
            tx.execute("INSERT INTO vision_scene_observations(scope,session,seq,text,turn) VALUES(?1,?2,?3,?4,?5)",params![scope,id,count,r.observation,ctx.turn_id])?;
        }
        "update_state" => {
            let updates: Value = serde_json::from_str(&r.state_updates)?;
            for (key, value) in updates.as_object().unwrap() {
                state[key] = value.clone();
            }
            let encoded = serde_json::to_string(&state)?;
            ensure!(
                encoded.len() <= 32768 && state.as_object().unwrap().len() <= 128,
                "scene state cap reached"
            );
            tx.execute(
                "UPDATE vision_scene_sessions SET state=?3 WHERE scope=?1 AND id=?2",
                params![scope, id, encoded],
            )?;
        }
        "close" => {
            tx.execute(
                "UPDATE vision_scene_sessions SET closed=1 WHERE scope=?1 AND id=?2",
                params![scope, id],
            )?;
        }
        "summarize" => {}
        _ => bail!("unknown action"),
    }
    tx.execute(
        "UPDATE vision_scene_sessions SET updated=unixepoch() WHERE scope=?1 AND id=?2",
        params![scope, id],
    )?;
    let total: i64 = tx.query_row(
        "SELECT COUNT(*) FROM vision_scene_observations WHERE scope=?1 AND session=?2",
        params![scope, id],
        |row| row.get(0),
    )?;
    let observations = {
        let mut stmt=tx.prepare("SELECT seq,text,turn FROM (SELECT seq,text,turn FROM vision_scene_observations WHERE scope=?1 AND session=?2 ORDER BY seq DESC LIMIT 32) ORDER BY seq")?;
        stmt.query_map(params![scope,id],|row|Ok(json!({"sequence":row.get::<_,i64>(0)?,"text":row.get::<_,String>(1)?,"turn_id":row.get::<_,String>(2)?,"trust":"caller_assertion"})))?.collect::<rusqlite::Result<Vec<_>>>()?
    };
    tx.commit()?;
    Ok(
        json!({"action":r.action,"task_id":id,"task_type":kind,"state":state,"observations":observations,"total_observations":total,"observations_truncated":total>32,"closed":closed!=0||r.action=="close","untrusted":true,"state_authority":"caller_assertions_only"}),
    )
}
