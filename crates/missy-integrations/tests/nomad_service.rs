use anyhow::Result;
use async_trait::async_trait;
use axum::{
    Router,
    body::{Body, to_bytes},
    extract::Request,
    response::Response,
};
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_integrations::*;
use missy_policy::PolicyHttpClient;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

struct Permit;
struct Deny;
#[async_trait]
impl ApprovalGate for Deny {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> Result<bool> {
        Ok(false)
    }
}
#[async_trait]
impl ApprovalGate for Permit {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> Result<bool> {
        Ok(true)
    }
}
fn context() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "turn".into(),
        cancellation: CancellationToken::new(),
    }
}
fn call(args: Value) -> ToolCall {
    ToolCall {
        id: "fixture".into(),
        name: "nomad_mutate".into(),
        arguments: args,
    }
}
fn graph() -> Value {
    json!({"ID":"svc","Name":"svc","Namespace":"lab","Type":"service","Meta":{"missy_service_owner":"operator-a"},"Datacenters":["dc1"],"TaskGroups":[{"Name":"web","Count":1,"EphemeralDisk":{"SizeMB":10},"Tasks":[{"Name":"app","Driver":"docker","Config":{"image":"example.invalid/pinned@sha256:abc"},"Resources":{"CPU":100,"MemoryMB":100}}]}]})
}
#[derive(Default)]
struct Fixture {
    requests: Vec<(String, String, Value)>,
    index: u64,
    expected: u64,
    foreign: bool,
    cpu: u64,
    conflict: bool,
    unknown: bool,
    dry_index: Option<u64>,
    writes: usize,
}
async fn fixture() -> (
    IntegrationTools,
    Arc<Mutex<Fixture>>,
    tokio::task::JoinHandle<()>,
) {
    fixture_at(7).await
}
async fn fixture_at(
    index: u64,
) -> (
    IntegrationTools,
    Arc<Mutex<Fixture>>,
    tokio::task::JoinHandle<()>,
) {
    let state = Arc::new(Mutex::new(Fixture {
        index,
        expected: index,
        cpu: 10000,
        ..Default::default()
    }));
    let captured = state.clone();
    let router=Router::new().fallback(move |req:Request| {
        let state=captured.clone();
        async move {
            let method=req.method().to_string(); let path=req.uri().to_string();
            let bytes=to_bytes(req.into_body(),262144).await.unwrap();
            let body=if bytes.is_empty(){Value::Null}else{serde_json::from_slice(&bytes).unwrap()};
            let mut s=state.lock().unwrap();
            s.requests.push((method.clone(),path.clone(),body.clone()));
            let mut status=200;
            let response=match (method.as_str(),path.as_str()) {
                ("GET","/v1/jobs?namespace=lab")=>if s.index==0 {json!([])} else {json!([{"ID":"svc","Namespace":"lab"}])},
                ("GET","/v1/job/svc?namespace=lab")=>{ let mut g=graph(); g["JobModifyIndex"]=json!(s.index); if s.foreign {g["Meta"]["missy_service_owner"]=json!("foreign");} g },
                ("GET","/v1/nodes")=>json!([{"ID":"node-a"},{"ID":"unreviewed"}]),
                ("GET","/v1/node/node-a")=>json!({"ID":"node-a","Status":"ready","SchedulingEligibility":"eligible","Datacenter":"dc1","NodePool":"default","NodeResources":{"Cpu":{"CpuShares":s.cpu},"Memory":{"MemoryMB":10000},"Disk":{"DiskMB":10000}},"ReservedResources":{"Cpu":{"CpuShares":100},"Memory":{"MemoryMB":100},"Disk":{"DiskMB":100}}}),
                ("GET","/v1/node/node-a/allocations")=>json!([{"ID":"foreign-alloc","NodeID":"node-a","ClientStatus":"running","Resources":{"CPU":100,"MemoryMB":100,"DiskMB":10}}]),
                ("POST","/v1/job/svc/plan?namespace=lab")=>json!({"JobModifyIndex":s.dry_index.unwrap_or(s.index),"FailedTGAllocs":{}}),
                ("POST","/v1/jobs?namespace=lab")=>{
                    s.writes+=1;
                    assert_eq!(body["EnforceIndex"],true);
                    assert_eq!(body["JobModifyIndex"],s.expected);
                    if s.conflict { status=409;json!({"error":"index conflict"}) }
                    else if s.unknown { status=503; json!({"error":"accepted then response lost"}) }
                    else { s.index=8;json!({"EvalID":"evaluation","JobModifyIndex":8}) }
                },
                _=>panic!("unexpected fixture request {method} {path}"),
            };
            Response::builder().status(status).header("content-type","application/json").body(Body::from(response.to_string())).unwrap()
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let cfg = IntegrationConfig {
        nomad: Some(NomadConfig {
            connection: ConnectionConfig {
                enabled: true,
                endpoint: endpoint.clone(),
                allow_mutations: true,
                ..Default::default()
            },
            namespace: "lab".into(),
            jobs: BTreeSet::from(["svc".into()]),
            nodes: BTreeSet::from(["node-a".into()]),
            deployments: BTreeMap::from([("svc".into(), graph())]),
            scales: BTreeMap::from([(
                "two".into(),
                NomadScale {
                    job: "svc".into(),
                    group: "web".into(),
                    count: 2,
                },
            )]),
            services: BTreeMap::from([(
                "svc".into(),
                NomadService {
                    owner: "operator-a".into(),
                    expected_modify_index: index,
                    rollout_overlap: 1,
                    allow_restart: true,
                    allow_stop: true,
                },
            )]),
        }),
        ..Default::default()
    };
    let http = PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![endpoint],
        allow_private_network: true,
        ..Default::default()
    })
    .unwrap();
    let tools = IntegrationTools::new(http, cfg)
        .unwrap()
        .with_approval(Arc::new(Permit));
    (tools, state, task)
}
async fn plan(tools: &IntegrationTools, action: &str) -> String {
    let args = if action == "plan_scale" {
        json!({"action":action,"grant":"two"})
    } else {
        json!({"action":action,"job":"svc"})
    };
    tools.execute(&context(), &call(args)).await.unwrap()["data"]["plan"]
        .as_str()
        .unwrap()
        .into()
}
#[tokio::test]
async fn scale_fresh_discovery_exact_graph_capacity_overlap_and_cas() {
    let (tools, s, task) = fixture().await;
    let token = plan(&tools, "plan_scale").await;
    let r = tools
        .execute(
            &context(),
            &call(json!({"action":"scale","grant":"two","plan":token})),
        )
        .await
        .unwrap();
    assert_eq!(r["data"]["state"], "submitted_not_yet_verified");
    let s = s.lock().unwrap();
    assert_eq!(
        s.requests.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(),
        vec![
            "/v1/jobs?namespace=lab",
            "/v1/job/svc?namespace=lab",
            "/v1/nodes",
            "/v1/node/node-a",
            "/v1/node/node-a/allocations",
            "/v1/job/svc/plan?namespace=lab",
            "/v1/jobs?namespace=lab",
            "/v1/job/svc?namespace=lab",
            "/v1/nodes",
            "/v1/node/node-a",
            "/v1/node/node-a/allocations",
            "/v1/jobs?namespace=lab"
        ]
    );
    let body = &s.requests.last().unwrap().2;
    assert_eq!(body["Job"]["TaskGroups"][0]["Count"], 2);
    assert_eq!(
        body["Job"]["TaskGroups"][0]["Tasks"][0],
        graph()["TaskGroups"][0]["Tasks"][0]
    );
    assert_eq!(body["Job"]["Constraints"][0]["RTarget"], "^(node-a)$");
    assert_eq!(s.writes, 1);
    task.abort();
}
#[tokio::test]
async fn foreign_owner_missing_capacity_and_dry_run_race_never_submit() {
    for mode in ["foreign", "capacity", "race"] {
        let (tools, s, task) = fixture().await;
        {
            let mut s = s.lock().unwrap();
            match mode {
                "foreign" => s.foreign = true,
                "capacity" => s.cpu = 399,
                _ => s.dry_index = Some(9),
            }
        }
        assert!(
            tools
                .execute(
                    &context(),
                    &call(json!({"action":"plan_scale","grant":"two"}))
                )
                .await
                .is_err()
        );
        assert_eq!(s.lock().unwrap().writes, 0);
        task.abort();
    }
}
#[tokio::test]
async fn concurrent_service_update_after_review_is_rejected_without_write() {
    let (tools, s, task) = fixture().await;
    let token = plan(&tools, "plan").await;
    s.lock().unwrap().index = 8;
    let c = call(json!({"action":"deploy","job":"svc","plan":token}));
    assert!(tools.execute(&context(), &c).await.is_err());
    assert!(tools.execute(&context(), &c).await.is_err());
    assert_eq!(s.lock().unwrap().writes, 0);
    task.abort();
}
#[tokio::test]
async fn scheduler_cas_conflict_and_unknown_effect_never_retry_or_replan() {
    for unknown in [false, true] {
        let (tools, s, task) = fixture().await;
        let token = plan(&tools, "plan").await;
        {
            let mut s = s.lock().unwrap();
            s.unknown = unknown;
            s.conflict = !unknown;
        }
        let c = call(json!({"action":"deploy","job":"svc","plan":token}));
        let error = tools.execute(&context(), &c).await.unwrap_err().to_string();
        assert!(error.contains("no retry"), "{error}");
        assert!(tools.execute(&context(), &c).await.is_err());
        assert!(
            tools
                .execute(&context(), &call(json!({"action":"plan","job":"svc"})))
                .await
                .is_err()
        );
        assert_eq!(s.lock().unwrap().writes, 1);
        task.abort();
    }
}
#[tokio::test]
async fn concurrent_plan_consumption_allows_at_most_one_registration() {
    let (tools, s, task) = fixture().await;
    let token = plan(&tools, "plan").await;
    let c = call(json!({"action":"deploy","job":"svc","plan":token}));
    let a = context();
    let b = context();
    let (one, two) = tokio::join!(tools.execute(&a, &c), tools.execute(&b, &c));
    assert_ne!(one.is_ok(), two.is_ok());
    assert_eq!(s.lock().unwrap().writes, 1);
    task.abort();
}
#[tokio::test]
async fn native_rolling_restart_and_stop_are_reviewed_cas_operations() {
    for stop in [false, true] {
        let (tools, s, task) = fixture().await;
        let token = plan(&tools, if stop { "plan_stop" } else { "plan_restart" }).await;
        tools
            .execute(
                &context(),
                &call(json!({"action":if stop{"stop"}else{"restart"},"job":"svc","plan":token})),
            )
            .await
            .unwrap();
        let s = s.lock().unwrap();
        let body = &s.requests.last().unwrap().2;
        if stop {
            assert_eq!(body["Job"]["Stop"], true);
        } else {
            assert_eq!(
                body["Job"]["TaskGroups"][0]["Tasks"][0]["Env"]["MISSY_SERVICE_RESTART_ID"],
                token
            );
        }
        assert!(
            !s.requests
                .iter()
                .any(|r| r.0 == "DELETE" || r.1.contains("/scale"))
        );
        task.abort();
    }
}
#[tokio::test]
async fn create_if_absent_pins_zero_and_rejects_foreign_arrival() {
    for arrival in [false, true] {
        let (tools, s, task) = fixture_at(0).await;
        let token = plan(&tools, "plan").await;
        if arrival {
            let mut s = s.lock().unwrap();
            s.index = 1;
            s.foreign = true;
        }
        let result = tools
            .execute(
                &context(),
                &call(json!({"action":"deploy","job":"svc","plan":token})),
            )
            .await;
        assert_eq!(result.is_ok(), !arrival);
        assert_eq!(s.lock().unwrap().writes, usize::from(!arrival));
        task.abort();
    }
}
#[tokio::test]
async fn capacity_and_foreign_owner_are_rechecked_after_review() {
    for foreign in [false, true] {
        let (tools, s, task) = fixture().await;
        let token = plan(&tools, "plan").await;
        {
            let mut s = s.lock().unwrap();
            if foreign {
                s.foreign = true;
            } else {
                s.cpu = 100;
            }
        }
        let c = call(json!({"action":"deploy","job":"svc","plan":token}));
        assert!(tools.execute(&context(), &c).await.is_err());
        assert!(tools.execute(&context(), &c).await.is_err());
        assert_eq!(s.lock().unwrap().writes, 0);
        task.abort();
    }
}
#[tokio::test]
async fn plans_bind_action_and_cannot_bypass_outer_approval() {
    let (tools, s, task) = fixture().await;
    let token = plan(&tools, "plan_scale").await;
    assert!(
        tools
            .execute(
                &context(),
                &call(json!({"action":"deploy","job":"svc","plan":token}))
            )
            .await
            .is_err()
    );
    assert!(
        tools
            .execute(
                &context(),
                &call(json!({"action":"scale","grant":"two","plan":token}))
            )
            .await
            .is_err()
    );
    assert_eq!(s.lock().unwrap().writes, 0);
    let tools = tools.with_approval(Arc::new(Deny));
    let requests = s.lock().unwrap().requests.len();
    assert!(
        tools
            .execute(&context(), &call(json!({"action":"plan","job":"svc"})))
            .await
            .is_err()
    );
    assert_eq!(s.lock().unwrap().requests.len(), requests);
    task.abort();
}
