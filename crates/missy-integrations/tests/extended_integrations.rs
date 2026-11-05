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
#[tokio::test]
async fn approved_storyboard_scene_single_submit_poll_download_and_decode() {
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let png = png.into_inner();
    let submits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = submits.clone();
    let app=Router::new().fallback(move|req:Request|{let png=png.clone();let count=count.clone();async move{
        let path=req.uri().to_string();
        if path.starts_with("/view?"){return Response::new(Body::from(png));}
        let response=if path=="/prompt" {count.fetch_add(1,std::sync::atomic::Ordering::SeqCst);let b=to_bytes(req.into_body(),8192).await.unwrap();let v:Value=serde_json::from_slice(&b).unwrap();assert_eq!(v["prompt"]["1"]["inputs"]["text"],"scene prompt");json!({"prompt_id":"scenejob","node_errors":{}})} else {assert_eq!(path,"/history/scenejob");json!({"scenejob":{"status":{"completed":true,"status_str":"success"},"outputs":{"2":{"images":[{"filename":"scene.png","subfolder":"","type":"output"}]}}}})};
        Response::new(Body::from(response.to_string()))
    }});
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("scene.png");
    let http = PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![endpoint.clone()],
        allow_private_network: true,
        write_roots: vec![dir.path().into()],
        max_file_bytes: 4096,
        ..Default::default()
    })
    .unwrap();
    let mut connection = conn(&endpoint);
    connection.allow_mutations = true;
    let tools=IntegrationTools::new(http,IntegrationConfig{comfyui:Some(ComfyConfig{connection,max_artifact_bytes:4096,max_tracked_jobs:10,allowed_classes:BTreeSet::from(["CLIPTextEncode".into(),"SaveImage".into()]),workflows:BTreeMap::from([("scene".into(),Workflow{graph:json!({"1":{"class_type":"CLIPTextEncode","inputs":{"text":"base"}},"2":{"class_type":"SaveImage","inputs":{"images":["1",0]}}}),prompt_node:Some("1".into())})]),..Default::default()}),..Default::default()}).unwrap();
    let path = tools
        .approved_scene(
            &ctx(),
            &SceneGenerationGrant {
                workflow: "scene".into(),
                output_node: "2".into(),
                width: 2,
                height: 2,
            },
            SceneImageRequest {
                prompt: "scene prompt".into(),
                output: output.clone(),
                width: 2,
                height: 2,
            },
        )
        .await
        .unwrap();
    assert_eq!(path, output);
    assert_eq!(image::open(path).unwrap().width(), 2);
    assert_eq!(submits.load(std::sync::atomic::Ordering::SeqCst), 1);
    task.abort();
}
#[tokio::test]
async fn comfy_operator_pinned_conditioning_multipart_and_graph_binding() {
    let seen: Seen = Default::default();
    let s = seen.clone();
    let app = Router::new().fallback(move |req: Request| {
        let s = s.clone();
        async move {
            let method = req.method().to_string();
            let path = req.uri().to_string();
            let content_type = req
                .headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            let body = to_bytes(req.into_body(), 4 * 1024 * 1024)
                .await
                .unwrap()
                .to_vec();
            s.lock().unwrap().push((method, path.clone(), body.clone()));
            let response = if path == "/upload/image" {
                assert!(content_type.starts_with("multipart/form-data; boundary=missy"));
                let text = String::from_utf8_lossy(&body);
                let name = text
                    .split("filename=\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap();
                let png_start = body
                    .windows(8)
                    .position(|w| w == b"\x89PNG\r\n\x1a\n")
                    .unwrap();
                assert_eq!(
                    image::load_from_memory(&body[png_start..]).unwrap().width(),
                    2
                );
                json!({"name":name,"subfolder":"","type":"input"})
            } else {
                assert_eq!(path, "/prompt");
                let graph: Value = serde_json::from_slice(&body).unwrap();
                assert!(
                    graph["prompt"]["1"]["inputs"]["image"]
                        .as_str()
                        .unwrap()
                        .starts_with("missy-")
                );
                assert_eq!(graph["prompt"]["1"]["class_type"], "LoadImage");
                json!({"prompt_id":"conditioned","node_errors":{}})
            };
            Response::new(Body::from(response.to_string()))
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.png");
    image::DynamicImage::new_rgb8(2, 2).save(&path).unwrap();
    let client = PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![endpoint.clone()],
        allow_private_network: true,
        read_roots: vec![dir.path().into()],
        max_file_bytes: 4096,
        ..Default::default()
    })
    .unwrap();
    let cfg = IntegrationConfig {
        comfyui: Some(ComfyConfig {
            connection: conn(&endpoint),
            max_tracked_jobs: 10,
            allowed_classes: BTreeSet::from(["LoadImage".into()]),
            workflows: BTreeMap::from([(
                "image".into(),
                Workflow {
                    graph: json!({"1":{"class_type":"LoadImage","inputs":{"image":"unused.png"}}}),
                    prompt_node: None,
                },
            )]),
            conditioning: BTreeMap::from([(
                "reference".into(),
                ImageConditioning {
                    workflow: "image".into(),
                    node: "1".into(),
                    path,
                },
            )]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let tools = IntegrationTools::new(client, cfg)
        .unwrap()
        .with_approval(Arc::new(Permit));
    tools
        .execute(
            &ctx(),
            &call(
                "comfyui_submit",
                json!({"workflow":"image","conditioning":"reference"}),
            ),
        )
        .await
        .unwrap();
    assert_eq!(seen.lock().unwrap().len(), 2);
    assert!(
        tools
            .execute(
                &ctx(),
                &call(
                    "comfyui_submit",
                    json!({"workflow":"image","conditioning":"unknown"})
                )
            )
            .await
            .is_err()
    );
    assert_eq!(seen.lock().unwrap().len(), 2);
    task.abort();
}
struct Permit;
// Reproduce the parent's dynamic side_effect catalogue and actual durable outer
// guard, including the privileged inner marker. The marker alone is not consent.
fn guarded(
    inner: IntegrationTools,
    registry: missy_governance::ApprovalRegistry,
) -> missy_governance::GuardedTools {
    let sensitive_tool_names = inner
        .definitions()
        .into_iter()
        .filter(|d| d.side_effect)
        .map(|d| d.name)
        .collect();
    missy_governance::GuardedTools::new(
        Arc::new(inner.with_approval(Arc::new(Permit))),
        registry,
        missy_governance::GuardConfig {
            sensitive_tool_names,
            context_fingerprint: "0".repeat(64),
            approval_ttl_seconds: 300,
        },
    )
    .unwrap()
}
async fn approve(
    guard: &missy_governance::GuardedTools,
    registry: &missy_governance::ApprovalRegistry,
    ctx: &ToolContext,
    call: &ToolCall,
) -> String {
    let err = guard.preflight(ctx, call).await.unwrap_err();
    let id = err
        .downcast_ref::<missy_core::ApprovalRequired>()
        .unwrap()
        .request_id
        .clone();
    let record = registry.get(&ctx.scope, &id).unwrap().unwrap();
    registry
        .decide(
            &ctx.scope,
            &id,
            &record.binding_hash,
            missy_governance::Decision::Approve,
            "operator",
        )
        .unwrap();
    id
}
#[tokio::test]
async fn actual_outer_guard_requires_exact_consumable_incus_operation_approval() {
    let (endpoint, seen, _, task) =
        server(json!({"type":"sync","status":"Success","metadata":{}})).await;
    let dir = tempfile::tempdir().unwrap();
    let registry =
        missy_governance::ApprovalRegistry::open(dir.path().join("approval.sqlite3")).unwrap();
    let guard = guarded(
        IntegrationTools::new(client(&endpoint), incus(&endpoint)).unwrap(),
        registry.clone(),
    );
    assert_eq!(
        guard.annotation("incus_operation"),
        missy_governance::PolicyAnnotation::OperatorApprovalAndInnerPolicyRequired
    );
    let context = ctx();
    let c = call("incus_operation", json!({"grant":"snap"}));
    assert!(guard.execute(&context, &c).await.is_err());
    assert!(seen.lock().unwrap().is_empty());
    let id = approve(&guard, &registry, &context, &c).await;
    let mut changed = c.clone();
    changed.arguments = json!({"grant":"restore"});
    assert!(guard.execute(&context, &changed).await.is_err());
    let mut changed_scope = ctx();
    changed_scope.scope.user = "different".into();
    assert!(guard.execute(&changed_scope, &c).await.is_err());
    assert!(seen.lock().unwrap().is_empty());
    guard.execute(&context, &c).await.unwrap();
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(
        registry.get(&context.scope, &id).unwrap().unwrap().status,
        missy_governance::ApprovalStatus::Consumed
    );
    assert!(guard.execute(&context, &c).await.is_err());
    assert_eq!(seen.lock().unwrap().len(), 1);
    task.abort();
}
#[tokio::test]
async fn actual_outer_guard_requires_approval_for_exact_comfy_artifact_output_path() {
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let png = png.into_inner();
    let seen: Seen = Default::default();
    let s = seen.clone();
    let expected = png.clone();
    let app=Router::new().fallback(move|req:Request|{let seen=s.clone();let png=png.clone();async move{
        let path=req.uri().to_string();let method=req.method().to_string();let body=to_bytes(req.into_body(),65536).await.unwrap();seen.lock().unwrap().push((method,path.clone(),body.to_vec()));
        let response=if path=="/prompt"{serde_json::to_vec(&json!({"prompt_id":"owned","node_errors":{}})).unwrap()}
        else if path=="/history/owned"{serde_json::to_vec(&json!({"owned":{"status":{"completed":true,"status_str":"success"},"outputs":{"1":{"images":[{"filename":"safe.png","subfolder":"","type":"output"}]}}}})).unwrap()}
        else {assert_eq!(path,"/view?filename=safe.png&subfolder=&type=output");png};Response::new(Body::from(response))
    }});
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("approved.png");
    let other = dir.path().join("unapproved.png");
    let registry =
        missy_governance::ApprovalRegistry::open(dir.path().join("approval.sqlite3")).unwrap();
    let http = PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![endpoint.clone()],
        allow_private_network: true,
        write_roots: vec![dir.path().into()],
        max_file_bytes: 4096,
        ..Default::default()
    })
    .unwrap();
    let cfg = IntegrationConfig {
        comfyui: Some(ComfyConfig {
            connection: conn(&endpoint),
            max_artifact_bytes: 1024,
            max_tracked_jobs: 10,
            allowed_classes: BTreeSet::from(["SaveImage".into()]),
            workflows: BTreeMap::from([(
                "image".into(),
                Workflow {
                    graph: json!({"1":{"class_type":"SaveImage","inputs":{"filename_prefix":"safe"}}}),
                    prompt_node: None,
                },
            )]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let guard = guarded(IntegrationTools::new(http, cfg).unwrap(), registry.clone());
    let context = ctx();
    assert_eq!(
        guard.annotation("comfyui_artifact"),
        missy_governance::PolicyAnnotation::OperatorApprovalAndInnerPolicyRequired
    );
    let submit = call("comfyui_submit", json!({"workflow":"image"}));
    approve(&guard, &registry, &context, &submit).await;
    guard.execute(&context, &submit).await.unwrap();
    let mut artifact = call(
        "comfyui_artifact",
        json!({"prompt_id":"owned","node":"1","kind":"images","output_index":0,"output_path":dest}),
    );
    artifact.id = "artifact-call".into();
    let n = seen.lock().unwrap().len();
    assert!(guard.execute(&context, &artifact).await.is_err());
    assert_eq!(n, seen.lock().unwrap().len());
    assert!(!dest.exists());
    let id = approve(&guard, &registry, &context, &artifact).await;
    let mut changed = artifact.clone();
    changed.arguments["output_path"] = json!(other);
    assert!(guard.execute(&context, &changed).await.is_err());
    assert_eq!(n, seen.lock().unwrap().len());
    assert!(!other.exists());
    guard.execute(&context, &artifact).await.unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), expected);
    assert_eq!(
        registry.get(&context.scope, &id).unwrap().unwrap().status,
        missy_governance::ApprovalStatus::Consumed
    );
    let n = seen.lock().unwrap().len();
    assert!(guard.execute(&context, &artifact).await.is_err());
    assert_eq!(n, seen.lock().unwrap().len());
    task.abort();
}
#[tokio::test]
async fn incus_exec_wait_recorded_logs_and_descriptor_confinement() {
    let id = "12345678-1234-4234-8234-123456789abc";
    let malicious = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let m = malicious.clone();
    let seen: Seen = Default::default();
    let s = seen.clone();
    let app=Router::new().fallback(move|req:Request|{let s=s.clone();let m=m.clone();async move{
        let method=req.method().to_string();let path=req.uri().to_string();let body=to_bytes(req.into_body(),65536).await.unwrap();s.lock().unwrap().push((method,path.clone(),body.to_vec()));
        let bytes=if path=="/1.0/instances/vm/exec?project=lab"{serde_json::to_vec(&json!({"type":"async","operation":format!("/1.0/operations/{id}"),"metadata":{}})).unwrap()}
        else if path==format!("/1.0/operations/{id}/wait?timeout=10&project=lab") {
            let stdout=if m.load(std::sync::atomic::Ordering::SeqCst){"http://evil/steal".into()}else{format!("/1.0/instances/vm/logs/exec_{id}.stdout")};
            serde_json::to_vec(&json!({"type":"sync","metadata":{"id":id,"status":"Success","status_code":200,"metadata":{"return":0,"output":{"1":stdout,"2":format!("/1.0/instances/vm/logs/exec_{id}.stderr")}}}})).unwrap()
        } else if path==format!("/1.0/instances/vm/logs/exec_{id}.stdout?project=lab"){b"0\n".to_vec()}
        else if path==format!("/1.0/instances/vm/logs/exec_{id}.stderr?project=lab"){vec![]}
        else{panic!("unexpected path {path}")};Response::new(Body::from(bytes))
    }});
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let tools = IntegrationTools::new(client(&endpoint), incus(&endpoint))
        .unwrap()
        .with_approval(Arc::new(Permit));
    let r = tools
        .execute(&ctx(), &call("incus_operation", json!({"grant":"exec"})))
        .await
        .unwrap();
    assert_eq!(r["data"]["exit_code"], 0);
    assert_eq!(r["data"]["stdout"], "0\n");
    assert_eq!(seen.lock().unwrap().len(), 4);
    malicious.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(
        tools
            .execute(&ctx(), &call("incus_operation", json!({"grant":"exec"})))
            .await
            .is_err()
    );
    assert_eq!(seen.lock().unwrap().len(), 6);
    task.abort();
}
#[async_trait]
impl ApprovalGate for Permit {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> Result<bool> {
        Ok(true)
    }
}
fn ctx() -> ToolContext {
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
fn call(name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: "c".into(),
        name: name.into(),
        arguments: args,
    }
}
fn conn(endpoint: &str) -> ConnectionConfig {
    ConnectionConfig {
        enabled: true,
        endpoint: endpoint.into(),
        allow_mutations: true,
        ..Default::default()
    }
}
fn client(endpoint: &str) -> PolicyHttpClient {
    PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![endpoint.into()],
        allow_private_network: true,
        ..Default::default()
    })
    .unwrap()
}
type Seen = Arc<Mutex<Vec<(String, String, Vec<u8>)>>>;
async fn server(response: Value) -> (String, Seen, Arc<Mutex<Value>>, tokio::task::JoinHandle<()>) {
    let seen: Seen = Default::default();
    let r = Arc::new(Mutex::new(response));
    let s = seen.clone();
    let rr = r.clone();
    let app = Router::new().fallback(move |req: Request| {
        let s = s.clone();
        let r = rr.clone();
        async move {
            let method = req.method().to_string();
            let path = req.uri().to_string();
            let b = to_bytes(req.into_body(), 4 * 1024 * 1024).await.unwrap();
            s.lock().unwrap().push((method, path, b.to_vec()));
            Response::new(Body::from(r.lock().unwrap().to_string()))
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen, r, task)
}
fn incus(endpoint: &str) -> IntegrationConfig {
    let fingerprint = "a".repeat(64);
    let vm = IncusResource::Instance { name: "vm".into() };
    let resources = BTreeSet::from([
        vm.clone(),
        IncusResource::Image {
            fingerprint: fingerprint.clone(),
        },
        IncusResource::Network { name: "net".into() },
        IncusResource::StoragePool {
            name: "pool".into(),
        },
        IncusResource::StorageVolume {
            pool: "pool".into(),
            name: "vol".into(),
        },
        IncusResource::Profile {
            name: "profile".into(),
        },
        IncusResource::Project { name: "lab".into() },
    ]);
    let operations = BTreeMap::from([
        (
            "launch".into(),
            IncusOperation::Create {
                resource: vm,
                definition: json!({"name":"vm","type":"container","source":{"type":"image","fingerprint":fingerprint}}),
            },
        ),
        (
            "exec".into(),
            IncusOperation::Exec {
                instance: "vm".into(),
                argv: vec!["/usr/bin/id".into(), "-u".into()],
            },
        ),
        (
            "write".into(),
            IncusOperation::FileWrite {
                file: IncusFile {
                    instance: "vm".into(),
                    path: "/etc/reviewed.txt".into(),
                },
                content: "pinned".into(),
            },
        ),
        (
            "snap".into(),
            IncusOperation::SnapshotCreate {
                instance: "vm".into(),
                snapshot: "before".into(),
            },
        ),
        (
            "restore".into(),
            IncusOperation::SnapshotRestore {
                instance: "vm".into(),
                snapshot: "before".into(),
            },
        ),
        (
            "drop-snap".into(),
            IncusOperation::SnapshotDelete {
                instance: "vm".into(),
                snapshot: "before".into(),
            },
        ),
        (
            "copy".into(),
            IncusOperation::Copy {
                instance: "vm".into(),
                target: "copy".into(),
            },
        ),
        (
            "move".into(),
            IncusOperation::Move {
                instance: "vm".into(),
                target: "copy".into(),
            },
        ),
        (
            "device".into(),
            IncusOperation::DeviceSet {
                instance: "vm".into(),
                device: "eth0".into(),
                definition: BTreeMap::from([
                    ("type".into(), "nic".into()),
                    ("network".into(), "net".into()),
                ]),
            },
        ),
        (
            "network".into(),
            IncusOperation::Create {
                resource: IncusResource::Network { name: "net".into() },
                definition: json!({"name":"net","type":"bridge","config":{}}),
            },
        ),
        (
            "storage".into(),
            IncusOperation::Create {
                resource: IncusResource::StoragePool {
                    name: "pool".into(),
                },
                definition: json!({"name":"pool","driver":"dir","config":{}}),
            },
        ),
        (
            "volume".into(),
            IncusOperation::Create {
                resource: IncusResource::StorageVolume {
                    pool: "pool".into(),
                    name: "vol".into(),
                },
                definition: json!({"name":"vol","type":"custom","config":{}}),
            },
        ),
        (
            "profile".into(),
            IncusOperation::Update {
                resource: IncusResource::Profile {
                    name: "profile".into(),
                },
                definition: json!({"config":{},"devices":{}}),
            },
        ),
        (
            "project".into(),
            IncusOperation::Update {
                resource: IncusResource::Project { name: "lab".into() },
                definition: json!({"config":{}}),
            },
        ),
        (
            "image".into(),
            IncusOperation::Update {
                resource: IncusResource::Image {
                    fingerprint: "a".repeat(64),
                },
                definition: json!({"public":false}),
            },
        ),
    ]);
    IntegrationConfig {
        incus: Some(IncusConfig {
            connection: conn(endpoint),
            project: "lab".into(),
            instances: BTreeSet::from(["vm".into(), "copy".into()]),
            resources,
            operations,
            file_reads: BTreeMap::from([(
                "read".into(),
                IncusFile {
                    instance: "vm".into(),
                    path: "/etc/reviewed.txt".into(),
                },
            )]),
            ..Default::default()
        }),
        ..Default::default()
    }
}
#[tokio::test]
async fn incus_all_typed_routes_pinned_bodies_and_fail_closed_approval() {
    let (endpoint, seen, response, task) =
        server(json!({"type":"sync","metadata":{"devices":{},"name":"vm"}})).await;
    let config = incus(&endpoint);
    let tools = IntegrationTools::new(client(&endpoint), config.clone()).unwrap();
    assert!(
        tools
            .definitions()
            .iter()
            .find(|d| d.name == "incus_operation")
            .unwrap()
            .side_effect
    );
    assert!(
        tools
            .execute(&ctx(), &call("incus_operation", json!({"grant":"launch"})))
            .await
            .is_err()
    );
    assert!(seen.lock().unwrap().is_empty());
    let tools = tools.with_approval(Arc::new(Permit));
    for grant in [
        "launch",
        "exec",
        "snap",
        "restore",
        "drop-snap",
        "copy",
        "move",
        "device",
        "network",
        "storage",
        "volume",
        "profile",
        "project",
        "image",
    ] {
        tools
            .execute(&ctx(), &call("incus_operation", json!({"grant":grant})))
            .await
            .unwrap();
    }
    let requests = seen.lock().unwrap().clone();
    assert_eq!(requests[0].1, "/1.0/instances?project=lab");
    let body: Value = serde_json::from_slice(&requests[1].2).unwrap();
    assert_eq!(body["command"], json!(["/usr/bin/id", "-u"]));
    assert_eq!(body["wait-for-websocket"], false);
    assert!(
        requests
            .iter()
            .any(|r| r.0 == "DELETE" && r.1 == "/1.0/instances/vm/snapshots/before?project=lab")
    );
    assert!(
        requests
            .iter()
            .any(|r| r.1 == "/1.0/storage-pools/pool/volumes/custom?project=lab")
    );
    let before = seen.lock().unwrap().len();
    for args in [
        json!({"grant":"exec","argv":["/bin/sh"]}),
        json!({"grant":"missing"}),
        json!({"grant":"launch","endpoint":"http://elsewhere"}),
    ] {
        assert!(
            tools
                .execute(&ctx(), &call("incus_operation", args))
                .await
                .is_err()
        );
    }
    assert_eq!(before, seen.lock().unwrap().len());
    *response.lock().unwrap() =
        json!({"type":"sync","metadata":[{"name":"net"},{"name":"private"}]});
    let r = tools
        .execute(
            &ctx(),
            &call("incus_discover", json!({"action":"list","kind":"network"})),
        )
        .await
        .unwrap();
    assert_eq!(r["data"].as_array().unwrap().len(), 1);
    assert!(
        tools
            .execute(
                &ctx(),
                &call(
                    "incus_discover",
                    json!({"action":"inspect","resource":"network/private"})
                )
            )
            .await
            .is_err()
    );
    task.abort();
}
#[test]
fn incus_bad_operator_grants_rejected_without_io() {
    for path in [
        "/etc/../shadow",
        "relative",
        "/etc//shadow",
        "/etc/a?x/../y",
    ] {
        let mut c = incus("http://127.0.0.1:1");
        c.incus
            .as_mut()
            .unwrap()
            .file_reads
            .get_mut("read")
            .unwrap()
            .path = path.into();
        assert!(IntegrationTools::new(client("http://127.0.0.1:1"), c).is_err());
    }
    let mut c = incus("http://127.0.0.1:1");
    c.incus.as_mut().unwrap().operations.insert("bad".into(),IncusOperation::Create{resource:IncusResource::Instance{name:"vm".into()},definition:json!({"name":"other","type":"container","source":{"type":"image","server":"http://evil"}})});
    assert!(IntegrationTools::new(client("http://127.0.0.1:1"), c).is_err());
}
#[tokio::test]
async fn nomad_nodes_plan_scale_allocations_exact_namespace_and_pins() {
    let (endpoint, seen, response, task) = server(json!([{"ID":"node"},{"ID":"private"}])).await;
    let config = IntegrationConfig {
        nomad: Some(NomadConfig {
            connection: conn(&endpoint),
            namespace: "lab".into(),
            jobs: BTreeSet::from(["worker".into()]),
            nodes: BTreeSet::from(["node".into()]),
            deployments: BTreeMap::from([(
                "worker".into(),
                json!({"ID":"worker","Namespace":"lab","TaskGroups":[{"Name":"group","Count":1,"Tasks":[]}]}),
            )]),
            scales: BTreeMap::from([(
                "two".into(),
                NomadScale {
                    job: "worker".into(),
                    group: "group".into(),
                    count: 2,
                },
            )]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let tools = IntegrationTools::new(client(&endpoint), config)
        .unwrap()
        .with_approval(Arc::new(Permit));
    let r = tools
        .execute(&ctx(), &call("nomad_read", json!({"action":"nodes"})))
        .await
        .unwrap();
    assert_eq!(r["data"].as_array().unwrap().len(), 1);
    *response.lock().unwrap() =
        json!({"ID":"node","NodeResources":{"Cpu":{"CpuShares":1000}},"Secret":"not projected"});
    let r = tools
        .execute(
            &ctx(),
            &call("nomad_read", json!({"action":"node","node":"node"})),
        )
        .await
        .unwrap();
    assert!(r["data"].get("Secret").is_none());
    *response.lock().unwrap() = json!({"JobModifyIndex":2,"FailedTGAllocs":{},"EvalID":"eval"});
    tools
        .execute(
            &ctx(),
            &call("nomad_read", json!({"action":"placement","job":"worker"})),
        )
        .await
        .unwrap();
    assert!(
        tools
            .execute(
                &ctx(),
                &call("nomad_mutate", json!({"action":"scale","grant":"two"})),
            )
            .await
            .is_err()
    );
    {
        let requests = seen.lock().unwrap();
        assert_eq!(requests[2].1, "/v1/job/worker/plan?namespace=lab");
        assert_eq!(requests.len(), 3); // no unfenced /scale shortcut
    }
    *response.lock().unwrap() = json!([{"ID":"a","JobID":"worker","Namespace":"lab","TaskStates":{"t":{"State":"dead","Failed":false}}},{"JobID":"private","Namespace":"lab"}]);
    let r = tools
        .execute(
            &ctx(),
            &call("nomad_read", json!({"action":"results","job":"worker"})),
        )
        .await
        .unwrap();
    assert_eq!(r["data"]["kind"], "allocation_metadata");
    assert_eq!(r["data"]["allocations"].as_array().unwrap().len(), 1);
    let n = seen.lock().unwrap().len();
    assert!(
        tools
            .execute(
                &ctx(),
                &call(
                    "nomad_mutate",
                    json!({"action":"scale","grant":"two","count":100})
                )
            )
            .await
            .is_err()
    );
    assert_eq!(n, seen.lock().unwrap().len());
    task.abort();
}
#[tokio::test]
async fn comfy_owned_artifact_decoded_atomic_policy_write_metadata_and_bounds() {
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 3)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let payload = Arc::new(Mutex::new(png.into_inner()));
    let data = payload.clone();
    let seen: Seen = Default::default();
    let s = seen.clone();
    let app=Router::new().fallback(move |req:Request|{let data=data.clone();let seen=s.clone();async move {
        let path=req.uri().to_string();let method=req.method().to_string();let bytes=to_bytes(req.into_body(),1024*1024).await.unwrap();
        seen.lock().unwrap().push((method,path.clone(),bytes.to_vec()));
        let body=if path=="/prompt"{serde_json::to_vec(&json!({"prompt_id":"owned","node_errors":{}})).unwrap()}
            else if path=="/history/owned"{serde_json::to_vec(&json!({"owned":{"status":{"completed":true,"status_str":"success"},"outputs":{"1":{"images":[{"filename":"image file.png","subfolder":"safe folder","type":"temp"}]}}}})).unwrap()}
            else if path.starts_with("/view?"){data.lock().unwrap().clone()}else{panic!("unexpected route {path}")};
        Response::new(Body::from(body))
    }});
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("artifact.png");
    let client = PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![endpoint.clone()],
        allow_private_network: true,
        write_roots: vec![dir.path().into()],
        max_file_bytes: 4096,
        ..Default::default()
    })
    .unwrap();
    let config = IntegrationConfig {
        comfyui: Some(ComfyConfig {
            connection: conn(&endpoint),
            max_artifact_bytes: 1024,
            max_tracked_jobs: 10,
            allowed_classes: BTreeSet::from(["SaveImage".into()]),
            workflows: BTreeMap::from([(
                "image".into(),
                Workflow {
                    graph: json!({"1":{"class_type":"SaveImage","inputs":{"filename_prefix":"safe"}}}),
                    prompt_node: None,
                },
            )]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let tools = IntegrationTools::new(client, config)
        .unwrap()
        .with_approval(Arc::new(Permit));
    tools
        .execute(&ctx(), &call("comfyui_submit", json!({"workflow":"image"})))
        .await
        .unwrap();
    let args =
        json!({"prompt_id":"owned","node":"1","kind":"images","output_index":0,"output_path":dest});
    let r = tools
        .execute(&ctx(), &call("comfyui_artifact", args.clone()))
        .await
        .unwrap();
    assert_eq!(r["data"]["media"]["width"], 2);
    assert_eq!(r["data"]["media"]["height"], 3);
    assert_eq!(r["data"]["artifact"]["storage_type"], "temp");
    assert_eq!(std::fs::read(&dest).unwrap(), *payload.lock().unwrap());
    let view = seen.lock().unwrap().last().unwrap().1.clone();
    assert_eq!(
        view,
        "/view?filename=image+file.png&subfolder=safe+folder&type=temp"
    );
    let n = seen.lock().unwrap().len();
    let mut other = ctx();
    other.scope.user = "other".into();
    assert!(
        tools
            .execute(&other, &call("comfyui_artifact", args.clone()))
            .await
            .is_err()
    );
    assert_eq!(n, seen.lock().unwrap().len());
    let mut denied = args.clone();
    denied["output_path"] = json!("/etc/not-allowed.png");
    assert!(
        tools
            .execute(&ctx(), &call("comfyui_artifact", denied))
            .await
            .is_err()
    );
    assert_eq!(n, seen.lock().unwrap().len());
    let mut missing = args.clone();
    missing["output_index"] = json!(1);
    assert!(
        tools
            .execute(&ctx(), &call("comfyui_artifact", missing))
            .await
            .is_err()
    );
    *payload.lock().unwrap() = b"not an image".to_vec();
    assert!(
        tools
            .execute(&ctx(), &call("comfyui_artifact", args.clone()))
            .await
            .is_err()
    );
    assert_ne!(std::fs::read(&dest).unwrap(), b"not an image");
    *payload.lock().unwrap() = vec![0; 1025];
    assert!(
        tools
            .execute(&ctx(), &call("comfyui_artifact", args.clone()))
            .await
            .unwrap_err()
            .to_string()
            .contains("bound")
    );
    use std::os::unix::fs::symlink;
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("target"), b"untouched").unwrap();
    std::fs::remove_file(&dest).unwrap();
    symlink(outside.path().join("target"), &dest).unwrap();
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 3)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    *payload.lock().unwrap() = png.into_inner();
    assert!(
        tools
            .execute(&ctx(), &call("comfyui_artifact", args))
            .await
            .is_err()
    );
    assert_eq!(
        std::fs::read(outside.path().join("target")).unwrap(),
        b"untouched"
    );
    task.abort();
}
#[tokio::test]
async fn incus_remote_text_read_write_exact_paths_not_host_paths() {
    let seen: Seen = Default::default();
    let s = seen.clone();
    let app = Router::new().fallback(move |req: Request| {
        let s = s.clone();
        async move {
            let method = req.method().to_string();
            let path = req.uri().to_string();
            let b = to_bytes(req.into_body(), 1024).await.unwrap();
            s.lock().unwrap().push((method.clone(), path, b.to_vec()));
            Response::new(Body::from(if method == "GET" {
                b"guest text".to_vec()
            } else {
                serde_json::to_vec(&json!({"type":"sync","error_code":0,"metadata":{}})).unwrap()
            }))
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let tools = IntegrationTools::new(client(&endpoint), incus(&endpoint))
        .unwrap()
        .with_approval(Arc::new(Permit));
    let r = tools
        .execute(&ctx(), &call("incus_file_read", json!({"grant":"read"})))
        .await
        .unwrap();
    assert_eq!(r["data"]["content"], "guest text");
    tools
        .execute(&ctx(), &call("incus_operation", json!({"grant":"write"})))
        .await
        .unwrap();
    let requests = seen.lock().unwrap();
    assert_eq!(
        requests[0].1,
        "/1.0/instances/vm/files?project=lab&path=%2Fetc%2Freviewed.txt"
    );
    assert_eq!(requests[1].2, b"pinned");
    drop(requests);
    task.abort();
}
