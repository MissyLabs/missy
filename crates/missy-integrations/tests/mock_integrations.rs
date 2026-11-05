use anyhow::Result;
use async_trait::async_trait;
use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request, State},
    http::StatusCode,
    response::Response,
};
use futures_util::{SinkExt, StreamExt};
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_integrations::*;
use missy_policy::PolicyHttpClient;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::{
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};
use tokio_tungstenite::{WebSocketStream, accept_async, tungstenite::Message};
use tokio_util::sync::CancellationToken;

struct Permit;
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
        id: "call-1".into(),
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
fn client(endpoint: &str, private: bool) -> PolicyHttpClient {
    PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![endpoint.replace("ws://", "http://")],
        allow_private_network: private,
        ..Default::default()
    })
    .unwrap()
}
fn nomad(endpoint: &str) -> IntegrationConfig {
    IntegrationConfig {
        nomad: Some(NomadConfig {
            connection: conn(endpoint),
            namespace: "lab".into(),
            jobs: BTreeSet::from(["worker".into()]),
            deployments: BTreeMap::from([(
                "worker".into(),
                json!({"ID":"worker","Namespace":"lab","TaskGroups":[]}),
            )]),
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn comfy(endpoint: &str) -> IntegrationConfig {
    IntegrationConfig {
        comfyui: Some(ComfyConfig {
            connection: conn(endpoint),
            max_tracked_jobs: 10,
            allowed_classes: BTreeSet::from(["CLIPTextEncode".into(), "SaveImage".into()]),
            workflows: BTreeMap::from([(
                "image".into(),
                Workflow {
                    graph: json!({"1":{"class_type":"CLIPTextEncode","inputs":{"text":"operator prompt"}},"2":{"class_type":"SaveImage","inputs":{"images":["1",0],"filename_prefix":"safe"}}}),
                    prompt_node: Some("1".into()),
                },
            )]),
            ..Default::default()
        }),
        ..Default::default()
    }
}
type RecordedRequest = (String, String, String, Value);
#[derive(Clone)]
struct Mock {
    seen: Arc<Mutex<Vec<RecordedRequest>>>,
    response: Arc<Mutex<Value>>,
    status: StatusCode,
}
struct Server {
    endpoint: String,
    task: JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn http(response: Value, status: StatusCode) -> (Server, Mock) {
    let state = Mock {
        seen: Default::default(),
        response: Arc::new(Mutex::new(response)),
        status,
    };
    async fn handler(State(s): State<Mock>, req: Request) -> Response {
        let method = req.method().to_string();
        let uri = req.uri().to_string();
        let auth = req
            .headers()
            .get("x-nomad-token")
            .or_else(|| req.headers().get("authorization"))
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let body = to_bytes(req.into_body(), 1_000_000).await.unwrap();
        let body = if body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&body).unwrap()
        };
        s.seen.lock().unwrap().push((method, uri, auth, body));
        Response::builder()
            .status(s.status)
            .header("content-type", "application/json")
            .body(Body::from(s.response.lock().unwrap().to_string()))
            .unwrap()
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().fallback(handler).with_state(state.clone());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (Server { endpoint, task }, state)
}

#[test]
fn defaults_and_config_are_fail_closed() {
    let tools = IntegrationTools::new(
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        IntegrationConfig::default(),
    )
    .unwrap();
    assert!(tools.definitions().is_empty());
    assert!(
        serde_json::from_value::<IntegrationConfig>(json!({"comfyui_host":"localhost"})).is_err()
    );
    for endpoint in [
        "http://user:password@example.com",
        "http://example.com/api",
        "http://example.com?token=bad",
        "file:///tmp/test",
        "ws://example.com",
    ] {
        assert!(
            IntegrationTools::new(
                PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
                nomad(endpoint)
            )
            .is_err()
        );
    }
    let mut config = nomad("http://127.0.0.1:1");
    config.nomad.as_mut().unwrap().connection.token_env = Some("BAD-NAME".into());
    assert!(
        IntegrationTools::new(
            PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
            config
        )
        .is_err()
    );
}
#[tokio::test]
async fn nomad_http_auth_projection_routes_approval() {
    let (server, state) = http(
        json!([{"ID":"worker","Name":"worker","Env":{"SECRET":"never returned"}},{"ID":"private"}]),
        StatusCode::OK,
    )
    .await;
    let mut config = nomad(&server.endpoint);
    config.nomad.as_mut().unwrap().connection.token_env = Some("USER".into());
    let tools = IntegrationTools::new(client(&server.endpoint, true), config).unwrap();
    assert!(
        tools
            .definitions()
            .iter()
            .find(|d| d.name == "nomad_mutate")
            .unwrap()
            .side_effect
    );
    let r = tools
        .execute(&ctx(), &call("nomad_read", json!({"action":"list"})))
        .await
        .unwrap();
    assert_eq!(r["data"].as_array().unwrap().len(), 1);
    assert!(!r.to_string().contains("never returned"));
    {
        let seen = state.seen.lock().unwrap();
        let req = &seen[0];
        assert_eq!(req.0, "GET");
        assert_eq!(req.1, "/v1/jobs?namespace=lab");
        assert_eq!(req.2, std::env::var("USER").unwrap());
    }
    assert!(
        tools
            .execute(
                &ctx(),
                &call("nomad_mutate", json!({"action":"stop","job":"worker"}))
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("approval")
    );
    let tools = tools.with_approval(Arc::new(Permit));
    *state.response.lock().unwrap() = json!({"EvalID":"evaluation","EvalCreateIndex":3});
    tools
        .execute(
            &ctx(),
            &call("nomad_mutate", json!({"action":"deploy","job":"worker"})),
        )
        .await
        .unwrap();
    tools
        .execute(
            &ctx(),
            &call("nomad_mutate", json!({"action":"stop","job":"worker"})),
        )
        .await
        .unwrap();
    let seen = state.seen.lock().unwrap();
    assert_eq!(seen[1].0, "POST");
    assert_eq!(seen[1].3["Job"]["ID"], "worker");
    assert_eq!(seen[2].0, "DELETE");
    assert_eq!(seen[2].1, "/v1/job/worker?namespace=lab&purge=false");
}
#[tokio::test]
async fn denied_resources_strict_arguments_and_routes_never_network() {
    let (server, state) = http(json!({}), StatusCode::OK).await;
    let tools = IntegrationTools::new(client(&server.endpoint, true), nomad(&server.endpoint))
        .unwrap()
        .with_approval(Arc::new(Permit));
    for c in [
        call("nomad_read", json!({"action":"status","job":"../private"})),
        call("nomad_read", json!({"action":"status","job":"private"})),
        call(
            "nomad_read",
            json!({"action":"list","endpoint":"http://evil"}),
        ),
        call(
            "nomad_mutate",
            json!({"action":"deploy","job":"worker","Job":{"ID":"evil"}}),
        ),
        call(
            "nomad_read",
            json!({"action":"call","route":"/v1/agent/self"}),
        ),
    ] {
        assert!(tools.execute(&ctx(), &c).await.is_err());
    }
    assert!(state.seen.lock().unwrap().is_empty());
}
#[tokio::test]
async fn http_policy_denial_output_limit_and_secret_errors() {
    let (server, state) = http(
        json!({"ID":"worker","StatusDescription":"x".repeat(2000)}),
        StatusCode::OK,
    )
    .await;
    let tools = IntegrationTools::new(
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        nomad(&server.endpoint),
    )
    .unwrap();
    assert!(
        tools
            .execute(&ctx(), &call("nomad_read", json!({"action":"list"})))
            .await
            .is_err()
    );
    assert!(state.seen.lock().unwrap().is_empty());
    let tools =
        IntegrationTools::new(client(&server.endpoint, false), nomad(&server.endpoint)).unwrap();
    assert!(
        tools
            .execute(&ctx(), &call("nomad_read", json!({"action":"list"})))
            .await
            .is_err()
    );
    assert!(state.seen.lock().unwrap().is_empty());
    let mut config = nomad(&server.endpoint);
    config.nomad.as_mut().unwrap().connection.max_output_bytes = 128;
    let tools = IntegrationTools::new(client(&server.endpoint, true), config).unwrap();
    assert!(
        tools
            .execute(
                &ctx(),
                &call("nomad_read", json!({"action":"status","job":"worker"}))
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("bound")
    );
    let (server, _) = http(
        json!({"error":"do not include this secret in error"}),
        StatusCode::UNAUTHORIZED,
    )
    .await;
    let tools =
        IntegrationTools::new(client(&server.endpoint, true), nomad(&server.endpoint)).unwrap();
    let error = tools
        .execute(&ctx(), &call("nomad_read", json!({"action":"list"})))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("401"));
    assert!(!error.contains("secret"));
}
#[tokio::test]
async fn incus_native_lifecycle_only() {
    let (server,state)=http(json!({"type":"sync","metadata":[{"name":"vm","status":"Running","config":{"secret":"not returned"}},{"name":"private"}]}),StatusCode::OK).await;
    let config = IntegrationConfig {
        incus: Some(IncusConfig {
            connection: conn(&server.endpoint),
            project: "lab".into(),
            instances: BTreeSet::from(["vm".into()]),
            actions: BTreeSet::from([InstanceAction::Stop]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let tools = IntegrationTools::new(client(&server.endpoint, true), config)
        .unwrap()
        .with_approval(Arc::new(Permit));
    let r = tools
        .execute(&ctx(), &call("incus_read", json!({"action":"list"})))
        .await
        .unwrap();
    assert_eq!(r["data"].as_array().unwrap().len(), 1);
    assert!(!r.to_string().contains("not returned"));
    *state.response.lock().unwrap() = json!({"type":"async","status":"Operation created","operation":"/1.0/operations/test","error_code":0});
    tools
        .execute(
            &ctx(),
            &call("incus_mutate", json!({"action":"stop","instance":"vm"})),
        )
        .await
        .unwrap();
    for action in ["delete", "exec", "restart"] {
        assert!(
            tools
                .execute(
                    &ctx(),
                    &call("incus_mutate", json!({"action":action,"instance":"vm"}))
                )
                .await
                .is_err()
        );
    }
    let seen = state.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].0, "PUT");
    assert_eq!(seen[1].1, "/1.0/instances/vm/state?project=lab");
    assert_eq!(seen[1].3["force"], false);
}
#[test]
fn comfy_validates_operator_graph_class_links_cycles_and_prompt_binding() {
    let gateway = client("http://127.0.0.1:1", true);
    for graph in [
        json!({"1":{"class_type":"ExecuteShell","inputs":{}}}),
        json!({"1":{"class_type":"SaveImage","inputs":{"images":["missing",0]}}}),
        json!({"1":{"class_type":"SaveImage","inputs":{"images":["2",0]}},"2":{"class_type":"SaveImage","inputs":{"images":["1",0]}}}),
    ] {
        let mut config = comfy("http://127.0.0.1:1");
        config
            .comfyui
            .as_mut()
            .unwrap()
            .workflows
            .get_mut("image")
            .unwrap()
            .graph = graph;
        assert!(IntegrationTools::new(gateway.clone(), config).is_err());
    }
}
#[tokio::test]
async fn comfy_submit_history_scope_and_artifact_path_validation() {
    let (server, state) = http(
        json!({"prompt_id":"prompt-1","node_errors":{}}),
        StatusCode::OK,
    )
    .await;
    let tools = IntegrationTools::new(client(&server.endpoint, true), comfy(&server.endpoint))
        .unwrap()
        .with_approval(Arc::new(Permit));
    let r = tools
        .execute(
            &ctx(),
            &call(
                "comfyui_submit",
                json!({"workflow":"image","prompt":"new prompt"}),
            ),
        )
        .await
        .unwrap();
    assert_eq!(r["data"]["prompt_id"], "prompt-1");
    assert_eq!(
        state.seen.lock().unwrap()[0].3["prompt"]["1"]["inputs"]["text"],
        "new prompt"
    );
    *state.response.lock().unwrap() = json!({"prompt-1":{"status":{"completed":true,"status_str":"success"},"outputs":{"2":{"images":[{"filename":"safe.png","subfolder":"images/test","type":"output"}]}}}});
    let r = tools
        .execute(
            &ctx(),
            &call("comfyui_history", json!({"prompt_id":"prompt-1"})),
        )
        .await
        .unwrap();
    assert_eq!(r["data"]["artifacts"][0]["filename"], "safe.png");
    assert_eq!(r["data"]["downloaded"], false);
    let mut other = ctx();
    other.scope.user = "other".into();
    assert!(
        tools
            .execute(
                &other,
                &call("comfyui_history", json!({"prompt_id":"prompt-1"}))
            )
            .await
            .is_err()
    );
    state.response.lock().unwrap()["prompt-1"]["outputs"]["2"]["images"][0]["filename"] =
        json!("../../passwd");
    assert!(
        tools
            .execute(
                &ctx(),
                &call("comfyui_history", json!({"prompt_id":"prompt-1"}))
            )
            .await
            .is_err()
    );
    for args in [
        json!({"workflow":"image","graph":{}}),
        json!({"workflow":"unknown"}),
    ] {
        assert!(
            tools
                .execute(&ctx(), &call("comfyui_submit", args))
                .await
                .is_err()
        );
    }
    assert!(
        state
            .seen
            .lock()
            .unwrap()
            .iter()
            .all(|r| r.1 == "/prompt" || r.1 == "/history/prompt-1")
    );
}

async fn ws_recv(ws: &mut WebSocketStream<TcpStream>) -> Value {
    let text = ws.next().await.unwrap().unwrap().into_text().unwrap();
    serde_json::from_str(&text).unwrap()
}
async fn ws_send(ws: &mut WebSocketStream<TcpStream>, v: Value) {
    ws.send(Message::Text(v.to_string().into())).await.unwrap()
}
async fn obs_server(auth: bool, bad_id: bool, large: bool) -> (Server, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("ws://{}", listener.local_addr().unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    let count2 = count.clone();
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        count2.fetch_add(1, Ordering::SeqCst);
        let mut ws = accept_async(stream).await.unwrap();
        let mut data = json!({"rpcVersion":1});
        if auth {
            data["authentication"] = json!({"salt":"salt","challenge":"challenge"});
        }
        ws_send(&mut ws, json!({"op":0,"d":data})).await;
        let identify = ws_recv(&mut ws).await;
        assert_eq!(identify["op"], 1);
        assert_eq!(identify["d"]["eventSubscriptions"], 0);
        if auth {
            use base64::{Engine, engine::general_purpose::STANDARD};
            use sha2::{Digest, Sha256};
            let pw = std::env::var("USER").unwrap();
            let secret = STANDARD.encode(Sha256::digest(format!("{pw}salt")));
            let expected = STANDARD.encode(Sha256::digest(format!("{secret}challenge")));
            assert_eq!(identify["d"]["authentication"], expected);
        }
        ws_send(&mut ws, json!({"op":2,"d":{"negotiatedRpcVersion":1}})).await;
        let request = ws_recv(&mut ws).await;
        assert_eq!(request["op"], 6);
        let data = if large {
            json!({"junk":"x".repeat(3000)})
        } else {
            json!({"scenes":[{"sceneName":"Studio","sceneIndex":0},{"sceneName":"Secret","sceneIndex":1}],"currentProgramSceneName":"Studio"})
        };
        ws_send(&mut ws,json!({"op":7,"d":{"requestId":if bad_id{json!("wrong-id")}else{request["d"]["requestId"].clone()},"requestType":request["d"]["requestType"],"requestStatus":{"result":true,"code":100},"responseData":data}})).await;
    });
    (Server { endpoint, task }, count)
}
fn obs(endpoint: &str, auth: bool) -> IntegrationConfig {
    let mut c = conn(endpoint);
    if auth {
        c.token_env = Some("USER".into());
    }
    IntegrationConfig {
        obs: Some(ObsConfig {
            connection: c,
            scenes: BTreeSet::from(["Studio".into()]),
            scene_items: BTreeSet::from(["Studio:12".into()]),
            actions: BTreeSet::from([ObsAction::SwitchScene]),
        }),
        ..Default::default()
    }
}
#[tokio::test]
async fn obs_real_websocket_v5_challenge_and_response_correlation() {
    let (server, _) = obs_server(true, false, false).await;
    let tools =
        IntegrationTools::new(client(&server.endpoint, true), obs(&server.endpoint, true)).unwrap();
    let r = tools
        .execute(&ctx(), &call("obs_read", json!({"action":"list_scenes"})))
        .await
        .unwrap();
    assert_eq!(r["data"]["scenes"].as_array().unwrap().len(), 1);
    assert!(!r.to_string().contains("Secret"));
    let (server, _) = obs_server(false, true, false).await;
    let tools = IntegrationTools::new(client(&server.endpoint, true), obs(&server.endpoint, false))
        .unwrap();
    assert!(
        tools
            .execute(&ctx(), &call("obs_read", json!({"action":"list_scenes"})))
            .await
            .unwrap_err()
            .to_string()
            .contains("correlation")
    );
}
#[tokio::test]
async fn obs_ws_bounds_policy_and_mutation_allowlist() {
    let (server, count) = obs_server(false, false, true).await;
    let tools = IntegrationTools::new(
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        obs(&server.endpoint, false),
    )
    .unwrap();
    assert!(
        tools
            .execute(&ctx(), &call("obs_read", json!({"action":"list_scenes"})))
            .await
            .is_err()
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    let mut config = obs(&server.endpoint, false);
    config.obs.as_mut().unwrap().connection.max_output_bytes = 512;
    let tools = IntegrationTools::new(client(&server.endpoint, true), config)
        .unwrap()
        .with_approval(Arc::new(Permit));
    for a in [
        json!({"action":"start_streaming"}),
        json!({"action":"switch_scene","scene":"Secret"}),
        json!({"action":"set_visibility","scene":"Studio","item_id":13,"visible":true}),
        json!({"action":"call","requestType":"OpenInputPropertiesDialog"}),
    ] {
        assert!(tools.execute(&ctx(), &call("obs_mutate", a)).await.is_err());
    }
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(
        tools
            .execute(&ctx(), &call("obs_read", json!({"action":"list_scenes"})))
            .await
            .is_err()
    );
}
#[tokio::test]
async fn vtube_real_ws_preprovisioned_auth_no_token_popups() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("ws://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let auth = ws_recv(&mut ws).await;
        assert_eq!(auth["messageType"], "AuthenticationRequest");
        assert_eq!(
            auth["data"]["authenticationToken"],
            std::env::var("USER").unwrap()
        );
        ws_send(&mut ws,json!({"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":auth["requestID"],"messageType":"AuthenticationResponse","data":{"authenticated":true}})).await;
        let req = ws_recv(&mut ws).await;
        assert_eq!(req["messageType"], "AvailableModelsRequest");
        ws_send(&mut ws,json!({"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":req["requestID"],"messageType":"AvailableModelsResponse","data":{"availableModels":[{"modelID":"allowed","modelName":"Allowed"},{"modelID":"private"}]}})).await;
    });
    let server = Server { endpoint, task };
    let mut connection = conn(&server.endpoint);
    connection.token_env = Some("USER".into());
    let config = IntegrationConfig {
        vtube: Some(VtubeConfig {
            connection,
            plugin_name: "Missy Test".into(),
            plugin_developer: "Test".into(),
            models: BTreeSet::from(["allowed".into()]),
            parameters: BTreeMap::from([(
                "MouthOpen".into(),
                ParameterRange { min: 0.0, max: 1.0 },
            )]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let tools = IntegrationTools::new(client(&server.endpoint, true), config)
        .unwrap()
        .with_approval(Arc::new(Permit));
    assert!(
        tools
            .execute(
                &ctx(),
                &call(
                    "vtube_mutate",
                    json!({"action":"set_parameter","parameter":"MouthOpen","value":2.0})
                )
            )
            .await
            .is_err()
    );
    assert!(
        tools
            .execute(
                &ctx(),
                &call(
                    "vtube_mutate",
                    json!({"action":"load_model","model":"private"})
                )
            )
            .await
            .is_err()
    );
    let r = tools
        .execute(&ctx(), &call("vtube_read", json!({"action":"list_models"})))
        .await
        .unwrap();
    assert_eq!(r["data"]["models"].as_array().unwrap().len(), 1);
}
#[tokio::test]
async fn cancellation_and_config_optin_deny_before_connect() {
    let (server, state) = http(json!({}), StatusCode::OK).await;
    let mut config = nomad(&server.endpoint);
    config.nomad.as_mut().unwrap().connection.allow_mutations = false;
    let tools = IntegrationTools::new(client(&server.endpoint, true), config)
        .unwrap()
        .with_approval(Arc::new(Permit));
    assert!(tools.definitions().iter().all(|d| !d.side_effect));
    assert!(
        tools
            .execute(
                &ctx(),
                &call("nomad_mutate", json!({"action":"stop","job":"worker"}))
            )
            .await
            .is_err()
    );
    let c = ctx();
    c.cancellation.cancel();
    assert!(
        tools
            .execute(&c, &call("nomad_read", json!({"action":"list"})))
            .await
            .is_err()
    );
    assert!(state.seen.lock().unwrap().is_empty());
}

#[test]
fn fingerprint_binds_endpoint_policy_resources_and_workflows() {
    let endpoint = "http://127.0.0.1:1";
    let config = nomad(endpoint);
    let tools = IntegrationTools::new(client(endpoint, true), config.clone()).unwrap();
    let mut changed = config.clone();
    changed.nomad.as_mut().unwrap().jobs.insert("other".into());
    assert_ne!(
        tools.configuration_fingerprint(),
        IntegrationTools::new(client(endpoint, true), changed)
            .unwrap()
            .configuration_fingerprint()
    );
    assert_ne!(
        tools.configuration_fingerprint(),
        IntegrationTools::new(client(endpoint, false), config)
            .unwrap()
            .configuration_fingerprint()
    );
}

#[tokio::test]
async fn upstream_credentials_redacted_even_in_unexpected_keys_and_errors() {
    let secret = std::env::var("USER").unwrap();
    let (server, _) = http(
        json!([{"ID":"worker","StatusDescription":format!("echo {secret}"),"Name":"worker"}]),
        StatusCode::OK,
    )
    .await;
    let mut config = nomad(&server.endpoint);
    config.nomad.as_mut().unwrap().connection.token_env = Some("USER".into());
    let tools = IntegrationTools::new(client(&server.endpoint, true), config).unwrap();
    let r = tools
        .execute(&ctx(), &call("nomad_read", json!({"action":"list"})))
        .await
        .unwrap();
    assert!(!r.to_string().contains(&secret));
    assert!(r.to_string().contains("[REDACTED]"));
}

#[tokio::test]
async fn websocket_deadline_bounds_stalled_protocol() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("ws://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let _ws = accept_async(stream).await.unwrap();
        std::future::pending::<()>().await;
    });
    let server = Server { endpoint, task };
    let mut config = obs(&server.endpoint, false);
    config.obs.as_mut().unwrap().connection.timeout_seconds = 1;
    let tools = IntegrationTools::new(client(&server.endpoint, true), config).unwrap();
    let start = std::time::Instant::now();
    assert!(
        tools
            .execute(&ctx(), &call("obs_read", json!({"action":"status"})))
            .await
            .unwrap_err()
            .to_string()
            .contains("deadline")
    );
    assert!(start.elapsed() < std::time::Duration::from_secs(3));
}

#[tokio::test]
async fn vtube_rejects_auth_failure_and_correlated_apierror_without_secret_details() {
    for auth_failure in [true, false] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("ws://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();
            let auth = ws_recv(&mut ws).await;
            ws_send(&mut ws,json!({"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":auth["requestID"],"messageType":"AuthenticationResponse","data":{"authenticated":!auth_failure,"reason":"SECRET_AUTH_DETAIL"}})).await;
            if !auth_failure {
                let req = ws_recv(&mut ws).await;
                ws_send(&mut ws,json!({"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":req["requestID"],"messageType":"APIError","data":{"message":"SECRET_API_DETAIL"}})).await;
            }
        });
        let server = Server { endpoint, task };
        let mut connection = conn(&server.endpoint);
        connection.token_env = Some("USER".into());
        let config = IntegrationConfig {
            vtube: Some(VtubeConfig {
                connection,
                plugin_name: "Test".into(),
                plugin_developer: "Test".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tools = IntegrationTools::new(client(&server.endpoint, true), config).unwrap();
        let e = tools
            .execute(&ctx(), &call("vtube_read", json!({"action":"status"})))
            .await
            .unwrap_err()
            .to_string();
        assert!(!e.contains("SECRET"));
    }
}

#[tokio::test]
async fn http_redirect_is_not_followed() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().fallback(|| async {
        Response::builder()
            .status(StatusCode::FOUND)
            .header("location", "http://127.0.0.1:9/private")
            .body(Body::from("SECRET_BODY"))
            .unwrap()
    });
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let server = Server { endpoint, task };
    let tools =
        IntegrationTools::new(client(&server.endpoint, true), nomad(&server.endpoint)).unwrap();
    let e = tools
        .execute(&ctx(), &call("nomad_read", json!({"action":"list"})))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("302"));
    assert!(!e.contains("SECRET_BODY"));
}

#[tokio::test]
async fn malformed_http_json_is_rejected_without_body_details() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().fallback(|| async {
        Response::builder()
            .status(StatusCode::OK)
            .body(Body::from("{SECRET_BODY"))
            .unwrap()
    });
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let server = Server { endpoint, task };
    let tools =
        IntegrationTools::new(client(&server.endpoint, true), nomad(&server.endpoint)).unwrap();
    let e = tools
        .execute(&ctx(), &call("nomad_read", json!({"action":"list"})))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("invalid JSON"));
    assert!(!e.contains("SECRET_BODY"));
}

#[tokio::test]
async fn preflight_definite_refusal_is_typed_and_never_sends() {
    let (server, state) = http(json!({}), StatusCode::OK).await;
    let tools =
        IntegrationTools::new(client(&server.endpoint, true), nomad(&server.endpoint)).unwrap();
    let e = tools
        .preflight(
            &ctx(),
            &call("nomad_mutate", json!({"action":"stop","job":"not-allowed"})),
        )
        .await
        .unwrap_err();
    assert!(
        e.downcast_ref::<missy_core::ToolDispatchRefused>()
            .is_some()
    );
    assert!(state.seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn obs_mutation_uses_exact_allowed_request_and_acknowledges() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("ws://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        ws_send(&mut ws, json!({"op":0,"d":{"rpcVersion":1}})).await;
        ws_recv(&mut ws).await;
        ws_send(&mut ws, json!({"op":2,"d":{"negotiatedRpcVersion":1}})).await;
        let req = ws_recv(&mut ws).await;
        assert_eq!(req["d"]["requestType"], "SetCurrentProgramScene");
        assert_eq!(req["d"]["requestData"], json!({"sceneName":"Studio"}));
        ws_send(&mut ws,json!({"op":7,"d":{"requestId":req["d"]["requestId"],"requestType":"SetCurrentProgramScene","requestStatus":{"code":100,"result":true}}})).await;
    });
    let server = Server { endpoint, task };
    let tools = IntegrationTools::new(client(&server.endpoint, true), obs(&server.endpoint, false))
        .unwrap()
        .with_approval(Arc::new(Permit));
    assert_eq!(
        tools
            .execute(
                &ctx(),
                &call(
                    "obs_mutate",
                    json!({"action":"switch_scene","scene":"Studio"})
                )
            )
            .await
            .unwrap()["data"]["acknowledged"],
        true
    );
}
