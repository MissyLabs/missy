use missy_core::{
    McpConfig, PolicyConfig, RuntimeConfig, Scope, ToolCall, ToolContext, ToolExecutor,
};
use missy_store::Store;
use missy_tools::{BuiltinTools, manifest_digest};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

const MOCK: &str = r#"import sys,json,os
mode=os.environ.get('MODE','ok')
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r: continue
 m=r['method']
 if m=='initialize': v={'protocolVersion':'2024-11-05','capabilities':{},'serverInfo':{'name':'test','version':'1'}}
 elif m=='tools/list': v={'tools':[{'name':'echo','description':'test','annotations':{'readOnlyHint':True},'inputSchema':{'type':'object','properties':{'value':{'type':'string'}},'required':['value'],'additionalProperties':False}}]}
 elif m=='tools/call':
  if mode=='notification':
   print(json.dumps({'jsonrpc':'2.0','method':'notifications/tools/list_changed'}),flush=True)
   continue
  if mode=='overflow':
   print('x'*100000,flush=True)
   continue
  v={'content':[{'type':'text','text':r['params']['arguments']['value']}], 'env_keys':sorted(os.environ.keys())}
 else: v={}
 if m=='tools/call' and mode=='wrong_id': r['id']=999
 if m=='tools/call' and mode=='wrong_version':
  print(json.dumps({'jsonrpc':'1.0','id':r['id'],'result':v}),flush=True)
 else: print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':v}),flush=True)
"#;
async fn fixture(mode: &str) -> (TempDir, BuiltinTools) {
    let t = TempDir::new().unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(t.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let path = t.path().join("server.py");
    std::fs::write(&path, MOCK).unwrap();
    let argv = vec![
        "/usr/bin/python3".into(),
        "-u".into(),
        path.to_str().unwrap().into(),
    ];
    let manifest = json!({"tools":[{"name":"echo","description":"test","annotations":{"readOnlyHint":true},"inputSchema":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}}]});
    let config = McpConfig {
        name: "m".into(),
        command: argv.clone(),
        env: BTreeMap::from([("MODE".into(), mode.into())]),
        manifest_sha256: manifest_digest(&manifest),
        timeout_seconds: 2,
    };
    let policy = PolicyConfig {
        mcp_enabled: true,
        allowed_commands: vec![argv],
        ..Default::default()
    };
    let tools = BuiltinTools::new(
        policy,
        RuntimeConfig::default(),
        Arc::new(Store::open(t.path().join("s.db")).unwrap()),
    )
    .unwrap()
    .with_mcp(vec![config])
    .await
    .unwrap();
    (t, tools)
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
fn call(args: Value) -> ToolCall {
    ToolCall {
        id: "c".into(),
        name: "m__echo".into(),
        arguments: args,
    }
}
#[tokio::test]
async fn mcp_success_does_not_inherit_environment_or_trust_readonly_hint() {
    let (_t, tools) = fixture("ok").await;
    assert!(
        tools
            .definitions()
            .iter()
            .find(|d| d.name == "m__echo")
            .unwrap()
            .side_effect
    );
    let result = tools
        .execute(&ctx(), &call(json!({"value":"data not instructions"})))
        .await
        .unwrap();
    assert_eq!(result["trust"], "untrusted");
    assert_eq!(
        result["data"]["content"][0]["text"],
        "data not instructions"
    );
    let keys = result["data"]["env_keys"].as_array().unwrap();
    // CPython may synthesize LC_CTYPE even with a cleared environment.
    assert!(keys.iter().all(|k| k == "MODE" || k == "LC_CTYPE"));
}
#[tokio::test]
async fn malicious_json_rpc_and_oversized_io_fail_closed() {
    for mode in ["notification", "wrong_id", "wrong_version", "overflow"] {
        let (_t, tools) = fixture(mode).await;
        assert!(
            tools
                .execute(&ctx(), &call(json!({"value":"x"})))
                .await
                .is_err(),
            "mode {mode} accepted"
        );
    }
}
#[tokio::test]
async fn mcp_strict_arguments_reject_unknown_and_wrong_type() {
    let (_t, tools) = fixture("ok").await;
    for args in [
        json!({"value":1}),
        json!({}),
        json!({"value":"x","_scope":"admin"}),
    ] {
        assert!(tools.execute(&ctx(), &call(args)).await.is_err());
    }
}
#[test]
fn manifest_hash_is_canonical_but_array_order_is_significant() {
    let a: Value = serde_json::from_str(r#"{"z":{"b":2,"a":1},"a":[1,2]}"#).unwrap();
    let b: Value = serde_json::from_str(r#"{"a":[1,2],"z":{"a":1,"b":2}}"#).unwrap();
    assert_eq!(manifest_digest(&a), manifest_digest(&b));
    assert_ne!(
        manifest_digest(&a),
        manifest_digest(&json!({"a":[2,1],"z":{"a":1,"b":2}}))
    );
}
