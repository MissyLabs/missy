#![cfg(unix)]
//! Local protocol fixture only: no provider, credentials, downloads or inference.
use missy_core::{Message, Provider};
use missy_provider_process::{AcpAdapterPin, CliProvider, CliProviderConfig, CliProviderKind};
use sha2::{Digest, Sha256};
use std::{fs, os::unix::fs::PermissionsExt, time::Duration};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

const FIXTURE: &str = r#"
const fs = require('node:fs');
const readline = require('node:readline');
const assert = require('node:assert/strict');
assert.equal(process.env.OPENAI_API_KEY, undefined);
assert.equal(process.env.NODE_OPTIONS, undefined);
assert.equal(process.env.HTTP_PROXY, undefined);
assert.equal(fs.statSync(process.cwd()).mode & 511, 448);
assert.equal(process.argv.length, 2);
const emit = v => process.stdout.write(JSON.stringify({jsonrpc:'2.0', ...v})+'\n');
const update = v => emit({method:'session/update', params:{sessionId:'fixture', update:v}});
readline.createInterface({input:process.stdin}).on('line', line => {
 const m = JSON.parse(line);
 if (m.method === 'initialize') {
   assert.equal(m.params.clientCapabilities.terminal, false);
   assert.equal(m.params.clientCapabilities.fs.readTextFile, false);
   emit({id:1,result:{protocolVersion:1,agentInfo:{name:'@agentclientprotocol/claude-agent-acp',version:'0.93.0'}}});
 } else if (m.method === 'session/new') {
   assert.deepEqual(m.params.mcpServers, []);
   assert.equal(m.params._meta.disableBuiltInTools, true);
   assert.ok(m.params._meta.systemPrompt.includes('Be precise'));
   const o=m.params._meta.claudeCode.options;
   assert.deepEqual(o.tools,[]);
   assert.deepEqual(o.settingSources,[]);
   assert.deepEqual(o.mcpServers,{});
   assert.equal(o.strictMcpConfig,true);
   assert.equal(o.settings.disableAllHooks,true);
   assert.equal(o.persistSession,false);
   assert.equal(o.maxTurns,1);
   assert.equal(o.extraArgs['disable-slash-commands'],'');
   emit({id:2,result:{sessionId:'fixture'}});
 } else if (m.method === 'session/set_config_option') {
   assert.equal(m.params.value,'fixture-model');
   emit({id:3,result:{configOptions:[{id:'model',currentValue:'fixture-model'}]}});
 } else if (m.method === 'session/prompt') {
   assert.equal(m.params.sessionId,'fixture');
   assert.equal(JSON.parse(m.params.prompt[0].text).conversation[0].content,'hello');
   ACTION
 } else if (m.id === 'permission') {
   assert.equal(m.result.outcome.outcome,'cancelled');
   fs.writeFileSync(process.env.HOME+'/denied','true');
   update({sessionUpdate:'agent_message_chunk',content:{type:'text',text:'must not succeed'}});
   emit({id:4,result:{stopReason:'end_turn'}});
 } else if (m.method === 'session/cancel') {
   fs.writeFileSync(process.env.HOME+'/cancelled','true');
 } else { throw Error('unexpected protocol'); }
});
"#;

fn fixture(action: &str) -> (TempDir, CliProviderConfig) {
    let directory = tempfile::tempdir().unwrap();
    fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let entrypoint = directory.path().join("adapter.cjs");
    let source = FIXTURE.replace("ACTION", action);
    fs::write(&entrypoint, &source).unwrap();
    let config = CliProviderConfig {
        name: "acp-fixture".into(),
        kind: CliProviderKind::Acpx,
        executable: "/usr/bin/node".into(),
        auth_home: directory.path().to_owned(),
        model: "fixture-model".into(),
        approved_executable: true,
        acp_adapter: Some(AcpAdapterPin {
            entrypoint,
            sha256: hex::encode(Sha256::digest(source.as_bytes())),
        }),
        timeout_seconds: 2,
        max_stdout_bytes: 1024 * 1024,
        max_stderr_bytes: 4096,
        max_input_bytes: 1024 * 1024,
        input_microusd_per_million: None,
        output_microusd_per_million: None,
    };
    (directory, config)
}
fn messages() -> Vec<Message> {
    vec![
        Message::text("system", "Be precise"),
        Message::text("user", "hello"),
    ]
}
const ANSWER: &str = "update({sessionUpdate:'agent_thought_chunk',content:{type:'text',text:'private'}}); update({sessionUpdate:'agent_message_chunk',content:{type:'text',text:'safe reply'}}); emit({id:4,result:{stopReason:'end_turn'}});";

#[tokio::test]
async fn native_acp_handshake_pinned_no_tools_options_and_text_only_reply() {
    let (_dir, config) = fixture(ANSWER);
    let reply = CliProvider::new(config)
        .unwrap()
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply.message.content, "safe reply");
    assert_eq!(reply.finish_reason, "stop");
    assert!(!reply.usage.cost_known);
}

#[tokio::test]
async fn acp_denies_even_allow_only_permission_options_then_rejects_completion() {
    let (dir, config) = fixture(
        "emit({id:'permission',method:'session/request_permission',params:{sessionId:'fixture',options:[{kind:'allow_once',optionId:'allow'}]}});",
    );
    let error = CliProvider::new(config)
        .unwrap()
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unsafe protocol"));
    assert_eq!(
        fs::read_to_string(dir.path().join("denied")).unwrap(),
        "true"
    );
}

#[tokio::test]
async fn acp_rejects_tool_events_fs_requests_wrong_session_model_and_stop() {
    for action in [
        "update({sessionUpdate:'tool_call',toolCallId:'bad',title:'secret'});",
        "emit({id:'bad',method:'fs/read_text_file',params:{sessionId:'fixture',path:'/etc/passwd'}});",
        "emit({method:'session/update',params:{sessionId:'other',update:{sessionUpdate:'agent_message_chunk',content:{type:'text',text:'secret'}}}});",
        "update({sessionUpdate:'config_option_update',configOptions:[{id:'model',currentValue:'other'}]});",
        "update({sessionUpdate:'agent_message_chunk',content:{type:'text',text:'secret'}});emit({id:4,result:{stopReason:'tool_use'}});",
        "emit({id:4,error:{code:-32603,message:'secret'}});",
        "emit({id:4,result:{stopReason:'end_turn'}});",
    ] {
        let (_dir, config) = fixture(action);
        let e = CliProvider::new(config)
            .unwrap()
            .complete(&messages(), &[], CancellationToken::new())
            .await
            .unwrap_err();
        assert!(e.to_string().contains("unsafe protocol"), "{e}");
        assert!(!e.to_string().contains("secret"));
    }
}

#[tokio::test]
async fn adapter_hash_rechecked_before_each_completion_and_version_is_exact() {
    let (_dir, config) = fixture(ANSWER);
    let provider = CliProvider::new(config.clone()).unwrap();
    fs::write(&config.acp_adapter.as_ref().unwrap().entrypoint, "changed").unwrap();
    assert!(
        provider
            .complete(&messages(), &[], CancellationToken::new())
            .await
            .unwrap_err()
            .to_string()
            .contains("SHA-256")
    );
    let (_dir, mut config) = fixture(ANSWER);
    let entry = &config.acp_adapter.as_ref().unwrap().entrypoint;
    let source = fs::read_to_string(entry)
        .unwrap()
        .replace("version:'0.93.0'", "version:'0.94.0'");
    fs::write(entry, &source).unwrap();
    config.acp_adapter.as_mut().unwrap().sha256 = hex::encode(Sha256::digest(source.as_bytes()));
    assert!(
        CliProvider::new(config)
            .unwrap()
            .complete(&messages(), &[], CancellationToken::new())
            .await
            .unwrap_err()
            .to_string()
            .contains("unsafe protocol")
    );
}

#[tokio::test]
async fn acp_timeout_cancellation_overflow_and_auth_refuse_boundedly() {
    let (_dir, mut config) = fixture("setInterval(()=>{},1000);");
    config.timeout_seconds = 1;
    let provider = CliProvider::new(config.clone()).unwrap();
    let e = tokio::time::timeout(
        Duration::from_secs(3),
        provider.complete(&messages(), &[], CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(e.to_string().contains("timed out"));
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(
        provider
            .complete(&messages(), &[], cancel)
            .await
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    assert!(
        CliProvider::probe_auth(config, CancellationToken::new())
            .await
            .is_err()
    );
    let (_dir, mut config) = fixture("process.stdout.write('x'.repeat(10000));");
    config.max_stdout_bytes = 1000;
    assert!(
        CliProvider::new(config)
            .unwrap()
            .complete(&messages(), &[], CancellationToken::new())
            .await
            .unwrap_err()
            .to_string()
            .contains("output limit")
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn acp_inflight_cancel_and_future_abort_reap_children() {
    for abort in [false, true] {
        let (dir, config) = fixture(
            "fs.writeFileSync(process.env.HOME+'/pid',String(process.pid));setInterval(()=>{},1000);",
        );
        let provider = CliProvider::new(config).unwrap();
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let task = tokio::spawn(async move { provider.complete(&messages(), &[], token).await });
        let pid_path = dir.path().join("pid");
        tokio::time::timeout(Duration::from_secs(2), async {
            while !pid_path.exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let pid = fs::read_to_string(pid_path).unwrap();
        if abort {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        } else {
            cancel.cancel();
            assert!(
                task.await
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("cancelled")
            );
        }
        assert!(
            !std::path::Path::new(&format!("/proc/{pid}")).exists(),
            "ACP child must be reaped"
        );
    }
}
