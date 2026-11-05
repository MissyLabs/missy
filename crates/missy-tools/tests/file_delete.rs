use missy_core::{PolicyConfig, RuntimeConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_store::Store;
use missy_tools::BuiltinTools;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{os::unix::fs::PermissionsExt, sync::Arc};
use tokio_util::sync::CancellationToken;

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
fn setup(enabled: bool) -> (tempfile::TempDir, BuiltinTools) {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let store = Arc::new(Store::open(tmp.path().join("store.db")).unwrap());
    std::fs::create_dir(tmp.path().join("root")).unwrap();
    let tools = BuiltinTools::new(
        PolicyConfig {
            allow_file_delete: enabled,
            write_roots: vec![tmp.path().join("root")],
            max_file_bytes: 64,
            ..Default::default()
        },
        RuntimeConfig::default(),
        store,
    )
    .unwrap();
    (tmp, tools)
}
fn call(arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: "call".into(),
        name: "file_delete".into(),
        arguments,
    }
}
fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[tokio::test]
async fn file_delete_opt_in_registration_schema_and_dispatch() {
    let (tmp, disabled) = setup(false);
    let path = tmp.path().join("root/file");
    std::fs::write(&path, b"safe").unwrap();
    assert!(
        !disabled
            .definitions()
            .iter()
            .any(|d| d.name == "file_delete")
    );
    assert!(
        disabled
            .execute(
                &context(),
                &call(json!({"path":path,"expected_sha256":digest(b"safe")}))
            )
            .await
            .is_err()
    );
    assert!(path.exists());
    let (tmp, tools) = setup(true);
    let definition = tools
        .definitions()
        .into_iter()
        .find(|d| d.name == "file_delete")
        .unwrap();
    assert!(definition.side_effect);
    assert_eq!(definition.parameters["additionalProperties"], false);
    assert_eq!(
        definition.parameters["required"],
        json!(["path", "expected_sha256"])
    );
    let path = tmp.path().join("root/file");
    std::fs::write(&path, b"safe").unwrap();
    let result = tools
        .execute(
            &context(),
            &call(json!({"path":path,"expected_sha256":digest(b"safe")})),
        )
        .await
        .unwrap();
    assert_eq!(result, json!({"trust":"untrusted","data":{"deleted":true}}));
    assert!(!path.exists());
    assert_eq!(
        std::fs::read_dir(tmp.path().join("root")).unwrap().count(),
        0
    );
}

#[tokio::test]
async fn file_delete_invalid_or_cancelled_calls_never_mutate_file() {
    let (tmp, tools) = setup(true);
    let path = tmp.path().join("root/file");
    std::fs::write(&path, b"safe").unwrap();
    for args in [
        json!({"path":path}),
        json!({"expected_sha256":digest(b"safe")}),
        json!({"path":path,"expected_sha256":"bad"}),
        json!({"path":path,"expected_sha256":digest(b"wrong")}),
        json!({"path":path,"expected_sha256":digest(b"safe"),"recursive":true}),
        json!({"path":path,"expected_sha256":false}),
    ] {
        assert!(tools.execute(&context(), &call(args)).await.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"safe");
        assert_eq!(
            std::fs::read_dir(tmp.path().join("root")).unwrap().count(),
            1
        );
    }
    let ctx = context();
    ctx.cancellation.cancel();
    assert!(
        tools
            .execute(
                &ctx,
                &call(json!({"path":path,"expected_sha256":digest(b"safe")}))
            )
            .await
            .is_err()
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"safe");
}
