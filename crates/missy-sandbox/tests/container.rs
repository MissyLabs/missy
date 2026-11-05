use missy_core::PolicyConfig;
use missy_sandbox::{ContainerConfig, SandboxConfig, build_container_command};
use std::path::{Path, PathBuf};
fn config() -> ContainerConfig {
    ContainerConfig {
        runtime: PathBuf::from("/approved/mock-docker"),
        image: format!("example/missy@sha256:{}", "a".repeat(64)),
        uid: 1000,
        gid: 1000,
        pids_limit: 32,
        memory_bytes: 128 * 1024 * 1024,
        cpus: 1,
    }
}
#[test]
fn hardened_argv_only_no_runtime_execution() {
    let temp = tempfile::tempdir().unwrap();
    let read = temp.path().join("read");
    let write = temp.path().join("write");
    std::fs::create_dir(&read).unwrap();
    std::fs::create_dir(&write).unwrap();
    let argv = vec![
        "/usr/bin/tool".into(),
        "--literal".into(),
        "a;not-a-shell".into(),
    ];
    let policy = PolicyConfig {
        shell_enabled: true,
        allowed_commands: vec![argv.clone()],
        read_roots: vec![read.clone()],
        write_roots: vec![write.clone()],
        ..Default::default()
    };
    let sandbox = SandboxConfig {
        read_roots: vec![read.clone()],
        write_roots: vec![write.clone()],
        ..Default::default()
    };
    let cfg = config();
    let plan =
        build_container_command(&policy, &sandbox, &cfg, &cfg.runtime, &argv, false).unwrap();
    for flag in [
        "--rm",
        "--pull=never",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--read-only",
        "--pids-limit",
        "--memory",
        "--cpus",
        "--entrypoint",
    ] {
        assert!(plan.args.iter().any(|arg| arg == flag), "missing {flag}");
    }
    assert_eq!(&plan.args[plan.args.len() - 2..], &argv[1..]);
    assert_eq!(
        plan.cleanup_args,
        vec![
            "rm".to_owned(),
            "--force".into(),
            "--".into(),
            plan.name.clone()
        ]
    );
    let other =
        build_container_command(&policy, &sandbox, &cfg, &cfg.runtime, &argv, false).unwrap();
    assert_ne!(plan.name, other.name);
    assert!(plan.args.contains(&format!(
        "type=bind,src={},dst={},bind-propagation=rprivate,readonly,bind-recursive=readonly",
        read.display(),
        read.display()
    )));
    let mut changed = cfg.clone();
    changed.image = "example/missy:latest".into();
    assert!(
        build_container_command(&policy, &sandbox, &changed, &cfg.runtime, &argv, false).is_err()
    );
    changed = cfg.clone();
    changed.uid = 0;
    assert!(
        build_container_command(&policy, &sandbox, &changed, &cfg.runtime, &argv, false).is_err()
    );
    assert!(
        build_container_command(
            &policy,
            &sandbox,
            &cfg,
            Path::new("/different/runtime"),
            &argv,
            false
        )
        .is_err()
    );
    assert!(
        build_container_command(
            &policy,
            &sandbox,
            &cfg,
            &cfg.runtime,
            &["/not-approved".into()],
            false
        )
        .is_err()
    );
    std::fs::write(read.join("docker.sock"), "not-even-a-socket").unwrap();
    assert!(build_container_command(&policy, &sandbox, &cfg, &cfg.runtime, &argv, false).is_err());
}
#[test]
fn rejects_socket_mounts_protected_roots() {
    let temp = tempfile::tempdir().unwrap();
    let argv = vec!["/usr/bin/tool".into()];
    let cfg = config();
    for root in [temp.path(), Path::new("/"), Path::new("/run")] {
        let policy = PolicyConfig {
            shell_enabled: true,
            allowed_commands: vec![argv.clone()],
            read_roots: vec![root.to_owned()],
            ..Default::default()
        };
        let sandbox = SandboxConfig {
            read_roots: vec![root.to_owned()],
            ..Default::default()
        };
        let _socket = std::os::unix::net::UnixListener::bind(
            temp.path().join(format!("sock-{}", uuid::Uuid::new_v4())),
        )
        .unwrap();
        assert!(
            build_container_command(&policy, &sandbox, &cfg, &cfg.runtime, &argv, false).is_err()
        );
    }
}
