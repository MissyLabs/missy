use missy_voice::{AudioFormat, CommandConfig, CommandStt, CommandTts, Stt, Tts};
use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, time::Duration};
use tokio_util::sync::CancellationToken;

fn compile_fixture(dir: &tempfile::TempDir) -> PathBuf {
    let source = dir.path().join("fixture.rs");
    let executable = dir.path().join("native-speech-fixture");
    fs::write(&source, include_str!("fixtures/native_engine.rs")).unwrap();
    assert!(
        std::process::Command::new("rustc")
            .arg("--edition=2024")
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .status()
            .unwrap()
            .success()
    );
    executable
}

#[tokio::test]
async fn native_process_adapters_use_private_files_exact_argv_and_stdin() {
    let dir = tempfile::tempdir().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let executable = compile_fixture(&dir);
    let config = CommandConfig {
        executable: executable.clone(),
        args: vec!["stt".into(), "{output}".into(), "{input}".into()],
        timeout_seconds: 5,
    };
    let stt = CommandStt::new(config).unwrap();
    let transcript = stt
        .transcribe(
            &[0, 1, 2, 3],
            AudioFormat {
                sample_rate: 16000,
                channels: 1,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(transcript, "real child process transcript");
    let tts = CommandTts::new(CommandConfig {
        executable,
        args: vec!["tts".into(), "{output}".into()],
        timeout_seconds: 5,
    })
    .unwrap();
    let speech = tts
        .synthesize("text ; $(never a command)", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(speech.sample_rate, 16000);
    assert_eq!(&speech.wav[44..], &[0, 1, 2, 3]);
    assert!(
        CommandStt::new(CommandConfig {
            executable: "/bin/sh".into(),
            args: vec!["{input}".into(), "{output}".into()],
            timeout_seconds: 5
        })
        .is_err()
    );
    assert!(
        CommandTts::new(CommandConfig {
            executable: "piper".into(),
            args: vec!["{output}".into()],
            timeout_seconds: 5
        })
        .is_err()
    );
}

#[tokio::test]
async fn cancelled_native_process_group_is_cleaned_and_temp_files_removed() {
    let dir = tempfile::tempdir().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let executable = compile_fixture(&dir);
    let signal = dir.path().join("fixture-pids");
    let tts = CommandTts::new(CommandConfig {
        executable,
        args: vec![
            "hang".into(),
            "{output}".into(),
            signal.to_string_lossy().into_owned(),
        ],
        timeout_seconds: 5,
    })
    .unwrap();
    let cancellation = CancellationToken::new();
    let cancel = cancellation.clone();
    let job = tokio::spawn(async move { tts.synthesize("cancel", cancellation).await });
    // Inspect only explicitly spawned fixture pids, never host processes/devices.
    let mut pids = vec![];
    let mut output = None;
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Ok(text) = fs::read_to_string(&signal) {
                let mut lines = text.lines();
                let ids: Vec<u32> = lines
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if let Some(path) = lines.next().filter(|_| ids.len() == 2) {
                    pids = ids;
                    output = Some(PathBuf::from(path));
                }
            }
            if output.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    cancel.cancel();
    assert!(job.await.unwrap().is_err());
    assert!(!output.unwrap().parent().unwrap().exists());
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let all_dead =
                pids.iter().all(
                    |pid| match fs::read_to_string(format!("/proc/{pid}/stat")) {
                        Ok(stat) => stat
                            .split(')')
                            .nth(1)
                            .is_some_and(|rest| rest.trim_start().starts_with('Z')),
                        Err(_) => true,
                    },
                );
            if all_dead {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}
