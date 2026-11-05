use std::io::{Read, Write};
fn main() {
    assert_eq!(
        std::env::var("PULSE_SERVER").unwrap(),
        "unix:/isolated-test-only"
    );
    assert!(std::env::var("DISPLAY").is_err());
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--format=json") {
        println!(
            r#"[{{"name":"virtual","description":"isolated fixture","mute":false,"volume":{{}},"state":"IDLE","driver":"module-null-sink.c"}},{{"name":"ungranted","description":"must not leak"}}]"#
        );
    } else if args.first().map(String::as_str) == Some("--device") {
        assert_eq!(args[1], "virtual");
        assert!(args.iter().any(|a| a == "--raw"));
        if args.iter().any(|a| a == "--volume") {
            let mut bytes = Vec::new();
            std::io::stdin().read_to_end(&mut bytes).unwrap();
            assert!(!bytes.is_empty());
        } else {
            std::io::stdout().write_all(&[0, 1].repeat(16000)).unwrap();
        }
    } else {
        assert!(
            args == ["set-sink-volume", "virtual", "20%"]
                || args == ["set-sink-volume", "virtual", "40%"]
        );
    }
}
