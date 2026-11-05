// Disposable speech-command process fixture, not a real recognizer/synthesizer.
use std::{fs, io::{self, Read}, os::unix::fs::PermissionsExt, path::PathBuf, time::Duration};
fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap();
    let output = PathBuf::from(args.next().unwrap());
    assert!(std::env::var_os("MISSY_VOICE_TEST_SECRET").is_none());
    assert!(std::env::var_os("PATH").is_none());
    assert!(std::env::var_os("HOME").is_none());
    assert_eq!(fs::metadata(std::env::current_dir().unwrap()).unwrap().permissions().mode() & 0o777, 0o700);
    if mode == "stt" {
        let input = args.next().unwrap();
        let wav = fs::read(input).unwrap();
        assert_eq!(fs::metadata(output.with_extension("txt")).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[44..], &[0,1,2,3]);
        fs::write(output.with_extension("txt"), "real child process transcript").unwrap();
    } else if mode == "tts" {
        assert_eq!(fs::metadata(&output).unwrap().permissions().mode() & 0o777, 0o600);
        let mut text = String::new();
        io::stdin().read_to_string(&mut text).unwrap();
        assert_eq!(text.trim(), "text ; $(never a command)");
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&40u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&4u32.to_le_bytes());
        wav.extend_from_slice(&[0,1,2,3]);
        fs::write(output, wav).unwrap();
    } else if mode == "hang" {
        let signal = args.next().unwrap();
        // The fixture is its own worker too, so a child demonstrably shares
        // the adapter-owned group without opening camera/audio devices.
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("worker").arg(&output).spawn().unwrap();
        fs::write(signal, format!("{} {}\n{}", std::process::id(), child.id(), output.display())).unwrap();
        std::thread::sleep(Duration::from_secs(30));
    } else if mode == "worker" {
        std::thread::sleep(Duration::from_secs(30));
    } else { panic!("unknown mode"); }
}
