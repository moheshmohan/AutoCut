//! Music from the shell: `exec music.suggest` ranks styles for a brief, its params generate the
//! track exactly as long as asked, and hostile briefs exit 1 with a message instead of crashing.

use std::process::{Command, Output};

use serde_json::Value;

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_filmcraft-cli")).args(args).output().expect("spawn filmcraft-cli")
}

fn json_out(o: &Output) -> Value {
    assert!(o.status.success(), "failed: {}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).expect("JSON on stdout")
}

#[test]
fn music_suggest_and_generate_from_the_shell() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/export-tests").join(format!("cli-music-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // key=value params
    let v = json_out(&cli(&["--compact", "exec", "music.suggest", "mood=chill", "duration=20", "count=2"]));
    let s = v["suggestions"].as_array().unwrap();
    assert_eq!(s.len(), 2);
    assert_eq!(s[0]["style"]["mood"], "dreamy");
    // the suggestion's params make the track: 20 s to the sample, ending on a bar
    let mut params = s[0]["params"].clone();
    params["place"] = Value::Bool(false);
    params["dir"] = Value::String(dir.to_string_lossy().into_owned());
    let g = json_out(&cli(&["--compact", "exec", "audio.generate", &params.to_string()]));
    assert_eq!(g["samples"], 960_000);
    assert_eq!(g["mood"], "dreamy");
    assert!(g["downbeats"].as_array().unwrap().contains(&g["ending"]));
    assert!(std::path::Path::new(g["path"].as_str().unwrap()).exists());
    // hostile briefs: exit 1, a message, no crash
    for bad in [r#"{"mood":"jazzy"}"#, r#"{"energy":[]}"#, r#"{"duration":-1}"#, r#"{"count":1000}"#, r#"{"bpm":"fast"}"#] {
        let o = cli(&["exec", "music.suggest", bad]);
        assert_eq!(o.status.code(), Some(1), "{bad}: {}", String::from_utf8_lossy(&o.stderr));
        assert!(String::from_utf8_lossy(&o.stderr).contains("music.suggest"), "{bad}");
    }
    let t = json_out(&cli(&["--compact", "exec", "music.taxonomy"]));
    assert_eq!(t["genres"].as_array().unwrap().len(), 6);
    let _ = std::fs::remove_dir_all(&dir);
}
