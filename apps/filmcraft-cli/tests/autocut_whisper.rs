//! AutoCut + Whisper end to end, through the real CLI binary: natural speech (macOS `say`,
//! with deliberate silences) is transcribed by the local Whisper model and edited by an
//! AutoCut prompt.
//!
//! Opt-in — it needs the whisper weights downloaded (a real model download), Ollama running
//! with the classifier model, and macOS `say`:
//!
//! ```sh
//! target/debug/filmcraft-cli exec transcript.downloadModel --demo   # whisper-base
//! AUTOCUT_WHISPER_LIVE=1 cargo test -p filmcraft-cli --test autocut_whisper -- --nocapture
//! ```

use std::path::PathBuf;
use std::process::Command;

use serde_json::{Value, json};

fn cli() -> Command {
    let exe: PathBuf = std::env::var("CARGO_BIN_EXE_filmcraft-cli").map(PathBuf::from).unwrap_or_else(|_| "target/debug/filmcraft-cli".into());
    Command::new(exe)
}

fn output(c: &mut Command) -> (String, String, bool) {
    let out = c.output().expect("spawn filmcraft-cli");
    (String::from_utf8_lossy(&out.stdout).to_string(), String::from_utf8_lossy(&out.stderr).to_string(), out.status.success())
}

/// Natural speech with 900ms silences between the sentences (`[[slnc]]`), via macOS `say`.
fn speech(dir: &std::path::Path) -> Option<PathBuf> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = dir.join("speech.aiff");
    Command::new("say")
        .arg("-o")
        .arg(&out)
        .arg("Hello world. [[slnc 900]] This first sentence is short. [[slnc 900]] And this third sentence talks about video editing, so the transcript has some length to it. [[slnc 900]] Thank you.")
        .status()
        .is_ok_and(|s| s.success())
        .then_some(out)
}

#[test]
fn autocut_removes_silence_from_natural_speech() {
    if std::env::var("AUTOCUT_WHISPER_LIVE").ok().as_deref() != Some("1") {
        return; // needs the whisper model + Ollama; see the module docs
    }
    let dir = std::env::temp_dir().join(format!("autocut-whisper-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let project = dir.join("speech.fcproj");
    let Some(speech) = speech(&dir) else {
        eprintln!("skipping: no macOS `say`");
        return;
    };

    // one process: a fresh sequence with the speech clip, then save
    let script = dir.join("setup.jsonl");
    let setup = concat_lines(&[
        r#"{"id": "file.newProject"}"#,
        r#"{"id": "file.newSequence", "params": {"width": 640, "height": 360}}"#,
        r#"{"id": "file.import", "params": {"paths": ["SPEECH"]}}"#,
        r#"{"id": "source.open", "params": {"item": 8}}"#,
        r#"{"id": "source.overwrite", "params": {}}"#,
    ])
    .replace("SPEECH", &speech.to_string_lossy());
    std::fs::write(&script, setup).unwrap();
    let (stdout, stderr, ok) = output(cli().arg("run").arg(&script).arg("--save-as").arg(&project));
    assert!(ok, "setup failed: {stdout}{stderr}");

    // another process: AutoCut transcribes (whisper), classifies (local LLM) and edits
    let (stdout, stderr, ok) = output(cli().arg("--project").arg(&project).arg("autocut").arg("remove the silences"));
    assert!(ok, "autocut failed: {stdout}{stderr}");
    let start = stdout.find('{').unwrap_or(0);
    let report: Value = serde_json::from_str(&stdout[start..]).expect("the report JSON");
    assert_eq!(report["intent"], json!("remove_silence"), "{report}");
    let results = report["results"].as_array().cloned().unwrap_or_default();
    let generated = results.iter().find(|r| r["id"] == json!("transcript.generate")).expect("whisper ran");
    let words = generated["result"]["items"][0]["words"].as_u64().unwrap_or(0);
    assert!(words >= 5, "whisper understood the natural speech: {generated}");
    let pauses = results.iter().find(|r| r["id"] == json!("transcript.removePauses")).expect("removePauses ran");
    let removed = pauses["result"]["removed"].as_u64().unwrap_or(0);
    assert!(removed >= 2, "the deliberate silences were cut: {pauses}");
    let _ = std::fs::remove_dir_all(&dir);
}

fn concat_lines(lines: &[&str]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}
