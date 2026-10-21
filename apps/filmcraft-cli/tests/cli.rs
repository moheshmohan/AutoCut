//! End-to-end tests of the CLI binary (headless backend).

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::Value;

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_filmcraft-cli")).args(args).output().expect("spawn filmcraft-cli")
}

fn json_out(o: &Output) -> Value {
    assert!(o.status.success(), "failed: {}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).expect("JSON on stdout")
}

/// Uploads without a Google Photos sign-in (a data dir of the test's own, its sign-in in
/// owner-only files, so the user's sign-in and Keychain are never read) fail before anything is
/// sent, naming the command that fixes it; usage errors are usage errors.
#[test]
fn photos_upload_without_a_sign_in_names_the_fix() {
    let dir = std::env::temp_dir().join(format!("filmcraft-cli-upload-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let data = dir.join("data");
    std::fs::create_dir_all(data.join("google-photos")).unwrap();
    let clip = dir.join("clip.mp4");
    std::fs::write(&clip, b"not really a movie").unwrap();
    let clip = clip.to_string_lossy().to_string();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
            .args(args)
            .env("FILMCRAFT_DATA_DIR", &data)
            .env("FILMCRAFT_SECRET_STORE", "file")
            .output()
            .expect("spawn filmcraft-cli")
    };
    let err = |o: &Output| String::from_utf8_lossy(&o.stderr).to_string();

    assert_eq!(run(&["photos", "upload"]).status.code(), Some(2));
    assert_eq!(run(&["photos", "upload", &clip, &clip]).status.code(), Some(2));
    // signing out with nothing signed in: said, nothing sent; --sign-out goes alone
    let o = run(&["photos", "auth", "--sign-out"]);
    assert!(o.status.success() && String::from_utf8_lossy(&o.stdout).contains("nothing to sign out"), "{}", err(&o));
    assert_eq!(run(&["photos", "auth", "--sign-out", "--upload"]).status.code(), Some(2));
    // an unreadable sign-in is still forgotten (and Google is not asked: nothing is sent)
    std::fs::write(data.join("google-photos/tokens.json"), "{not json").unwrap();
    let o = run(&["photos", "auth", "--sign-out"]);
    assert!(o.status.success() && String::from_utf8_lossy(&o.stdout).contains("signed out of Google Photos here"), "{}", err(&o));
    assert!(err(&o).contains("could not be read") && err(&o).contains("myaccount.google.com"), "{}", err(&o));
    assert!(!data.join("google-photos/tokens.json").exists());
    let o = run(&["photos", "upload", &clip, "--description", "Robots"]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("not connected") && err(&o).contains("photos auth --upload"), "{}", err(&o));
    // signed in for the picker only: the upload needs consent
    std::fs::write(data.join("google-photos/client_secret.json"), r#"{"installed":{"client_id":"id","client_secret":"s"}}"#).unwrap();
    let picker = r#"{"access_token":"a","refresh_token":"r","expires_at":1,"scope":"https://www.googleapis.com/auth/photospicker.mediaitems.readonly"}"#;
    std::fs::write(data.join("google-photos/tokens.json"), picker).unwrap();
    let o = run(&["photos", "upload", &clip]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("consent") && err(&o).contains("photos auth --upload"), "{}", err(&o));
    // the engine command says the same, as its reason for being disabled
    let o = run(&["exec", "photos.upload", &format!("path={clip}")]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("photos auth --upload"), "{}", err(&o));
    // showcase --upload: needs --export, and fails before the build without consent
    let report = dir.join("report.json");
    std::fs::write(&report, r#"{"subject":"robot","clips":[]}"#).unwrap();
    let report = report.to_string_lossy().to_string();
    assert_eq!(run(&["showcase", "--report", &report, "--upload"]).status.code(), Some(2));
    let o = run(&["showcase", "--report", &report, "--export", &dir.join("out.mp4").to_string_lossy(), "--upload"]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("--upload") && err(&o).contains("photos auth --upload"), "{}", err(&o));
    assert!(!dir.join("out.mp4").exists(), "nothing was built");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Album commands without the album permissions (a data dir of the test's own) fail before
/// anything is sent, naming `photos auth --albums`; malformed ones are usage errors.
#[test]
fn photos_albums_without_consent_name_the_fix() {
    let dir = std::env::temp_dir().join(format!("filmcraft-cli-albums-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let data = dir.join("data");
    std::fs::create_dir_all(data.join("google-photos")).unwrap();
    let clip = dir.join("clip.mp4");
    std::fs::write(&clip, b"not really a movie").unwrap();
    let clip = clip.to_string_lossy().to_string();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
            .args(args)
            .env("FILMCRAFT_DATA_DIR", &data)
            .env("FILMCRAFT_SECRET_STORE", "file")
            .output()
            .expect("spawn filmcraft-cli")
    };
    let err = |o: &Output| String::from_utf8_lossy(&o.stderr).to_string();

    // usage errors: exit 2, nothing read
    let too_long = "t ".repeat(400);
    for args in [
        vec!["photos", "album"],
        vec!["photos", "album", "frobnicate"],
        vec!["photos", "album", "create"],
        vec!["photos", "album", "create", "  "],
        vec!["photos", "album", "create", &too_long],
        vec!["photos", "album", "add", "Robots"],
        vec!["photos", "album", "add", "Robots", "not an id"],
        vec!["photos", "albums", "extra"],
        vec!["photos", "upload", &clip, "--album"],
        vec!["photos", "upload", &clip, "--album", "  "],
        vec!["photos", "search", "robots"],
        // a pick searches (--auto) or takes a whole album (--album), never both
        vec!["photos", "pick", "--auto", "robots", "--album", "Robot showcase"],
        vec!["photos", "pick", "--album", "Robot showcase", "--no-browser"],
        vec!["photos", "pick", "--videos-only"],
        vec!["photos", "pick", "--album"],
    ] {
        let o = run(&args);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", err(&o));
    }
    // not connected
    for args in [vec!["photos", "albums"], vec!["photos", "album", "create", "Robot", "showcase"], vec!["photos", "album", "add", "Robots", "AF1QipItem"]] {
        let o = run(&args);
        assert_eq!(o.status.code(), Some(1), "{args:?}: {}", err(&o));
        assert!(err(&o).contains("not connected") && err(&o).contains("photos auth --albums"), "{args:?}: {}", err(&o));
    }
    // signed in for uploads only: albums (and uploads into them) need `--albums`
    std::fs::write(data.join("google-photos/client_secret.json"), r#"{"installed":{"client_id":"id","client_secret":"s"}}"#).unwrap();
    let upload_only = r#"{"access_token":"a","refresh_token":"r","expires_at":1,"scope":"https://www.googleapis.com/auth/photospicker.mediaitems.readonly https://www.googleapis.com/auth/photoslibrary.appendonly"}"#;
    std::fs::write(data.join("google-photos/tokens.json"), upload_only).unwrap();
    for args in [vec!["photos", "albums", "--json"], vec!["photos", "album", "create", "Robots"], vec!["photos", "upload", &clip, "--album", "Robots"]] {
        let o = run(&args);
        assert_eq!(o.status.code(), Some(1), "{args:?}: {}", err(&o));
        assert!(err(&o).contains("consent") && err(&o).contains("photos auth --albums"), "{args:?}: {}", err(&o));
    }
    // the engine commands say the same, as their reason for being disabled
    let o = run(&["exec", "photos.albums"]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("photos auth --albums"), "{}", err(&o));
    // showcase --album goes with --upload
    let report = dir.join("report.json");
    std::fs::write(&report, r#"{"subject":"robot","clips":[]}"#).unwrap();
    let report = report.to_string_lossy().to_string();
    let out = dir.join("out.mp4").to_string_lossy().to_string();
    assert_eq!(run(&["showcase", "--report", &report, "--export", &out, "--album", "Robots"]).status.code(), Some(2));
    let o = run(&["showcase", "--report", &report, "--export", &out, "--upload", "--album", "Robots"]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("photos auth --albums"), "{}", err(&o));
    assert!(!dir.join("out.mp4").exists(), "nothing was built");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `showcase --album`: usage errors (exit 2) before anything is read; without a sign-in it says
/// so before opening anything; signed in, the analysis job checks its vision model before the
/// album is picked (no Ollama here: nothing is picked, no browser opens). Offline: a data dir of
/// the test's own, and an Ollama address nothing answers on.
#[test]
fn showcase_from_an_album_checks_before_it_picks() {
    let dir = std::env::temp_dir().join(format!("filmcraft-cli-showcase-album-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let data = dir.join("data");
    std::fs::create_dir_all(data.join("google-photos")).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
            .args(args)
            .env("FILMCRAFT_DATA_DIR", &data)
            .env("FILMCRAFT_SECRET_STORE", "file")
            .env("AUTOCUT_OLLAMA_HOST", "http://127.0.0.1:9")
            .output()
            .expect("spawn filmcraft-cli")
    };
    let err = |o: &Output| String::from_utf8_lossy(&o.stderr).to_string();
    let report = dir.join("report.json");
    std::fs::write(&report, r#"{"subject":"robot","clips":[]}"#).unwrap();
    let report = report.to_string_lossy().to_string();
    let out = dir.join("out.mp4").to_string_lossy().to_string();

    for args in [
        vec!["showcase"],
        vec!["showcase", "--album"],
        vec!["showcase", "--album", "  "],
        // the album to read, or a report: not both (--upload-album names the upload's album)
        vec!["showcase", "--album", "Robots", "--report", &report],
        vec!["showcase", "--album", "Robots", "--report", &report, "--export", &out, "--upload", "--upload-album", "Showcases"],
        // what the video is of
        vec!["showcase", "--album", "Robots"],
        vec!["showcase", "--album", "Robots", "--prompt", " "],
        // the photos are kept (--with-photos) or left out (--videos-only): not both
        vec!["showcase", "--album", "Robots", "--prompt", "a video of every robot", "--with-photos", "--videos-only"],
        vec!["showcase", "--album", "Robots", "--subject", "robot", "--since", "last year"],
        vec!["showcase", "--album", "Robots", "--subject", "robot", "--frames", "many"],
        vec!["showcase", "--album", "Robots", "--subject", "robot", "--upload-album", "Showcases"],
        vec!["showcase", "--album", "Robots", "--subject", "robot", "--upload"],
        // a report already says what it found
        vec!["showcase", "--report", &report, "--prompt", "a video of every robot"],
        vec!["showcase", "--report", &report, "--videos-only"],
        vec!["showcase", "--report", &report, "--export", &out, "--upload-album", "Showcases"],
    ] {
        let o = run(&args);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", err(&o));
    }
    // the export into the album the clips come from: a warning (the next run would pick it)
    let o = run(&["showcase", "--album", "Robots", "--subject", "robot", "--export", &out, "--upload", "--upload-album", "Robots"]);
    assert_eq!(o.status.code(), Some(1), "not signed in: {}", err(&o));
    assert!(err(&o).contains("warning") && err(&o).contains("picks this video too"), "{}", err(&o));
    // not connected: said before a project or a model is touched
    let o = run(&["--demo", "showcase", "--album", "Robot showcase", "--prompt", "a video of every robot since 2020"]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("not connected") && err(&o).contains("photos auth"), "{}", err(&o));
    // signed in for the picker: the job starts, and stops at the missing vision model before
    // anything is picked
    std::fs::write(data.join("google-photos/client_secret.json"), r#"{"installed":{"client_id":"id","client_secret":"s"}}"#).unwrap();
    let picker = r#"{"access_token":"a","refresh_token":"r","expires_at":1,"scope":"https://www.googleapis.com/auth/photospicker.mediaitems.readonly"}"#;
    std::fs::write(data.join("google-photos/tokens.json"), picker).unwrap();
    let o = run(&["--demo", "showcase", "--album", "Robot showcase", "--subject", "robot", "--export", &out]);
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(err(&o).contains("vision model") && err(&o).contains("ollama pull"), "{}", err(&o));
    assert!(!err(&o).contains("picking the Google Photos album"), "nothing was picked: {}", err(&o));
    assert!(!dir.join("out.mp4").exists(), "nothing was built");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn help_and_usage_errors() {
    let h = cli(&["help"]);
    assert!(h.status.success());
    assert!(String::from_utf8_lossy(&h.stdout).contains("exec <id>"));
    assert_eq!(cli(&[]).status.code(), Some(2));
    assert_eq!(cli(&["frobnicate"]).status.code(), Some(2));
    assert_eq!(cli(&["exec", "file.newBin", "notkv"]).status.code(), Some(2));
    // Invalid --bridge is a usage error (exit 2) for both exec and mcp (#921).
    assert_eq!(cli(&["--bridge", "0.0.0.0:9876", "exec", "inspect.project"]).status.code(), Some(2));
    assert_eq!(cli(&["mcp", "--bridge", "0.0.0.0:9876"]).status.code(), Some(2));
}

#[test]
fn version_after_global_options() {
    let want = format!("filmcraft-cli {}\n", env!("CARGO_PKG_VERSION"));
    for args in [&["--version"][..], &["-V"], &["version"], &["--demo", "--version"], &["--compact", "-V"], &["--demo", "version"]] {
        let o = cli(args);
        assert!(o.status.success(), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
        assert_eq!(String::from_utf8_lossy(&o.stdout), want, "{args:?}");
    }
}

#[test]
fn long_help_matches_help_command() {
    let reference = cli(&["help"]);
    assert!(reference.status.success());
    let long_help = cli(&["--help"]);
    assert!(long_help.status.success(), "--help failed: {}", String::from_utf8_lossy(&long_help.stderr));
    assert_eq!(long_help.stdout, reference.stdout);
    assert!(long_help.stderr.is_empty());
    let short_help = cli(&["-h"]);
    assert!(short_help.status.success());
    assert_eq!(short_help.stdout, reference.stdout);
    let sub_help = cli(&["export", "--help"]);
    assert!(sub_help.status.success(), "export --help failed: {}", String::from_utf8_lossy(&sub_help.stderr));
    assert_eq!(sub_help.stdout, reference.stdout);
}

#[test]
fn probe_reports_mpeg_transport_and_program_streams() {
    let Some(ffmpeg) = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"].into_iter().find(|p| std::path::Path::new(p).exists()) else {
        eprintln!("SKIPPED (probe_reports_mpeg_transport_and_program_streams): ffmpeg not found");
        return;
    };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/cli-tests").join(format!("mpeg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, fmt, extra) in [("a.ts", "mpegts", &["-mpegts_m2ts_mode", "0"][..]), ("b.vob", "vob", &[][..])] {
        let out = dir.join(name);
        let st = Command::new(ffmpeg)
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
            .args(["-t", "0.5", "-c:v", "mpeg2video", "-c:a", "mp2", "-ac", "2", "-f", fmt])
            .args(extra)
            .arg(&out)
            .status()
            .unwrap();
        assert!(st.success());
        let v = json_out(&cli(&["probe", out.to_str().unwrap()]));
        assert!(v["video"]["codec"].as_str().unwrap().starts_with("MPEG-2 Video"), "{v}");
        assert_eq!(v["audio_streams"][0]["codec"], "MPEG Audio");
        let streams = v["mpeg"]["streams"].as_array().unwrap();
        assert_eq!(streams.len(), 2, "{v}");
        assert_eq!(streams[0]["codec"], "MPEG-2 Video");
        assert!(streams[0]["pictures"]["I"].as_u64().unwrap() >= 1);
        assert_eq!(v["mpeg"]["format"], if fmt == "mpegts" { "MPEG-2 TS" } else { "MPEG-2 PS" });
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn exec_inspect_and_describe() {
    let seq = json_out(&cli(&["--demo", "inspect", "sequence", "--compact"]));
    assert!(seq["tracks"].is_array() || seq.is_object(), "{seq}");
    let d = json_out(&cli(&["describe", "timeline.razor"]));
    assert_eq!(d["id"], "timeline.razor");
    assert!(d["params"].is_string());
    let list = json_out(&cli(&["commands", "razor", "--json"]));
    assert!(list.as_array().unwrap().iter().any(|c| c["id"] == "timeline.razor"));
    // An unknown command fails with exit status 1.
    let bad = cli(&["--demo", "exec", "no.such.command"]);
    assert_eq!(bad.status.code(), Some(1));
}

#[test]
fn exec_save_as_then_reopen() {
    let dir = std::env::temp_dir().join(format!("fc-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let proj = dir.join("p.fcproj");
    let p = proj.to_str().unwrap();
    json_out(&cli(&["--demo", "exec", "file.newBin", "name=CLI Selects", "--save-as", p]));
    let tree = json_out(&cli(&["--project", p, "inspect", "project"]));
    assert!(tree.to_string().contains("CLI Selects"), "{tree}");
    // `run` from stdin, editing and saving back in place.
    let mut child = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
        .args(["--project", p, "--save", "run", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"# comment\n{\"id\":\"file.newBin\",\"params\":{\"name\":\"Second\"}}\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let line: Value = serde_json::from_slice(out.stdout.split(|b| *b == b'\n').next().unwrap()).unwrap();
    assert_eq!(line["ok"], true);
    assert!(json_out(&cli(&["--project", p, "inspect", "project"])).to_string().contains("Second"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn run_keep_going_reports_failures() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
        .args(["--demo", "--keep-going", "run", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{\"id\":\"no.such\"}\n{\"id\":\"file.newBin\",\"params\":{\"name\":\"x\"}}\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let lines: Vec<Value> = String::from_utf8_lossy(&out.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["ok"], false);
    assert_eq!(lines[1]["ok"], true);
}

#[test]
fn export_wav_waits_for_job() {
    let dir = std::env::temp_dir().join(format!("fc-cli-x-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("still.wav");
    let o = cli(&["--demo", "export", out.to_str().unwrap()]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(std::fs::metadata(&out).map(|m| m.len() > 44).unwrap_or(false));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `showcase --qa` (no export): the built show is checked and the report is in the output; the
/// narration cache in `--data-dir` serves the second build; `--no-qa` skips the check. Offline:
/// the script is given and the model host is a closed local port.
#[test]
fn showcase_qa_and_the_narration_cache() {
    let dir = std::env::temp_dir().join(format!("fc-cli-showcase-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let clip = |item: u64, label: &str, unix: i64| {
        serde_json::json!({"item": item, "name": format!("{item}.mp4"), "seconds": 14.0, "created": null, "unix": unix, "present": true, "coverage": 1.0,
            "label": label, "kind": "humanoid", "description": label, "best": 7.0, "frames": [{"seconds": 7.0, "present": true, "area": 0.4, "center": [0.5, 0.5]}]})
    };
    let report = dir.join("report.json");
    std::fs::write(
        &report,
        serde_json::json!({"subject": "robot", "clips": [clip(5, "Pepper humanoid robot", 1_600_000_000), clip(6, "red toy robot", 1_665_841_668)]})
            .to_string(),
    )
    .unwrap();
    let script = dir.join("script.json");
    std::fs::write(&script, r#"{"intro": "Every robot, in order.", "lines": ["The first one.", "The second one."], "outro": "Thanks for watching."}"#).unwrap();
    let data = dir.join("data");
    let run = |extra: &[&str]| {
        let mut args = vec!["--demo", "--compact", "--data-dir", data.to_str().unwrap(), "showcase", "--report", report.to_str().unwrap()];
        args.extend(["--script-in", script.to_str().unwrap(), "--voice", "basic-male"]);
        args.extend(extra);
        let o = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli")).args(&args).env("AUTOCUT_OLLAMA_HOST", "http://127.0.0.1:9").output().unwrap();
        (json_out(&o), String::from_utf8_lossy(&o.stderr).into_owned())
    };
    let (v, err) = run(&["--qa"]);
    assert_eq!(v["narration_lines"], 4, "{v}");
    assert_eq!(v["narration_cached"], 0);
    assert_eq!(v["qa"]["errors"], 0, "{}", v["qa"]);
    assert!(v["qa"]["checks"].as_array().unwrap().iter().any(|c| c["name"] == "subject" && c["status"] == "skipped"), "{}", v["qa"]);
    assert!(v["qa"]["loudness"]["integratedLufs"].is_number());
    assert!(err.contains("showcase: QA: 0 error(s)"), "{err}");
    // the same build again: every line from the cache, and no QA when asked not to; over the
    // user's own music this time (a 12 s track, remixed to the show's length)
    let g = cli(&[
        "--data-dir",
        data.to_str().unwrap(),
        "exec",
        "audio.generate",
        &serde_json::json!({"kind": "music", "duration": 12, "bpm": 120, "place": false, "dir": dir}).to_string(),
    ]);
    let track = json_out(&g)["path"].as_str().unwrap().to_string();
    let (v, err) = run(&["--no-qa", "--music", &track]);
    assert_eq!(v["narration_cached"], 4, "{v}");
    assert!(v.get("qa").is_none());
    assert_eq!(v["music"]["fit"], "remixed", "{}\n{err}", v["music"]);
    assert!(v["music"]["source"].as_str().is_some_and(|s| s.ends_with(".wav") && std::path::Path::new(s).is_absolute()), "{}", v["music"]);
    assert!(err.contains("cutting on its beat"), "{err}");
    // a music file that is not there fails before anything is built
    let missing = dir.join("missing.mp3");
    let o = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
        .args(["--demo", "showcase", "--report", report.to_str().unwrap(), "--script-in", script.to_str().unwrap(), "--music", missing.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("missing.mp3"), "{}", String::from_utf8_lossy(&o.stderr));
    // a target length: what it came to next to what was asked; a nonsense one is a usage error
    let (v, err) = run(&["--no-qa", "--target-minutes", "0.5"]);
    assert_eq!(v["target_seconds"], 30.0, "{v}");
    assert!(v["seconds"].as_f64().is_some_and(|s| (s - 30.0).abs() < 8.0), "{}", v["seconds"]);
    assert!(err.contains("for a target of 0:30"), "{err}");
    for bad in ["ten", "-1", "0", "inf"] {
        let o = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
            .args(["--demo", "showcase", "--report", report.to_str().unwrap(), "--target-minutes", bad])
            .output()
            .unwrap();
        assert_eq!(o.status.code(), Some(2), "{bad}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `export --preset`: a built-in preset by name (dash or en dash), a custom range, a user preset
/// from `--data-dir`, `--list-presets`, `--queue`, and a clean failure for an unknown preset.
#[test]
fn export_with_presets() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/export-tests").join(format!("cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let data = dir.join("data");
    let data_s = data.to_str().unwrap();
    let out = dir.join("proxy.mov");
    let o = cli(&["--demo", "--data-dir", data_s, "export", out.to_str().unwrap(), "--preset", "Apple ProRes 422 Proxy", "--start", "0", "--end", "0.25"]);
    let v = json_out(&o);
    assert_eq!(v["preset"], "Apple ProRes 422 Proxy");
    assert!(std::fs::metadata(&out).map(|m| m.len() > 1000).unwrap_or(false));
    if let Some(ffprobe) = ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe", "/usr/bin/ffprobe"].into_iter().find(|p| std::path::Path::new(p).exists()) {
        let p = Command::new(ffprobe)
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,profile,width", "-of", "csv=p=0", out.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&p.stdout).trim(), "prores,Proxy,1920");
    }
    // the extension comes from the preset; a hyphen finds the en-dash name
    let base = dir.join("adaptive");
    json_out(&cli(&[
        "--demo",
        "--data-dir",
        data_s,
        "export",
        base.to_str().unwrap(),
        "--preset",
        "Match Source - Adaptive Low Bitrate",
        "--start",
        "0",
        "--end",
        "0.2",
    ]));
    assert!(dir.join("adaptive.mp4").exists());
    // a user preset saved into the data directory is found by later runs
    json_out(&cli(&[
        "--data-dir",
        data_s,
        "exec",
        "export.presets.save",
        "name=Tiny WAV",
        "from=Waveform Audio 48 kHz 16-bit",
        "settings={\"audio\":{\"channels\":1}}",
    ]));
    let list = json_out(&cli(&["--data-dir", data_s, "export", "--list-presets", "tiny"]));
    assert_eq!(list.as_array().unwrap().len(), 1, "{list}");
    let wav = dir.join("tiny.wav");
    json_out(&cli(&["--demo", "--data-dir", data_s, "export", wav.to_str().unwrap(), "--preset", "Tiny WAV", "--start", "0", "--end", "0.5"]));
    let b = std::fs::read(&wav).unwrap();
    assert_eq!(u16::from_le_bytes([b[22], b[23]]), 1, "mono from the user preset");
    // through the queue
    let q = dir.join("queued.wav");
    let v = json_out(&cli(&["--demo", "--data-dir", data_s, "export", q.to_str().unwrap(), "--preset", "Tiny WAV", "--queue", "--range", "entire"]));
    assert_eq!(v["items"][0]["status"], "done", "{v}");
    assert!(q.exists());
    // a format that changes the extension: the completion message names the file written (#341)
    let apv = dir.join("apv.mp4");
    let o =
        cli(&["--demo", "--data-dir", data_s, "export", apv.to_str().unwrap(), "--preset", "APV 422-10", "--start", "0", "--end", "0.1", "--scale", "0.125"]);
    let v = json_out(&o);
    let written = v["path"].as_str().unwrap().to_string();
    assert!(written.ends_with("apv.mp4.mov") && std::path::Path::new(&written).exists(), "{v}");
    let msg = String::from_utf8_lossy(&o.stderr);
    assert!(msg.contains(&format!("exported {written} in ")), "{msg}");
    let qa = dir.join("queued-apv.mp4");
    let o = cli(&[
        "--demo",
        "--data-dir",
        data_s,
        "export",
        qa.to_str().unwrap(),
        "--preset",
        "APV 422-10",
        "--queue",
        "--start",
        "0",
        "--end",
        "0.1",
        "--scale",
        "0.125",
    ]);
    let v = json_out(&o);
    let written = v["items"][0]["path"].as_str().unwrap().to_string();
    assert!(written.ends_with("queued-apv.mp4.mov"), "{v}");
    assert!(String::from_utf8_lossy(&o.stderr).contains(&format!("exported {written} in ")), "{}", String::from_utf8_lossy(&o.stderr));
    // unknown preset: exit status 1 and a message naming it
    let bad = cli(&["--demo", "--data-dir", data_s, "export", dir.join("x.mp4").to_str().unwrap(), "--preset", "No Such Preset"]);
    assert_eq!(bad.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("No Such Preset"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `filmcraft-cli commands | head -c 10` panicked with "failed printing to stdout: Broken pipe
/// (os error 32)" and exit status 101 (#285).
#[test]
fn closed_stdout_ends_quietly() {
    use std::io::Read;
    for args in [&["commands"][..], &["help"], &["--demo", "commands", "--json"]] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli")).args(args).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let mut first = [0u8; 10];
        child.stdout.as_mut().unwrap().read_exact(&mut first).unwrap();
        drop(child.stdout.take());
        let out = child.wait_with_output().unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success() && err.is_empty(), "{args:?}: {:?}: {err}", out.status);
    }
}

/// `run -` with stdout already closed: `run` reads its script to the end before it prints, so its
/// first write fails for sure. The rest of the script and the save still happen, and the exit status
/// still reports a failing command.
fn run_with_closed_stdout(args: &[&str], script: &[u8]) -> Output {
    let mut child =
        Command::new(env!("CARGO_BIN_EXE_filmcraft-cli")).args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    drop(child.stdout.take());
    child.stdin.take().unwrap().write_all(script).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn closed_stdout_still_runs_the_script_and_saves() {
    let dir = std::env::temp_dir().join(format!("fc-cli-pipe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let proj = dir.join("p.fcproj");
    let p = proj.to_str().unwrap();
    let script = b"{\"id\":\"file.newBin\",\"params\":{\"name\":\"Piped One\"}}\n{\"id\":\"file.newBin\",\"params\":{\"name\":\"Piped Two\"}}\n";
    let out = run_with_closed_stdout(&["--demo", "--save-as", p, "run", "-"], script);
    assert!(out.status.success(), "{:?}: {}", out.status, String::from_utf8_lossy(&out.stderr));
    let tree = json_out(&cli(&["--project", p, "inspect", "project"])).to_string();
    assert!(tree.contains("Piped One") && tree.contains("Piped Two"), "{tree}");

    // a failing command is still a failure, whether or not anyone reads stdout
    let out = run_with_closed_stdout(&["--demo", "run", "-"], b"{\"id\":\"no.such.command\"}\n");
    assert_eq!(out.status.code(), Some(1));
    let out = run_with_closed_stdout(
        &["--demo", "--keep-going", "run", "-"],
        b"{\"id\":\"no.such.command\"}\n{\"id\":\"file.newBin\",\"params\":{\"name\":\"x\"}}\n",
    );
    assert_eq!(out.status.code(), Some(1));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `render` with a non-numeric or non-finite `--seconds` / `--scale` is a usage error (exit 2) and
/// writes nothing, instead of silently rendering frame 0 at full scale.
#[test]
fn render_rejects_bad_numbers() {
    let dir = std::env::temp_dir().join(format!("fc-cli-render-nan-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let still = dir.join("s.png").to_str().unwrap().to_string();
    for (opt, bad) in
        [("--seconds", "1s"), ("--seconds", "1,5"), ("--seconds", "abc"), ("--seconds", "NaN"), ("--seconds", "inf"), ("--scale", "xyz"), ("--scale", "NaN")]
    {
        let (secs, scale) = if opt == "--seconds" { (bad, "0.1") } else { ("1", bad) };
        let o = cli(&["--demo", "render", "--seconds", secs, "--scale", scale, "--out", &still]);
        let err = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(2), "{opt} {bad}: {err}");
        assert!(err.contains("filmcraft-cli:") && err.contains(opt), "{opt} {bad}: {err}");
        assert!(!std::path::Path::new(&still).exists(), "{opt} {bad} wrote a file");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// #1054: exporting or rendering a project whose media is missing fails (exit 1) and names the
/// offline items on stderr and in the result's `offlineMedia`, unless `--allow-offline` accepts the
/// placeholder frames; then it still warns.
#[test]
fn export_with_missing_media_fails_unless_allowed() {
    let dir = std::env::temp_dir().join(format!("fc-cli-1054-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let f = |n: &str| dir.join(n).to_str().unwrap().to_string();
    let (still, proj) = (f("still.png"), f("p.fcproj"));
    let o = cli(&["--demo", "render", "--seconds", "1", "--scale", "0.1", "--out", &still]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    json_out(&cli(&["exec", "file.newSequence", "name=s", "width=64", "height=36", "fps=24", "--save-as", &proj]));
    let paths = format!("paths=[\"{}\"]", still.replace('\\', "\\\\"));
    let item = json_out(&cli(&["--project", &proj, "--save", "exec", "file.import", &paths]))["items"][0].as_u64().unwrap();
    json_out(&cli(&["--project", &proj, "--save", "exec", "timeline.place", &format!("item={item}"), "track=V1", "frame=0", "audio=false"]));
    let export = |out: &str, extra: &[&str]| {
        let mut args = vec!["--project", proj.as_str(), "--compact", "export", out, "--format", "png", "--start", "0", "--end", "0.05"];
        args.extend_from_slice(extra);
        cli(&args)
    };

    let on = export(&f("on.png"), &[]);
    let v = json_out(&on);
    assert_eq!(v["offlineMedia"], serde_json::json!([]), "{v}");
    assert!(!String::from_utf8_lossy(&on.stderr).contains("warning"), "{}", String::from_utf8_lossy(&on.stderr));

    std::fs::remove_file(&still).unwrap();
    let off = export(&f("off.png"), &[]);
    let err = String::from_utf8_lossy(&off.stderr);
    assert_eq!(off.status.code(), Some(1), "{err}");
    assert!(err.contains("still.png") && err.contains("missing") && err.contains("--allow-offline"), "{err}");
    let v: Value = serde_json::from_slice(&off.stdout).expect("the result JSON is still printed");
    assert_eq!(v["offlineMedia"][0]["item"], item, "{v}");
    assert_eq!(v["offlineMedia"][0]["status"], "missing", "{v}");

    let allowed = export(&f("allowed.png"), &["--allow-offline"]);
    let err = String::from_utf8_lossy(&allowed.stderr);
    assert!(allowed.status.success(), "{err}");
    assert!(err.contains("still.png") && err.contains("missing"), "still warns: {err}");
    assert_eq!(json_out(&allowed)["offlineMedia"][0]["status"], "missing");

    let render = cli(&["--project", &proj, "render", "--seconds", "0", "--out", &f("r.png")]);
    let err = String::from_utf8_lossy(&render.stderr);
    assert_eq!(render.status.code(), Some(1), "{err}");
    assert!(err.contains("still.png") && err.contains("--allow-offline"), "{err}");
    let render = cli(&["--project", &proj, "render", "--seconds", "0", "--out", &f("r.png"), "--allow-offline"]);
    assert!(render.status.success(), "{}", String::from_utf8_lossy(&render.stderr));
    assert!(String::from_utf8_lossy(&render.stderr).contains("still.png"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `showcase --outro-*`: bad flags are usage errors; good ones become the plan's `outro`
/// (written out with `--plan`), over the reviewed plan's own fields.
#[test]
fn showcase_outro_flags_become_the_plans_outro() {
    let dir = std::env::temp_dir().join(format!("filmcraft-cli-showcase-outro-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
            .args(args)
            .env("FILMCRAFT_DATA_DIR", dir.join("data"))
            .env("AUTOCUT_OLLAMA_HOST", "http://127.0.0.1:9")
            .output()
            .expect("spawn filmcraft-cli")
    };
    let err = |o: &Output| String::from_utf8_lossy(&o.stderr).to_string();
    let report = dir.join("report.json");
    std::fs::write(&report, r#"{"subject":"robot","clips":[]}"#).unwrap();
    let report = report.to_string_lossy().to_string();
    let logo = dir.join("logo.png");
    std::fs::write(&logo, b"\x89PNG").unwrap();
    let logo = logo.to_string_lossy().to_string();
    let plan = dir.join("plan.json");
    std::fs::write(&plan, r#"{"entries": [], "outro": {"channel": "From the plan", "tagline": "kept"}}"#).unwrap();
    let plan = plan.to_string_lossy().to_string();
    for bad in [
        vec!["--outro-logo", "/no/such/logo.png"],
        vec!["--outro-seconds", "soon"],
        vec!["--outro-background", "x:1:2"],
        vec!["--outro-background", "4830:a"],
        vec!["--outro-background", "4830:1:2:3"],
        vec!["--outro-channel", "H4hacks", "--outro-seconds", "3"],
        vec!["--outro-channel", "two\nlines"],
    ] {
        let mut args = vec!["showcase", "--report", &report, "--plan"];
        args.extend(bad.iter().copied());
        let o = run(&args);
        assert_eq!(o.status.code(), Some(2), "{bad:?}: {}", err(&o));
        assert!(err(&o).contains("outro"), "{bad:?}: {}", err(&o));
    }
    let o = run(&[
        "showcase",
        "--report",
        &report,
        "--script-in",
        &plan,
        "--plan",
        "--outro-logo",
        &logo,
        "--outro-channel",
        "H4hacks",
        "--outro-handle",
        "@H4hacks",
        "--outro-cta",
        "Subscribe",
        "--outro-background",
        "4830:5:25",
        "--outro-background",
        "4831",
        "--outro-seconds",
        "11",
    ]);
    let v = json_out(&o);
    let outro = &v["outro"];
    assert_eq!(outro["channel"], "H4hacks", "{v}");
    assert_eq!(outro["tagline"], "kept", "the plan's own fields stay");
    assert_eq!(outro["background"], serde_json::json!([{"item": 4830, "from": 5.0, "to": 25.0}, {"item": 4831}]));
    assert_eq!(outro["seconds"], 11.0);
    assert!(outro["logo"].as_str().is_some_and(|p| std::path::Path::new(p).is_absolute()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `ml-lint` checks a Malayalam lines file (against the English plan), fixes it, and with
/// `--strict` fails unless it is clean; `ml-script` keeps approved lines as examples and names
/// the model it cannot reach. No model runs here.
#[test]
fn malayalam_scripts_are_linted_and_remembered() {
    let dir = std::env::temp_dir().join(format!("fc-cli-ml-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let plan = dir.join("plan.json");
    std::fs::write(
        &plan,
        r#"{"entries": [{"label": "Pepper"}], "script": {"intro": "Hi.", "lines": ["Pepper greets visitors; SoftBank made twenty-seven thousand."], "outro": "Thanks for watching."}}"#,
    )
    .unwrap();
    let lines = dir.join("script-ml.json");
    std::fs::write(&lines, r#"{"intro": "ഹായ്.", "outro": "കണ്ടതിന് നന്ദി.", "lines": {"Pepper": "പെപ്പർ സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു. സോഫ്റ്റ്ബാങ്ക് പത്ത് എണ്ണം ഉണ്ടാക്കി."}}"#)
        .unwrap();
    let (p, l) = (plan.to_str().unwrap(), lines.to_str().unwrap());
    let o = cli(&["ml-lint", l, "--plan", p]);
    let v = json_out(&o);
    assert_eq!((v["warnings"].as_u64(), v["factsMissing"].as_u64(), v["numbersAdded"].as_u64()), (Some(2), Some(1), Some(1)), "{v}");
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("Pepper: formal: «സന്ദർശകരെ» → വിസിറ്റേഴ്സിനെ") && err.contains("Pepper: facts: missing: twenty-seven thousand; added: 10"), "{err}");
    assert_eq!(cli(&["ml-lint", l, "--plan", p, "--strict"]).status.code(), Some(1));
    // the safe fixes, written as a lines file again
    let fixed = dir.join("fixed.json");
    json_out(&cli(&["ml-lint", l, "--fix", "--out", fixed.to_str().unwrap()]));
    let f: Value = serde_json::from_slice(&std::fs::read(&fixed).unwrap()).unwrap();
    assert_eq!(f["lines"]["Pepper"], "പെപ്പർ വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്നു. സോഫ്റ്റ്ബാങ്ക് പത്ത് എണ്ണം ഉണ്ടാക്കി.");
    assert_eq!(f["outro"], "കണ്ടതിന് താങ്ക്സ്.");
    assert_eq!(cli(&["ml-lint"]).status.code(), Some(2));
    assert_eq!(cli(&["ml-lint", l, "--fix"]).status.code(), Some(2), "--fix needs --out");
    assert_eq!(cli(&["ml-lint", "/no/such/script.json"]).status.code(), Some(1));
    // approved lines become examples
    let examples = dir.join("examples.json");
    let v = json_out(&cli(&["ml-script", "--plan", p, "--remember", fixed.to_str().unwrap(), "--examples", examples.to_str().unwrap()]));
    assert_eq!((v["remembered"].as_u64(), v["kept"].as_u64()), (Some(3), Some(3)), "{v}");
    let e: Value = serde_json::from_slice(&std::fs::read(&examples).unwrap()).unwrap();
    assert_eq!(e["pairs"][1]["en"], "Pepper greets visitors; SoftBank made twenty-seven thousand.");
    // usage errors, and a model that is not there
    assert_eq!(cli(&["ml-script", "--out", "x.json"]).status.code(), Some(2));
    assert_eq!(cli(&["ml-script", "--plan", p]).status.code(), Some(2));
    assert_eq!(cli(&["ml-script", "--plan", p, "--out", "x.json", "--batch", "0"]).status.code(), Some(2));
    let o = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
        .args(["ml-script", "--plan", p, "--out", dir.join("out.json").to_str().unwrap()])
        .env("AUTOCUT_OLLAMA_HOST", "http://127.0.0.1:9")
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("127.0.0.1:9"), "{}", String::from_utf8_lossy(&o.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `showcase-plan`: a real round's notes (trimmed, crates/autocut/tests/data/plan_patch) applied to
/// its plan give the plans it made, English and Malayalam; bad notes fail naming the entry.
#[test]
fn showcase_plan_applies_notes_and_writes_the_other_language() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/autocut/tests/data/plan_patch");
    let f = |n: &str| data.join(n).to_string_lossy().to_string();
    let dir = std::env::temp_dir().join(format!("fc-cli-plan-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("plan-v5.json").to_string_lossy().to_string();
    let o = cli(&[
        "showcase-plan",
        "--base",
        &f("base.json"),
        "--patch",
        &f("notes.json"),
        "--clips",
        &f("clips.json"),
        "--out",
        &out,
        "--lines",
        &f("lines-ml.json"),
    ]);
    let v = json_out(&o);
    assert_eq!((v["entries"].as_u64(), v["robots"].as_u64()), (Some(10), Some(9)), "{v}");
    let read = |p: &std::path::Path| -> Value { serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap() };
    assert_eq!(read(std::path::Path::new(&out)), read(&data.join("expected.json")));
    assert_eq!(read(&dir.join("plan-v5-ml.json")), read(&data.join("expected-ml.json")));
    // the places `photos locations` read fill the entries without one, by their first shot
    let locations = dir.join("locations.json");
    std::fs::write(&locations, r#"[{"fileName": "org_video_6553912_1642930732000.MP4", "place": "Expo City, Dubai", "estimated": false}]"#).unwrap();
    let placed = dir.join("placed.json");
    let v = json_out(&cli(&[
        "showcase-plan",
        "--base",
        &f("base.json"),
        "--patch",
        &f("notes.json"),
        "--clips",
        &f("clips.json"),
        "--places-from-locations",
        locations.to_str().unwrap(),
        "--out",
        placed.to_str().unwrap(),
    ]));
    assert_eq!(v["placesFilled"], serde_json::json!(["Micropolis M02"]), "{v}");
    let p = read(&placed);
    let m02 = p["entries"].as_array().unwrap().iter().find(|e| e["label"] == "Micropolis M02").unwrap();
    assert_eq!(m02["place"], "Expo City, Dubai");
    // a note for an entry the plan does not have
    let bad = dir.join("bad.json");
    std::fs::write(&bad, r#"{"entries": {"No such robot": {"line": "x"}}}"#).unwrap();
    let o = cli(&["showcase-plan", "--base", &f("base.json"), "--patch", bad.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("No such robot"));
    assert_eq!(cli(&["showcase-plan", "--patch", &f("notes.json")]).status.code(), Some(2));
    assert_eq!(cli(&["showcase-plan", "--base", &f("base.json"), "--lines", &f("lines-ml.json")]).status.code(), Some(2), "--lines needs somewhere to go");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `sheet` over the demo project and `frames` of a file exported from it: labelled sheets.
#[test]
fn sheets_of_clips_and_frames_of_a_video() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/export-tests").join(format!("sheet-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let sheet = dir.join("sheet.png");
    let v = json_out(&cli(&["--demo", "sheet", "--items", "5,5:0.5-1.5", "--n", "3", "--side", "120", "--out", sheet.to_str().unwrap()]));
    assert_eq!((v["frames"].as_u64(), v["missing"].as_u64()), (Some(6), Some(0)), "{v}");
    assert!(std::fs::read(&sheet).unwrap().starts_with(b"\x89PNG"));
    assert_eq!(cli(&["--demo", "sheet", "--items", "5:9-1", "--out", sheet.to_str().unwrap()]).status.code(), Some(2));
    assert_eq!(cli(&["--demo", "sheet", "--items", "999", "--out", sheet.to_str().unwrap()]).status.code(), Some(1));
    let video = dir.join("clip.mp4");
    json_out(&cli(&["--demo", "export", video.to_str().unwrap(), "--start", "0", "--end", "1", "--scale", "0.25"]));
    let jpg = dir.join("frames.jpg");
    let v = json_out(&cli(&["frames", video.to_str().unwrap(), "--at", "0.2,0.6", "--out", jpg.to_str().unwrap()]));
    assert_eq!(v["frames"].as_u64(), Some(2), "{v}");
    assert!(std::fs::read(&jpg).unwrap().starts_with(&[0xFF, 0xD8]));
    assert_eq!(cli(&["frames", video.to_str().unwrap(), "--out", jpg.to_str().unwrap()]).status.code(), Some(2), "--at or --n");
    assert_eq!(cli(&["frames", dir.join("none.mp4").to_str().unwrap(), "--at", "1", "--out", jpg.to_str().unwrap()]).status.code(), Some(1));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `showcase --languages en,ml`: one run builds the show twice from one plan, the second in
/// Malayalam from its lines by label; what it needs is checked before anything is built.
/// Offline: no voice, the model host a closed local port.
#[test]
fn showcase_in_two_languages_from_one_plan() {
    let dir = std::env::temp_dir().join(format!("fc-cli-languages-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let shot = |item: u64| serde_json::json!({"item": item, "name": format!("{item}.mp4"), "clip_seconds": 14.0, "best": 7.0, "seen": [7.0], "unseen": []});
    let entry = |n: usize, label: &str, item: u64, unix: i64| serde_json::json!({"number": n, "label": label, "kind": "humanoid", "description": label, "unix": unix, "shots": [shot(item)]});
    let plan = serde_json::json!({
        "entries": [entry(1, "Pepper", 5, 1_600_000_000), entry(2, "Toy robot", 6, 1_665_841_668)],
        "script": {"intro": "Every robot, in order.", "lines": ["The first one.", "The second one."], "outro": "Thanks for watching."}
    });
    let report = dir.join("report.json");
    std::fs::write(&report, r#"{"subject": "robot", "clips": []}"#).unwrap();
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan.to_string()).unwrap();
    let lines = dir.join("script-ml.json");
    std::fs::write(&lines, r#"{"intro": "എല്ലാ റോബോട്ടുകളും.", "outro": "നന്ദി.", "lines": {"Pepper": "ആദ്യത്തേത്.", "Toy robot": "രണ്ടാമത്തേത്."}}"#).unwrap();
    let run = |extra: &[&str]| {
        let mut args = vec!["--demo", "--compact", "showcase", "--report", report.to_str().unwrap(), "--script-in", plan_path.to_str().unwrap(), "--no-qa"];
        args.extend(extra);
        Command::new(env!("CARGO_BIN_EXE_filmcraft-cli")).args(&args).env("AUTOCUT_OLLAMA_HOST", "http://127.0.0.1:9").output().unwrap()
    };
    let o = run(&["--languages", "en,ml", "--lines-ml", lines.to_str().unwrap()]);
    let v = json_out(&o);
    let builds = v["languages"].as_array().unwrap();
    assert_eq!(builds.len(), 2, "{v}");
    assert_eq!((builds[0].get("language"), builds[1]["language"].as_str()), (None, Some("ml")));
    assert_ne!(builds[0]["sequence"], builds[1]["sequence"], "two sequences");
    assert_eq!((builds[0]["entries"].as_u64(), builds[1]["entries"].as_u64()), (Some(2), Some(2)));
    assert!(String::from_utf8_lossy(&o.stderr).contains("showcase (Malayalam):"), "{}", String::from_utf8_lossy(&o.stderr));
    // what it needs, before anything is built
    assert_eq!(run(&["--languages", "en,ml"]).status.code(), Some(2), "Malayalam needs its lines");
    assert_eq!(run(&["--languages", "en,fr"]).status.code(), Some(2));
    assert_eq!(run(&["--languages", "en", "--language", "ml"]).status.code(), Some(2));
    let no_plan = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
        .args(["--demo", "showcase", "--report", report.to_str().unwrap(), "--languages", "en"])
        .output()
        .unwrap();
    assert_eq!(no_plan.status.code(), Some(2));
    // a spoken entry without a Malayalam line
    let short = dir.join("short-ml.json");
    std::fs::write(&short, r#"{"intro": "a", "outro": "b", "lines": {"Pepper": "x"}}"#).unwrap();
    let o = run(&["--languages", "en,ml", "--lines-ml", short.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("Toy robot"), "{}", String::from_utf8_lossy(&o.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}
