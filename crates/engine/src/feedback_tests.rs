use serde_json::{Value, json};

use super::*;
use crate::Session;

/// A fresh folder under the temp directory.
fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fc-feedback-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The demo project, with its data directory (preferences, Logs) in `dir`.
fn session(dir: &Path) -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.prefs_path = Some(dir.join("preferences.json"));
    s
}

/// Every file of a bundle, by its path inside it.
fn read_bundle(dir: &Path) -> Vec<(String, String)> {
    fn walk(root: &Path, d: &Path, out: &mut Vec<(String, String)>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(root, &p, out);
            } else {
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
                out.push((rel, String::from_utf8_lossy(&std::fs::read(&p).unwrap()).into_owned()));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

fn file<'a>(files: &'a [(String, String)], name: &str) -> &'a str {
    files
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, t)| t.as_str())
        .unwrap_or_else(|| panic!("no {name} in {:?}", files.iter().map(|f| &f.0).collect::<Vec<_>>()))
}

#[test]
fn the_bundle_holds_every_part() {
    let dir = tmp("parts");
    let mut s = session(&dir);
    let logs = dir.join("Logs");
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::write(logs.join("filmcraft.log"), "[2026-10-21T10:00:00Z WARN filmcraft_engine] decoder fell back\n").unwrap();
    std::fs::write(logs.join("filmcraft.1.log"), "previous run\n").unwrap();
    std::fs::write(logs.join("crash-20000.log"), "=== FilmCraft 0.10 (unix 1)\nindex out of bounds (at crates/x.rs:1, thread main)\n").unwrap();
    std::fs::write(logs.join("session-42.log"), "{\"command\":\"timeline.razor\"}\n").unwrap();
    // never read: not a log
    std::fs::write(logs.join("tokens.json"), "{\"refresh_token\": \"1//0NEVERREADTHISFILE\"}").unwrap();
    // an AutoCut QA report
    let qa = dir.join("robots.qa.json");
    std::fs::write(&qa, r#"{"summary": "2 warnings", "checks": [{"name": "black frames", "ok": true}]}"#).unwrap();
    // some history: two edits, one undone
    s.execute("sequence.addEdit", json!({"seconds": 1.0})).unwrap();
    s.execute("sequence.addEdit", json!({"seconds": 2.0})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();

    let out = dir.join("bundle");
    let r = s.execute("app.feedbackBundle", json!({"path": out.to_string_lossy(), "qaReport": qa.to_string_lossy()})).unwrap();
    assert_eq!(r["path"], out.to_string_lossy().as_ref());
    let files = read_bundle(&out);
    let names: Vec<&str> = files.iter().map(|f| f.0.as_str()).collect();
    for want in [
        "README.txt",
        "manifest.json",
        "system.json",
        "history.json",
        "events.json",
        "jobs.json",
        "perf.json",
        "project.json",
        "preferences.json",
        "qa.json",
        "logs/filmcraft.log",
        "logs/filmcraft.1.log",
        "logs/session-42.log",
        "crashes/crash-20000.log",
    ] {
        assert!(names.contains(&want), "{want} missing: {names:?}");
    }
    assert!(!names.iter().any(|n| n.contains("tokens")), "only logs are read: {names:?}");
    // version and OS
    let manifest: Value = serde_json::from_str(file(&files, "manifest.json")).unwrap();
    assert_eq!(manifest["app"], format!("FilmCraft {}", env!("CARGO_PKG_VERSION")));
    assert_eq!(manifest["os"], std::env::consts::OS);
    assert_eq!(manifest["files"].as_array().unwrap().len(), names.len() - 2, "every part but README and manifest is listed");
    let system: Value = serde_json::from_str(file(&files, "system.json")).unwrap();
    assert_eq!(system["build"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(system["cpuThreads"].as_u64().is_some());
    // undo history with times
    let history: Value = serde_json::from_str(file(&files, "history.json")).unwrap();
    assert_eq!(history["undo"].as_array().unwrap().len(), s.history.undo.len());
    assert_eq!(history["redo"].as_array().unwrap().len(), 1);
    let log = history["log"].as_array().unwrap();
    let actions: Vec<&str> = log.iter().filter_map(|e| e["action"].as_str()).collect();
    assert!(actions.ends_with(&["edit", "edit", "undo"]), "{actions:?}");
    assert!(log.iter().all(|e| e["unix_ms"].as_u64().is_some_and(|t| t > 1_600_000_000_000)), "{log:?}");
    assert!(history["journal"].as_array().unwrap().iter().any(|c| c["command"] == "sequence.addEdit"));
    // perf, project, QA, logs, crashes
    assert!(serde_json::from_str::<Value>(file(&files, "perf.json")).unwrap().is_object());
    let project: Value = serde_json::from_str(file(&files, "project.json")).unwrap();
    assert_eq!(project["name"], s.project.name);
    let qa: Value = serde_json::from_str(file(&files, "qa.json")).unwrap();
    assert_eq!(qa["report"]["summary"], "2 warnings");
    assert!(file(&files, "logs/filmcraft.log").contains("decoder fell back"));
    assert!(file(&files, "crashes/crash-20000.log").contains("index out of bounds"));
    assert!(file(&files, "README.txt").contains("FilmCraft feedback bundle"));
    // history.list shows the times too
    let l = s.execute("history.list", json!({})).unwrap();
    assert_eq!(l["log"].as_array().unwrap().last().unwrap()["action"], "undo");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fake_secrets_and_the_home_folder_never_reach_the_bundle() {
    let dir = tmp("secrets");
    let mut s = session(&dir);
    let home = home_dir(&s).expect("a home directory");
    // secrets in preferences (string fields), the command journal and the logs
    s.prefs.media_browser.favorites.push(format!("{home}/Movies/clips?access_token=ya29.FAKEPREFTOKEN0123456789abcdef"));
    s.prefs.general.recent_projects.push(format!("{home}/Projects/secret client_secret=FAKECLIENTSECRET42.fcproj"));
    s.journal.push(("instagram.connect".into(), json!({"token": "IGQVJFAKEINSTAGRAMTOKEN1234567890", "account": "me"})));
    s.journal.push(("youtube.connect".into(), json!({"auth": {"refresh_token": "1//0FAKEREFRESHINJOURNAL"}, "apiKey": "FAKEAPIKEY"})));
    let logs = dir.join("Logs");
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::write(
        logs.join("filmcraft.log"),
        format!(
            "GET https://photos.example/v1/items?pageToken=FAKEPAGETOKEN&key=AIzaFAKEGOOGLEAPIKEY000000000000000\n\
             Authorization: Bearer FAKEBEARERTOKEN.abc.def\n\
             upload failed: {{\"error\":\"invalid_grant\",\"refresh_token\":\"1//0FAKEREFRESHTOKENINLOG\"}}\n\
             password = FAKEPASSWORD99\n\
             signed in as FAKEUSER@example.com\n\
             jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJGQUtFIn0.FAKESIGNATUREabc\n\
             opened {home}/Movies/AutoCut/robots.mp4 (H.264 1920x1080, 29.97 fps)\n\
             also /Users/someoneelse/Desktop/x.mov and C:\\Users\\Bob\\Videos\\y.mp4\n"
        ),
    )
    .unwrap();
    std::fs::write(logs.join("crash-20001.log"), format!("panic at {home}/src/x.rs: token=FAKECRASHTOKEN\n")).unwrap();
    let out = dir.join("bundle");
    let r = s.execute("app.feedbackBundle", json!({"path": out.to_string_lossy()})).unwrap();
    assert!(r["redactions"].as_u64().unwrap() >= 10, "{r}");
    let files = read_bundle(&out);
    for (name, text) in &files {
        assert!(
            !text.contains("FAKE") && !text.contains("fake"),
            "{name} leaks a secret:\n{}",
            text.lines().filter(|l| l.contains("FAKE")).collect::<Vec<_>>().join("\n")
        );
        assert!(!text.contains(&home), "{name} shows the home folder");
        assert!(!text.contains("someoneelse") && !text.contains("\\Bob\\"), "{name} shows a user name");
    }
    // what is useful stays
    let log = file(&files, "logs/filmcraft.log");
    assert!(log.contains("~/Movies/AutoCut/robots.mp4 (H.264 1920x1080, 29.97 fps)"), "{log}");
    assert!(log.contains("Authorization: Bearer [redacted]"), "{log}");
    assert!(log.contains("[email]"), "{log}");
    let prefs = file(&files, "preferences.json");
    assert!(prefs.contains("~/Movies/clips?access_token=[redacted]"), "{prefs}");
    let history: Value = serde_json::from_str(file(&files, "history.json")).unwrap();
    let j = &history["journal"];
    assert!(j.to_string().contains("instagram.connect") && j.to_string().contains("\"account\":\"me\""), "{j}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_bundle_is_capped() {
    let dir = tmp("cap");
    let mut s = session(&dir);
    let logs = dir.join("Logs");
    std::fs::create_dir_all(&logs).unwrap();
    // three 5 MiB logs and twenty 1 MiB crash reports
    let line = "[INFO filmcraft] frame 123 decoded in 4 ms\n";
    let big = line.repeat((5 << 20) / line.len());
    for n in ["filmcraft.log", "filmcraft.1.log", "filmcraft.2.log"] {
        std::fs::write(logs.join(n), &big).unwrap();
    }
    for d in 0..20 {
        std::fs::write(logs.join(format!("crash-{}.log", 20_000 + d)), line.repeat((1 << 20) / line.len())).unwrap();
    }
    let out = dir.join("bundle");
    let r = s.execute("app.feedbackBundle", json!({"path": out.to_string_lossy()})).unwrap();
    assert!(r["bytes"].as_u64().unwrap() <= MAX_BUNDLE as u64 + 64 * 1024, "{}", r["bytes"]);
    let files = read_bundle(&out);
    let crashes = files.iter().filter(|f| f.0.starts_with("crashes/")).count();
    assert_eq!(crashes, MAX_CRASHES);
    assert!(files.iter().any(|f| f.0 == "crashes/crash-20019.log"), "the newest crash reports");
    let log = file(&files, "logs/filmcraft.log");
    assert!(log.len() <= LOG_TAIL && log.starts_with("[INFO"), "a tail from a line start ({} bytes)", log.len());
    let manifest: Value = serde_json::from_str(file(&files, "manifest.json")).unwrap();
    assert!(manifest["files"].as_array().unwrap().iter().any(|f| f["note"].as_str().is_some_and(|n| n.contains("last"))), "cut parts say so");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_are_errors() {
    let dir = tmp("hostile");
    let mut s = session(&dir);
    let a_file = dir.join("not-a-folder");
    std::fs::write(&a_file, "x").unwrap();
    let big = dir.join("big.qa.json");
    std::fs::write(&big, "{").unwrap();
    for p in [
        json!({"path": ""}),
        json!({"path": "   "}),
        json!({"path": 42}),
        json!({"path": "a\u{0}b"}),
        json!({"path": "x".repeat(5000)}),
        json!({"path": a_file.to_string_lossy()}),
        json!({"path": dir.join("b1").to_string_lossy(), "perf": "fast"}),
        json!({"path": dir.join("b2").to_string_lossy(), "system": [1, 2]}),
        json!({"path": dir.join("b3").to_string_lossy(), "qaReport": dir.join("missing.json").to_string_lossy()}),
        json!({"path": dir.join("b4").to_string_lossy(), "qaReport": big.to_string_lossy()}),
        json!([1, 2, 3]),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("app.feedbackBundle", p.clone())));
        assert!(matches!(r, Ok(Err(_))), "{p}: {r:?}");
    }
    // the app's own perf stats and system report are taken (and redacted)
    let out = dir.join("ok");
    s.execute(
        "app.feedbackBundle",
        json!({"path": out.to_string_lossy(), "perf": {"ui": {"fps": 60}, "apiToken": "FAKE"}, "system": {"gpu": {"name": "Test GPU"}}}),
    )
    .unwrap();
    let files = read_bundle(&out);
    assert!(file(&files, "perf.json").contains("\"fps\": 60") && !file(&files, "perf.json").contains("FAKE"));
    assert!(file(&files, "system.json").contains("Test GPU"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn redaction_rules() {
    let r = |t: &str| redact_text(t, Some("/Users/ann")).0;
    assert_eq!(r("/Users/ann/Movies/a.mp4"), "~/Movies/a.mp4");
    assert_eq!(r("C:\\Users\\Ann Lee\\x"), "~\\x", "Windows user folders may hold spaces");
    assert_eq!(r("saw /Users/bob now"), "saw ~ now");
    assert_eq!(r("D:/Users/bob/clip.mov"), "~/clip.mov");
    assert_eq!(r("/home/carol/.config"), "~/.config");
    assert_eq!(r("/Users/Shared/Footage"), "/Users/Shared/Footage");
    assert_eq!(r("token=abc123"), "token=[redacted]");
    assert_eq!(r("?access_token=xyz&page=2"), "?access_token=[redacted]&page=2");
    assert_eq!(r("\"client_secret\": \"s3cr3t\""), "\"client_secret\": \"[redacted]\"");
    assert_eq!(r("Authorization: Bearer abc.def"), "Authorization: Bearer [redacted]");
    assert_eq!(r("password = hunter2, user ok"), "password = [redacted], user ok");
    assert_eq!(r("mail me@example.org now"), "mail [email] now");
    assert_eq!(r("key ya29.a0AfH6SMBx and sk-ant-api03-abcdefghijklmnop"), "key [redacted] and [redacted]");
    assert_eq!(r("ghp_abcdefghijklmnopqrstuvwxyz0123456789"), "[redacted]");
    assert_eq!(r("AKIAIOSFODNN7EXAMPLE"), "[redacted]");
    // ordinary diagnostics stay
    for keep in [
        "H.264 High 4.1, 1920x1080, 29.97 fps",
        "timeline.razor {\"seconds\": 3}",
        "/tmp/fc-cap-1/out.srt",
        "123e4567-e89b-12d3-a456-426614174000",
        "https://github.com/storytold/filmcraft/issues",
        "decoded 120 frames in 4.2 ms (p95 7.1 ms)",
        "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "key frame at 00:00:01:00",
    ] {
        assert_eq!(r(keep), keep);
    }
    // JSON: secret keys at any depth, strings redacted, nesting bounded
    let mut n = 0;
    let v = redact_json(
        &json!({"a": {"refreshToken": "x", "list": [{"password": 5}], "ok": "/Users/ann/x"}, "apiKey": null, "secretEnabled": true}),
        Some("/Users/ann"),
        &mut n,
    );
    assert_eq!(v, json!({"a": {"refreshToken": "[redacted]", "list": [{"password": "[redacted]"}], "ok": "~/x"}, "apiKey": null, "secretEnabled": true}));
    assert_eq!(n, 3);
    let mut deep = json!("leaf");
    for _ in 0..200 {
        deep = json!([deep]);
    }
    let out = redact_json(&deep, None, &mut n);
    assert!(out.to_string().contains("too deep"));
    // hostile text never panics
    for t in ["", "=", ":", "token", "token:", "Bearer", "\u{0}\u{ffff}", "/Users/", "C:\\Users\\", "é@é.é", &"a=".repeat(100_000), &"/Users/x".repeat(10_000)]
    {
        let _ = redact_text(t, Some("/"));
        let _ = redact_text(t, None);
    }
}

#[test]
fn dates_are_civil() {
    assert_eq!(stamp(0), "1970-01-01 00-00-00");
    assert_eq!(stamp(1_792_580_645), "2026-10-21 11-04-05");
    assert_eq!(stamp(951_782_400), "2000-02-29 00-00-00");
    let _ = stamp(u64::MAX);
}
