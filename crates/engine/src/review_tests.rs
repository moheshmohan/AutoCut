//! The review inbox (PX8.1): comments pinned to frames, kept in `build/review/inbox.jsonl` beside
//! the saved project, on sequences built here from generated stills and titles (no media
//! committed); looks render on the CPU reference renderer.
use std::path::{Path, PathBuf};

use filmcraft_time::Tick;
use serde_json::{Value, json};

use super::*;
use crate::Session;

fn secs(s: f64) -> i64 {
    Tick::from_seconds_f64(s).0
}

/// A session saved (by path) in a folder of its own, with a 320×180 30 fps sequence (two video
/// tracks).
fn session(name: &str) -> (Session, PathBuf) {
    let dir = crate::media_test_util::tmp_dir(name);
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 320, "height": 180, "fps": 30, "video": 2, "audio": 1})).unwrap();
    s.path = Some(dir.join("edit.fcproj").display().to_string());
    (s, dir)
}

/// A still of one colour placed on `track` from `from` to `to` seconds: its clip id.
fn still(s: &mut Session, dir: &Path, name: &str, rgb: [u8; 3], track: &str, from: f64, to: f64) -> u64 {
    let path = dir.join(name);
    image::save_buffer(&path, &rgb.repeat(320 * 180), 320, 180, image::ColorType::Rgb8).unwrap();
    let item = crate::commands::import_streamed(s, path.to_str().unwrap(), None).unwrap();
    let r =
        s.execute("timeline.place", json!({"item": item.0, "track": track, "time": secs(from), "duration": secs(to) - secs(from), "audio": false})).unwrap();
    r["clips"][0].as_u64().or_else(|| r["clip"].as_u64()).unwrap()
}

fn title(s: &mut Session, text: &str, at: f64, seconds: f64) -> u64 {
    s.execute("graphics.newText", json!({"text": text, "size": 40, "seconds": seconds, "time": secs(at), "track": "V2"})).unwrap()["clip"].as_u64().unwrap()
}

fn inbox_file(dir: &Path) -> PathBuf {
    dir.join("build").join("review").join("inbox.jsonl")
}

fn ids(v: &Value) -> Vec<String> {
    v["comments"].as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap().to_string()).collect()
}

/// A stored comment line, as another writer (the app, another agent) would add it.
fn line(id: &str, seconds: f64, text: &str) -> String {
    let c = Comment {
        id: id.into(),
        sequence: 1,
        time: secs(seconds),
        seconds,
        timecode: "00:00:00:00".into(),
        clip: None,
        area: None,
        text: text.into(),
        author: "user".into(),
        created: utc_now(),
        status: Status::Open,
        note: None,
        done_at: None,
    };
    serde_json::to_string(&c).unwrap()
}

#[test]
fn comments_are_kept_in_the_build_folder_and_closed_with_a_note() {
    let (mut s, dir) = session("review-inbox");
    let a = still(&mut s, &dir, "a.png", [60, 60, 60], "V1", 0.0, 2.0);
    still(&mut s, &dir, "b.png", [200, 200, 200], "V1", 2.0, 4.0);
    let (steps, revision) = (s.history.undo.len(), s.revision);
    let r = s
        .execute("review.comment", json!({"seconds": 1.01, "clip": a, "box": [10, 20, 100, 50], "text": "  The title\tsits too low  ", "author": "user"}))
        .unwrap();
    let c = &r["comment"];
    assert_eq!(c["id"], "c1", "{r:#}");
    assert_eq!(c["time"], secs(1.0), "the start of the frame it falls in: {r:#}");
    assert_eq!((c["seconds"].as_f64(), c["timecode"].as_str()), (Some(1.0), Some("00:00:01:00")));
    assert_eq!((c["clip"].as_u64(), c["box"].clone()), (Some(a), json!([10.0, 20.0, 100.0, 50.0])));
    assert_eq!((c["text"].as_str(), c["author"].as_str(), c["status"].as_str()), (Some("The title sits too low"), Some("user"), Some("open")));
    assert!(parse_utc(c["created"].as_str().unwrap()).is_some(), "{r:#}");
    assert_eq!(r["undoable"], false, "a comment is no undo step: {r:#}");
    assert!(r["note"].as_str().unwrap().contains("review.delete"), "{r:#}");
    // one line in build/review/inbox.jsonl, with no folder in it
    let file = std::fs::read_to_string(inbox_file(&dir)).unwrap();
    assert_eq!(file.lines().count(), 1, "{file}");
    assert!(!file.contains(dir.to_str().unwrap()) && !file.contains("edit.fcproj"), "no paths are stored: {file}");
    assert_eq!(r["inbox"], inbox_file(&dir).display().to_string());
    assert_eq!((s.history.undo.len(), s.revision), (steps, revision), "the project did not change");
    // without a time: the playhead; without an author: the agent
    s.execute("playhead.set", json!({"seconds": 3.0})).unwrap();
    let r = s.execute("review.comment", json!({"text": "Hold this shot longer"})).unwrap();
    assert_eq!((r["comment"]["id"].as_str(), r["comment"]["time"].as_i64(), r["comment"]["author"].as_str()), (Some("c2"), Some(secs(3.0)), Some("agent")));
    assert!(r["comment"].get("clip").is_none() && r["comment"].get("box").is_none(), "{r:#}");
    let v = s.execute("review.inbox", json!({})).unwrap();
    assert_eq!(ids(&v), ["c1", "c2"]);
    assert_eq!((v["open"].as_u64(), v["done"].as_u64()), (Some(2), Some(0)));
    assert!(parse_utc(v["now"].as_str().unwrap()).is_some(), "{v:#}");
    // done, with a note of what was changed
    let d = s.execute("review.done", json!({"id": "c1", "note": "raised the title by 40 px"})).unwrap();
    assert_eq!(
        (d["comment"]["status"].as_str(), d["comment"]["note"].as_str(), d["again"].as_bool()),
        (Some("done"), Some("raised the title by 40 px"), Some(false))
    );
    assert!(parse_utc(d["comment"]["doneAt"].as_str().unwrap()).is_some(), "{d:#}");
    assert_eq!(ids(&s.execute("review.inbox", json!({})).unwrap()), ["c2"]);
    assert_eq!(ids(&s.execute("review.inbox", json!({"status": "done"})).unwrap()), ["c1"]);
    assert_eq!(ids(&s.execute("review.inbox", json!({"status": "all"})).unwrap()), ["c1", "c2"]);
    let again = s.execute("review.done", json!({"id": "c1", "note": "and checked it"})).unwrap();
    assert_eq!(again["again"], true);
    // markers for the app: the open ones
    let open: Vec<String> = open_comments(&s).into_iter().map(|c| c.id).collect();
    assert_eq!(open, ["c2"]);
    // `since` lists what came after
    let now = v["now"].as_str().unwrap().to_string();
    std::thread::sleep(std::time::Duration::from_millis(5));
    s.execute("review.comment", json!({"seconds": 0.5, "text": "Too dark"})).unwrap();
    assert_eq!(ids(&s.execute("review.inbox", json!({"since": now})).unwrap()), ["c3"]);
}

#[test]
fn an_unsaved_project_has_no_inbox() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 320, "height": 180, "fps": 30})).unwrap();
    for (cmd, p) in [
        ("review.comment", json!({"text": "x", "seconds": 0})),
        ("review.inbox", json!({})),
        ("review.look", json!({"id": "c1"})),
        ("review.done", json!({"id": "c1", "note": "x"})),
        ("review.delete", json!({"id": "c1"})),
    ] {
        let e = s.execute(cmd, p).unwrap_err().to_string();
        assert!(e.contains("not saved") || e.contains("empty"), "{cmd}: {e}");
        assert!(!e.contains("internal error"), "{cmd}: {e}");
    }
    assert!(open_comments(&s).is_empty());
}

#[test]
fn hostile_parameters_are_errors_not_crashes() {
    let (mut s, dir) = session("review-hostile");
    let a = still(&mut s, &dir, "a.png", [60, 60, 60], "V1", 0.0, 2.0);
    s.execute("review.comment", json!({"seconds": 0.5, "text": "fine"})).unwrap();
    let long = "x".repeat(MAX_TEXT + 1);
    let cases = [
        ("review.comment", json!({"seconds": 0.5})),
        ("review.comment", json!({"seconds": 0.5, "text": ""})),
        ("review.comment", json!({"seconds": 0.5, "text": " \n\t "})),
        ("review.comment", json!({"seconds": 0.5, "text": 5})),
        ("review.comment", json!({"seconds": 0.5, "text": long})),
        ("review.comment", json!({"seconds": -1, "text": "x"})),
        ("review.comment", json!({"seconds": 2.0, "text": "x"})),
        ("review.comment", json!({"seconds": 1e300, "text": "x"})),
        ("review.comment", json!({"seconds": "soon", "text": "x"})),
        ("review.comment", json!({"time": i64::MIN, "text": "x"})),
        ("review.comment", json!({"frame": i64::MAX, "text": "x"})),
        ("review.comment", json!({"timecode": "99:99", "text": "x"})),
        ("review.comment", json!({"seconds": 0.5, "frame": 3, "text": "x"})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "box": [1, 2, 3]})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "box": [1, 2, 3, "4"]})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "box": [-1, 2, 3, 4]})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "box": [300, 2, 100, 4]})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "box": [0, 0, 1e308, 1e308]})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "box": "all"})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "clip": 999_999})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "clip": "a"})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "author": "<script>"})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "author": "a".repeat(49)})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "author": ""})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "sequence": 999_999})),
        ("review.comment", json!({"seconds": 0.5, "text": "x", "colour": "red"})),
        ("review.comment", json!("text")),
        ("review.inbox", json!({"status": "closed"})),
        ("review.inbox", json!({"since": "yesterday"})),
        ("review.inbox", json!({"since": 5})),
        ("review.inbox", json!({"wait": true, "timeout": -1})),
        ("review.inbox", json!({"wait": true, "timeout": 601})),
        ("review.inbox", json!({"wait": true, "timeout": "x"})),
        ("review.inbox", json!({"wait": "yes"})),
        ("review.inbox", json!({"job": 1})),
        ("review.inbox", json!({"limit": 3})),
        ("review.done", json!({"id": "c1"})),
        ("review.done", json!({"id": "c1", "note": ""})),
        ("review.done", json!({"id": "c1", "note": "y".repeat(MAX_NOTE + 1)})),
        ("review.done", json!({"id": "c9", "note": "x"})),
        ("review.done", json!({"id": "x", "note": "x"})),
        ("review.done", json!({"id": "c0", "note": "x"})),
        ("review.done", json!({"id": "c01", "note": "x"})),
        ("review.done", json!({"id": "c99999999999", "note": "x"})),
        ("review.done", json!({"id": 1, "note": "x"})),
        ("review.look", json!({"id": "c1", "scale": 0})),
        ("review.look", json!({"id": "c1", "scale": 2})),
        ("review.look", json!({"id": "c1", "out": ""})),
        ("review.look", json!({"id": "c1", "inline": "yes"})),
        ("review.look", json!({"id": "c7"})),
        ("review.delete", json!({"id": "c7"})),
        ("review.delete", json!({})),
    ];
    for (cmd, p) in cases {
        let r = s.execute(cmd, p.clone());
        let e = r.expect_err(&format!("{cmd} {p} is refused")).to_string();
        assert!(!e.contains("internal error"), "{cmd} {p}: {e}");
    }
    // a box half a pixel over the edge is kept inside the frame
    let r = s.execute("review.comment", json!({"seconds": 0.5, "text": "edge", "box": [300, 170, 20.4, 10.3], "clip": a})).unwrap();
    assert_eq!(r["comment"]["box"], json!([300.0, 170.0, 20.0, 10.0]));
    assert_eq!(ids(&s.execute("review.inbox", json!({})).unwrap()), ["c1", "c2"], "only the good ones were kept");
}

#[test]
fn damaged_lines_are_skipped_with_a_note_and_kept() {
    let (mut s, dir) = session("review-damaged");
    still(&mut s, &dir, "a.png", [60, 60, 60], "V1", 0.0, 2.0);
    let file = inbox_file(&dir);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    let empty_text = line("c2", 0.5, "x").replace("\"text\":\"x\"", "\"text\":\"\"");
    let deleted = r#"{"id":"c5","deleted":"2026-11-23T10:00:00.000Z"}"#;
    let body = [line("c1", 0.5, "keep me"), "{not json".into(), empty_text, String::new(), deleted.into(), line("c1", 0.7, "twice")].join("\n");
    std::fs::write(&file, &body).unwrap();
    let v = s.execute("review.inbox", json!({"status": "all"})).unwrap();
    assert_eq!(ids(&v), ["c1"], "{v:#}");
    assert_eq!(v["damaged"]["count"], 3, "{v:#}");
    let lines: Vec<u64> = v["damaged"]["lines"].as_array().unwrap().iter().map(|d| d["line"].as_u64().unwrap()).collect();
    assert_eq!(lines, [2, 3, 6], "{v:#}");
    // a new comment takes the id after the highest ever given (the deleted c5 too)
    let r = s.execute("review.comment", json!({"seconds": 1.0, "text": "new"})).unwrap();
    assert_eq!(r["comment"]["id"], "c6");
    let after = std::fs::read_to_string(&file).unwrap();
    assert!(after.contains("{not json") && after.contains(deleted), "damaged and deleted lines are written back: {after}");
    // an inbox over its size limit is refused, never read into memory whole
    std::fs::write(&file, vec![b' '; (MAX_INBOX_BYTES + 1) as usize]).unwrap();
    let e = s.execute("review.inbox", json!({})).unwrap_err().to_string();
    assert!(e.contains("limit"), "{e}");
    // and a full inbox takes no more comments
    let full: Vec<String> = (1..=MAX_COMMENTS).map(|n| line(&format!("c{n}"), 0.5, "x")).collect();
    std::fs::write(&file, full.join("\n")).unwrap();
    let e = s.execute("review.comment", json!({"seconds": 1.0, "text": "one more"})).unwrap_err().to_string();
    assert!(e.contains("holds 2000"), "{e}");
}

#[test]
fn deleted_comments_leave_only_their_id() {
    let (mut s, dir) = session("review-delete");
    still(&mut s, &dir, "a.png", [60, 60, 60], "V1", 0.0, 2.0);
    s.execute("review.comment", json!({"seconds": 0.2, "text": "first"})).unwrap();
    s.execute("review.comment", json!({"seconds": 0.4, "text": "a private aside", "box": [1, 1, 5, 5]})).unwrap();
    let r = s.execute("review.delete", json!({"id": "c2"})).unwrap();
    assert_eq!((r["deleted"].as_str(), r["open"].as_u64()), (Some("c2"), Some(1)));
    let file = std::fs::read_to_string(inbox_file(&dir)).unwrap();
    assert!(!file.contains("private aside") && file.contains(r#""id":"c2","deleted""#), "{file}");
    assert_eq!(ids(&s.execute("review.inbox", json!({"status": "all"})).unwrap()), ["c1"]);
    assert_eq!(s.execute("review.comment", json!({"seconds": 0.6, "text": "third"})).unwrap()["comment"]["id"], "c3", "c2 is never given again");
    for (cmd, p) in [("review.delete", json!({"id": "c2"})), ("review.done", json!({"id": "c2", "note": "x"})), ("review.look", json!({"id": "c2"}))] {
        let e = s.execute(cmd, p).unwrap_err().to_string();
        assert!(e.contains("deleted"), "{cmd}: {e}");
    }
}

#[test]
fn a_wait_returns_when_a_new_comment_arrives_or_at_the_timeout() {
    let (mut s, dir) = session("review-wait");
    still(&mut s, &dir, "a.png", [60, 60, 60], "V1", 0.0, 2.0);
    s.execute("review.comment", json!({"seconds": 0.2, "text": "already here"})).unwrap();
    // nothing new: the timeout
    let r = s.execute("review.inbox", json!({"wait": true, "timeout": 0})).unwrap();
    assert_eq!((r["timedOut"].as_bool(), r["new"].as_array().map(Vec::len)), (Some(true), Some(0)), "{r:#}");
    assert_eq!(ids(&r), ["c1"]);
    // another writer adds one while the call waits
    let file = inbox_file(&dir);
    let writer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        let mut body = std::fs::read_to_string(&file).unwrap();
        body.push_str(&line("c2", 1.0, "from the app"));
        body.push('\n');
        std::fs::write(&file, body).unwrap();
    });
    let t0 = std::time::Instant::now();
    let r = s.execute("review.inbox", json!({"wait": true, "timeout": 20})).unwrap();
    writer.join().unwrap();
    assert_eq!(r["new"], json!(["c2"]), "{r:#}");
    assert_eq!(r["timedOut"], false);
    assert!(t0.elapsed() < std::time::Duration::from_secs(10), "it returned when the comment came: {:?}", t0.elapsed());
    assert!(r["waited"].as_f64().unwrap() >= 0.2, "{r:#}");
    // a comment already there after `since` answers at once
    let r = s.execute("review.inbox", json!({"wait": true, "timeout": 20, "since": "2000-01-01T00:00:00Z"})).unwrap();
    assert_eq!(r["new"], json!(["c1", "c2"]), "{r:#}");
}

/// The finished job's `jobs.list` entry (at most `limit`).
fn finished(s: &mut Session, job: u64, limit: std::time::Duration) -> Value {
    let t0 = std::time::Instant::now();
    loop {
        let jobs = s.execute("jobs.list", json!({})).unwrap();
        let j = jobs.as_array().unwrap().iter().find(|j| j["id"].as_u64() == Some(job)).cloned().expect("the job is listed");
        if j["finished"] == true {
            return j;
        }
        assert!(t0.elapsed() < limit, "the wait did not end: {j:#}");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn a_wait_as_a_job_never_holds_the_session() {
    let (mut s, dir) = session("review-job");
    still(&mut s, &dir, "a.png", [60, 60, 60], "V1", 0.0, 2.0);
    let r = s.execute("review.inbox", json!({"job": true, "timeout": 20})).unwrap();
    let job = r["job"].as_u64().unwrap();
    assert_eq!((r["poll"].as_str(), r["cancel"].as_str()), (Some("jobs.list"), Some("jobs.cancel")));
    // the session keeps answering: the comment comes through it
    std::thread::sleep(std::time::Duration::from_millis(100));
    s.execute("review.comment", json!({"seconds": 0.3, "text": "while it waits"})).unwrap();
    let j = finished(&mut s, job, std::time::Duration::from_secs(10));
    assert_eq!(j["review"]["new"], json!(["c1"]), "{j:#}");
    assert_eq!(j["review"]["comments"][0]["text"], "while it waits");
    assert!(j["status"].as_str().unwrap().contains("1 new review comment"), "{j:#}");
    // cancelled: it ends with what is there
    let r = s.execute("review.inbox", json!({"job": true, "timeout": 600})).unwrap();
    let job = r["job"].as_u64().unwrap();
    s.execute("jobs.cancel", json!({"job": job})).unwrap();
    let j = finished(&mut s, job, std::time::Duration::from_secs(5));
    assert_eq!((j["review"]["cancelled"].as_bool(), j["review"]["timedOut"].as_bool()), (Some(true), Some(false)), "{j:#}");
}

/// The pixel at (x, y) of a PNG.
fn pixel(path: &str, x: u32, y: u32) -> [u8; 3] {
    let img = image::open(path).unwrap().to_rgb8();
    img.get_pixel(x, y).0
}

#[test]
fn a_look_shows_the_frame_with_its_box_and_what_is_there() {
    let (mut s, dir) = session("review-look");
    let a = still(&mut s, &dir, "a.png", [60, 60, 60], "V1", 0.0, 2.0);
    let b = still(&mut s, &dir, "b.png", [200, 200, 200], "V1", 2.0, 4.0);
    let t = title(&mut s, "Hello", 0.5, 1.2);
    s.execute("sequence.applyVideoTransition", json!({"clip": a, "edge": "out", "effect": "cross_dissolve", "frames": 12})).unwrap();
    let srt = dir.join("words.srt");
    std::fs::write(&srt, "1\n00:00:00,500 --> 00:00:01,500\nHello there\n\n").unwrap();
    s.execute("file.import", json!({"paths": [srt.to_string_lossy()]})).unwrap();
    s.execute("review.comment", json!({"seconds": 1.0, "clip": a, "box": [40, 30, 120, 60], "text": "Title too big", "author": "user"})).unwrap();
    let r = s.execute("review.look", json!({"id": "c1", "inline": true})).unwrap();
    let f = &r["frame"];
    let path = f["path"].as_str().unwrap();
    assert_eq!(path, dir.join("build").join("review").join("look-c1.png").display().to_string());
    assert_eq!((f["width"].as_u64(), f["height"].as_u64(), f["time"].as_i64()), (Some(320), Some(180), Some(secs(1.0))), "{r:#}");
    assert_eq!(f["box"], json!([40, 30, 120, 60]));
    assert_eq!(pixel(path, 100, 30), BOX_RGB, "the outline");
    assert_eq!(pixel(path, 100, 28), EDGE_RGB, "its dark edge");
    assert_eq!(pixel(path, 40, 60), BOX_RGB, "the left side");
    assert!(r["pngBase64"].as_str().is_some_and(|p| p.len() > 100));
    let c = &r["context"];
    let under: Vec<(u64, &str, &str)> =
        c["clips"].as_array().unwrap().iter().map(|x| (x["clip"].as_u64().unwrap(), x["track"].as_str().unwrap(), x["kind"].as_str().unwrap())).collect();
    assert_eq!(under, [(a, "V1", "media"), (t, "V2", "graphic")], "{c:#}");
    assert_eq!(c["text"][0]["text"], "Hello", "{c:#}");
    assert_eq!(c["captions"][0]["text"], "Hello there", "{c:#}");
    assert_eq!(c["cuts"]["before"]["time"], secs(0.5), "the title's start: {c:#}");
    assert_eq!(c["cuts"]["after"]["time"], secs(1.7), "the title's end: {c:#}");
    assert!(c["transition"].is_null(), "{c:#}");
    assert!(c["motion"]["findings"].is_array() && c["motion"].get("error").is_none(), "{c:#}");
    // inside the dissolve, a point: a ring
    s.execute("review.comment", json!({"seconds": 2.0, "box": [160, 90, 0, 0], "text": "Jumpy dissolve"})).unwrap();
    let r = s.execute("review.look", json!({"id": "c2", "scale": 0.5})).unwrap();
    let c = &r["context"];
    assert_eq!(c["transition"]["effect"], "cross_dissolve", "{c:#}");
    assert_eq!((c["transition"]["from"].as_u64(), c["transition"]["to"].as_u64()), (Some(a), Some(b)));
    assert!((c["transition"]["progress"].as_f64().unwrap() - 0.5).abs() < 0.1, "{c:#}");
    assert_eq!((r["frame"]["width"].as_u64(), r["frame"]["box"].clone()), (Some(160), json!([80, 45, 0, 0])));
    assert!(r.get("pngBase64").is_none());
    let path = r["frame"]["path"].as_str().unwrap();
    let ring = (160 / 30).clamp(6, 60);
    assert_eq!(pixel(path, 80 + ring - 1, 45), BOX_RGB, "the ring");
    // a shorter sequence: its last frame, said so
    s.execute("review.comment", json!({"seconds": 3.5, "text": "end"})).unwrap();
    s.execute("timeline.select", json!({"clips": [b]})).unwrap();
    s.execute("edit.clear", json!({})).unwrap();
    let r = s.execute("review.look", json!({"id": "c3"})).unwrap();
    assert!(r["notes"][0].as_str().unwrap().contains("shorter"), "{r:#}");
    assert!(r["frame"]["time"].as_i64().unwrap() < secs(2.0), "{r:#}");
}
