//! The finish reviewer (PX8.3): answers parsed against the contract, rounds recorded in
//! `build/review/rounds.json`, packets gathered from sequences built here (no media committed;
//! frames on the CPU reference renderer).
use std::path::{Path, PathBuf};

use filmcraft_time::Tick;
use serde_json::{Value, json};

use super::*;
use crate::Session;

fn secs(s: f64) -> i64 {
    Tick::from_seconds_f64(s).0
}

fn session(name: &str) -> (Session, PathBuf) {
    let dir = crate::media_test_util::tmp_dir(name);
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 320, "height": 180, "fps": 30, "video": 2, "audio": 1})).unwrap();
    s.path = Some(dir.join("edit.fcproj").display().to_string());
    (s, dir)
}

fn still(s: &mut Session, dir: &Path, name: &str, rgb: [u8; 3], from: f64, to: f64) -> u64 {
    let path = dir.join(name);
    image::save_buffer(&path, &rgb.repeat(320 * 180), 320, 180, image::ColorType::Rgb8).unwrap();
    let item = crate::commands::import_streamed(s, path.to_str().unwrap(), None).unwrap();
    let r = s.execute("timeline.place", json!({"item": item.0, "track": "V1", "time": secs(from), "duration": secs(to) - secs(from), "audio": false})).unwrap();
    r["clips"][0].as_u64().or_else(|| r["clip"].as_u64()).unwrap()
}

const FIX_TWO: &str = "The opening works; two things slow it down.\n\
FIX 1: 00:00:01:12 (c3) the title comes in too fast: ease it over half a second\n\
FIX 2: 12.5s the dissolve dips to grey; use a straight cut\n\
KEEP: the music hit at 0:42\n\
GATE: fix 2 keep 1\n\n";

#[test]
fn answers_follow_the_contract() {
    let a = parse_answer(FIX_TWO).unwrap();
    assert_eq!(a.verdict, Verdict::Fix);
    assert_eq!(a.fixes.len(), 2);
    assert_eq!((a.fixes[0].comments.clone(), a.fixes[0].times.clone()), (vec!["c3".to_string()], vec!["00:00:01:12".to_string()]));
    assert_eq!(a.fixes[1].times, ["12.5s"]);
    assert_eq!((a.keep.len(), a.keep[0].times.clone()), (1, vec!["0:42".to_string()]));
    assert!(a.warnings.is_empty(), "{:?}", a.warnings);
    // ship, with a note; markdown and case around the gate
    let a = parse_answer("Clean, it reads well.\n**GATE:** SHIP — nothing left").unwrap();
    assert_eq!((a.verdict, a.summary.as_str()), (Verdict::Ship, "nothing left"));
    // counts left out are taken from the list
    let a = parse_answer("- FIX: the logo at c1 is cut off\nGATE: fix").unwrap();
    assert_eq!(a.fixes.len(), 1);
    for (text, verdict, why) in [
        ("GATE: rebuild - the story starts in the middle", Verdict::Rebuild, "the story starts in the middle"),
        ("GATE: recapture — the drone shot is soft", Verdict::Recapture, "the drone shot is soft"),
        ("> GATE: recapture: a cleaner take of line 3", Verdict::Recapture, "a cleaner take of line 3"),
    ] {
        let a = parse_answer(text).unwrap();
        assert_eq!((a.verdict, a.summary.as_str()), (verdict, why), "{text}");
    }
    // warnings, not refusals: fixes that say not where, numbers out of order, fixes on a rebuild
    let a = parse_answer("FIX 2: tighten the middle\nFIX 1: at c4 lift the voice\nGATE: fix 2").unwrap();
    assert_eq!(a.warnings.len(), 2, "{:?}", a.warnings);
    let a = parse_answer("FIX 1: at 3s x\nGATE: rebuild - it drags").unwrap();
    assert!(a.warnings[0].contains("notes"), "{:?}", a.warnings);
}

#[test]
fn malformed_answers_are_refused() {
    let nine: String = (1..=9).map(|n| format!("FIX {n}: at {n}s change {n}\n")).collect::<String>() + "GATE: fix 9";
    let long = format!("{}\nGATE: ship", "x".repeat(MAX_ANSWER));
    for (text, says) in [
        ("Looks good to me.", "no `GATE:`"),
        ("GATE: fix 1\nFIX 1: at 2s x\nGATE: fix 1", "2 `GATE:` lines"),
        ("FIX 1: at 2s x\nGATE: fix 1\nThanks!", "not the last line"),
        ("GATE: maybe", "not a verdict"),
        ("GATE:", "no verdict"),
        ("GATE: fix", "lists its fixes"),
        (nine.as_str(), "at most 8"),
        ("FIX 1: at 2s x\nKEEP: a\nKEEP: b\nGATE: fix 1", "KEEP lines"),
        ("FIX 1: at 2s x\nFIX 2: at 3s y\nGATE: fix 3", "says fix 3"),
        ("FIX 1: at 2s x\nGATE: fix 1 keep 1", "says keep 1"),
        ("FIX 1: at 2s x\nGATE: ship", "a ship answer lists no FIX"),
        ("GATE: rebuild", "say why"),
        ("GATE: recapture", "say what"),
        ("FIX 1: at 2s x\nGATE: fix 1 please", "unexpected `please`"),
        ("FIX 1:\nGATE: fix 1", "has no text"),
        ("KEEP:\nGATE: ship", "KEEP line has no text"),
        (long.as_str(), "bytes"),
    ] {
        let e = parse_answer(text).expect_err(text);
        assert!(e.contains(says), "{text:?}: {e}");
    }
}

#[test]
fn a_review_run_has_two_rounds_at_most() {
    let (mut s, dir) = session("review-gate");
    still(&mut s, &dir, "a.png", [60, 60, 60], 0.0, 2.0);
    let file = dir.join("build").join("review").join(ROUNDS);
    let r = s.execute("review.gate", json!({"text": FIX_TWO, "packet": dir.join("build/review/packet-1").to_str().unwrap()})).unwrap();
    assert_eq!((r["recorded"].as_bool(), r["run"].as_u64(), r["round"].as_u64()), (Some(true), Some(1), Some(1)), "{r:#}");
    assert_eq!((r["gate"]["verdict"].as_str(), r["blocksDelivery"].as_bool()), (Some("fix"), Some(false)));
    assert!(r["closed"].is_null() && r["next"].as_str().unwrap().contains("round 2"), "{r:#}");
    let stored = std::fs::read_to_string(&file).unwrap();
    assert!(stored.contains("\"packet\": \"packet-1\"") && !stored.contains(dir.to_str().unwrap()), "the packet's name only: {stored}");
    // the round is checked; a malformed answer records nothing
    let e = s.execute("review.gate", json!({"text": FIX_TWO, "round": 1})).unwrap_err().to_string();
    assert!(e.contains("round 2 of review run 1"), "{e}");
    let e = s.execute("review.gate", json!({"text": "Fine.\nGATE: perhaps"})).unwrap_err().to_string();
    assert!(e.contains("not a verdict"), "{e}");
    assert_eq!(std::fs::read_to_string(&file).unwrap(), stored, "nothing recorded");
    // a dry run records nothing either
    let r = s.execute("review.gate", json!({"text": "GATE: ship", "dryRun": true})).unwrap();
    assert_eq!((r["recorded"].as_bool(), r["round"].as_u64()), (Some(false), Some(2)));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), stored);
    // round two without a ship: the run is with the user
    let r = s.execute("review.gate", json!({"text": "FIX 1: at 1s still too fast\nGATE: fix 1", "round": 2})).unwrap();
    assert_eq!((r["round"].as_u64(), r["closed"]["by"].as_str(), r["closed"]["verdict"].as_str()), (Some(2), Some("limit"), Some("fix")), "{r:#}");
    for p in [json!({"text": "GATE: ship"}), json!({"text": "GATE: ship", "round": 1})] {
        let e = s.execute("review.gate", p).unwrap_err().to_string();
        assert!(e.contains("the user decides"), "{e}");
    }
    let state = s.execute("review.gate", json!({})).unwrap();
    assert!(state["next"]["refused"].as_str().is_some_and(|w| w.contains("user")), "{state:#}");
    // the user's own words end it, and a new run can start
    let r = s.execute("review.gate", json!({"user": "Ship it, it is fine as it is.", "verdict": "ship"})).unwrap();
    assert_eq!((r["closed"]["by"].as_str(), r["closed"]["words"].as_str()), (Some("user"), Some("Ship it, it is fine as it is.")));
    let r = s.execute("review.gate", json!({"text": "GATE: ship"})).unwrap();
    assert_eq!((r["run"].as_u64(), r["round"].as_u64(), r["closed"]["by"].as_str()), (Some(2), Some(1), Some("gate")), "{r:#}");
    let e = s.execute("review.gate", json!({"user": "stop"})).unwrap_err().to_string();
    assert!(e.contains("closed already"), "{e}");
    // the user's words close an open run early
    s.execute("review.gate", json!({"text": FIX_TWO})).unwrap();
    let r = s.execute("review.gate", json!({"user": "Good enough, I will post it."})).unwrap();
    assert!(r["note"].as_str().unwrap().contains("closed early"), "{r:#}");
    let state = s.execute("review.gate", json!({})).unwrap();
    assert_eq!(state["runs"].as_array().unwrap().len(), 3, "{state:#}");
    assert_eq!(state["next"], json!({"run": 4, "round": 1}));
    // a third round is refused outright, and so are hostile parameters
    for p in [
        json!({"text": "GATE: ship", "round": 3}),
        json!({"text": "GATE: ship", "round": 0}),
        json!({"text": "GATE: ship", "round": "one"}),
        json!({"text": 5}),
        json!({"text": "GATE: ship", "user": "x"}),
        json!({"user": ""}),
        json!({"user": "x".repeat(1001)}),
        json!({"user": "fine", "verdict": "maybe"}),
        json!({"verdict": "ship"}),
        json!({"text": "GATE: ship", "verdict": "ship"}),
        json!({"text": "GATE: ship", "packet": ""}),
        json!({"text": "GATE: ship", "dryRun": "yes"}),
        json!({"text": "GATE: ship", "colour": 1}),
    ] {
        let e = s.execute("review.gate", p.clone()).expect_err(&p.to_string()).to_string();
        assert!(!e.contains("internal error"), "{p}: {e}");
    }
    // a damaged rounds file is refused, not overwritten
    std::fs::write(&file, b"{not json").unwrap();
    let e = s.execute("review.gate", json!({"text": "GATE: ship"})).unwrap_err().to_string();
    assert!(e.contains("move it aside"), "{e}");
    assert_eq!(std::fs::read(&file).unwrap(), b"{not json");
    // an unsaved project has no rounds
    s.path = None;
    assert!(s.execute("review.gate", json!({"text": "GATE: ship"})).unwrap_err().to_string().contains("not saved"));
}

/// The finished job's `jobs.list` entry.
fn finished(s: &mut Session, job: u64) -> Value {
    let t0 = std::time::Instant::now();
    loop {
        let jobs = s.execute("jobs.list", json!({})).unwrap();
        let j = jobs.as_array().unwrap().iter().find(|j| j["id"].as_u64() == Some(job)).cloned().expect("listed");
        if j["finished"] == true {
            return j;
        }
        assert!(t0.elapsed() < std::time::Duration::from_secs(60), "{j:#}");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn a_packet_gathers_the_evidence_for_a_fresh_reviewer() {
    let (mut s, dir) = session("review-packet");
    let a = still(&mut s, &dir, "a.png", [60, 60, 60], 0.0, 2.0);
    still(&mut s, &dir, "b.png", [200, 200, 200], 2.0, 4.0);
    s.execute("sequence.applyVideoTransition", json!({"clip": a, "edge": "out", "effect": "cross_dissolve", "frames": 12})).unwrap();
    let t = s.execute("graphics.newText", json!({"text": "Hello", "size": 40, "seconds": 1.2, "time": secs(0.5), "track": "V2"})).unwrap()["clip"]
        .as_u64()
        .unwrap();
    s.execute("graphics.animateText", json!({"clip": t, "in": "fade"})).unwrap();
    s.execute("review.comment", json!({"seconds": 1.0, "text": "Title too big", "author": "user"})).unwrap();
    s.execute("review.comment", json!({"seconds": 3.0, "text": "Fine now"})).unwrap();
    s.execute("review.done", json!({"id": "c2", "note": "checked"})).unwrap();
    let build = dir.join("build");
    std::fs::write(build.join("decisions.json"), br#"{"waivers":[]}"#).unwrap();
    std::fs::write(build.join("edit.qa.json"), br#"{"format":"autocut.deliveryQa","passed":true,"summary":{"floor":0}}"#).unwrap();
    let r = s.execute("review.packet", json!({"wait": true})).unwrap();
    let folder = PathBuf::from(r["folder"].as_str().unwrap());
    assert_eq!(folder, build.join("review").join("packet-1"));
    let files: Vec<&str> = r["files"].as_array().unwrap().iter().map(|f| f.as_str().unwrap()).collect();
    for f in ["brief.md", "rounds.json", "sequence.json", "frames.jpg", "motion.json", "qa.json", "decisions.json", "comments.json", "manifest.json"] {
        assert!(files.contains(&f), "{f} in {files:?}");
        assert!(folder.join(f).is_file(), "{f}");
    }
    assert!(r["missing"].as_array().unwrap().is_empty(), "{r:#}");
    assert_eq!((r["round"].as_u64(), r["run"].as_u64()), (Some(1), Some(1)));
    assert!(std::fs::read_to_string(folder.join("brief.md")).unwrap().contains("this packet is round 1"));
    assert_eq!(r["qa"]["from"], "edit.qa.json", "the newest QA report, by name: {r:#}");
    assert!(r["strips"]["sheets"].as_u64().unwrap() >= 2, "the title's fade and the dissolve: {r:#}");
    assert!(folder.join("strips").is_dir());
    let comments: Value = serde_json::from_slice(&std::fs::read(folder.join("comments.json")).unwrap()).unwrap();
    assert_eq!((comments["open"][0]["id"].as_str(), comments["done"][0]["note"].as_str()), (Some("c1"), Some("checked")));
    // a frame of each shot: 0–0.5, 0.5–1.7, 1.7–2, 2–4, spread to eight
    let sheet = &r["sheet"];
    assert_eq!(sheet["frames"], 8, "{sheet:#}");
    let img = image::open(folder.join("frames.jpg")).unwrap();
    assert_eq!((u64::from(img.width()), u64::from(img.height())), (sheet["width"].as_u64().unwrap(), sheet["height"].as_u64().unwrap()));
    let seq: Value = serde_json::from_slice(&std::fs::read(folder.join("sequence.json")).unwrap()).unwrap();
    assert_eq!((seq["shots"].as_u64(), seq["titles"][0]["text"].as_str()), (Some(4), Some("Hello")), "{seq:#}");
    // the next packet is the next folder; a folder with files in it is refused
    let r = s.execute("review.packet", json!({"strips": false, "frames": 3})).unwrap();
    assert!(r["folder"].as_str().unwrap().ends_with("packet-2"), "{r:#}");
    assert_eq!(r["sheet"]["frames"], 3);
    assert!(r["missing"].as_array().unwrap().iter().any(|m| m["file"] == "strips/"), "{r:#}");
    let e = s.execute("review.packet", json!({"out": folder.to_str().unwrap()})).unwrap_err().to_string();
    assert!(e.contains("not empty"), "{e}");
    // filmstrips as a job: its id, and what it made in jobs.list
    let r = s.execute("review.packet", json!({"out": dir.join("p3").to_str().unwrap()})).unwrap();
    let job = r["job"].as_u64().unwrap_or_else(|| panic!("a strips job: {r:#}"));
    let j = finished(&mut s, job);
    assert!(j["motionStrips"]["sheets"].as_array().is_some_and(|a| a.len() >= 2), "{j:#}");
    // no QA report and no decisions: said what is missing
    std::fs::remove_file(build.join("edit.qa.json")).unwrap();
    std::fs::remove_file(build.join("decisions.json")).unwrap();
    let r = s.execute("review.packet", json!({"strips": false})).unwrap();
    let missing: Vec<&str> = r["missing"].as_array().unwrap().iter().map(|m| m["file"].as_str().unwrap()).collect();
    assert_eq!(missing, ["qa.json", "decisions.json", "strips/"], "{r:#}");
    // after a gate, the next packet is round 2
    s.execute("review.gate", json!({"text": FIX_TWO})).unwrap();
    let r = s.execute("review.packet", json!({"strips": false})).unwrap();
    assert_eq!(r["round"], 2);
    // hostile parameters are refused
    for p in [
        json!({"frames": 0}),
        json!({"frames": 121}),
        json!({"frames": "x"}),
        json!({"out": ""}),
        json!({"qa": ""}),
        json!({"qa": "a.qa.json", "preflight": true}),
        json!({"wait": "yes"}),
        json!({"sequence": 999_999}),
        json!({"pages": 2}),
    ] {
        let e = s.execute("review.packet", p.clone()).expect_err(&p.to_string()).to_string();
        assert!(!e.contains("internal error"), "{p}: {e}");
    }
    let e = s.execute("review.packet", json!({"qa": dir.join("missing.qa.json").to_str().unwrap(), "strips": false})).unwrap();
    assert!(e["missing"][0]["why"].as_str().unwrap().contains("could not be read"), "{e:#}");
}

/// The contract and the brief say the same thing.
#[test]
fn the_contract_doc_states_the_brief_and_its_grammar() {
    let doc = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/finish-reviewer.md")).unwrap();
    for line in BRIEF.lines().filter(|l| l.trim_start().starts_with("GATE:")) {
        assert!(doc.contains(line.trim()), "the doc has `{}`", line.trim());
    }
    for needle in ["most eight fixes", "one keep", "two rounds", "user's own words", "review.packet", "review.gate", "blocksDelivery"] {
        assert!(doc.contains(needle), "the doc says {needle}");
    }
}
