//! Showcase QA against a real headless session: a deliberately broken sequence gets each of its
//! defects reported, a clean showcase gets no errors.

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::qa::{self, Severity};
use filmcraft_autocut::showcase::{self, Options, Script};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use serde_json::{Value, json};

const TPS: f64 = 254_016_000_000.0;

fn demo() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    SessionExecutor::from(s)
}

fn ticks(s: f64) -> i64 {
    (s * TPS).round() as i64
}

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-qa-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().into_owned()
}

async fn run(ex: &SessionExecutor, id: &str, p: Value) -> Value {
    ex.execute(id, p.clone()).await.unwrap_or_else(|e| panic!("{id} {p}: {e}"))
}

/// `item`'s `[src, src + dur)` at `t` on `track` (a video track: its sound, if any, on `audio`;
/// an audio track: the sound alone).
async fn place(ex: &SessionExecutor, item: u64, track: &str, audio: Option<&str>, t: f64, src: f64, dur: f64) -> Vec<u64> {
    let mut p = json!({"item": item, "track": track, "time": ticks(t), "sourceIn": ticks(src), "duration": ticks(dur)});
    match audio {
        Some(a) => p["audioTrack"] = json!(a),
        None if track.starts_with('V') => p["audio"] = json!(false),
        None => {}
    }
    let r = run(ex, "timeline.place", p).await;
    r["clips"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_broken_sequence_gets_each_defect_reported() {
    let ex = demo();
    let dir = tmp("broken");
    run(&ex, "file.newSequence", json!({"name": "Broken", "width": 640, "height": 360, "fps": 30, "video": 3, "audio": 3})).await;
    // V2: a shot, half a second of nothing, a shot, a black slate, a shot
    place(&ex, 5, "V2", Some("A1"), 0.0, 1.0, 2.0).await;
    place(&ex, 6, "V2", Some("A1"), 2.5, 1.0, 2.0).await;
    let slate = run(&ex, "file.newBlackVideo", json!({"seconds": 3})).await["item"].as_u64().unwrap();
    place(&ex, slate, "V2", None, 4.5, 0.0, 1.5).await;
    place(&ex, 7, "V2", Some("A1"), 6.0, 1.0, 3.0).await;
    // a counter-like title (V3) over the slate: black under it is still black
    let title = json!({"text": "ROBOT #2", "seconds": 6, "time": ticks(3.0), "size": 28, "position": [560, 30], "track": 2});
    run(&ex, "graphics.newText", title).await;
    // two narration lines on A2, and between them a tone pushed far past full scale on A3
    let voice = |text: &str, t: f64| json!({"text": text, "voice": "basic-male", "time": ticks(t), "track": "A2", "dir": dir});
    run(&ex, "tts.create", voice("Hello and welcome to the show, everyone.", 0.0)).await;
    run(&ex, "tts.create", voice("And here is the very next one.", 6.0)).await;
    let bars = run(&ex, "file.newBarsAndTone", json!({"seconds": 10})).await["item"].as_u64().unwrap();
    let tone = place(&ex, bars, "A3", None, 3.0, 0.0, 2.8).await;
    run(&ex, "clip.audioGain", json!({"clips": tone, "mode": "set", "db": 30})).await;

    let opts = qa::Options { frame_side: 160, work_dir: Some(dir.clone().into()), ..qa::Options::default() };
    let mut lines = Vec::new();
    let r = qa::check(&ex, &opts, None, &mut |l| lines.push(l.to_string())).await;
    println!("{}", r.summary());
    for f in &r.findings {
        println!("{:?} {} @{:.3}: {}", f.severity, f.check, f.seconds, f.message);
    }
    let near = |check: &str, a: f64, b: f64| r.findings.iter().any(|f| f.check == check && f.severity == Severity::Error && f.seconds >= a && f.seconds <= b);
    // (a) the gap on V2, and nothing on any video track there
    assert!(near("joins", 1.99, 2.01), "the gap after the first shot");
    assert!(r.findings.iter().any(|f| f.check == "joins" && f.message.contains("nothing on any video track") && (f.seconds - 2.0).abs() < 0.01));
    // (b) black frames: in the gap and on the slate
    assert!(near("black", 1.99, 2.5), "black in the gap");
    assert!(near("black", 4.5, 6.0), "the black slate");
    assert!(!near("black", 0.0, 1.9), "the first shot is not black");
    // (c) the mix clips, and the voice is quieter than what plays between its lines
    let mix = r.of("mix", Severity::Error);
    assert!(mix.iter().any(|f| f.message.contains("clips")), "{mix:?}");
    assert!(mix.iter().any(|f| f.message.contains("quieter than")), "{mix:?}");
    let l = r.loudness.as_ref().expect("the mix was measured");
    assert!(l.true_peak_dbtp.unwrap() > 0.0 && l.first_over.is_some_and(|t| (2.9..5.9).contains(&t)), "{l:?}");
    assert!(l.voice_margin_db.unwrap() < 0.0, "{l:?}");
    // (d) no model here: recorded as skipped
    let subject = r.checks.iter().find(|c| c.name == "subject").unwrap();
    assert_eq!((subject.status, subject.note.as_deref()), ("skipped", Some("model unavailable")));
    // a contact sheet, one tile per shot (no entries given: each clip on V2 is one), marked
    let sheet = image::load_from_memory(r.contact_sheet_png.as_ref().unwrap()).unwrap();
    assert_eq!(sheet.width(), 4 * (160 + 4) + 4);
    assert!(r.errors >= 5 && r.frames_rendered > 4, "{}", r.summary());
    assert!(lines.iter().any(|l| l.contains("the mix")), "{lines:?}");
    // the temporary mixdown is gone
    assert!(std::fs::read_dir(&dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().starts_with("filmcraft-qa-")));
    // and the report goes next to the export
    let out = std::path::Path::new(&dir).join("broken.mp4");
    let mut r = r;
    let (json_path, png_path) = qa::write_next_to(&mut r, &out).unwrap();
    let v: Value = serde_json::from_slice(&std::fs::read(&json_path).unwrap()).unwrap();
    assert_eq!(v["errors"].as_u64().unwrap() as usize, r.errors);
    assert!(png_path.unwrap().exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn shots_the_model_cannot_find_the_subject_in_are_flagged() {
    let ex = demo();
    run(&ex, "file.newSequence", json!({"name": "Shots", "width": 640, "height": 360, "fps": 30, "video": 3, "audio": 3})).await;
    for (i, item) in [5u64, 6, 7, 5].iter().enumerate() {
        place(&ex, *item, "V2", None, i as f64 * 2.0, 1.0, 2.0).await;
    }
    let found = json!({"present": true, "bbox_2d": [100, 100, 500, 400]});
    // the second shot shows no robot; the fourth gets no answer (the model ran out)
    let model = filmcraft_autocut::llm::FixedModel::new(vec![found.clone(), json!({"present": false, "bbox_2d": null}), found]);
    let opts = qa::Options { frame_side: 160, mix: false, ..qa::Options::default() };
    let r = qa::check(&ex, &opts, Some(&model), &mut |_| {}).await;
    let missing = r.of("subject", Severity::Warning);
    assert_eq!(missing.len(), 1, "{:#?}", r.findings);
    assert_eq!((missing[0].entry, missing[0].seconds.floor()), (Some(2), 3.0), "the middle of the second shot");
    assert!(missing[0].message.contains("no robot found"), "{}", missing[0].message);
    let c = r.checks.iter().find(|c| c.name == "subject").unwrap();
    assert_eq!(c.status, "findings");
    assert!(c.note.as_deref().is_some_and(|n| n.contains("3 shots checked") && n.contains("1 unanswered")), "{c:?}");
    // what the model saw: the grounding prompt with a rendered frame
    let asked = model.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 4);
    assert!(asked[0].1.contains("Is there a robot in this image?"));
    // a model that never answers: the check is skipped, not failed
    let mute = filmcraft_autocut::llm::FixedModel::new(vec![]);
    let r = qa::check(&ex, &opts, Some(&mute), &mut |_| {}).await;
    let c = r.checks.iter().find(|c| c.name == "subject").unwrap();
    assert_eq!(c.status, "skipped", "{c:?}");
    assert!(r.of("subject", Severity::Info).is_empty());
    assert_eq!(r.errors, 0, "{:#?}", r.findings);
}

fn fact(item: u64, label: &str, unix: i64, seconds: f64) -> ClipFacts {
    ClipFacts {
        item,
        name: format!("{item}.mp4"),
        seconds,
        created: None,
        unix: Some(unix),
        present: true,
        coverage: 1.0,
        label: label.into(),
        kind: "humanoid".into(),
        description: format!("a {label}"),
        best: seconds / 2.0,
        frames: vec![FrameFact { seconds: seconds / 2.0, present: true, area: 0.4, center: [0.5, 0.5] }],
        ..ClipFacts::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_clean_showcase_has_no_errors() {
    let ex = demo();
    let dir = tmp("clean");
    // the setup of showcase_build.rs: three "robots" over two years in the demo footage
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 18.0),
        fact(6, "Boston Dynamics Spot", 1_665_741_668, 16.0),
        fact(7, "red toy robot", 1_665_841_668, 14.0),
    ];
    let entries = showcase::group(&facts, "robot", 0.3);
    let script = Script {
        intro: "Every robot I have filmed, in order.".into(),
        lines: vec!["One.".into(), "Two.".into(), "Three.".into()],
        outro: "Thanks for watching.".into(),
        templated: 0,
        ..Script::default()
    };
    let opts = Options { title: "Robots".into(), width: 640, height: 360, ..Options::default() };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    assert_eq!(built.layout.len(), 3, "{:?}", built.layout);
    assert!(built.layout.iter().all(|e| !e.shots.is_empty() && e.end > e.start));
    let q = qa::Options { frame_side: 160, work_dir: Some(dir.clone().into()), ..qa::Options::for_showcase(&opts, &built) };
    let r = qa::check(&ex, &q, None, &mut |_| {}).await;
    println!("{}", r.summary());
    for f in &r.findings {
        println!("{:?} {} @{:.3}: {}", f.severity, f.check, f.seconds, f.message);
    }
    assert_eq!(r.errors, 0, "{:#?}", r.findings);
    for name in ["joins", "black", "mix"] {
        assert_ne!(r.checks.iter().find(|c| c.name == name).map(|c| c.status), Some("skipped"), "{name} ran: {:?}", r.checks);
    }
    assert!(r.loudness.as_ref().is_some_and(|l| l.integrated_lufs.is_some()));
    // one tile per entry
    let sheet = image::load_from_memory(r.contact_sheet_png.as_ref().unwrap()).unwrap();
    assert_eq!(sheet.width(), 3 * (160 + 4) + 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_narrated_showcase_keeps_the_voice_on_top() {
    let ex = demo();
    let dir = tmp("narrated");
    // short clips with long lines (slowed shots), as in showcase_build.rs, and a third robot
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 3.5),
        fact(6, "Boston Dynamics Spot", 1_600_100_000, 3.5),
        fact(7, "red toy robot", 1_665_841_668, 14.0),
    ];
    let entries = showcase::group(&facts, "robot", 0.3);
    let long = "This one walked right up to me in the middle of a very busy hall and then it started to dance for everyone who was watching it.";
    let script = Script {
        intro: "Hello and welcome, this is every robot I have met.".into(),
        lines: vec![long.into(), long.into(), "A small red toy robot that I found at a market stall.".into()],
        outro: "Bye for now, see you soon at the next one.".into(),
        templated: 0,
        ..Script::default()
    };
    let opts = Options { title: "Robots".into(), voice: Some("basic-male".into()), width: 640, height: 360, ..Options::default() };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    let q = qa::Options { frame_side: 160, work_dir: Some(dir.clone().into()), ..qa::Options::for_showcase(&opts, &built) };
    let r = qa::check(&ex, &q, None, &mut |_| {}).await;
    println!("{}\n{:?}", r.summary(), r.loudness);
    assert_eq!(r.errors, 0, "{:#?}", r.findings);
    // the mix puts the voice well above the music and the shots' sound between its lines, and
    // the limiter keeps the true peak at -1 dBTP
    let l = r.loudness.as_ref().unwrap();
    assert!(l.voice_margin_db.is_some_and(|m| m >= 4.0), "{l:?}");
    assert!(l.true_peak_dbtp.is_some_and(|tp| tp <= -0.95), "{l:?}");
    assert!(r.of("mix", Severity::Warning).iter().all(|f| !f.message.contains("true peak")), "{:?}", r.findings);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_showcase_on_the_users_music_has_no_errors() {
    let ex = demo();
    let dir = tmp("music");
    // a track shorter than the show (remixed to length), loud: it is levelled and ducked
    let g = run(
        &ex,
        "audio.generate",
        json!({"kind": "music", "duration": 14, "bpm": 110, "seed": 5, "intensity": 1.0, "soundfont": "oscillators", "place": false, "dir": dir}),
    )
    .await;
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 18.0),
        fact(6, "Boston Dynamics Spot", 1_665_741_668, 16.0),
        fact(7, "red toy robot", 1_665_841_668, 14.0),
    ];
    let entries = showcase::group(&facts, "robot", 0.3);
    let script = Script {
        intro: "Hello and welcome, this is every robot I have met.".into(),
        lines: vec!["A friendly humanoid at a shop.".into(), "Spot walked right up to me.".into(), "A small red toy robot from a market stall.".into()],
        outro: "Bye for now, see you soon at the next one.".into(),
        ..Script::default()
    };
    let music = g["path"].as_str().map(str::to_string);
    let opts = Options { title: "Robots".into(), voice: Some("basic-male".into()), music, width: 640, height: 360, ..Options::default() };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    assert_eq!(built.music.as_ref().map(|m| m.fit.as_str()), Some("remixed"), "{:?}", built.music);
    let q = qa::Options { frame_side: 160, work_dir: Some(dir.clone().into()), ..qa::Options::for_showcase(&opts, &built) };
    let r = qa::check(&ex, &q, None, &mut |_| {}).await;
    println!("{}\n{:?}", r.summary(), r.loudness);
    assert_eq!(r.errors, 0, "{:#?}", r.findings);
    let l = r.loudness.as_ref().unwrap();
    assert!(l.voice_margin_db.is_some_and(|m| m >= 4.0), "{l:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
