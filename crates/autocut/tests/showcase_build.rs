//! The showcase build against a real headless session: every engine command it issues must
//! exist, accept its parameters and leave a playable sequence.

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::qa;
use filmcraft_autocut::showcase::{self, Options, Script};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use serde_json::json;

fn demo() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    SessionExecutor::from(s)
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
async fn builds_a_counted_showcase_on_the_demo_media() {
    let ex = demo();
    // three "robots" over two years, in the demo's footage (items 5, 6, 7)
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 18.0),
        fact(6, "Boston Dynamics Spot", 1_665_741_668, 16.0),
        fact(7, "red toy robot", 1_665_841_668, 14.0),
    ];
    let entries = showcase::group(&facts, "robot", 0.3);
    assert_eq!(entries.len(), 3);
    let script = Script {
        intro: "Every robot I have filmed, in order.".into(),
        lines: vec!["One.".into(), "Two.".into(), "Three.".into()],
        outro: "Thanks for watching.".into(),
        templated: 0,
        ..Script::default()
    };
    let opts = Options { title: "Robots".into(), ..Options::default() };
    let mut lines = Vec::new();
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |l| lines.push(l.to_string())).await.expect("the showcase builds");
    assert_eq!(built.entries, 3);
    assert!(built.seconds > 15.0, "{built:?}");
    assert!(built.notes.is_empty(), "every transition applied: {:?}", built.notes);
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    assert_eq!((seq["settings"]["width"].as_u64(), seq["settings"]["height"].as_u64()), (Some(1920), Some(1080)));
    let names: Vec<String> = seq["video"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|t| t["items"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["name"].as_str().map(str::to_string))
        .collect();
    assert!(names.iter().any(|n| n == "Counter"), "the counter is on screen: {names:?}");
    let audio: Vec<String> = seq["audio"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|t| t["items"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["name"].as_str().map(str::to_string))
        .collect();
    assert!(audio.iter().any(|n| n.starts_with("Music")), "a music bed: {audio:?}");
    assert!(audio.iter().filter(|n| n.starts_with("Whoosh")).count() >= 2, "whooshes at the cuts: {audio:?}");
    assert!(audio.iter().any(|n| n.starts_with("Impact")), "impacts: {audio:?}");
    // a frame renders mid-show
    let png = ex.render(built.seconds / 2.0, 320).await;
    assert!(png.is_some_and(|p| p.starts_with(b"\x89PNG")));
    assert_eq!(lines.iter().filter(|l| l.starts_with('[')).count(), 3, "a line per entry: {lines:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn narration_never_overlaps_and_short_footage_is_slowed_to_fit() {
    let ex = demo();
    // short clips (3.5 s) with lines too long for them
    let facts = vec![fact(5, "Pepper humanoid robot", 1_600_000_000, 3.5), fact(6, "Boston Dynamics Spot", 1_600_100_000, 3.5)];
    let entries = showcase::group(&facts, "robot", 0.3);
    let long = "This one walked right up to me in the middle of a very busy hall and then it started to dance for everyone who was watching it.";
    let script = Script {
        intro: "Hello and welcome.".into(),
        lines: vec![long.into(), long.into()],
        outro: "Bye for now, see you soon.".into(),
        templated: 0,
        ..Script::default()
    };
    let opts = Options { title: "Robots".into(), voice: Some("basic-male".into()), ..Options::default() };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    assert_eq!(built.narration_lines, 4, "intro, two lines, outro: {built:?}");
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    // the voice track (A2) has its four narrations in order, none overlapping
    let a2 = seq["audio"][1]["items"].as_array().cloned().unwrap_or_default();
    assert_eq!(a2.len(), 4, "{a2:?}");
    for w in a2.windows(2) {
        assert!(w[0]["end"].as_i64() <= w[1]["start"].as_i64(), "narrations overlap: {w:?}");
    }
    // the shots of the short clips were slowed down (speed below 100 %)
    let v2 = seq["video"][1]["items"].as_array().cloned().unwrap_or_default();
    assert!(v2.iter().any(|c| c["speed"].as_f64().is_some_and(|s| s < 0.999)), "{v2:?}");
    // shots meet on frames: no clip of V2 overlaps the next, and no gap is a frame or longer
    let frame = 254_016_000_000 / 30;
    for w in v2.windows(2) {
        let (end, start) = (w[0]["end"].as_i64().unwrap(), w[1]["start"].as_i64().unwrap());
        assert!(end <= start && start - end < frame, "shots overlap or leave a gap: {w:?}");
    }
    // the mix: the voice is dialogue, the shots' sound ambience ducked under it, and a limiter
    // keeps the mix below full scale
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    let es = ex.execute("essentialSound.inspect", json!({"clips": a2.iter().filter_map(|c| c["clip"].as_u64()).collect::<Vec<_>>()})).await.unwrap();
    assert!(es.to_string().contains("dialogue") || es.to_string().contains("Dialogue"), "{es}");
    let mixer = ex.execute("mixer.inspect", json!({})).await.unwrap();
    assert!(mixer.to_string().contains("hard_limiter"), "{mixer}");
}

/// The three demo "robots" of the first test.
fn three() -> Vec<showcase::Entry> {
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 18.0),
        fact(6, "Boston Dynamics Spot", 1_665_741_668, 16.0),
        fact(7, "red toy robot", 1_665_841_668, 14.0),
    ];
    showcase::group(&facts, "robot", 0.3)
}

fn short_script() -> Script {
    Script {
        intro: "Every robot I have filmed, in order.".into(),
        lines: vec!["One.".into(), "Two.".into(), "Three.".into()],
        outro: "Thanks for watching.".into(),
        ..Script::default()
    }
}

/// Seconds from `t` to the nearest multiple of `beat` after `offset`.
fn off_grid(t: f64, beat: f64, offset: f64) -> f64 {
    let x = (t - offset) / beat;
    (x - x.round()).abs() * beat
}

#[tokio::test(flavor = "multi_thread")]
async fn cuts_land_on_the_beat_of_the_generated_bed() {
    let ex = demo();
    let opts = Options { title: "Robots".into(), bpm: 120.0, width: 640, height: 360, ..Options::default() };
    let built = showcase::build(&ex, &opts, &three(), &short_script(), &mut |_| {}).await.expect("the showcase builds");
    let m = built.music.clone().expect("music");
    assert_eq!((m.source.as_str(), m.fit.as_str(), m.bpm), ("generated", "generated", Some(120.0)));
    assert!(m.cuts >= 6 && m.cuts_on_beat == m.cuts, "{m:?}");
    // every entry starts (its transition, its year card) within a frame of the 0.5 s grid
    for e in &built.layout {
        assert!(off_grid(e.start, 0.5, 0.0) <= 1.0 / 30.0 + 1e-6, "entry {} at {}", e.number, e.start);
        assert!(e.shots.iter().all(|s| s.end > s.start));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_music_file_is_fitted_to_the_show_and_cut_on_its_beat() {
    let dir = std::env::temp_dir().join(format!("filmcraft-music-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (seconds, fit) in [(12.0, "remixed"), (90.0, "trimmed")] {
        let ex = demo();
        // the user's track: 128 bpm music written to a file (not placed)
        let g = ex
            .execute("audio.generate", json!({"kind": "music", "duration": seconds, "bpm": 128, "seed": 11, "place": false, "dir": dir.to_string_lossy()}))
            .await
            .unwrap();
        let path = g["path"].as_str().unwrap().to_string();
        let opts =
            Options { title: "Robots".into(), music: Some(path.clone()), voice: Some("basic-male".into()), width: 640, height: 360, ..Options::default() };
        let mut lines = Vec::new();
        let built = showcase::build(&ex, &opts, &three(), &short_script(), &mut |l| lines.push(l.to_string())).await.expect("the showcase builds");
        assert!(built.notes.is_empty(), "{:?}", built.notes);
        let m = built.music.clone().expect("music");
        assert_eq!((m.source.as_str(), m.fit.as_str()), (path.as_str(), fit), "{lines:?}");
        assert!(m.bpm.is_some_and(|b| (b - 128.0).abs() < 2.0), "{m:?}");
        assert!(m.cuts >= 6 && m.cuts_on_beat == m.cuts, "{m:?} {lines:?}");
        // the track under the whole show on A3, nothing generated, faded out, levelled as music
        let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
        let a3 = seq["audio"][2]["items"].as_array().cloned().unwrap_or_default();
        let file = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().into_owned();
        assert!(!a3.is_empty() && a3.iter().all(|c| c["name"].as_str() == Some(file.as_str())), "only the track, nothing generated: {a3:?}");
        let end = a3.iter().filter_map(|c| c["end"].as_i64()).max().unwrap() as f64 / 254_016_000_000.0;
        assert!(end >= built.seconds && end <= built.seconds + 1.0, "music ends at {end}, the show at {}", built.seconds);
        let es = ex.execute("essentialSound.inspect", json!({"clips": a3.iter().filter_map(|c| c["clip"].as_u64()).collect::<Vec<_>>()})).await.unwrap();
        assert!(es.to_string().to_lowercase().contains("music"), "{es}");
        // entries start on its beat: at 128 bpm, within a frame of its grid
        let beats = ex.execute("media.beats", json!({"item": a3[0]["item"]})).await.unwrap();
        let first = beats["beats"][0].as_f64().unwrap();
        for e in &built.layout {
            assert!(off_grid(e.start, 60.0 / 128.0, first) <= 1.0 / 30.0 + 0.015, "entry {} at {} (first beat {first})", e.number, e.start);
        }
    }
    // a file that is not there: a clear error, nothing half-built
    let ex = demo();
    let opts = Options { music: Some(dir.join("missing.wav").to_string_lossy().into_owned()), ..Options::default() };
    let err = showcase::build(&ex, &opts, &three(), &short_script(), &mut |_| {}).await.unwrap_err().to_string();
    assert!(err.contains("missing.wav"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The keyframe count of `param` of `effect` on a clip of `sequence.inspect`.
fn keyframes(clip: &serde_json::Value, effect: &str, param: &str) -> u64 {
    clip["effects"].as_array().and_then(|es| es.iter().find(|e| e["effect"] == effect)).and_then(|e| e["params"][param]["keyframes"].as_u64()).unwrap_or(0)
}

fn has_effect(clip: &serde_json::Value, effect: &str) -> bool {
    clip["effects"].as_array().is_some_and(|es| es.iter().any(|e| e["effect"] == effect))
}

/// Mean luma (0..1) of the bottom-left quarter of a PNG frame, where the lower third sits.
fn lower_left_luma(png: &[u8]) -> f64 {
    let img = image::load_from_memory(png).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in h * 3 / 4..h {
        for x in 0..w / 3 {
            let p = img.get_pixel(x, y);
            sum += (0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64) / 255.0;
            n += 1.0;
        }
    }
    sum / n
}

#[tokio::test(flavor = "multi_thread")]
async fn shots_push_in_and_lower_thirds_slide_in_and_out() {
    let ex = demo();
    let opts = Options { title: "Robots".into(), width: 640, height: 360, ..Options::default() };
    let built = showcase::build(&ex, &opts, &three(), &short_script(), &mut |_| {}).await.expect("the showcase builds");
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let tps = 254_016_000_000i64;
    let frame = tps / 30;
    // every shot pushes in: two scale keyframes, eased, at its first and last frames
    let v2 = seq["video"][1]["items"].as_array().cloned().unwrap();
    assert!(v2.len() >= 6, "{}", v2.len());
    for c in &v2 {
        assert_eq!(keyframes(c, "motion", "scale"), 2, "{c}");
        let (start, end) = (c["start"].as_i64().unwrap(), c["end"].as_i64().unwrap());
        for (t, ease) in [(start, "easeOut"), (end - frame, "easeIn")] {
            let r = ex
                .execute("effects.setInterpolation", json!({"clip": c["clip"], "effect": "motion", "param": "scale", "time": t, "interpolation": ease}))
                .await;
            assert!(r.is_ok(), "a keyframe at {t} on clip {}: {r:?}", c["clip"]);
        }
        assert!(!has_effect(c, "warp_stabilizer"), "not stabilized unless asked");
    }
    // the blurred copies behind portrait shots stay still
    for c in seq["video"][0]["items"].as_array().cloned().unwrap_or_default() {
        assert_eq!(keyframes(&c, "motion", "scale"), 0, "{c}");
    }
    // each entry's lower third slides and fades in and out
    let thirds: Vec<serde_json::Value> =
        seq["video"][2]["items"].as_array().cloned().unwrap().into_iter().filter(|c| keyframes(c, "motion", "position") == 4).collect();
    assert_eq!(thirds.len(), 3, "{:?}", seq["video"][2]);
    assert!(thirds.iter().all(|c| keyframes(c, "opacity", "opacity") == 4));
    // …so it is not there on its first and last frames, coming and going in between, and
    // fully there a second in
    let secs = |c: &serde_json::Value, k: &str| c[k].as_i64().unwrap() as f64 / tps as f64;
    let (start, end) = (secs(&thirds[1], "start"), secs(&thirds[1], "end"));
    let at = async |t: f64| lower_left_luma(&ex.render(t, 640).await.unwrap());
    let (bare, first, second, shown) = (at(start - 0.2).await, at(start).await, at(start + 1.0 / 30.0).await, at(start + 1.0).await);
    let (last, after) = (at(end - 1.0 / 30.0).await, at(end).await);
    println!("lower third: before {bare:.3}, frames 0 {first:.3}, 1 {second:.3}, 30 {shown:.3}, last {last:.3}, after {after:.3}");
    assert!(shown - bare > 0.1, "it shows: {bare} → {shown}");
    assert!((first - bare).abs() < 0.01 && (last - after).abs() < 0.01, "not there on its first ({first} vs {bare}) and last ({last} vs {after}) frames");
    assert!(second > first + 0.01 && second < shown - 0.05, "fading in: {first} {second} {shown}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_thumbnail_is_composed_and_exported_through_the_engine() {
    let ex = demo();
    let opts = Options { title: "Every Robot I've Met".into(), width: 640, height: 360, ..Options::default() };
    let entries = three();
    let built = showcase::build(&ex, &opts, &entries, &short_script(), &mut |_| {}).await.expect("the showcase builds");
    let dir = std::env::temp_dir().join(format!("filmcraft-thumbnail-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Pepper and Spot are famous, the toy robot is not; as large as Spot, Pepper comes first
    assert_eq!(showcase::thumbnail_frame(&entries, "robot").map(|p| p.2), Some(1));
    let png = dir.join("robots.thumbnail.png");
    let r = showcase::thumbnail(&ex, &opts, &entries, built.sequence, &png.to_string_lossy()).await.expect("a thumbnail");
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(1280), Some(720)), "{r}");
    // white (the title's capitals) and black (their outline) pixels in rows `ys` of a PNG
    let title_pixels = |png: &std::path::Path, ys: std::ops::Range<u32>| {
        let img = image::open(png).unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (1280, 720));
        let (mut white, mut black) = (0, 0);
        for y in ys {
            for x in 100..1180 {
                let p = img.get_pixel(x, y);
                let l = 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
                white += usize::from(l > 235.0);
                black += usize::from(l < 20.0);
            }
        }
        (white, black)
    };
    // the robot in the middle: the title low in the picture
    let (white, black) = title_pixels(&png, 420..700);
    assert!(white > 15_000 && black > 5_000, "white {white}, black {black}");
    assert!(title_pixels(&png, 30..300).0 < 2_000, "nothing white at the top");
    // a robot low in the frame: the title goes to the top
    let mut low = entries.clone();
    for e in &mut low {
        for s in &mut e.shots {
            if let Some(m) = s.largest.as_mut() {
                m.center = [0.5, 0.8];
            }
        }
    }
    showcase::thumbnail(&ex, &opts, &low, built.sequence, &png.to_string_lossy()).await.expect("a thumbnail");
    assert!(title_pixels(&png, 30..300).0 > 15_000 && title_pixels(&png, 420..700).0 < 2_000);
    // a JPEG for YouTube (well under its 2 MB)
    let jpg = dir.join("robots.thumbnail.jpg");
    showcase::thumbnail(&ex, &opts, &entries, built.sequence, &jpg.to_string_lossy()).await.expect("a thumbnail");
    let bytes = std::fs::read(&jpg).unwrap();
    assert!(bytes.starts_with(&[0xFF, 0xD8, 0xFF]) && bytes.len() > 20_000 && bytes.len() < 2_000_000, "{}", bytes.len());
    // the show is the active sequence again, and its thumbnail sequence is in the project
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    assert_eq!(seq["name"], "Every Robot I've Met");
    assert_eq!(seq["id"].as_u64(), built.sequence);
    let project = ex.execute("project.inspect", json!({})).await.unwrap();
    assert!(project.to_string().contains("Every Robot I've Met — Thumbnail"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_target_length_stretches_or_tightens_the_entries_within_the_robots_in_view() {
    // three robots, each in view only between 4 s and 13.5 s of its 18 s clip
    let partly = |item: u64, label: &str, unix: i64| ClipFacts {
        frames: [(2.0, false), (6.0, true), (9.0, true), (12.0, true), (15.0, false)]
            .iter()
            .map(|(s, present)| FrameFact { seconds: *s, present: *present, area: 0.4, center: [0.5, 0.5] })
            .collect(),
        best: 9.0,
        ..fact(item, label, unix, 18.0)
    };
    let facts =
        vec![partly(5, "Pepper humanoid robot", 1_600_000_000), partly(6, "Boston Dynamics Spot", 1_665_741_668), partly(7, "red toy robot", 1_665_841_668)];
    let entries = showcase::group(&facts, "robot", 0.3);
    let script = Script {
        intro: "Every robot I have met, in order.".into(),
        lines: vec!["A friendly humanoid at a shop.".into(), "Spot walked right up to me.".into(), "A small red toy robot.".into()],
        outro: "Thanks for watching.".into(),
        ..Script::default()
    };
    let build = async |minutes: Option<f64>| {
        let ex = demo();
        let opts = Options { title: "Robots".into(), voice: Some("basic-male".into()), width: 640, height: 360, target_minutes: minutes, ..Options::default() };
        let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
        assert!(built.notes.is_empty(), "{:?}", built.notes);
        (ex, built)
    };
    let (_, plain) = build(None).await;
    // a target the robots allow: met within a couple of seconds (beats and frames)
    let (_, near) = build(Some(0.6)).await;
    println!("plain {:.1} s; 36 s target: {:.1} s", plain.seconds, near.seconds);
    assert_eq!(near.target_seconds, Some(36.0));
    assert!((near.seconds - 36.0).abs() < 2.0, "{}", near.seconds);
    // tighter than the default: shorter entries, every narration still inside its entry
    let (ex, tight) = build(Some(0.3)).await;
    println!("18 s target: {:.1} s", tight.seconds);
    assert!(tight.seconds < plain.seconds - 3.0, "{} vs {}", tight.seconds, plain.seconds);
    for (e, line) in tight.layout.iter().zip(tight.narration.iter().skip(1)) {
        assert!(line.end <= e.end + 0.05, "entry {} ends at {} before its words do ({})", e.number, e.end, line.end);
    }
    let q = qa::Options { frame_side: 160, mix: false, ..qa::Options::for_showcase(&Options::default(), &tight) };
    assert_eq!(qa::check(&ex, &q, None, &mut |_| {}).await.errors, 0);
    // longer than the robots are in view: as long as they are, no longer — and never out of view
    let (ex, long) = build(Some(5.0)).await;
    println!("5 min target: {:.1} s", long.seconds);
    // (the robots are in view 3 × 9.5 s; the intro and outro take the rest)
    assert!(long.seconds > near.seconds + 2.0 && long.seconds < 45.0, "{}", long.seconds);
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let tps = 254_016_000_000.0;
    for e in &long.layout {
        for s in &e.shots {
            let c = seq["video"][1]["items"].as_array().unwrap().iter().find(|c| c["clip"].as_u64() == Some(s.clip)).unwrap();
            let (a, b) = (c["sourceIn"].as_f64().unwrap() / tps, c["sourceOut"].as_f64().unwrap() / tps);
            assert!(a >= 4.0 - 1e-3 && b <= 13.5 + 0.05, "entry {} shows {a:.2}–{b:.2} s of its clip, outside 4–13.5", e.number);
            assert!((c["speed"].as_f64().unwrap() - 1.0).abs() < 1e-9, "no slowed or held frames: {c}");
        }
    }
    let q = qa::Options { frame_side: 160, mix: false, ..qa::Options::for_showcase(&Options::default(), &long) };
    assert_eq!(qa::check(&ex, &q, None, &mut |_| {}).await.errors, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn stabilize_and_colour_match_when_asked() {
    let ex = demo();
    // one robot filmed twice the same day (one entry, two clips), then another
    let day = 1_600_000_000;
    let facts = vec![
        fact(5, "Pepper humanoid robot", day, 18.0),
        fact(6, "Pepper humanoid robot", day + 3600, 16.0),
        fact(7, "red toy robot", day + 86_400 * 40, 14.0),
    ];
    let entries = showcase::group(&facts, "robot", 0.3);
    assert_eq!(entries.iter().map(|e| e.shots.len()).collect::<Vec<_>>(), vec![2, 1]);
    let script = Script {
        lines: vec!["This is Pepper, filmed twice at the same little shop on the very same afternoon, waving at everyone.".into(), "And a toy.".into()],
        ..short_script()
    };
    let opts =
        Options { title: "Robots".into(), voice: Some("basic-male".into()), width: 640, height: 360, stabilize: true, match_color: true, ..Options::default() };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    assert_eq!(built.layout[0].shots.iter().map(|s| s.item).collect::<Vec<_>>(), vec![5, 6], "{:?}", built.layout);
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let v2 = seq["video"][1]["items"].as_array().cloned().unwrap();
    let outro = v2.last().unwrap();
    assert!(has_effect(outro, "gaussian_blur") && !has_effect(outro, "warp_stabilizer"), "the blurred outro is not stabilized");
    assert!(v2[..v2.len() - 1].iter().all(|c| has_effect(c, "warp_stabilizer")), "every other shot is");
    // the first entry's second clip is matched to its first; nothing else is
    let e1 = &built.layout[0];
    let matched: Vec<u64> = v2.iter().filter(|c| has_effect(c, "lumetri")).filter_map(|c| c["clip"].as_u64()).collect();
    let second = e1.shots.iter().find(|s| s.item == 6).map(|s| s.clip);
    assert_eq!(matched.len(), 1, "{matched:?}");
    assert_eq!(Some(matched[0]), second);
    // a stabilized shot renders (its motion analysed on first use)
    let mid = (e1.shots[0].start + e1.shots[0].end) / 2.0;
    assert!(ex.render(mid, 320).await.is_some_and(|p| p.starts_with(b"\x89PNG")));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_voice_says_respellings_and_the_subtitles_show_the_text() {
    let ex = demo();
    let facts = vec![fact(5, "Zaku II model kit", 1_600_000_000, 12.0), fact(6, "NodeMCU board", 1_665_741_668, 12.0)];
    let entries = showcase::group(&facts, "robot", 0.3);
    let script = Script {
        intro: "Every robot I have met, in order.".into(),
        lines: vec![
            "This is the Zaku II, a model kit I built over three long weekends in March. It still stands on my desk.".into(),
            "A NodeMCU board drives this little Go1 replica.".into(),
        ],
        outro: "Thanks for watching.".into(),
        pronounce: [("Zaku II", "Zaku Two"), ("NodeMCU", "Node M C U"), ("Go1", "Go One")].into_iter().map(|(a, b)| (a.into(), b.into())).collect(),
        say: [("outro".to_string(), "Thanks for watching, see you at the next one.".to_string())].into_iter().collect(),
        ..Script::default()
    };
    let opts = Options { title: "Robots".into(), voice: Some("basic-male".into()), width: 640, height: 360, ..Options::default() };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    // the voice read the respellings and the outro's own spoken version…
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let mut said = Vec::new();
    for c in seq["audio"][1]["items"].as_array().unwrap() {
        let n = ex.execute("tts.inspect", json!({"clip": c["clip"]})).await.unwrap();
        said.push(n["narration"]["text"].as_str().unwrap_or_default().to_string());
    }
    assert_eq!(said.len(), 4, "{said:?}");
    assert!(said[1].contains("Zaku Two") && !said[1].contains("Zaku II"), "{said:?}");
    assert!(said[2].contains("Node M C U") && said[2].contains("Go One"), "{said:?}");
    assert_eq!(said[3], "Thanks for watching, see you at the next one.");
    // …and the subtitles show the text as written, timed inside the narration
    for c in &built.subtitles {
        println!("{:7.3} → {:7.3}  {}", c.start, c.end, c.text.replace('\n', " / "));
    }
    let shown: String = built.subtitles.iter().map(|c| c.text.replace('\n', " ")).collect::<Vec<_>>().join(" ");
    assert!(shown.contains("Zaku II,") && shown.contains("NodeMCU") && shown.contains("Go1") && !shown.contains("Zaku Two"), "{shown}");
    assert!(shown.ends_with("Thanks for watching."), "{shown}");
    assert_eq!(built.narration.len(), 4);
    for c in &built.subtitles {
        assert!(c.text.lines().count() <= 2 && c.text.lines().all(|l| l.chars().count() <= 42), "{c:?}");
        assert!(c.end - c.start >= 1.0 - 1e-9 && c.end - c.start <= 7.0 + 1e-9, "{c:?}");
        assert!(built.narration.iter().any(|l| c.start >= l.start - 1.0 / 30.0 && c.start < l.end), "a cue starts while its line is spoken: {c:?}");
    }
    for w in built.subtitles.windows(2) {
        assert!(w[0].end <= w[1].start + 1e-9, "{w:?}");
    }
    assert!(built.subtitles.len() > built.narration.len(), "the long line is split: {:#?}", built.subtitles);
    // on a caption track of the sequence, which exports as SubRip and WebVTT
    let list = ex.execute("captions.list", json!({})).await.unwrap();
    let track = &list["tracks"][0];
    assert_eq!(track["name"], "Subtitles");
    assert_eq!(track["captions"].as_array().map(Vec::len), Some(built.subtitles.len()));
    let dir = std::env::temp_dir().join(format!("filmcraft-subtitles-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for ext in ["srt", "vtt"] {
        let path = dir.join(format!("robots.{ext}"));
        ex.execute("captions.export", json!({"path": path.to_string_lossy(), "track": track["id"]})).await.unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("Zaku II") && !text.contains("Zaku Two"), "{text}");
        assert!(if ext == "vtt" { text.starts_with("WEBVTT") } else { text.starts_with('1') }, "{text}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
