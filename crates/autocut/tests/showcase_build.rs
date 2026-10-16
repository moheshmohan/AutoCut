//! The showcase build against a real headless session: every engine command it issues must
//! exist, accept its parameters and leave a playable sequence.

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
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
    let script = Script { intro: "Hello and welcome.".into(), lines: vec![long.into(), long.into()], outro: "Bye for now, see you soon.".into(), templated: 0 };
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
