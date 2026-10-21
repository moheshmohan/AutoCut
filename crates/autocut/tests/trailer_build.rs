//! A trailer of a synthetic showcase against a real headless session: the show is built from the
//! demo project's generated footage, then `trailer::make` cuts its vertical trailer — every engine
//! command it issues must exist and accept its parameters, and the result must be a 9:16 sequence
//! of the asked length, its cuts on the music's beats, its words inside the safe zone, and an
//! end card.

use filmcraft_autocut::showcase::{self, Entry, Moment, Options, Script, Shot};
use filmcraft_autocut::{Executor as _, SessionExecutor, trailer};
use filmcraft_engine::Session;
use serde_json::{Value, json};

const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;

fn demo() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    SessionExecutor::from(s)
}

/// A robot entry over demo footage `item` (`seconds` long), filmed in `year`.
fn entry(number: usize, label: &str, kind: &str, year: i64, item: u64, seconds: f64) -> Entry {
    let best = seconds / 2.0;
    Entry {
        number,
        label: label.into(),
        kind: kind.into(),
        description: format!("a {kind}"),
        unix: filmcraft_autocut::look::unix_of(year, 5, 1),
        shots: vec![Shot {
            item,
            name: format!("{item}.mp4"),
            clip_seconds: seconds,
            best,
            seen: vec![best - 2.0, best, best + 2.0],
            largest: Some(Moment { seconds: best, area: 0.3, center: [0.5, 0.5] }),
            ..Shot::default()
        }],
        ..Entry::default()
    }
}

/// The demo's footage as a show of five robots from 2014 to 2025 (the last one's sponsor logo
/// hidden, as the Promobot's is), and a small logo PNG for the end card.
async fn show(ex: &SessionExecutor) -> (Value, Value, String) {
    let mut entries = vec![
        entry(1, "Toy rover", "wheeled", 2014, 5, 18.0),
        entry(2, "Walker", "humanoid", 2018, 6, 16.0),
        entry(3, "Robot dog", "quadruped", 2021, 7, 14.0),
        entry(4, "Arm", "arm", 2023, 8, 15.0),
        entry(5, "Greeter", "humanoid", 2025, 9, 12.0),
    ];
    entries[4].redact = Some(json!({"targets": ["Acme logo"]}));
    let script = Script {
        intro: "Every robot I have met.".into(),
        lines: vec!["One.".into(), "Two.".into(), "Three.".into(), "Four.".into(), "Five.".into()],
        outro: "Thanks.".into(),
        templated: 0,
        ..Script::default()
    };
    let opts = Options { title: "Robots".into(), ..Options::default() };
    let built = showcase::build(ex, &opts, &entries, &script, &mut |_| {}).await.expect("the show builds");
    let mut report = serde_json::to_value(&built).unwrap();
    report["sequence"] = json!(built.sequence);
    let plan = json!({"title": "Robots", "entries": entries, "outro": {"channel": "Robots", "handle": "@robots"}});
    let dir = std::env::temp_dir().join(format!("filmcraft-trailer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let logo = dir.join("logo.png");
    image::RgbaImage::from_fn(64, 64, |x, y| {
        if (x as i32 - 32).pow(2) + (y as i32 - 32).pow(2) < 900 { image::Rgba([200, 60, 255, 255]) } else { image::Rgba([0, 0, 0, 0]) }
    })
    .save(&logo)
    .unwrap();
    (plan, report, logo.display().to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_synthetic_show_gets_a_vertical_trailer_on_the_beat() {
    let ex = demo();
    let (plan, mut report, logo) = show(&ex).await;
    // the show's privacy pass hid a face in the robot dog's clip: a masked blur on its shot
    let dog = report["layout"].as_array().unwrap().iter().find(|e| e["label"] == "Robot dog").unwrap()["shots"][0]["clip"].as_u64().unwrap();
    ex.execute("effects.apply", json!({"clips": [dog], "effect": "gaussian_blur"})).await.unwrap();
    ex.execute("masks.add", json!({"clip": dog, "effect": "gaussian_blur", "shape": "ellipse", "center": [300, 200], "size": [80, 100]})).await.unwrap();
    report["privacy"] = json!([{"clip": dog, "name": "7.mp4", "faces": 1, "plates": 0, "named": 0, "masks": 1, "largestMask": 0.01, "mostHidden": 0.01}]);
    let p = json!({"seconds": 16, "plan": plan, "report": report, "logo": logo, "hookVoice": false, "redact": false, "bpm": 120});
    let mut lines = Vec::new();
    let made = trailer::make(&ex, &p, None, None, &mut |l| lines.push(l.to_string())).await.unwrap_or_else(|e| panic!("{e}: {lines:?}"));
    // a 9:16 sequence of about the length asked for
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    assert_eq!((seq["settings"]["width"].as_u64(), seq["settings"]["height"].as_u64()), (Some(1080), Some(1920)), "{made}");
    let seconds = made["seconds"].as_f64().unwrap();
    assert!((12.0..=20.0).contains(&seconds), "{seconds}");
    let length = seq["duration"].as_f64().unwrap() / TPS;
    assert!((length - seconds).abs() < 0.1, "the sequence runs {length} s, the trailer {seconds} s");
    assert!(seq["name"].as_str().unwrap().ends_with("Reel trailer"), "{}", seq["name"]);
    // the hook and the robots, the hook the most striking (a humanoid), the rest in date order
    let shots = made["shots"].as_array().unwrap();
    assert!(shots.len() >= 4, "{made}");
    assert_eq!(shots[0]["number"], 0);
    let kind_of = |label: &str| plan["entries"].as_array().unwrap().iter().find(|e| e["label"] == label).unwrap()["kind"].as_str().unwrap().to_string();
    assert_eq!(kind_of(shots[0]["label"].as_str().unwrap()), "humanoid", "{made}");
    let years: Vec<i64> = shots.iter().skip(1).filter_map(|s| s["year"].as_i64()).collect();
    assert!(years.windows(2).all(|w| w[0] <= w[1]), "{years:?}");
    // every cut on the music's beat
    assert_eq!(made["cutsOnBeat"], made["cuts"], "{made}");
    let bpm = made["bpm"].as_f64().unwrap();
    assert!((bpm - 120.0).abs() < 1.0, "{bpm}");
    let beat = 60.0 / bpm;
    for s in shots {
        let start = s["start"].as_f64().unwrap();
        let k = (start / beat).round();
        assert!((start - k * beat).abs() <= 1.0 / 30.0 + 1e-6, "a shot at {start} s is off the beat");
    }
    // the shots play inside the windows the show played (the redaction cache and the show's choices)
    assert!(shots.iter().all(|s| s["inShowWindow"] == true), "{made}");
    // the picture track has no gap from the first frame to the end card
    let timing = &made["timing"];
    let hit = timing["endCard"][0].as_f64().unwrap();
    let pictures: Vec<&Value> =
        seq["video"].as_array().unwrap().iter().flat_map(|t| t["items"].as_array().unwrap()).filter(|c| shots.iter().any(|s| s["clip"] == c["clip"])).collect();
    let mut spans: Vec<(f64, f64)> = pictures.iter().map(|c| (c["start"].as_f64().unwrap() / TPS, c["end"].as_f64().unwrap() / TPS)).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(spans.first().unwrap().0.abs() < 1e-6 && (spans.last().unwrap().1 - hit).abs() < 1e-6, "{spans:?}");
    assert!(spans.windows(2).all(|w| (w[0].1 - w[1].0).abs() < 1e-6), "{spans:?}");
    // the music under all of it, its final hit where the end card starts
    let music = made["music"].clone();
    assert!(music["clip"].as_u64().is_some(), "{music}");
    if let Some(ending) = music["ending"].as_f64() {
        assert!((ending - hit).abs() <= 1.5 / 30.0, "the final hit at {ending}, the end card at {hit}");
    }
    // every word inside Reels' safe zone
    let sz = &made["safeZones"];
    assert_eq!(sz["ok"], true, "{sz}");
    let check = ex.execute("sequence.checkSafeZones", json!({"platform": "reels"})).await.unwrap();
    assert_eq!(check["overlaps"].as_array().map(Vec::len), Some(0), "{check}");
    // the end card: the logo, the words and the handle from the plan's outro, at the end
    let names: Vec<(String, f64)> = seq["video"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|t| t["items"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| Some((c["name"].as_str()?.to_string(), c["start"].as_f64()? / TPS)))
        .collect();
    assert!(names.iter().any(|(n, t)| n == "logo.png" && (t - hit).abs() < 0.05), "{names:?}");
    assert!(names.iter().any(|(n, t)| n.starts_with("Full video on YouTube") && *t >= hit), "{names:?}");
    assert!(sz["texts"].as_array().unwrap().iter().any(|t| t["what"] == "end card"), "{sz}");
    let title: Vec<&str> = made["title"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert_eq!(title, ["5 robots", "2014 → 2025"]);
    // a frame of the end card renders
    let png = ex.render(hit + 1.0, 480).await;
    assert!(png.is_some_and(|p| p.starts_with(b"\x89PNG")));
    // the show's blur, with its mask, is carried onto the trailer's shot of the dog (and its
    // blurred copy when it is shown in a band), and the trailer is still the active sequence
    let dog_shot = shots.iter().find(|s| s["label"] == "Robot dog").expect("the dog is in the trailer");
    let carried: Vec<u64> = made["carried"].as_array().unwrap().iter().filter_map(Value::as_u64).collect();
    assert!(carried.contains(&dog_shot["clip"].as_u64().unwrap()), "{made}");
    assert_eq!(seq["id"], made["sequence"]);
    let masks = ex.execute("masks.list", json!({"clip": dog_shot["clip"]})).await.unwrap();
    assert!(masks["masks"].as_array().unwrap().iter().any(|m| m["effectId"] == "gaussian_blur"), "{masks}");
}

#[tokio::test(flavor = "multi_thread")]
async fn hostile_trailer_requests_fail_cleanly() {
    let ex = demo();
    for p in [
        json!({}),
        json!({"plan": {"entries": []}}),
        json!({"plan": {"entries": [{"number": 1, "label": "x", "kind": "scene", "shots": []}]}}),
        json!({"plan": {"entries": [{"number": 1, "label": "R", "kind": "humanoid", "shots": [{"item": 99999, "name": "x", "clip_seconds": 20.0, "best": 3.0}]}]}}),
        json!({"plan": {"entries": [{"number": 1, "label": "R", "kind": "humanoid", "shots": [{"item": 5, "name": "x", "clip_seconds": 1e300, "best": f64::MAX}]}]}, "hook": "Nobody"}),
        json!({"seconds": 1e9}),
        json!({"sequence": 424242}),
    ] {
        let r = trailer::make(&ex, &p, None, None, &mut |_| {}).await;
        assert!(r.is_err(), "{p} gave {r:?}");
    }
}
