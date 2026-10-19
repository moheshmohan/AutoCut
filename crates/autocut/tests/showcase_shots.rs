//! The plan's per-shot controls (`from`/`to`, `speed`, `fit`, `focus`, `mute`, `volume`) and an
//! entry's `place` against a real headless session: windows placed as asked, sped shots shorter
//! and silent, a portrait clip whole over its blurred copy, hostile values refused before
//! anything is built.

use std::sync::Arc;

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::look::{Look, Style};
use filmcraft_autocut::showcase::{self, Built, Entry, Options, Script};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use filmcraft_engine::project::{ClipId, ItemId, Label, ParamValue, Sequence, TrackItem};
use filmcraft_engine::time::Tick;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND};
use serde_json::json;

const FPS: f64 = 30.0;
const FRAME: f64 = 1.0 / FPS;

/// The demo project, with a portrait clip (360×640, 20 s) added: its item.
fn demo() -> (SessionExecutor, u64) {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let src = GeneratorSource::new(Generator::Demo(DemoScene::Aurora), 360, 640, FrameRate::FPS_30, Tick(20 * TICKS_PER_SECOND));
    let p = Arc::make_mut(&mut s.project);
    let portrait = filmcraft_engine::demo::add_generator(p, &s.media, src, "portrait.mp4", Label::Iris, None);
    (SessionExecutor::from(s), portrait.0)
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

/// Three robots: Pepper (item 5), Spot (6) and a portrait one; then the plan's controls.
fn robots(portrait: u64) -> Vec<Entry> {
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 18.0),
        fact(6, "Boston Dynamics Spot", 1_665_741_668, 16.0),
        fact(portrait, "Robot arm game", 1_700_900_000, 20.0),
    ];
    let mut e = showcase::group(&facts, "robot", 0.3);
    assert_eq!(e.len(), 3);
    // Pepper: the window where it sprays the mist, a little quieter
    e[0].shots[0].from = Some(3.0);
    e[0].shots[0].to = Some(7.5);
    e[0].shots[0].volume = Some(-12.0);
    // Spot: a wide shot fast-forwarded three times
    e[1].shots[0].from = Some(2.0);
    e[1].shots[0].to = Some(14.0);
    e[1].shots[0].speed = Some(3.0);
    // the portrait clip whole, over its blurred copy, its sound muted; filmed in Busan at an expo
    e[2].shots[0].fit = Some("full".into());
    e[2].shots[0].mute = Some(true);
    e[2].event = Some("G-Star 2023".into());
    e[2].place = Some("Busan, South Korea".into());
    e
}

fn script() -> Script {
    Script {
        intro: "Every robot I have met, in order.".into(),
        lines: vec!["Pepper sprays mist.".into(), "Spot walks.".into(), "A robot arm game.".into()],
        outro: "Thanks for watching.".into(),
        ..Script::default()
    }
}

fn opts(look: Look) -> Options {
    Options { title: "Every Robot I've Met".into(), voice: Some("basic-male".into()), bpm: 120.0, width: 640, height: 360, look, ..Options::default() }
}

fn with_seq<T>(ex: &SessionExecutor, built: &Built, f: impl FnOnce(&Sequence) -> T) -> T {
    let g = ex.0.lock().unwrap();
    let seq = g.project.sequence(ItemId(built.sequence.unwrap())).unwrap();
    f(seq)
}

fn item(seq: &Sequence, clip: u64) -> TrackItem {
    seq.find_item(ClipId(clip)).unwrap().1.clone()
}

fn track_of(seq: &Sequence, clip: u64) -> usize {
    seq.video_tracks.iter().position(|t| t.items.iter().any(|i| i.id == ClipId(clip))).unwrap()
}

fn keys(it: &TrackItem, effect: &str, param: &str) -> usize {
    it.effect(effect).and_then(|e| e.param(param)).map_or(0, |p| p.keyframes.len())
}

/// Luma (0..1) of every pixel of a PNG in `[x0, x1) × [y0, y1)` (pixels of the PNG).
fn lumas(png: &[u8], (x0, y0, x1, y1): (u32, u32, u32, u32)) -> Vec<f64> {
    let img = image::load_from_memory(png).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let mut out = Vec::new();
    for y in y0.min(h)..y1.min(h) {
        for x in x0.min(w)..x1.min(w) {
            let p = img.get_pixel(x, y).0;
            out.push((0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])) / 255.0);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn shot_windows_speeds_fits_and_sound_follow_the_plan() {
    let (ex, portrait) = demo();
    let entries = robots(portrait);
    let built = showcase::build(&ex, &opts(Look { style: Some(Style::tech()), ..Look::default() }), &entries, &script(), &mut |_| {}).await.expect("it builds");
    let notes: Vec<&String> = built.notes.iter().filter(|n| !n.starts_with("style: the font")).collect();
    assert!(notes.is_empty(), "{notes:?}");
    let (pepper, spot, arm) = (&built.layout[0], &built.layout[1], &built.layout[2]);

    // ---- Pepper: exactly its window, with its sound, 12 dB down
    let s = pepper.shots.iter().find(|s| s.fixed).expect("the plan's window");
    let [a, b] = s.source.unwrap();
    assert!((a - 3.0).abs() < 1e-9 && (b - 7.5).abs() < 1e-9, "{s:?}");
    assert!((s.end - s.start - 4.5).abs() <= FRAME + 1e-6, "{s:?}");
    assert_eq!((s.speed, s.volume), (None, Some(-12.0)));
    let sound = s.sound.expect("its own sound");
    // the words fit: more of the entry after the fixed window carries the rest
    assert!(pepper.end - pepper.start >= built.narration[1].end - pepper.start - 1e-6, "{pepper:?} {:?}", built.narration[1]);

    // ---- Spot: twelve seconds of its clip in four, frames sampled, no sound of its own
    let s = &spot.shots[0];
    assert_eq!((s.speed, s.fixed, s.sound), (Some(3.0), true, None), "{s:?}");
    assert!((s.end - s.start - 4.0).abs() <= FRAME + 1e-6, "placed {:.3} s", s.end - s.start);
    // ---- the portrait clip: whole, over its blurred copy; muted
    let s = &arm.shots[0];
    assert_eq!((s.fit, s.sound), (Some(filmcraft_autocut::shot_edit::Fit::Full), None), "{s:?}");
    let bg = s.background.expect("a blurred copy behind it");
    with_seq(&ex, &built, |seq| {
        let sped = item(seq, spot.shots[0].clip);
        assert_eq!(sped.speed, 3.0);
        assert_eq!(sped.time_interpolation.name(), filmcraft_engine::project::TimeInterpolation::FrameSampling.name());
        // nothing plays on A1 under the sped and the muted shots
        let a1 = &seq.audio_tracks[0];
        for quiet in [&spot.shots[0], &arm.shots[0]] {
            let over = a1.items.iter().filter(|c| c.start.seconds() < quiet.end - 1e-6 && c.end().seconds() > quiet.start + 1e-6).count();
            assert_eq!(over, 0, "sound under {quiet:?}");
        }
        // the quieter one: 12 dB under another shot's level
        let louder = built.layout.iter().flat_map(|e| e.shots.iter()).filter_map(|s| s.sound).find(|c| *c != sound);
        if let Some(other) = louder {
            let (q, l) = (item(seq, sound).gain_db, item(seq, other).gain_db);
            assert!((l - q - 12.0).abs() < 3.0, "{q} dB vs {l} dB");
        }
        // whole: no push-in on it; the copy behind it on V1, blurred, dimmed and pushing in
        let fg = item(seq, arm.shots[0].clip);
        assert_eq!(keys(&fg, "motion", "scale"), 0, "the whole picture stays whole");
        let back = item(seq, bg);
        assert_eq!(track_of(seq, bg), 0);
        assert!(back.effect("gaussian_blur").is_some());
        assert_eq!(keys(&back, "motion", "scale"), 2, "the copy behind it moves");
        assert_eq!(back.effect("opacity").and_then(|e| e.param("opacity")).map(|p| p.value.clone()), Some(ParamValue::Float(45.0)));
        // its lower third says where, not the expo
        let third = item(seq, arm.lower_third.unwrap());
        let texts: Vec<String> = third
            .effects
            .iter()
            .filter(|e| e.effect == "graphic_text")
            .filter_map(|e| match &e.param("text")?.value {
                ParamValue::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t == "Busan, South Korea · November 2023"), "{texts:?}");
    });
    // the place starts no chapter of its own: the expo does
    assert!(built.chapters.iter().any(|c| c.title == "G-Star 2023") && !built.chapters.iter().any(|c| c.title.contains("Busan")), "{:?}", built.chapters);
    // the portrait picture at full height in the middle, the blurred copy filling the sides
    let mid = (arm.shots[0].start + arm.shots[0].end) / 2.0;
    let png = ex.render(mid, 640).await.unwrap();
    let sides = lumas(&png, (0, 0, 40, 360));
    let lit = sides.iter().filter(|l| **l > 0.04).count() as f64 / sides.len() as f64;
    assert!(lit > 0.8, "the sides show the blurred copy, not black: {lit}");
    // and the build's layout says it all
    let v = serde_json::to_value(&built.layout).unwrap();
    assert_eq!(v[1]["shots"][0]["speed"], 3.0);
    assert_eq!(v[2]["shots"][0]["fit"], "full");
    assert_eq!(v[0]["shots"][0]["source"], json!([3.0, 7.5]));
}

#[tokio::test(flavor = "multi_thread")]
async fn hostile_shot_controls_are_refused_before_anything_is_built() {
    let (ex, portrait) = demo();
    let sequences = async || ex.execute("project.inspect", json!({})).await.unwrap().to_string().matches("\"type\":\"Sequence\"").count();
    let before = sequences().await;
    type Edit = fn(&mut showcase::Shot);
    let hostile: [(&str, Edit); 12] = [
        ("negative from", |s| s.from = Some(-1.0)),
        ("NaN from", |s| s.from = Some(f64::NAN)),
        ("infinite to", |s| s.to = Some(f64::INFINITY)),
        ("to before from", |s| (s.from, s.to) = (Some(8.0), Some(4.0))),
        ("to past the clip", |s| (s.from, s.to) = (Some(2.0), Some(99.0))),
        ("speed 0", |s| s.speed = Some(0.0)),
        ("speed 1000", |s| s.speed = Some(1000.0)),
        ("speed NaN", |s| s.speed = Some(f64::NAN)),
        ("fit stretch", |s| s.fit = Some("stretch".into())),
        ("focus outside", |s| s.focus = Some([1.5, 0.5])),
        ("volume +40", |s| s.volume = Some(40.0)),
        ("a window too long", |s| (s.clip_seconds, s.from, s.to) = (500.0, Some(0.0), Some(400.0))),
    ];
    for (what, edit) in hostile {
        let mut entries = robots(portrait);
        edit(&mut entries[1].shots[0]);
        let r = showcase::build(&ex, &opts(Look::default()), &entries, &script(), &mut |_| {}).await;
        let err = r.expect_err(what).to_string();
        assert!(err.contains("entry 2, shot 1"), "{what}: {err}");
    }
    assert_eq!(sequences().await, before, "nothing was built");
    // …and in a plan file: a wrong type is a plan error, never a panic
    for bad in [json!({"from": "soon"}), json!({"focus": [0.5]}), json!({"mute": "yes"})] {
        let mut v = serde_json::to_value(&robots(portrait)[0]).unwrap();
        for (k, x) in bad.as_object().unwrap() {
            v["shots"][0][k] = x.clone();
        }
        assert!(serde_json::from_value::<Entry>(v).is_err(), "{bad}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn fixed_windows_shorter_than_the_words_run_on() {
    // a one-second window under a long line: the clip runs on after it so the words still fit
    let (ex, portrait) = demo();
    let mut entries = robots(portrait);
    entries[0].shots[0].volume = None;
    entries[0].shots[0].from = Some(2.0);
    entries[0].shots[0].to = Some(3.0);
    let mut sc = script();
    sc.lines[0] = "Pepper, the humanoid robot from SoftBank, greeted every visitor at the stand and sprayed a fine mist over the crowd.".into();
    let built = showcase::build(&ex, &opts(Look::default()), &entries, &sc, &mut |_| {}).await.expect("it builds");
    let pepper = &built.layout[0];
    let words = &built.narration[1];
    assert!(pepper.end >= words.end, "the entry ({:.2}–{:.2}) holds its words (to {:.2})", pepper.start, pepper.end, words.end);
    assert!(pepper.shots[0].fixed && (pepper.shots[0].end - pepper.shots[0].start - 1.0).abs() <= FRAME + 1e-6);
    let run_on = &pepper.shots[1];
    assert!((run_on.source.unwrap()[0] - 3.0).abs() < 1e-6, "the clip after the window: {run_on:?}");
    assert!(built.notes.iter().any(|n| n.contains("entry 1") && n.contains("runs on")), "{:?}", built.notes);
}
