//! The branded channel outro ([`filmcraft_autocut::outro`]) against a real headless session: its
//! elements placed, timed, inside the frame and clear of YouTube's end-screen regions from the
//! end screen's start; a gradient and no logo when its footage and logo are not there.

use std::sync::Arc;

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::look::{Look, Style};
use filmcraft_autocut::outro::{self, Outro};
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

/// A 400×400 logo with a transparent background: a magenta disc in a ring of dots.
fn logo(dir: &std::path::Path) -> String {
    let mut img = image::RgbaImage::new(400, 400);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let (dx, dy) = (x as f64 - 200.0, y as f64 - 200.0);
        let r = (dx * dx + dy * dy).sqrt();
        *p = if r < 120.0 {
            image::Rgba([220, 40, 200, 255])
        } else if r > 160.0 && r < 190.0 && ((dx.atan2(dy) * 12.0).sin() > 0.5) {
            image::Rgba([40, 200, 255, 255])
        } else {
            image::Rgba([0, 0, 0, 0])
        };
    }
    let path = dir.join("logo.png");
    img.save(&path).unwrap();
    path.to_string_lossy().into_owned()
}

fn branded(logo: Option<String>, background: serde_json::Value) -> Outro {
    let mut v = json!({
        "channel": "H4HACKS", "handle": "@H4hacks", "tagline": "Hacking, gadgets and robots in English and Malayalam",
        "cta": "SUBSCRIBE", "ctaDone": "SUBSCRIBED", "terminal": "> h4hacks.com", "background": background, "seconds": 10
    });
    if let Some(l) = logo {
        v["logo"] = json!(l);
    }
    Outro::from_plan(&v).unwrap().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_branded_outro_leaves_the_end_screen_clear() {
    let (ex, portrait) = demo();
    let dir = std::env::temp_dir().join(format!("filmcraft-outro-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let o = branded(Some(logo(&dir)), json!([{"item": 7, "from": 1.0, "to": 9.0}]));
    let look = Look { style: Some(Style::tech()), outro: Some(o), ..Look::default() };
    let mut lines = Vec::new();
    let built = showcase::build(&ex, &opts(look), &robots(portrait), &script(), &mut |l| lines.push(l.to_string())).await.expect("it builds");
    let notes: Vec<&String> = built.notes.iter().filter(|n| !n.starts_with("style: the font")).collect();
    assert!(notes.is_empty(), "{notes:?}\n{lines:#?}");
    let out = built.outro.clone().expect("the branded outro");
    assert_eq!(out.background, "footage");
    assert!((out.end - built.seconds).abs() < 2.0 * FRAME, "it ends the show: {} vs {}", out.end, built.seconds);
    assert!(out.end - out.start >= 10.0 - FRAME, "at least its own length");
    let [es0, es1] = out.end_screen;
    assert!(es1 - es0 >= outro::END_SCREEN_MIN - 1e-6 && es1 - es0 <= outro::END_SCREEN_MAX + 1e-6, "{es0}–{es1}");
    let regions = out.regions.unwrap();
    let names: Vec<&str> = out.elements.iter().map(|e| e.name).collect();
    for want in ["badge", "ring", "burst", "lockup", "logo", "button", "cursor", "ripple", "tagline", "terminal", "fade"] {
        assert!(names.contains(&want), "{want} in {names:?}");
    }
    assert_eq!(names.iter().filter(|n| **n == "button").count(), 2, "SUBSCRIBE, then SUBSCRIBED");
    let overlap =
        |a: &filmcraft_autocut::look::Bounds, b: &filmcraft_autocut::look::Bounds| a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
    with_seq(&ex, &built, |seq| {
        for e in &out.elements {
            let it = item(seq, e.clip);
            // placed and timed as reported, inside the outro, on frames
            assert!(
                (it.start.seconds() - e.from).abs() <= FRAME + 1e-6 && (it.end().seconds() - e.to).abs() <= FRAME + 1e-6,
                "{e:?}: {:?}–{:?}",
                it.start,
                it.end()
            );
            assert!(e.from >= out.start - 1e-6 && e.to <= out.end + FRAME, "{e:?}");
            // where it sits from the end screen on: inside the frame, clear of the regions
            if e.to > es0 + FRAME && e.name != "fade" {
                let b = e.settled.unwrap_or_else(|| panic!("{} is on screen in the end screen: where?", e.name));
                assert!(b.left >= 0.0 && b.top >= 0.0 && b.right <= 640.0 && b.bottom <= 360.0, "{e:?}");
                assert!(!overlap(&b, &regions.video) && !overlap(&b, &regions.subscribe), "{} at {b:?} covers an end-screen region", e.name);
            }
        }
        // the logo glitches in (held flickers) and glides up into the lockup
        let logo = out.elements.iter().find(|e| e.name == "logo").unwrap();
        let it = item(seq, logo.clip);
        assert!(keys(&it, "opacity", "opacity") >= 6 && keys(&it, "motion", "position") >= 7 && keys(&it, "motion", "scale") >= 5);
        // the terminal line types itself out to the whole line
        let term = out.elements.iter().find(|e| e.name == "terminal").unwrap();
        let it = item(seq, term.clip);
        let text = it.effects.iter().find(|e| e.effect == "graphic_text").and_then(|e| e.param("text")).unwrap();
        assert!(text.keyframes.len() > 13, "{}", text.keyframes.len());
        let at = |t: f64| text.value_at(it.source_time_at(Tick::from_seconds_f64(t)));
        assert_eq!(at(term.from + FRAME * 0.5), ParamValue::Text("_".into()));
        assert!(matches!(at(out.end - 0.2), ParamValue::Text(t) if t.starts_with("> h4hacks.com")));
        // no plain "3 ROBOTS" card
        let plain = seq.video_tracks.iter().flat_map(|t| t.items.iter()).any(|c| {
            c.effects
                .iter()
                .filter(|e| e.effect == "graphic_text")
                .any(|e| matches!(e.param("text").map(|p| &p.value), Some(ParamValue::Text(t)) if t.contains("ROBOTS")))
        });
        assert!(!plain);
    });
    // in the end screen the regions show only the darkened footage; the lockup is bright
    let png = ex.render(es0 + 1.5, 640).await.unwrap();
    let px = |b: &filmcraft_autocut::look::Bounds| (b.left as u32 + 4, b.top as u32 + 4, b.right as u32 - 4, b.bottom as u32 - 4);
    for r in [regions.video, regions.subscribe] {
        let l = lumas(&png, px(&r));
        let bright = l.iter().filter(|v| **v > 0.6).count();
        assert_eq!(bright, 0, "nothing bright in the end-screen region {r:?}");
    }
    let lockup = out.elements.iter().find(|e| e.name == "lockup").and_then(|e| e.settled).unwrap();
    assert!(lumas(&png, px(&lockup)).iter().any(|v| *v > 0.5), "the channel name is there");
    // and QA finds nothing wrong with it: no gap, no black outside the last fade
    let q = filmcraft_autocut::qa::Options {
        frame_side: 160,
        work_dir: Some(dir.clone()),
        ..filmcraft_autocut::qa::Options::for_showcase(&opts(Look::default()), &built)
    };
    let r = filmcraft_autocut::qa::check(&ex, &q, None, &mut |_| {}).await;
    let errors: Vec<String> = r
        .findings
        .iter()
        .filter(|f| f.severity == filmcraft_autocut::qa::Severity::Error)
        .map(|f| format!("{} @{:.2}: {}", f.check, f.seconds, f.message))
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_branded_outro_without_its_footage_or_logo_falls_back() {
    let (ex, portrait) = demo();
    // an item that is not in the project, a logo file that is not there
    let o = branded(Some("/no/such/dir/logo.png".into()), json!([{"item": 999_999}]));
    let look = Look { outro: Some(o), ..Look::default() };
    let built = showcase::build(&ex, &opts(look), &robots(portrait), &script(), &mut |_| {}).await.expect("it still builds");
    let out = built.outro.clone().unwrap();
    assert_eq!(out.background, "gradient");
    assert!(built.notes.iter().any(|n| n.contains("background item 999999")), "{:?}", built.notes);
    assert!(built.notes.iter().any(|n| n.contains("logo") && n.contains("left out")), "{:?}", built.notes);
    assert!(!out.elements.iter().any(|e| e.name == "logo"));
    // the name alone, centred, still above the regions; and the picture track is never empty
    let lockup = out.elements.iter().find(|e| e.name == "lockup").and_then(|e| e.settled).unwrap();
    assert!(lockup.bottom < out.regions.unwrap().video.top);
    with_seq(&ex, &built, |seq| {
        let v2 = &seq.video_tracks[1];
        assert!(v2.items.iter().any(|c| c.start.seconds() <= out.start + FRAME && c.end().seconds() >= out.end - FRAME), "the gradient on V2");
    });
    // a plan's hostile outro is refused with what is wrong
    for bad in [json!({"outro": {"channel": "x", "seconds": 2}}), json!({"outro": {"channel": "x", "background": [{"item": "a"}]}}), json!({"outro": 7})] {
        let err = Look::from_plan(&bad).unwrap_err().to_string();
        assert!(err.contains("outro"), "{bad}: {err}");
    }
}
