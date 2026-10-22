//! The pillars — the blurred, dimmed copy on V1 behind a picture shown whole (`fit: "full"`) —
//! stay through the transitions out of and into that picture, against a real headless session:
//! a portrait clip and a portrait photo, each next to landscape shots, never leave the sides of
//! the frame black while the pictures change over, and dip to black with the picture into and
//! out of a chapter. The media are drawn here (never committed).

use std::path::{Path, PathBuf};

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::qa;
use filmcraft_autocut::showcase::{self, Built, Entry, Options, Script};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use serde_json::{Value, json};

const TPS: f64 = 254_016_000_000.0;
const FPS: f64 = 30.0;
/// The frame: 640 × 360.
const SIDE: u32 = 640;

/// A fresh folder per call: the clock alone is not enough (macOS ticks in microseconds, and two
/// tests starting together shared one folder, one rewriting a photo while the other read it).
fn tmp_dir(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("filmcraft-showcase-pillars-{tag}-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A portrait picture: a dark blue ground, a green robot (box `[x, y, w, h]`, 0..1) and a band of
/// orange at the top, so its blurred copy has colour all the way to the frame's sides.
fn picture(w: u32, h: u32, robot: [f64; 4]) -> image::RgbaImage {
    image::RgbaImage::from_fn(w, h, |x, y| {
        let (u, v) = ((f64::from(x) + 0.5) / f64::from(w), (f64::from(y) + 0.5) / f64::from(h));
        if u >= robot[0] && u < robot[0] + robot[2] && v >= robot[1] && v < robot[1] + robot[3] {
            image::Rgba([40, 200, 80, 255])
        } else if v < 0.15 {
            image::Rgba([220, 120, 40, 255])
        } else {
            image::Rgba([30, 40, 70, 255])
        }
    })
}

/// A portrait clip, 12 s: numbered stills (the robot drifting down a little) imported as an image
/// sequence. Its first still's path.
fn portrait_clip(dir: &Path) -> String {
    let frames = 360;
    for k in 0..frames {
        let drift = 0.1 * f64::from(k) / f64::from(frames);
        picture(90, 160, [0.25, 0.2 + drift, 0.5, 0.6]).save(dir.join(format!("tall_{k:04}.png"))).unwrap();
    }
    dir.join("tall_0000.png").to_string_lossy().into_owned()
}

/// A portrait photo of a tall robot (a crop would cut it, so it is shown whole).
fn portrait_photo(dir: &Path) -> String {
    let path = dir.join("tall-photo.png");
    picture(600, 900, [0.25, 0.05, 0.5, 0.9]).save(&path).unwrap();
    path.to_string_lossy().into_owned()
}

struct Media {
    ex: SessionExecutor,
    clip: u64,
    photo: u64,
    dir: PathBuf,
}

/// The demo project (its 16:9 clips 5 and 6) with the portrait clip and photo imported.
fn session() -> Media {
    let dir = tmp_dir("media");
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let r = s.execute("file.import", json!({"paths": [portrait_clip(&dir)], "imageSequence": true})).unwrap();
    let clip = r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}"));
    let r = s.execute("file.import", json!({"paths": [portrait_photo(&dir)]})).unwrap();
    let photo = r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}"));
    Media { ex: SessionExecutor::from(s), clip, photo, dir }
}

fn video(item: u64, label: &str, unix: i64, seconds: f64) -> ClipFacts {
    ClipFacts {
        item,
        name: format!("{item}.mp4"),
        seconds,
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

/// Five robots: the portrait clip shown whole, a landscape clip, the portrait photo (whole),
/// the photo again (another robot in it), and — the next year — another landscape clip. Their
/// transitions: a dip to black into the clip (its chapter), a whip out of it, a flash into the
/// photo, a push from the photo to the photo (both whole), and a dip to black out of it into the
/// next chapter (the owner's photo demo, where the sides went black first).
fn entries(m: &Media) -> Vec<Entry> {
    let robot = [0.25, 0.05, 0.5, 0.9];
    let photo = |label: &str, unix: i64| ClipFacts {
        name: "tall-photo.png".into(),
        seconds: 5.0,
        frames: vec![FrameFact { seconds: 0.0, present: true, area: robot[2] * robot[3], center: [0.5, 0.5] }],
        photo: true,
        subject: Some(robot),
        ..video(m.photo, label, unix, 5.0)
    };
    let facts = vec![
        video(m.clip, "Tall delivery robot", 1_600_000_000, 12.0),
        video(5, "Pepper humanoid robot", 1_600_100_000, 18.0),
        photo("Tall reception robot", 1_600_200_000),
        photo("Second reception robot", 1_600_300_000),
        video(6, "Boston Dynamics Spot", 1_612_000_000, 16.0),
    ];
    let mut e = showcase::group(&facts, "robot", 0.3);
    assert_eq!(e.len(), 5, "{e:?}");
    // the portrait clip shown whole (as the owner's plan asked for two robots); the photo is
    // whole of its own accord (its tall robot would be cut)
    e[0].shots[0].fit = Some("full".into());
    e
}

fn script() -> Script {
    Script {
        intro: "Robots I met.".into(),
        lines: vec!["A tall one.".into(), "Pepper.".into(), "Another tall one.".into(), "One more.".into(), "Spot.".into()],
        outro: "Thanks.".into(),
        ..Script::default()
    }
}

fn opts() -> Options {
    Options { title: "Robots".into(), bpm: 120.0, width: SIDE, height: 360, fps: FPS, ..Options::default() }
}

/// The frame's left and right tenths, rows 15–70 % (clear of the counter, the lower third and
/// the year card): the share of their pixels that are not black, and their mean luma (0..1).
fn sides(png: &[u8]) -> (f64, f64) {
    let img = image::load_from_memory(png).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let (mut lit, mut luma, mut n) = (0u32, 0.0, 0u32);
    for y in h * 15 / 100..h * 70 / 100 {
        for x in (0..w / 10).chain(w - w / 10..w) {
            let p = img.get_pixel(x, y).0;
            n += 1;
            lit += u32::from(p.iter().any(|c| *c > 8));
            luma += (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])) / 255.0;
        }
    }
    let n = f64::from(n.max(1));
    (f64::from(lit) / n, luma / n)
}

/// A transition next to a picture shown whole: its effect and span (seconds), and whether the
/// picture shown whole is the one going out (else the one coming in).
#[derive(Debug)]
struct Change {
    effect: String,
    start: f64,
    end: f64,
    out_of_whole: bool,
    /// The pillars' mean luma in the middle of that picture's shot.
    pillars: f64,
}

/// The transitions on V2 and V1 next to a picture shown whole (its clip or its blurred copy's),
/// one per cut (the earliest-starting where both tracks have one).
async fn changes(ex: &SessionExecutor, built: &Built) -> Vec<Change> {
    let whole: Vec<&showcase::PlacedShot> = built.layout.iter().flat_map(|e| &e.shots).filter(|s| s.background.is_some()).collect();
    assert!(whole.len() >= 2, "a clip and a photo shown whole: {:?}", built.layout);
    let shot_of = |c: Option<u64>| c.and_then(|c| whole.iter().find(|s| s.clip == c || s.background == Some(c)));
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let mut out: Vec<Change> = Vec::new();
    // (the pictures' own first: where both tracks have one, it names the change)
    for x in seq["video"][1]["transitions"].as_array().unwrap().iter().chain(seq["video"][0]["transitions"].as_array().unwrap()) {
        let (from, to) = (shot_of(x["from"].as_u64()), shot_of(x["to"].as_u64()));
        let Some(shot) = from.or(to) else { continue };
        let (start, dur) = (x["start"].as_f64().unwrap() / TPS, x["duration"].as_f64().unwrap() / TPS);
        let png = ex.render((shot.start + shot.end) / 2.0, SIDE).await.unwrap();
        let change = Change { effect: x["effect"].as_str().unwrap().into(), start, end: start + dur, out_of_whole: from.is_some(), pillars: sides(&png).1 };
        match out.iter_mut().find(|c| c.start < change.end && change.start < c.end) {
            Some(c) if change.start < c.start => *c = change,
            Some(_) => {}
            None => out.push(change),
        }
    }
    out.sort_by(|a, b| a.start.total_cmp(&b.start));
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn the_pillars_stay_through_the_transitions_of_a_picture_shown_whole() {
    let m = session();
    let entries = entries(&m);
    let built = showcase::build(&m.ex, &opts(), &entries, &script(), &mut |_| {}).await.expect("it builds");
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    let whole: Vec<(bool, bool)> = built.layout.iter().map(|e| (e.shots[0].background.is_some(), e.shots[0].photo)).collect();
    assert_eq!(whole, [(true, false), (false, false), (true, true), (true, true), (false, false)], "{:?}", built.layout);
    let changes = changes(&m.ex, &built).await;
    let kinds: Vec<(&str, bool)> = changes.iter().map(|c| (c.effect.as_str(), c.out_of_whole)).collect();
    // into the clip's chapter, out of the clip, into the photo, from the photo to the photo, out
    // of it into the next chapter
    assert_eq!(kinds, [("dip_to_black", false), ("whip", true), ("flash", false), ("push", true), ("dip_to_black", true)], "{changes:#?}");
    let mut bad = Vec::new();
    for c in &changes {
        let frames = ((c.end - c.start) * FPS).round() as i64;
        for k in 0..frames {
            let t = c.start + k as f64 / FPS;
            let (lit, luma) = sides(&m.ex.render(t, SIDE).await.unwrap());
            // (the transition's own progress at this frame: its middle frame is black)
            let p = (t - c.start) / (c.end - c.start);
            let ok = if c.effect.starts_with("dip_to") {
                // a dip: over the half that shows the picture shown whole, its pillars fade with
                // it — as bright as they are mid-shot, times how far the picture has faded
                let shown = if c.out_of_whole { 1.0 - 2.0 * p } else { 2.0 * p - 1.0 };
                shown <= 1e-6 || luma >= 0.3 * c.pillars * shown
            } else {
                // any other transition: never black at the sides
                lit >= 0.95
            };
            println!(
                "{:>12} {t:7.3} s: sides {:5.1} % lit, luma {luma:.4} (pillars {:.4}){}",
                c.effect,
                lit * 100.0,
                c.pillars,
                if ok { "" } else { "  BLACK" }
            );
            if !ok {
                bad.push((c.effect.clone(), t, lit, luma));
            }
        }
    }
    assert!(bad.is_empty(), "the sides go black during a transition: {bad:?}");
    // the blurred copies never overlap one another on V1
    let seq = m.ex.execute("sequence.inspect", json!({})).await.unwrap();
    let v1: Vec<Value> = seq["video"][0]["items"].as_array().cloned().unwrap_or_default();
    for w in v1.windows(2) {
        assert!(w[0]["end"].as_i64() <= w[1]["start"].as_i64(), "{w:?}");
    }
    let _ = std::fs::remove_dir_all(&m.dir);
}

/// Put the old bug back into a built show: each blurred copy only under its own picture, and the
/// dips over the pictures again (on V2, where a dip's black covers the pillars at once).
async fn stop_the_pillars_short(ex: &SessionExecutor, built: &Built) {
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let items = |t: usize| seq["video"][t]["items"].as_array().cloned().unwrap_or_default();
    let (copies, pictures) = (items(0), items(1));
    let span =
        |list: &[Value], id: u64| list.iter().find(|c| c["clip"].as_u64() == Some(id)).map(|c| (c["start"].as_i64().unwrap(), c["end"].as_i64().unwrap()));
    // the dips back on the pictures' track, where they were
    for x in seq["video"][0]["transitions"].as_array().unwrap().iter().filter(|x| x["effect"].as_str().unwrap().starts_with("dip_to")) {
        let (start, duration) = (x["start"].as_i64().unwrap(), x["duration"].as_i64().unwrap());
        let cut = pictures.iter().find(|c| (start..=start + duration).contains(&c["end"].as_i64().unwrap())).unwrap_or_else(|| panic!("a picture ends in {x}"));
        let r = ex
            .execute(
                "sequence.applyVideoTransition",
                json!({"clip": cut["clip"], "edge": "out", "effect": x["effect"], "frames": (duration as f64 / TPS * FPS).round()}),
            )
            .await
            .unwrap();
        ex.execute("sequence.setTransition", json!({"transition": r["transition"], "start": start, "duration": duration})).await.unwrap();
        ex.execute("sequence.removeTransition", json!({"transitions": [x["id"]]})).await.unwrap();
    }
    // each copy trimmed back to its picture's span
    for s in built.layout.iter().flat_map(|e| &e.shots) {
        let Some(b) = s.background else { continue };
        let ((cs, ce), (ps, pe)) = (span(&copies, b).unwrap(), span(&pictures, s.clip).unwrap());
        if ce > pe {
            ex.execute("timeline.trim", json!({"clip": b, "edge": "out", "mode": "regular", "delta": pe - ce})).await.unwrap();
        }
        if cs < ps {
            ex.execute("timeline.trim", json!({"clip": b, "edge": "in", "mode": "regular", "delta": ps - cs})).await.unwrap();
        }
    }
    let v1 = ex.execute("sequence.inspect", json!({})).await.unwrap()["video"][0].clone();
    for s in built.layout.iter().flat_map(|e| &e.shots).filter(|s| s.background.is_some()) {
        let b = span(v1["items"].as_array().unwrap(), s.background.unwrap()).unwrap();
        assert_eq!(b, span(&pictures, s.clip).unwrap(), "the copy only under its picture");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn qa_finds_the_sides_going_black_where_the_pillars_stop_short() {
    let m = session();
    let entries = entries(&m);
    let built = showcase::build(&m.ex, &opts(), &entries, &script(), &mut |_| {}).await.expect("it builds");
    let changes = changes(&m.ex, &built).await;
    let qa_opts = qa::Options { mix: false, ..qa::Options::for_showcase(&opts(), &built) };
    let sides = |r: &qa::QaReport| r.of("sides", qa::Severity::Warning).into_iter().cloned().collect::<Vec<_>>();
    // as built: the pillars stay through every transition
    let r = qa::check(&m.ex, &qa_opts, None, &mut |_| {}).await;
    let check = r.checks.iter().find(|c| c.name == "sides").unwrap_or_else(|| panic!("{:?}", r.checks));
    println!("as built: {check:?}");
    assert_eq!(check.status, "ok", "{:#?}", sides(&r));
    assert!(r.frames_rendered > 0);
    // with the old bug: the sides go black in the transitions next to the pictures shown whole
    stop_the_pillars_short(&m.ex, &built).await;
    let r = qa::check(&m.ex, &qa_opts, None, &mut |_| {}).await;
    let found = sides(&r);
    for f in &found {
        println!("{:7.3} s {:?}: {}", f.seconds, f.until, f.message);
    }
    assert!(found.iter().all(|f| f.severity == qa::Severity::Error && f.message.contains(" side")), "{found:#?}");
    // the pillars going black are found in each dip next to a picture shown whole (from its first
    // frame out of one, over its second half into one) and in the whip out of the clip (on the
    // frame after the cut, as lane F6 saw them), and nothing outside the transitions next to one
    let fd = 1.0 / FPS;
    let near = |c: &Change, f: &qa::Finding| f.seconds >= c.start - fd && f.seconds < c.end + fd;
    let pillars = |f: &qa::Finding| f.message.contains("left") || f.message.contains("right");
    for c in changes.iter().filter(|c| c.effect.starts_with("dip_to") || c.effect == "whip") {
        assert!(found.iter().any(|f| near(c, f) && pillars(f)), "nothing found in {c:?}: {found:#?}");
    }
    assert!(found.iter().all(|f| changes.iter().any(|c| near(c, f))), "{found:#?}");
    let _ = std::fs::remove_dir_all(&m.dir);
}
