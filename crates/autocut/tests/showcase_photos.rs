//! Photos in a showcase, against a real headless session: a robot that was only photographed
//! is an entry of its photos, each held with a slow pan and zoom (`clip.kenBurns`) that ends
//! framed on its subject, the moves taking turns from one photo to the next; a portrait photo
//! whose robot the crop would cut is shown whole over its blurred copy; an entry may mix photos
//! and video; the privacy pass looks at a photo once; QA finds no dip in the shots' sound where
//! a photo is silent on purpose. The photos are drawn here (never committed media).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::privacy::Redact;
use filmcraft_autocut::qa;
use filmcraft_autocut::shot_edit::Fit;
use filmcraft_autocut::showcase::{self, Built, Entry, Options, PHOTO_REACH, Script, Shot};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use filmcraft_engine::ken_burns::{Pose, shown};
use filmcraft_engine::project::{ClipId, ItemId, ParamValue, Sequence, TrackItem};
use serde_json::{Value, json};

const FPS: f64 = 30.0;
const FRAME: f64 = 1.0 / FPS;
/// The robot's colour (green: not the fake face detector's warm pixels).
const ROBOT: [u8; 4] = [40, 200, 80, 255];

/// A fresh folder per call: the clock alone is not enough (macOS ticks in microseconds, and two
/// tests starting together shared one folder, one rewriting a photo while the other read it).
fn tmp_dir(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("filmcraft-showcase-photos-{tag}-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A `w` × `h` photo on a dark background: the robot (a green box, `robot` as `[x, y, w, h]`
/// 0..1) and, with `face`, a visitor's face (a warm textured disc) in its lower left.
fn photo(dir: &Path, name: &str, (w, h): (u32, u32), robot: [f64; 4], face: bool) -> String {
    let img = image::RgbaImage::from_fn(w, h, |x, y| {
        let (u, v) = ((f64::from(x) + 0.5) / f64::from(w), (f64::from(y) + 0.5) / f64::from(h));
        let (fx, fy, r) = (0.15 * f64::from(w), 0.8 * f64::from(h), 0.06 * f64::from(h));
        if face && ((f64::from(x) - fx).powi(2) + (f64::from(y) - fy).powi(2)).sqrt() < r {
            let v = if (x / 2 + y / 2) % 2 == 0 { 250 } else { 160 };
            image::Rgba([v, v / 2, 0, 255])
        } else if u >= robot[0] && u < robot[0] + robot[2] && v >= robot[1] && v < robot[1] + robot[3] {
            image::Rgba(ROBOT)
        } else {
            image::Rgba([30, 40, 70, 255])
        }
    });
    let path = dir.join(name);
    img.save(&path).unwrap();
    path.to_string_lossy().into_owned()
}

/// The fake face detector: the warm pixels' box.
fn warm(rgba: &[u8], w: usize, h: usize) -> Vec<filmcraft_vision::Detection> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, p) in rgba.as_chunks::<4>().0.iter().enumerate() {
        if p[0] > 140 && p[2] < 30 && p[1] < 140 {
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    if x1 < x0 {
        return Vec::new();
    }
    // not sure of it (0.6): a photo is looked at once, and that is enough
    let d = filmcraft_vision::Detection {
        kind: filmcraft_vision::Kind::Face,
        score: 0.6,
        bbox: [x0 as f32, y0 as f32, (x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32],
        points: Vec::new(),
    };
    vec![d.normalized(w as f32, h as f32)]
}

struct Photos {
    ex: SessionExecutor,
    /// The items: a landscape photo with a visitor, a portrait one, a 4:3 one; the demo's clip 5.
    wide: u64,
    tall: u64,
    small: u64,
    dir: PathBuf,
}

fn session() -> Photos {
    let dir = tmp_dir("build");
    let paths = [
        photo(&dir, "wide.png", (1200, 800), [0.6, 0.2, 0.2, 0.4], true),
        photo(&dir, "tall.png", (600, 900), [0.25, 0.05, 0.5, 0.9], false),
        photo(&dir, "small.png", (1000, 750), [0.35, 0.3, 0.3, 0.4], false),
    ];
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let r = s.execute("file.import", json!({"paths": paths})).unwrap();
    let ids: Vec<u64> = r["items"].as_array().unwrap().iter().filter_map(Value::as_u64).collect();
    assert_eq!(ids.len(), 3, "{r}");
    s.redacts.detector = Some(Arc::new(|rgba: &[u8], w: usize, h: usize, _: usize| warm(rgba, w, h)));
    Photos { ex: SessionExecutor::from(s), wide: ids[0], tall: ids[1], small: ids[2], dir }
}

fn photo_fact(item: u64, label: &str, unix: i64, robot: [f64; 4]) -> ClipFacts {
    let center = [robot[0] + robot[2] / 2.0, robot[1] + robot[3] / 2.0];
    ClipFacts {
        item,
        name: format!("{item}.png"),
        seconds: 5.0,
        unix: Some(unix),
        present: true,
        coverage: 1.0,
        label: label.into(),
        kind: "humanoid".into(),
        description: format!("a {label}"),
        frames: vec![FrameFact { seconds: 0.0, present: true, area: robot[2] * robot[3], center }],
        photo: true,
        subject: Some(robot),
        ..ClipFacts::default()
    }
}

fn video_fact(item: u64, label: &str, unix: i64) -> ClipFacts {
    ClipFacts {
        item,
        name: format!("{item}.mp4"),
        seconds: 18.0,
        unix: Some(unix),
        present: true,
        coverage: 1.0,
        label: label.into(),
        kind: "wheeled".into(),
        description: format!("a {label}"),
        best: 9.0,
        frames: vec![FrameFact { seconds: 9.0, present: true, area: 0.4, center: [0.5, 0.5] }],
        ..ClipFacts::default()
    }
}

/// Four robots: three only photographed, one filmed (with a photo of it added to its entry).
fn entries(p: &Photos) -> Vec<Entry> {
    let facts = vec![
        photo_fact(p.wide, "Ameca humanoid", 1_600_000_000, [0.6, 0.2, 0.2, 0.4]),
        photo_fact(p.tall, "Tall reception robot", 1_610_000_000, [0.25, 0.05, 0.5, 0.9]),
        photo_fact(p.small, "Small toy robot", 1_620_000_000, [0.35, 0.3, 0.3, 0.4]),
        video_fact(5, "Delivery robot", 1_630_000_000),
    ];
    let mut e = showcase::group(&facts, "robot", 0.3);
    assert_eq!(e.len(), 4);
    assert!(e[..3].iter().all(|e| e.shots.len() == 1 && e.shots[0].photo), "an entry of its photos: {e:?}");
    assert_eq!((e[0].shots[0].clip_seconds, e[0].shots[0].subject), (PHOTO_REACH, Some([0.6, 0.2, 0.2, 0.4])));
    // the plan: the first held four seconds, moving in
    e[0].shots[0].hold = Some(4.0);
    e[0].shots[0].motion = Some("in".into());
    // the filmed robot's entry: its video, then a photo of it panned across
    let extra = Shot { hold: Some(3.0), motion: Some("pan".into()), ..e[2].shots[0].clone() };
    e[3].shots.push(extra);
    e
}

fn script() -> Script {
    Script {
        intro: "Robots I met.".into(),
        lines: vec!["Ameca.".into(), "A tall one.".into(), "A toy.".into(), "A delivery robot.".into()],
        outro: "Thanks.".into(),
        ..Script::default()
    }
}

fn opts() -> Options {
    Options {
        title: "Every Robot I've Met".into(),
        voice: Some("basic-male".into()),
        bpm: 120.0,
        width: 640,
        height: 360,
        fps: FPS,
        redact: Some(Redact { plates: false, verify: false, ..Redact::default() }),
        ..Options::default()
    }
}

fn with_seq<T>(ex: &SessionExecutor, built: &Built, f: impl FnOnce(&Sequence, &Session) -> T) -> T {
    let g = ex.0.lock().unwrap();
    let seq = g.project.sequence(ItemId(built.sequence.unwrap())).unwrap();
    f(seq, &g)
}

fn item(seq: &Sequence, clip: u64) -> TrackItem {
    seq.find_item(ClipId(clip)).unwrap().1.clone()
}

/// A clip's Motion at its first and last keyframes (or its static value twice).
fn poses(it: &TrackItem) -> (Pose, Pose) {
    let m = it.effect("motion").unwrap();
    let at = |param: &str, k: usize| {
        let p = m.param(param).unwrap();
        p.keyframes.get(k).map_or_else(|| p.value.clone(), |kf| kf.value.clone())
    };
    let pose = |k: usize| {
        let scale = match at("scale", k) {
            ParamValue::Float(v) => v,
            other => panic!("{other:?}"),
        };
        let position = match at("position", k) {
            ParamValue::Vec2(v) => [v.x, v.y],
            other => panic!("{other:?}"),
        };
        Pose { position, scale }
    };
    (pose(0), pose(1))
}

fn contains(outer: [f64; 4], inner: [f64; 4]) -> bool {
    let e = 1e-6;
    inner[0] >= outer[0] - e && inner[1] >= outer[1] - e && inner[0] + inner[2] <= outer[0] + outer[2] + e && inner[1] + inner[3] <= outer[1] + outer[3] + e
}

#[tokio::test(flavor = "multi_thread")]
async fn photos_are_held_with_a_slow_move_that_ends_on_their_robot() {
    let p = session();
    let entries = entries(&p);
    let o = opts();
    let built = showcase::build(&p.ex, &o, &entries, &script(), &mut |_| {}).await.expect("it builds");
    let (wide, tall, small, mixed) = (&built.layout[0], &built.layout[1], &built.layout[2], &built.layout[3]);

    // ---- the first photo: held exactly as planned, moving in, no sound
    let s = &wide.shots[0];
    assert!(s.photo && s.fixed && s.sound.is_none() && s.speed.is_none(), "{s:?}");
    assert_eq!(s.motion.as_deref(), Some("in"));
    assert!((s.end - s.start - 4.0).abs() <= FRAME + 1e-6, "held {:.3} s", s.end - s.start);
    with_seq(&p.ex, &built, |seq, g| {
        let it = item(seq, s.clip);
        assert!((it.duration.seconds() - 4.0).abs() <= FRAME + 1e-6, "the still clip is as long as its hold");
        let m = it.effect("motion").unwrap();
        for param in ["scale", "position"] {
            assert_eq!(m.param(param).unwrap().keyframes.len(), 2, "{param} eased from one end to the other");
        }
        let (a, b) = poses(&it);
        assert!(b.scale > a.scale && b.scale / a.scale < 1.2, "in, slowly: {a:?} → {b:?}");
        let picture = g.project.conformed_source_size(it.item, seq.settings.par).unwrap();
        let frame = (f64::from(seq.settings.width), f64::from(seq.settings.height));
        let end = shown(picture, frame, b);
        assert!(contains(end, [0.6, 0.2, 0.2, 0.4]), "it ends framed on the robot: {end:?}");
        assert!(end[2] < 0.95, "closer than the whole picture: {end:?}");
        // the visitor's face: found in the one look at the photo, hidden for the whole hold
        let fx = it.effects.iter().find(|e| filmcraft_engine::redact::is_auto_redaction(e)).expect("the face is hidden");
        let mask = fx.masks.first().expect("a mask");
        assert!(mask.opacity.keyframes.is_empty() && mask.path.keyframes.len() <= 2, "one detection for the whole hold: {mask:?}");
    });

    // ---- the portrait photo: its tall robot would be cut by a crop, so it is whole over pillars
    let s = &tall.shots[0];
    assert_eq!((s.fit, s.photo), (Some(Fit::Full), true), "{s:?}");
    let bg = s.background.expect("the blurred copy behind it");
    with_seq(&p.ex, &built, |seq, _| {
        let back = item(seq, bg);
        assert!(back.effect("gaussian_blur").is_some());
        assert_eq!(seq.video_tracks.iter().position(|t| t.items.iter().any(|i| i.id == ClipId(bg))), Some(0), "on V1, under it");
        // the picture itself only breathes
        let (a, b) = poses(&item(seq, s.clip));
        let (lo, hi) = (a.scale.min(b.scale), a.scale.max(b.scale));
        assert!(hi > lo && hi / lo <= 1.0 + filmcraft_engine::ken_burns::MAX_FULL_ZOOM + 1e-9, "{a:?} {b:?}");
        assert!((lo - 40.0).abs() < 1e-6, "whole at full height (360 / 900 = 40 %): {lo}");
    });
    let png = p.ex.render((s.start + s.end) / 2.0, 640).await.unwrap();
    let side = image::load_from_memory(&png).unwrap().to_rgb8();
    let lit = (0..side.height()).filter(|y| side.get_pixel(5, *y).0.iter().any(|c| *c > 8)).count() as f64 / f64::from(side.height());
    assert!(lit > 0.8, "the pillars show the blurred copy, not black: {lit}");

    // ---- the 4:3 photo fills the frame
    assert_eq!(small.shots[0].fit, Some(Fit::Fill), "{:?}", small.shots[0]);

    // ---- photos without a move of their own take turns: in, out, in…
    let defaults: Vec<&str> = built.layout.iter().flat_map(|e| &e.shots).filter(|s| s.photo).filter_map(|s| s.motion.as_deref()).collect();
    assert_eq!(defaults.len(), 4, "{defaults:?}");
    // (the first is the plan's "in", the last its "pan"; the two between go the other way each time)
    assert_eq!(defaults, ["in", "out", "in", "pan"], "consecutive photos move the other way");

    // ---- an entry of video and a photo
    assert_eq!(mixed.shots.len(), 2, "{mixed:?}");
    let (v, ph) = (&mixed.shots[0], &mixed.shots[1]);
    assert!(!v.photo && v.sound.is_some(), "{v:?}");
    assert!(ph.photo && ph.sound.is_none() && ph.motion.as_deref() == Some("pan"), "{ph:?}");
    assert!(ph.end - ph.start >= 3.0 - FRAME - 1e-6, "{ph:?}");

    // ---- QA: no dip in the shots' sound where a photo is silent on purpose, nothing black
    let q = qa::Options { frame_side: 160, mix: false, ..qa::Options::for_showcase(&o, &built) };
    let r = qa::check(&p.ex, &q, None, &mut |_| {}).await;
    let photo_spans: Vec<(f64, f64)> = built.layout.iter().flat_map(|e| &e.shots).filter(|s| s.photo).map(|s| (s.start, s.end)).collect();
    for f in r.findings.iter().filter(|f| f.message.contains("sound clips")) {
        assert!(!photo_spans.iter().any(|(a, b)| f.seconds >= a - 1e-3 && f.seconds <= b + 1e-3), "{f:?}");
    }
    assert!(r.of("black", qa::Severity::Error).is_empty(), "{:?}", r.of("black", qa::Severity::Error));
    assert!(r.of("joins", qa::Severity::Error).is_empty(), "{:?}", r.of("joins", qa::Severity::Error));
    let _ = std::fs::remove_dir_all(&p.dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn hostile_photo_controls_are_refused_before_anything_is_built() {
    let p = session();
    type Edit = fn(&mut Shot);
    let hostile: [(&str, Edit); 7] = [
        ("hold 0", |s| s.hold = Some(0.0)),
        ("hold NaN", |s| s.hold = Some(f64::NAN)),
        ("hold an hour", |s| s.hold = Some(3600.0)),
        ("move sideways", |s| s.motion = Some("sideways".into())),
        ("a photo's window", |s| (s.from, s.to) = (Some(0.0), Some(2.0))),
        ("a photo's speed", |s| s.speed = Some(2.0)),
        ("focus outside", |s| s.focus = Some([0.5, 2.0])),
    ];
    for (what, edit) in hostile {
        let mut e = entries(&p);
        edit(&mut e[1].shots[0]);
        let err = showcase::build(&p.ex, &opts(), &e, &script(), &mut |_| {}).await.expect_err(what).to_string();
        assert!(err.contains("entry 2, shot 1"), "{what}: {err}");
    }
    // a video shot with a photo's controls
    let mut e = entries(&p);
    e[3].shots[0].hold = Some(3.0);
    let err = showcase::build(&p.ex, &opts(), &e, &script(), &mut |_| {}).await.expect_err("hold on a video").to_string();
    assert!(err.contains("photo"), "{err}");
    let _ = std::fs::remove_dir_all(&p.dir);
}
