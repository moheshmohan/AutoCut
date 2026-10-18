use filmcraft_project::{ItemId, ParamValue, SequenceSettings, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use crate::Session;
use crate::media_test_util::tmp_dir;

const W: u32 = 320;
const H: u32 = 180;
const FRAMES: u32 = 48;

fn hash(x: i64, y: i64, seed: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

fn noise(u: f64, v: f64, seed: u64) -> f32 {
    let (x0, y0) = (u.floor(), v.floor());
    let (fx, fy) = ((u - x0) as f32, (v - y0) as f32);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let (a, b, c, d) = (hash(x0, y0, seed), hash(x0 + 1, y0, seed), hash(x0, y0 + 1, seed), hash(x0 + 1, y0 + 1, seed));
    let top = a + (b - a) * fx;
    let bot = c + (d - c) * fx;
    top + (bot - top) * fy
}

fn texture(u: f64, v: f64, seed: u64) -> f32 {
    0.5 * noise(u / 8.0, v / 8.0, seed) + 0.3 * noise(u / 3.5, v / 3.5, seed + 1) + 0.2 * noise(u / 1.6, v / 1.6, seed + 2)
}

/// The subject's centre in frame `k` (pixels): left to right across the picture.
fn subject_at(k: u32) -> (f64, f64) {
    (60.0 + 200.0 * k as f64 / (FRAMES - 1) as f64, 90.0)
}

/// A frame: a cool textured background and a warm textured disc (radius 26) at `subject_at`.
fn frame(k: u32, with_subject: bool) -> image::RgbaImage {
    let (cx, cy) = subject_at(k);
    image::RgbaImage::from_fn(W, H, |x, y| {
        let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
        let b = texture(px + 400.0, py + 200.0, 5);
        let mut c = [0.2 + 0.25 * b, 0.3 + 0.3 * b, 0.45 + 0.35 * b];
        let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
        if with_subject && d < 26.0 {
            let t = texture(px - cx + 900.0, py - cy + 100.0, 77);
            c = [0.6 + 0.4 * t, 0.3 + 0.35 * t, 0.1 + 0.15 * t];
        }
        image::Rgba([(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8, 255])
    })
}

/// A session holding a 16:9 image-sequence clip of the moving subject on V1 of a 320×180
/// sequence (flat frames without it when `with_subject` is false).
fn session(with_subject: bool) -> (Session, ItemId, ItemId) {
    let dir = tmp_dir(if with_subject { "reframe-subject" } else { "reframe-flat" });
    for k in 0..FRAMES {
        let img = if with_subject { frame(k, true) } else { image::RgbaImage::from_pixel(W, H, image::Rgba([90, 90, 90, 255])) };
        img.save(dir.join(format!("shot{k:04}.png"))).unwrap();
    }
    let mut s = Session::default();
    let first = dir.join("shot0000.png");
    let r = s.execute("file.importImageSequence", json!({"path": first.to_string_lossy()})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}")));
    let rate = s.project.item(item).unwrap().as_media().unwrap().info.video.clone().unwrap().frame_rate;
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Talk", SequenceSettings { width: W, height: H, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let d = p.item(item).unwrap().duration();
    let mut v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, d), rate).unwrap();
    for e in &mut v.effects {
        filmcraft_project::resolve_auto_points(e, (W, H), (W, H));
    }
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    (s, item, seq)
}

/// Where the warm disc is in the active sequence's frame `k` (centroid of warm pixels), as a share
/// of the frame width.
fn disc_x(s: &mut Session, k: i64) -> Option<f64> {
    let (w, _, px) = crate::media_test_util::frame_rgba(s, k, 1.0);
    let (mut sx, mut n) = (0.0, 0.0);
    for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
        if p[0] as i32 > p[2] as i32 + 60 {
            sx += (i % w) as f64;
            n += 1.0;
        }
    }
    (n > 20.0).then(|| sx / n / w as f64)
}

fn motion(s: &Session, seq: ItemId) -> filmcraft_project::EffectInstance {
    let q = s.project.sequence(seq).unwrap();
    q.video_tracks[0].items[0].effect("motion").unwrap().clone()
}

#[test]
fn reframes_a_sequence_to_vertical_following_the_subject() {
    let (mut s, _, src) = session(true);
    let undo_before = s.history.undo.len();
    let r = s.execute("sequence.autoReframe", json!({"aspect": "9:16", "wait": true})).unwrap();
    let out = &r["reframe"];
    let seq = ItemId(out["sequence"].as_u64().unwrap_or_else(|| panic!("{r}")));
    assert_ne!(seq, src);
    assert_eq!((out["width"].as_u64(), out["height"].as_u64()), (Some(180), Some(320)), "{out}");
    assert_eq!(out["followed"], json!(1), "{out}");
    assert_eq!(s.history.undo.len(), undo_before + 1, "one undo step");
    assert_eq!(s.history.undo.last().unwrap().0, "Auto Reframe Sequence");
    assert_eq!(s.state.active_sequence, Some(seq), "the new sequence is open");
    let m = motion(&s, seq);
    let pos = m.param("position").unwrap();
    assert!(pos.keyframes.len() >= 2, "the position follows: {:?}", pos.keyframes.len());
    // the subject stays near the middle of the narrow frame from start to end
    for k in [2, 12, 24, 36, 46] {
        let x = disc_x(&mut s, k).unwrap_or_else(|| panic!("frame {k}: the subject is in view"));
        assert!((x - 0.5).abs() < 0.2, "frame {k}: subject at {:.0}% of the width", x * 100.0);
    }
    // the picture covers the frame everywhere (no bars) and the source sequence is untouched
    let q = s.project.sequence(seq).unwrap();
    let it = &q.video_tracks[0].items[0];
    for k in 0..FRAMES as i64 {
        let t = q.settings.frame_rate.tick_of(k);
        let mt = it.source_time_at(t);
        let p = m.vec2_at("position", mt);
        let sc = m.f64_at("scale", mt) / 100.0;
        assert!(p.x - W as f64 * sc / 2.0 <= 0.5 && p.x + W as f64 * sc / 2.0 >= 180.0 - 0.5, "frame {k}: {p:?} at {sc}");
        assert!(sc * H as f64 >= 320.0 - 0.5, "frame {k}: fills the height");
    }
    assert!(motion(&s, src).param("position").unwrap().keyframes.is_empty());
    // undo takes the copy away
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project.sequence(seq).is_none());
}

#[test]
fn a_range_is_reframed_on_its_own() {
    let (mut s, _, _) = session(true);
    let rate = s.sequence_rate();
    let (a, b) = (rate.tick_of(10), rate.tick_of(30));
    let r = s.execute("sequence.autoReframe", json!({"aspect": "1:1", "range": {"start": a.0, "end": b.0}, "name": "Square bit", "wait": true})).unwrap();
    let seq = ItemId(r["reframe"]["sequence"].as_u64().unwrap());
    let q = s.project.sequence(seq).unwrap();
    assert_eq!((q.settings.width, q.settings.height), (180, 180));
    assert_eq!(q.duration(), b - a, "only the range, from zero");
    assert_eq!(s.project.item(seq).unwrap().name, "Square bit");
}

#[test]
fn clips_reframe_in_place_and_keep_their_scale() {
    let (mut s, _, seq) = session(true);
    // a 9:16 sequence holding the 16:9 clip filled, with a push-in on its scale
    let mut p = (*s.project).clone();
    {
        let q = p.sequence_mut(seq).unwrap();
        q.settings.width = 180;
        q.settings.height = 320;
        let m = q.video_tracks[0].items[0].effect_mut("motion").unwrap();
        let prm = m.param_mut("scale").unwrap();
        prm.value = ParamValue::Float(178.0);
        prm.put_keyframe(Tick::ZERO, ParamValue::Float(178.0));
        prm.put_keyframe(Tick::from_seconds_f64(1.5), ParamValue::Float(190.0));
    }
    s.project = std::sync::Arc::new(p);
    let clip = s.project.sequence(seq).unwrap().video_tracks[0].items[0].id;
    let r = s.execute("clip.autoReframe", json!({"clips": [clip.0], "keepScale": true, "motion": "responsive", "wait": true})).unwrap();
    assert_eq!(r["reframe"]["followed"], json!(1), "{r}");
    let m = motion(&s, seq);
    assert_eq!(m.param("scale").unwrap().keyframes.len(), 2, "the push-in stays");
    assert!(m.param("position").unwrap().keyframes.len() >= 2);
    assert_eq!(s.history.undo.last().unwrap().0, "Auto Reframe");
    for k in [4, 24, 44] {
        let x = disc_x(&mut s, k).unwrap();
        assert!((x - 0.5).abs() < 0.2, "frame {k}: subject at {:.0}%", x * 100.0);
    }
}

#[test]
fn a_clip_without_a_subject_is_letterboxed() {
    let (mut s, _, _) = session(false);
    let r = s.execute("sequence.autoReframe", json!({"wait": true})).unwrap();
    let out = &r["reframe"];
    assert_eq!(out["clips"][0]["letterbox"], json!(true), "{out}");
    let seq = ItemId(out["sequence"].as_u64().unwrap());
    let m = motion(&s, seq);
    // fitted whole: 320 wide into 180
    assert!((m.f64_at("scale", Tick::ZERO) - 180.0 / 320.0 * 100.0).abs() < 1e-6);
    assert_eq!(m.vec2_at("position", Tick::ZERO), filmcraft_geom::Vec2::new(90.0, 160.0));
    // centred instead, on request
    let r = s.execute("sequence.autoReframe", json!({"sequence": out["sequence"], "fallback": "center", "aspect": "16:9", "wait": true})).unwrap();
    assert_eq!(r["reframe"]["clips"][0]["letterbox"], json!(true));
}

#[test]
fn a_seed_box_starts_the_tracker_on_the_subject() {
    let (mut s, _, _) = session(true);
    let (cx, cy) = subject_at(0);
    let b = json!({"x": (cx - 26.0) / W as f64, "y": (cy - 26.0) / H as f64, "w": 52.0 / W as f64, "h": 52.0 / H as f64});
    let r = s.execute("sequence.autoReframe", json!({"subject": b, "motion": "responsive", "wait": true})).unwrap();
    assert_eq!(r["reframe"]["followed"], json!(1), "{r}");
    let x = disc_x(&mut s, 40).unwrap();
    assert!((x - 0.5).abs() < 0.15, "{x}");
}

#[test]
fn runs_as_a_job_and_lands_when_polled() {
    let (mut s, _, _) = session(true);
    let r = s.execute("sequence.autoReframe", json!({"analysisFps": 8})).unwrap();
    let job = r["job"].as_u64().unwrap();
    let t0 = std::time::Instant::now();
    let entry = loop {
        let jobs = s.execute("jobs.list", json!({})).unwrap();
        let j = jobs.as_array().unwrap().iter().find(|j| j["id"].as_u64() == Some(job)).cloned().unwrap();
        if j["finished"] == json!(true) {
            break j;
        }
        assert!(t0.elapsed().as_secs() < 120, "the job finishes");
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let seq = entry["reframe"]["sequence"].as_u64().unwrap_or_else(|| panic!("{entry}"));
    assert!(s.project.sequence(ItemId(seq)).is_some());
}

#[test]
fn a_cancelled_job_changes_nothing() {
    let (mut s, _, _) = session(true);
    let items = s.project.items.len();
    let r = s.execute("sequence.autoReframe", json!({})).unwrap();
    let job = r["job"].as_u64().unwrap();
    s.execute("jobs.cancel", json!({"job": job})).unwrap();
    let t0 = std::time::Instant::now();
    while s.jobs.iter().find(|j| j.id == job).is_some_and(|j| !j.progress.finished.load(std::sync::atomic::Ordering::Relaxed)) {
        assert!(t0.elapsed().as_secs() < 60);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    s.execute("jobs.list", json!({})).unwrap();
    // it may have finished before the cancel landed; if not, nothing was made
    let made = crate::reframe::outcome_json(&s, job).and_then(|o| o["sequence"].as_u64());
    assert_eq!(s.project.items.len(), items + usize::from(made.is_some()));
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, _, _) = session(true);
    let bad: Vec<Value> = vec![
        json!({"aspect": "0:0"}),
        json!({"aspect": "abc"}),
        json!({"aspect": 916}),
        json!({"aspect": "1:100"}),
        json!({"motion": 7}),
        json!({"motion": "wobbly"}),
        json!({"fallback": "stretch"}),
        json!({"subject": "robot"}),
        json!({"subject": {"x": 0.1}}),
        json!({"subject": {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.5}}),
        json!({"subject": {"x": f64::MAX, "y": 0.1, "w": 0.2, "h": 0.5}}),
        json!({"subject": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.5, "seconds": -4}}),
        json!({"subject": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.5, "clip": "x"}}),
        json!({"subject": vec![json!({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}); crate::reframe::MAX_SUBJECTS + 1]}),
        json!({"range": {"start": 100, "end": 10}}),
        json!({"range": {"start": -5, "end": 10}}),
        json!({"range": {"startSeconds": 1e300, "endSeconds": 1e301}}),
        json!({"range": {"startSeconds": 1000, "endSeconds": 2000}}),
        json!({"range": "all"}),
        json!({"analysisFps": 0}),
        json!({"analysisFps": 1000}),
        json!({"analysisFps": "fast"}),
        json!({"name": ""}),
        json!({"name": "x".repeat(500)}),
        json!({"name": "two\nlines"}),
        json!({"sequence": 999_999}),
        json!({"sequence": "main"}),
        json!([1, 2, 3]),
    ];
    for p in bad {
        let r = s.execute("sequence.autoReframe", p.clone());
        assert!(r.is_err(), "{p} was accepted: {r:?}");
    }
    for p in [json!({"clips": "all"}), json!({"clips": [987_654]}), json!({"clips": ["a"]}), json!({"motion": []}), json!(17)] {
        assert!(s.execute("clip.autoReframe", p.clone()).is_err(), "{p}");
    }
    // a subject box at the very edge, a NaN-free huge seconds and odd-but-valid values work
    let ok = json!({"subject": [{"x": -0.5, "y": 1.5, "w": 1.0, "h": 1.0, "seconds": 9e6}], "aspect": "4:5", "analysisFps": 1, "wait": true});
    s.execute("sequence.autoReframe", ok).unwrap();
    // nothing open
    let mut empty = Session::default();
    assert!(empty.execute("sequence.autoReframe", json!({})).is_err());
    assert!(empty.execute("clip.autoReframe", json!({})).is_err());
}
