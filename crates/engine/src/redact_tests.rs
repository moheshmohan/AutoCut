use std::sync::Arc;

use filmcraft_project::{ClipId, ItemId, SequenceSettings, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use filmcraft_vision::{Detection, Kind};
use serde_json::{Value, json};

use super::*;
use crate::Session;
use crate::media_test_util::tmp_dir;

const W: u32 = 320;
const H: u32 = 180;
const FRAMES: u32 = 40;
/// The disc's radius, pixels.
const R: f64 = 24.0;

fn hash(x: i64, y: i64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// The "face" (a warm, finely textured disc) in frame `k`: left to right across the picture.
fn face_at(k: u32) -> (f64, f64) {
    (60.0 + 200.0 * k as f64 / (FRAMES - 1) as f64, 90.0)
}

fn frame(k: u32) -> image::RgbaImage {
    let (cx, cy) = face_at(k);
    image::RgbaImage::from_fn(W, H, |x, y| {
        let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
        if ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() < R {
            // a checkerboard the blur flattens
            let n = hash(x as i64 / 2, y as i64 / 2);
            let v = if n > 0.5 { 255 } else { 120 };
            image::Rgba([v, v / 2, 0, 255])
        } else {
            image::Rgba([40, 60, 110, 255])
        }
    })
}

/// A session holding one image-sequence clip of the moving face on V1 of a 320×180 sequence,
/// selected; a fake detector that finds the warm disc.
fn session() -> (Session, ClipId, ItemId) {
    let dir = tmp_dir("redact-face");
    for k in 0..FRAMES {
        frame(k).save(dir.join(format!("shot{k:04}.png"))).unwrap();
    }
    let mut s = Session::default();
    let first = dir.join("shot0000.png");
    let r = s.execute("file.importImageSequence", json!({"path": first.to_string_lossy()})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}")));
    let rate = s.project.item(item).unwrap().as_media().unwrap().info.video.clone().unwrap().frame_rate;
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Visit", SequenceSettings { width: W, height: H, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let d = p.item(item).unwrap().duration();
    let mut v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, d), rate).unwrap();
    for e in &mut v.effects {
        filmcraft_project::resolve_auto_points(e, (W, H), (W, H));
    }
    let clip = v.id;
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    s.state.selection = vec![clip];
    s.redacts.detector = Some(Arc::new(warm_disc));
    (s, clip, seq)
}

/// The fake detector: the warm pixels' bounding box as a face.
fn warm_disc(rgba: &[u8], w: usize, h: usize) -> Vec<Detection> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, p) in rgba.as_chunks::<4>().0.iter().enumerate() {
        if p[0] > 100 && p[2] < 30 {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x1 < x0 {
        return Vec::new();
    }
    let d = Detection { kind: Kind::Face, score: 0.95, bbox: [x0 as f32, y0 as f32, (x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32], points: Vec::new() };
    vec![d.normalized(w as f32, h as f32)]
}

fn clip_effects(s: &Session, seq: ItemId, clip: ClipId) -> Vec<filmcraft_project::EffectInstance> {
    s.project.sequence(seq).unwrap().find_item(clip).unwrap().1.effects.clone()
}

/// How busy the picture is in a box around the face in sequence frame `k` (the blur flattens it).
fn busy(s: &mut Session, k: u32) -> f64 {
    let (w, _, px) = crate::media_test_util::frame_rgba(s, k as i64, 1.0);
    let (cx, cy) = face_at(k);
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in (cy - R * 0.6) as usize..(cy + R * 0.6) as usize {
        for x in (cx - R * 0.6) as usize..(cx + R * 0.6) as usize {
            let (a, b) = (px[(y * w + x) * 4] as f64, px[(y * w + x + 1) * 4] as f64);
            sum += (a - b).abs();
            n += 1.0;
        }
    }
    sum / n
}

struct Judge(std::result::Result<Option<bool>, String>, std::sync::atomic::AtomicUsize);

impl FaceJudge for Judge {
    fn name(&self) -> String {
        "test judge".into()
    }
    fn judge(&self, _: Kind, crops: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String> {
        self.1.fetch_add(crops.len(), std::sync::atomic::Ordering::Relaxed);
        assert!(crops.iter().all(|c| c.width > 0 && c.height > 0 && c.rgba.len() == (c.width * c.height * 4) as usize));
        self.0.clone().map(|a| vec![a; crops.len()])
    }
}

#[test]
fn hides_a_moving_face_in_one_undo_step_and_the_blur_follows_it() {
    let (mut s, clip, seq) = session();
    let sharp: Vec<f64> = [3, 20, 36].iter().map(|k| busy(&mut s, *k)).collect();
    let undo_before = s.history.undo.len();
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    let out = &r["redact"];
    assert_eq!(out["hidden"], json!(1), "{out}");
    assert_eq!(out["clips"][0]["faces"], json!(1), "{out}");
    assert_eq!(out["verify"]["state"], json!("off"));
    assert_eq!(s.history.undo.len(), undo_before + 1, "one undo step");
    assert_eq!(s.history.undo.last().unwrap().0, LABEL);
    let fx = clip_effects(&s, seq, clip);
    let blur = fx.iter().find(|e| e.effect == "gaussian_blur").expect("a blur");
    assert!(is_auto_redaction(blur));
    assert_eq!(blur.masks.len(), 1);
    let m = &blur.masks[0];
    assert_eq!(m.name, "Redact face 1");
    assert!(m.path.keyframes.len() >= 10, "a path keyframe at each analysed frame: {}", m.path.keyframes.len());
    // the mask's centre is on the face (clip pixels) at analysed and in-between frames
    let it = s.project.sequence(seq).unwrap().find_item(clip).unwrap().1.clone();
    let rate = s.project.sequence(seq).unwrap().settings.frame_rate;
    for k in [1u32, 7, 20, 33, 38] {
        let mt = it.source_time_at(rate.tick_of(k as i64));
        let c = m.path_at(mt).centroid();
        let (fx, fy) = face_at(k);
        assert!((c.x - fx).abs() < 6.0 && (c.y - fy).abs() < 4.0, "frame {k}: mask at {c:?}, face at ({fx}, {fy})");
        assert!(m.opacity.f64_at(mt) > 99.0, "frame {k}: the mask is on");
    }
    // the rendered picture: the face is flattened in every frame
    for (i, k) in [3u32, 20, 36].iter().enumerate() {
        let b = busy(&mut s, *k);
        assert!(b < sharp[i] * 0.35, "frame {k}: detail {b:.1} left of {:.1}", sharp[i]);
    }
    // undo and redo
    s.execute("edit.undo", json!({})).unwrap();
    assert!(!clip_effects(&s, seq, clip).iter().any(is_auto_redaction));
    s.execute("edit.redo", json!({})).unwrap();
    assert!(clip_effects(&s, seq, clip).iter().any(is_auto_redaction));
    // running again replaces it (one redaction, one more undo step)
    s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "style": "mosaic"})).unwrap();
    let fx = clip_effects(&s, seq, clip);
    assert_eq!(fx.iter().filter(|e| is_auto_redaction(e)).count(), 1);
    assert!(fx.iter().any(|e| e.effect == "mosaic"));
    // …unless asked to keep it
    s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "replace": false})).unwrap();
    assert_eq!(clip_effects(&s, seq, clip).iter().filter(|e| is_auto_redaction(e)).count(), 2);
}

#[test]
fn a_dry_run_reports_the_tracks_and_changes_nothing() {
    let (mut s, clip, seq) = session();
    let rev = s.revision;
    let undo = s.history.undo.len();
    let r = s.execute("sequence.autoRedact", json!({"wait": true, "dryRun": true, "verify": false})).unwrap();
    let out = &r["redact"];
    assert_eq!(out["dryRun"], json!(true));
    let t = &out["clips"][0]["tracks"][0];
    assert_eq!(t["kind"], json!("face"), "{out}");
    assert_eq!(t["hidden"], json!(true));
    assert!(t["path"].as_array().is_some_and(|p| p.len() >= 10), "{t}");
    assert_eq!((s.revision, s.history.undo.len()), (rev, undo));
    assert!(clip_effects(&s, seq, clip).iter().all(|e| e.masks.is_empty()));
    // the outcome is in jobs.list too
    let jobs = s.execute("jobs.list", json!({})).unwrap();
    assert!(jobs.as_array().unwrap().iter().any(|j| j["redact"]["dryRun"] == json!(true)), "{jobs}");
}

#[test]
fn the_vision_check_leaves_robots_alone() {
    let (mut s, clip, seq) = session();
    let judge = Arc::new(Judge(Ok(Some(false)), Default::default()));
    s.redacts.judge = Some(judge.clone());
    let r = s.execute("clip.autoRedact", json!({"wait": true})).unwrap();
    let out = &r["redact"];
    assert_eq!(out["hidden"], json!(0), "{out}");
    assert_eq!(out["verify"]["state"], json!("done"));
    assert_eq!(out["verify"]["notPeople"], json!(1));
    assert_eq!(out["clips"][0]["skipped"]["notAPerson"], json!(1));
    assert_eq!(judge.1.load(std::sync::atomic::Ordering::Relaxed), 1, "one crop per track");
    assert!(!clip_effects(&s, seq, clip).iter().any(is_auto_redaction));
    // a person is hidden; an unsure answer hides it too
    for a in [Some(true), None] {
        s.redacts.judge = Some(Arc::new(Judge(Ok(a), Default::default())));
        let r = s.execute("clip.autoRedact", json!({"wait": true})).unwrap();
        assert_eq!(r["redact"]["hidden"], json!(1), "{a:?}: {r}");
    }
    // a judge that fails hides everything and says so
    s.redacts.judge = Some(Arc::new(Judge(Err("Ollama is unreachable".into()), Default::default())));
    let r = s.execute("clip.autoRedact", json!({"wait": true})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(1));
    assert_eq!(r["redact"]["verify"]["state"], json!("failed"));
    // no judge: skipped, noted, everything hidden
    s.redacts.judge = None;
    let r = s.execute("clip.autoRedact", json!({"wait": true})).unwrap();
    assert_eq!(r["redact"]["verify"]["state"], json!("skipped"));
    assert_eq!(r["redact"]["hidden"], json!(1));
}

#[test]
fn small_faces_and_excluded_boxes_are_left_alone() {
    let (mut s, _, _) = session();
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "minFaceHeight": 0.5})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(0));
    assert_eq!(r["redact"]["clips"][0]["skipped"]["tooSmall"], json!(1), "{r}");
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "exclude": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}})).unwrap();
    assert_eq!(r["redact"]["clips"][0]["skipped"]["excluded"], json!(1), "{r}");
    // an exclusion at a moment the face is elsewhere does not apply
    let r =
        s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "exclude": [{"x": 0.0, "y": 0.0, "w": 0.3, "h": 1.0, "seconds": 100.0}]})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(1), "{r}");
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, _, _) = session();
    for p in [
        json!({"kinds": "elephants"}),
        json!({"kinds": []}),
        json!({"kinds": 3}),
        json!({"minFaceHeight": -1}),
        json!({"minFaceHeight": 1e308}),
        json!({"strength": 0}),
        json!({"padding": "wide"}),
        json!({"style": "smudge"}),
        json!({"analysisFps": 1000}),
        json!({"verify": "yes"}),
        json!({"exclude": 4}),
        json!({"exclude": {"x": 0.1}}),
        json!({"exclude": {"x": 0.0, "y": 0.0, "w": -1.0, "h": 1.0}}),
        json!({"exclude": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "seconds": -3}}),
        json!({"clips": "all"}),
        json!({"clips": [-1]}),
        json!({"clip": "x"}),
        json!([1, 2]),
    ] {
        assert!(s.execute("clip.autoRedact", p.clone()).is_err(), "{p}");
    }
    for p in [json!({"sequence": "x"}), json!({"sequence": 999_999}), json!({"faceScore": f64::MAX})] {
        assert!(s.execute("sequence.autoRedact", p.clone()).is_err(), "{p}");
    }
    // clips that are not in the sequence: nothing to redact
    assert!(s.execute("clip.autoRedact", json!({"clips": [987_654]})).is_err());
    assert!(!options_p("t", &json!({"kinds": "faces"})).unwrap().plates);
    assert!(!options_p("t", &json!({"kinds": ["plates"]})).unwrap().faces);
}

#[test]
fn needs_a_selection_and_says_how_to_get_the_models() {
    let (mut s, _, _) = session();
    s.state.selection.clear();
    assert!(!s.is_enabled("clip.autoRedact"));
    assert!(s.is_enabled("sequence.autoRedact"));
    assert!(s.is_enabled("redact.status"));
    let st = s.execute("redact.status", json!({})).unwrap();
    assert_eq!(st["judge"], Value::Null);
    assert!(st["license"].as_str().unwrap().contains("MIT"));
    // without the test detector and without downloaded models, the command says what to run
    s.redacts.detector = None;
    if !installed() {
        let e = s.execute("sequence.autoRedact", json!({})).unwrap_err().to_string();
        assert!(e.contains("redact.downloadModels") || e.contains("not available"), "{e}");
    }
}

#[test]
fn masks_switch_on_and_off_with_their_track() {
    use filmcraft_vision::track::TrackPoint;
    let pt = |t: f64, x: f64| TrackPoint { t, bbox: [x, 0.4, 0.1, 0.2], corners: vec![], seen: true };
    let track = Track { kind: Kind::Face, points: vec![pt(2.0, 0.1), pt(2.1, 0.2), pt(2.2, 0.3)], hits: 3, best_score: 0.9 };
    let c = ClipRedact {
        clip: ClipId(1),
        name: "a".into(),
        size: (1000.0, 500.0),
        tracks: vec![(track.clone(), None)],
        frames: 50,
        interval: 0.1,
        media: (Tick::from_seconds_f64(1.0), Tick::from_seconds_f64(6.0)),
        missed: 0,
    };
    let m = mask_for(&track, 1, &c, 0.2).unwrap();
    let at = |s: f64| m.opacity.f64_at(Tick::from_seconds_f64(s));
    assert_eq!((at(1.5), at(1.95), at(2.15), at(2.35)), (0.0, 100.0, 100.0, 0.0));
    // padded ellipse: 0.1 × 1000 px wide, plus a fifth each side
    let (lo, hi) = m.path_at(Tick::from_seconds_f64(2.0)).bounds();
    assert!(((hi.x - lo.x) - 140.0).abs() < 1.0 && ((hi.y - lo.y) - 140.0).abs() < 1.0, "{lo:?} {hi:?}");
    // a track from the clip's first frame is on from the start, one to its end stays on
    let early = Track { points: vec![pt(1.0, 0.1), pt(1.1, 0.1)], ..track.clone() };
    let m = mask_for(&early, 1, &c, 0.2).unwrap();
    assert_eq!(m.opacity.f64_at(Tick::from_seconds_f64(1.0)), 100.0);
    let late = Track { points: vec![pt(5.9, 0.1), pt(6.0, 0.1)], ..track };
    let m = mask_for(&late, 1, &c, 0.2).unwrap();
    assert_eq!(m.opacity.f64_at(Tick::from_seconds_f64(6.0)), 100.0);
    assert_eq!(m.opacity.f64_at(Tick::from_seconds_f64(5.0)), 0.0);
}

#[test]
fn crops_stay_inside_the_frame() {
    let rgba = vec![200u8; 64 * 48 * 4];
    let c = crop_around(&rgba, 64, 48, &[0.4, 0.4, 0.2, 0.2], false).unwrap();
    assert!(c.width <= CROP_SIDE && c.height <= CROP_SIDE && c.rgba.iter().all(|v| *v == 200));
    // a plate's crop outlines it in red
    let c = crop_around(&rgba, 64, 48, &[0.4, 0.4, 0.2, 0.2], true).unwrap();
    assert!(c.rgba.as_chunks::<4>().0.contains(&[255, 0, 0, 255]));
    for b in [[f64::NAN, 0.0, 0.1, 0.1], [5.0, 5.0, 0.1, 0.1], [-3.0, -3.0, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 1e300, 1e300], [0.0, 0.0, 1e300, 1.0]] {
        let _ = crop_around(&rgba, 64, 48, &b, true);
    }
    assert!(crop_around(&rgba, 0, 48, &[0.4, 0.4, 0.2, 0.2], true).is_none());
    assert!(crop_around(&rgba[..10], 64, 48, &[0.4, 0.4, 0.2, 0.2], false).is_none());
    let big = vec![0u8; 4000 * 3000 * 4];
    let c = crop_around(&big, 4000, 3000, &[0.0, 0.0, 1.0, 1.0], false).unwrap();
    assert!(c.width.max(c.height) == CROP_SIDE);
}

#[test]
fn stopping_changes_nothing() {
    let (mut s, clip, seq) = session();
    // a detector that cancels the job on its first frame
    let jobs: Arc<std::sync::Mutex<Option<Arc<filmcraft_export::Progress>>>> = Arc::default();
    let j2 = jobs.clone();
    s.redacts.detector = Some(Arc::new(move |rgba: &[u8], w: usize, h: usize| {
        // wait for the test to know the job, then stop it
        for _ in 0..500 {
            if let Some(p) = j2.lock().unwrap().as_ref() {
                p.cancel.store(true, Ordering::Relaxed);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        warm_disc(rgba, w, h)
    }));
    let r = s.execute("clip.autoRedact", json!({"verify": false})).unwrap();
    let id = r["job"].as_u64().unwrap();
    *jobs.lock().unwrap() = s.jobs.iter().find(|j| j.id == id).map(|j| j.progress.clone());
    for _ in 0..500 {
        poll(&mut s);
        if outcome_json(&s, id).is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let o = outcome_json(&s, id).expect("finished");
    assert!(o["error"].as_str().is_some_and(|e| e.contains("stopped")), "{o}");
    assert!(!clip_effects(&s, seq, clip).iter().any(is_auto_redaction));
}
