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
    s.redacts.detector = Some(Arc::new(|rgba: &[u8], w: usize, h: usize, _: usize| warm_disc(rgba, w, h)));
    (s, clip, seq)
}

/// The session with a blurred background copy of the clip on V1 (the clip itself moved up to a
/// new V2), as the showcase lays out a portrait shot shown whole.
fn session_with_background() -> (Session, ClipId, ClipId, ItemId) {
    let (mut s, clip, seq) = session();
    let mut p = (*s.project).clone();
    let q = p.sequence(seq).unwrap();
    let mut bg = q.video_tracks[0].items[0].clone();
    bg.id = ClipId(p.next_id);
    p.next_id += 1;
    bg.effects.push(filmcraft_project::find_effect("gaussian_blur").unwrap().instance());
    let bg_id = bg.id;
    let q = p.sequence_mut(seq).unwrap();
    let mut v2 = q.video_tracks[0].clone();
    v2.id = filmcraft_project::TrackId(p.next_id);
    p.next_id += 1;
    let q = p.sequence_mut(seq).unwrap();
    // the foreground stays on top (V2), the copy goes under it (V1)
    q.video_tracks[0].items = vec![bg];
    q.video_tracks.push(v2);
    q.check().unwrap();
    s.project = std::sync::Arc::new(p);
    s.state.selection = vec![clip, bg_id];
    (s, clip, bg_id, seq)
}

#[test]
fn a_background_copy_shares_the_analysis_and_gets_the_masks_first() {
    let (mut s, clip, bg, seq) = session_with_background();
    let one = s.execute("clip.autoRedact", json!({"clips": [clip.0], "wait": true, "verify": false, "dryRun": true})).unwrap();
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    assert_eq!(r["frames"], one["frames"], "the copy is analysed once: {r}");
    assert_eq!(r["redact"]["clips"].as_array().unwrap().len(), 1, "{r}");
    let c = &r["redact"]["clips"][0];
    let mut both = vec![c["clip"].as_u64().unwrap(), c["also"][0].as_u64().unwrap()];
    both.sort();
    assert_eq!(both, vec![clip.0.min(bg.0), clip.0.max(bg.0)], "{c}");
    assert_eq!(r["redact"]["clipsChanged"], json!(2));
    for id in [clip, bg] {
        let fx = clip_effects(&s, seq, id);
        let at = fx.iter().position(is_auto_redaction).unwrap_or_else(|| panic!("{id:?} has no redaction"));
        // ahead of every other standard effect (the copy's own blur)
        let first_standard = fx.iter().position(|e| e.def().is_some_and(|d| !d.intrinsic)).unwrap();
        assert_eq!(at, first_standard, "{id:?}: {:?}", fx.iter().map(|e| e.effect.as_str()).collect::<Vec<_>>());
    }
}

#[test]
fn a_photo_is_looked_at_once_and_hidden_for_its_whole_hold() {
    let dir = tmp_dir("redact-photo");
    let path = dir.join("visit.png");
    frame(10).save(&path).unwrap();
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
    let item = r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}"));
    s.execute("file.newSequence", json!({"name": "Photos", "width": W, "height": H, "fps": 25})).unwrap();
    let four = Tick::from_seconds_f64(4.0).0;
    let placed = s.execute("timeline.place", json!({"item": item, "track": "V1", "time": 0, "sourceIn": 0, "duration": four, "audio": false})).unwrap();
    let clip = placed["clips"][0].as_u64().unwrap();
    // a detector that is not sure of it (0.6): on a photo, seen once is enough
    s.redacts.detector = Some(Arc::new(|rgba: &[u8], w: usize, h: usize, _: usize| {
        warm_disc(rgba, w, h)
            .into_iter()
            .map(|mut d| {
                d.score = 0.6;
                d
            })
            .collect()
    }));
    let r = s.execute("clip.autoRedact", json!({"clips": [clip], "wait": true, "verify": false})).unwrap();
    assert_eq!(r["frames"], json!(1), "one look for the whole hold: {r}");
    assert_eq!(r["redact"]["clips"][0]["faces"], json!(1), "{r}");
    let seq = s.state.active_sequence.unwrap();
    let fx = clip_effects(&s, seq, ClipId(clip));
    let m = &fx.iter().find(|e| is_auto_redaction(e)).unwrap().masks[0];
    // the same mask from the first frame to the last, never switched off
    assert!(m.path.keyframes.len() <= 2, "{} path keyframes", m.path.keyframes.len());
    assert!(m.opacity.keyframes.is_empty(), "{:?}", m.opacity.keyframes);
    assert_eq!(m.opacity.value, filmcraft_project::ParamValue::Float(100.0));
    let _ = std::fs::remove_dir_all(&dir);
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
    let d =
        Detection { kind: Kind::Face, score: 0.95, bbox: [x0 as f32, y0 as f32, (x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32], points: Vec::new(), pose: false };
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

impl VisionJudge for Judge {
    fn name(&self) -> String {
        "test judge".into()
    }
    fn judge(&self, _: Kind, crops: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String> {
        self.1.fetch_add(crops.len(), std::sync::atomic::Ordering::Relaxed);
        assert!(crops.iter().all(|c| c.width > 0 && c.height > 0 && c.rgba.len() == (c.width * c.height * 4) as usize));
        self.0.clone().map(|a| vec![a; crops.len()])
    }
    /// The warm disc, for the first target.
    fn find(&self, frame: &Crop, targets: &[String]) -> std::result::Result<Vec<(usize, [f64; 4])>, String> {
        assert!(!targets.is_empty());
        Ok(warm_disc(&frame.rgba, frame.width as usize, frame.height as usize)
            .into_iter()
            .map(|d| (0, [f64::from(d.bbox[0]), f64::from(d.bbox[1]), f64::from(d.bbox[2]), f64::from(d.bbox[3])]))
            .collect())
    }
}

#[test]
fn a_named_target_is_found_followed_and_hidden_under_a_rectangle() {
    let (mut s, clip, seq) = session();
    // without a vision model, named targets cannot be looked for
    let e = s.execute("clip.autoRedact", json!({"wait": true, "kinds": [], "targets": ["the logo"]})).unwrap_err().to_string();
    assert!(e.contains("vision model"), "{e}");
    s.redacts.judge = Some(Arc::new(Judge(Ok(Some(true)), Default::default())));
    let r = s.execute("clip.autoRedact", json!({"wait": true, "kinds": [], "targets": ["the logo"], "targetEvery": 0.5})).unwrap();
    let c = &r["redact"]["clips"][0];
    assert_eq!(c["targets"], json!(1), "{r}");
    assert_eq!(c["faces"], json!(0));
    assert!(c["looks"]["asked"].as_u64().unwrap() >= 2, "{c}");
    let fx = clip_effects(&s, seq, clip);
    let m = &fx.iter().find(|e| is_auto_redaction(e)).unwrap().masks[0];
    assert_eq!(m.name, "Redact the logo 1");
    // a rectangle (four corners) that follows the disc
    let it = s.project.sequence(seq).unwrap().find_item(clip).unwrap().1.clone();
    let rate = s.project.sequence(seq).unwrap().settings.frame_rate;
    for k in [4u32, 20, 35] {
        let path = m.path_at(it.source_time_at(rate.tick_of(k as i64)));
        assert_eq!(path.vertices.len(), 4);
        let c = path.centroid();
        let (fx, fy) = face_at(k);
        assert!((c.x - fx).abs() < 10.0 && (c.y - fy).abs() < 8.0, "frame {k}: {c:?} for ({fx}, {fy})");
    }
    // hostile targets
    for t in [json!("a"), json!([""]), json!([7]), json!([{"text": "x", "clips": "all"}]), json!(["a\nb"]), json!(vec!["x"; 17])] {
        assert!(s.execute("clip.autoRedact", json!({"targets": t.clone()})).is_err(), "{t}");
    }
    assert!(s.execute("clip.autoRedact", json!({"kinds": []})).is_err(), "nothing to hide");
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
    // the robot: the subject's box, where the face passes
    let robot = json!({"wait": true, "exclude": {"x": 0.0, "y": 0.2, "w": 1.0, "h": 0.6}});
    let r = s.execute("clip.autoRedact", robot).unwrap();
    let out = &r["redact"];
    assert_eq!(out["hidden"], json!(0), "{out}");
    assert_eq!(out["verify"]["state"], json!("done"));
    assert_eq!(out["verify"]["notPeople"], json!(1));
    assert_eq!(out["verify"]["onSubject"], json!(1));
    assert_eq!(out["clips"][0]["skipped"]["notAPerson"], json!(1));
    assert_eq!(judge.1.load(std::sync::atomic::Ordering::Relaxed), 2, "one crop per track, and one more look at it alone");
    assert!(!clip_effects(&s, seq, clip).iter().any(is_auto_redaction));
    // away from the subject, a face the detector is sure of (0.95) is hidden whatever the model
    // says; one it is not sure of is left alone
    let r = s.execute("clip.autoRedact", json!({"wait": true})).unwrap();
    assert_eq!((r["redact"]["hidden"].as_u64(), r["redact"]["verify"]["overruled"].as_u64()), (Some(1), Some(1)), "{r}");
    let sure = s
        .redacts
        .detector
        .replace(Arc::new(|rgba: &[u8], w: usize, h: usize, _: usize| warm_disc(rgba, w, h).into_iter().map(|d| Detection { score: 0.7, ..d }).collect()));
    let r = s.execute("clip.autoRedact", json!({"wait": true})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(0), "{r}");
    s.redacts.detector = sure;
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

/// A judge whose grid answer (`first`) and second look (`second`) differ.
struct TwoLooks(Option<bool>, Option<bool>, std::sync::atomic::AtomicUsize);

impl VisionJudge for TwoLooks {
    fn name(&self) -> String {
        "two looks".into()
    }
    fn judge(&self, _: Kind, crops: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String> {
        Ok(vec![self.0; crops.len()])
    }
    fn confirm(&self, _: Kind, crop: &Crop) -> std::result::Result<Option<bool>, String> {
        assert!(crop.rgba.as_chunks::<4>().0.contains(&[255, 0, 0, 255]), "the second look is outlined");
        self.2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(self.1)
    }
}

/// The bug of 2026-10-21 (Every Robot I've Met, Ameca at GITEX): visitors crowding round the
/// robot stood inside its box (its largest analysed box, taken square: 57 % of the frame's width
/// and all its height), and every face there was left sharp as the robot's, unasked. Two of them,
/// at 0.92, filled a sixth of the frame for six seconds.
#[test]
fn a_visitor_inside_the_robots_box_is_hidden_when_the_model_sees_a_person() {
    let (mut s, clip, seq) = session();
    let robot = json!({"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "seconds": 0.5});
    s.redacts.judge = Some(Arc::new(Judge(Ok(Some(true)), Default::default())));
    let r = s.execute("clip.autoRedact", json!({"wait": true, "exclude": robot})).unwrap();
    let out = &r["redact"];
    assert_eq!(out["hidden"], json!(1), "the visitor was left sharp as the robot's face: {out}");
    assert_eq!((out["verify"]["onSubject"].as_u64(), out["clips"][0]["skipped"]["excluded"].as_u64()), (Some(1), Some(0)), "{out}");
    assert!(clip_effects(&s, seq, clip).iter().any(is_auto_redaction));
    // the robot's box is in the outcome, for the check of the finished video
    assert_eq!(out["clips"][0]["robots"][0]["w"], json!(1.0), "{out}");
    // the grid called the visitor a robot (a robot's head beside it), the second look a person
    let two = Arc::new(TwoLooks(Some(false), Some(true), Default::default()));
    s.redacts.judge = Some(two.clone());
    let r = s.execute("clip.autoRedact", json!({"wait": true, "exclude": robot})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(1), "{r}");
    assert_eq!((two.2.load(std::sync::atomic::Ordering::Relaxed), r["redact"]["verify"]["lookedAgain"].as_u64()), (1, Some(1)));
    // an unclear answer hides it too
    s.redacts.judge = Some(Arc::new(Judge(Ok(None), Default::default())));
    let r = s.execute("clip.autoRedact", json!({"wait": true, "exclude": robot})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(1), "{r}");
    // without a vision model the box alone decides, as before: the robot's face is left alone
    s.redacts.judge = None;
    let r = s.execute("clip.autoRedact", json!({"wait": true, "exclude": robot})).unwrap();
    assert_eq!((r["redact"]["hidden"].as_u64(), r["redact"]["clips"][0]["skipped"]["excluded"].as_u64()), (Some(0), Some(1)), "{r}");
    // a model that fails leaves the robot's face to its box, too
    s.redacts.judge = Some(Arc::new(Judge(Err("Ollama is unreachable".into()), Default::default())));
    let r = s.execute("clip.autoRedact", json!({"wait": true, "exclude": robot})).unwrap();
    assert_eq!((r["redact"]["hidden"].as_u64(), r["redact"]["verify"]["state"].as_str()), (Some(0), Some("failed")), "{r}");
}

/// The face detector looks again twice as close where the picture allows it (a visitor 7 % of
/// the frame tall was missed at 640 px and found at 1280), never past 1920 or the picture.
#[test]
fn the_face_detector_looks_closer_where_the_picture_allows() {
    assert_eq!(close_side(640, 1920), Some(1280));
    assert_eq!(close_side(1280, 3840), Some(1920));
    assert_eq!(close_side(640, 640), None, "no closer picture to look at");
    assert_eq!(close_side(640, 760), None, "not enough closer to be worth it");
    assert_eq!(close_side(1920, 4000), None);
    assert_eq!(close_side(usize::MAX, usize::MAX), None);
    assert_eq!(close_side(0, 1920), None, "a degenerate side is left alone");
}

/// The second looks are kept in the cache like the first: a rebuilt show asks nothing again.
#[test]
fn second_looks_come_from_the_cache() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (mut s, _, _) = cached_session(calls);
    let two = Arc::new(TwoLooks(Some(false), Some(false), Default::default()));
    s.redacts.judge = Some(two.clone());
    let p = json!({"wait": true, "dryRun": true, "exclude": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}});
    let first = s.execute("clip.autoRedact", p.clone()).unwrap();
    assert_eq!(first["redact"]["hidden"], json!(0), "{first}");
    let second = s.execute("clip.autoRedact", p).unwrap();
    assert_eq!(two.2.load(std::sync::atomic::Ordering::Relaxed), 1, "looked again once");
    assert_eq!(second["redact"]["verify"]["lookedAgainFromCache"], json!(1), "{second}");
    assert_eq!(tracks_of(&first), tracks_of(&second));
}

#[test]
fn small_faces_and_excluded_boxes_are_left_alone() {
    let (mut s, clip, seq) = session();
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "minFaceHeight": 0.5})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(0));
    assert_eq!(r["redact"]["clips"][0]["skipped"]["tooSmall"], json!(1), "{r}");
    // the same face on a clip zoomed in to 300 % is shown three times as tall: hidden
    let mut p = (*s.project).clone();
    let (_, it) = p.sequence_mut(seq).unwrap().find_item_mut(clip).unwrap();
    *it.effect_mut("motion").unwrap().param_mut("scale").unwrap() = filmcraft_project::Param::new(filmcraft_project::ParamValue::Float(300.0));
    s.project = std::sync::Arc::new(p);
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "minFaceHeight": 0.5})).unwrap();
    assert_eq!(r["redact"]["hidden"], json!(1), "{r}");
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
    // the profile pass: true, false or not said
    for bad in [json!("yes"), json!(1), json!([true]), json!({"on": true})] {
        let p = json!({"profiles": bad});
        assert!(s.execute("clip.autoRedact", p.clone()).is_err(), "{p}");
        assert!(s.execute("sequence.autoRedact", p.clone()).is_err(), "{p}");
        assert!(s.execute("redact.downloadModels", p.clone()).is_err(), "{p}");
        assert!(s.execute("sequence.checkPrivacy", json!({"seconds": [0.0], "profiles": bad})).is_err(), "{p}");
    }
    assert!(s.execute("redact.downloadModels", json!({"wait": "soon"})).is_err());
    assert!(s.execute("redact.downloadModels", json!(7)).is_err());
    assert_eq!(options_p("t", &json!({})).unwrap().profiles, None);
    assert_eq!(options_p("t", &json!({"profiles": null})).unwrap().profiles, None);
    assert_eq!(options_p("t", &json!({"profiles": false})).unwrap().profiles, Some(false));
    // not looking for faces: no profile pass whatever is asked
    assert!(!profiles_on(&options_p("t", &json!({"kinds": "plates", "profiles": true})).unwrap()));
    assert!(!profiles_on(&options_p("t", &json!({"profiles": false})).unwrap()));
}

/// The profile pass's models are in the detections' cache key: entries made without it (or with
/// other models) are analysed again once; without it, the key is the one entries always had.
#[test]
fn the_profile_models_are_in_the_cache_key() {
    let off = Options { profiles: Some(false), ..Options::default() };
    let unset = Options::default();
    let on = Options { profiles: Some(true), ..Options::default() };
    let key = |o: &Options| detections_key("models:abc", "file:1:2|1920x1080", 640, 0.5, 1280, o);
    assert_eq!(key(&off), key(&unset));
    assert_eq!(
        key(&off),
        format!(
            "det v{}|file:1:2|1920x1080|models:abc|faces=true plates=true fs=0.600 ps=0.800 side=640 pside=640 scale=0.50000 close=1280",
            crate::redact_cache::VERSION
        )
    );
    let k = key(&on);
    assert_ne!(k, key(&off));
    for f in filmcraft_vision::catalog::PROFILE_FILES {
        assert!(k.contains(f.name) && k.contains(&f.sha256[..16]), "{k}");
    }
}

/// Faces the profile pass found (a head from a pose: in profile, half hidden) are followed,
/// checked and hidden like the face detector's, and the outcome says which they were.
#[test]
fn heads_found_from_poses_are_hidden_and_reported() {
    let (mut s, clip, seq) = session();
    s.redacts.detector = Some(Arc::new(|rgba: &[u8], w: usize, h: usize, _: usize| {
        warm_disc(rgba, w, h).into_iter().map(|d| Detection { pose: true, score: 0.6, ..d }).collect()
    }));
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    let out = &r["redact"];
    assert_eq!(out["hidden"], json!(1), "{out}");
    let c = &out["clips"][0];
    assert_eq!((c["faces"].as_u64(), c["facesFromPose"].as_u64(), c["facesOnlyFromPose"].as_u64()), (Some(1), Some(1), Some(1)), "{c}");
    assert!(c["tracks"][0]["fromPose"].as_u64().unwrap() >= 2, "{c}");
    // a host's own detector is the whole detector: no profile pass of the engine's
    assert_eq!(out["profiles"], json!(false));
    assert_eq!(clip_effects(&s, seq, clip).iter().filter(|e| is_auto_redaction(e)).count(), 1);
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
    // the profile pass's models: their licence, sizes and files before any download
    let pr = &st["profiles"];
    assert!(pr["license"].as_str().unwrap().contains("Apache-2.0") && pr["files"].as_array().unwrap().len() == 3, "{pr}");
    assert!(pr["missingBytes"].as_u64().unwrap() <= filmcraft_vision::catalog::profile_total_bytes());
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
    let track = Track { kind: Kind::Face, label: None, points: vec![pt(2.0, 0.1), pt(2.1, 0.2), pt(2.2, 0.3)], hits: 3, best_score: 0.9, pose_hits: 0 };
    let c = ClipRedact {
        clip: ClipId(1),
        also: Vec::new(),
        name: "a".into(),
        size: (1000.0, 500.0),
        tracks: vec![(track.clone(), None)],
        frames: 50,
        interval: 0.1,
        media: (Tick::from_seconds_f64(1.0), Tick::from_seconds_f64(6.0)),
        missed: 0,
        looks: 0,
        found: 0,
        look_errors: Vec::new(),
        robots: Vec::new(),
    };
    let m = mask_for(&track, 1, &c, 0.2).unwrap();
    let at = |s: f64| m.opacity.f64_at(Tick::from_seconds_f64(s));
    // on LINGER (0.4 s) before the first sighting and after the last: a face going behind
    // something is lost before it is gone
    assert_eq!((at(1.5), at(1.65), at(1.95), at(2.15), at(2.55), at(2.65)), (0.0, 100.0, 100.0, 100.0, 100.0, 0.0));
    // padded ellipse: 0.1 × 1000 px wide, plus a fifth each side, and CHIN (15 px) further down
    let (lo, hi) = m.path_at(Tick::from_seconds_f64(2.0)).bounds();
    assert!(((hi.x - lo.x) - 140.0).abs() < 1.0 && ((hi.y - lo.y) - 155.0).abs() < 1.0, "{lo:?} {hi:?}");
    // a track from the clip's first frame is on from the start, one to its end stays on
    let early = Track { points: vec![pt(1.0, 0.1), pt(1.1, 0.1)], ..track.clone() };
    let m = mask_for(&early, 1, &c, 0.2).unwrap();
    assert_eq!(m.opacity.f64_at(Tick::from_seconds_f64(1.0)), 100.0);
    let late = Track { points: vec![pt(5.9, 0.1), pt(6.0, 0.1)], ..track };
    let m = mask_for(&late, 1, &c, 0.2).unwrap();
    assert_eq!(m.opacity.f64_at(Tick::from_seconds_f64(6.0)), 100.0);
    assert_eq!(m.opacity.f64_at(Tick::from_seconds_f64(5.0)), 0.0);
}

/// A clip of 1000 × 500 px analysed from 1 s to 6 s every 0.1 s, with `tracks`.
fn clip_redact(tracks: Vec<(Track, Option<filmcraft_vision::track::Skip>)>) -> ClipRedact {
    ClipRedact {
        clip: ClipId(1),
        also: Vec::new(),
        name: "a".into(),
        size: (1000.0, 500.0),
        tracks,
        frames: 50,
        interval: 0.1,
        media: (Tick::from_seconds_f64(1.0), Tick::from_seconds_f64(6.0)),
        missed: 0,
        looks: 0,
        found: 0,
        look_errors: Vec::new(),
        robots: Vec::new(),
    }
}

/// The blur's coverage of mask `m` at clip pixel `(x, y)` at `s` seconds (the renderer's
/// falloff of the signed distance to its shape).
fn cover(m: &filmcraft_project::Mask, s: f64, (x, y): (f64, f64)) -> f64 {
    let t = Tick::from_seconds_f64(s);
    let poly: Vec<(f64, f64)> = m.path_at(t).flatten(0.5).iter().map(|p| (p.x, p.y)).collect();
    let n = poly.len();
    let (mut inside, mut best) = (false, f64::INFINITY);
    for i in 0..n {
        let ((ax, ay), (bx, by)) = (poly[i], poly[(i + n - 1) % n]);
        if (ay > y) != (by > y) && x < ax + (y - ay) * (bx - ax) / (by - ay) {
            inside = !inside;
        }
        let (dx, dy) = (bx - ax, by - ay);
        let u = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy).max(1e-12)).clamp(0.0, 1.0);
        best = best.min((ax + u * dx - x).hypot(ay + u * dy - y));
    }
    let sd = if inside { best } else { -best };
    let (f, e) = (m.feather.f64_at(t), m.expansion.f64_at(t));
    f64::from(filmcraft_render::mask::falloff((sd + e) as f32, f as f32)) * m.opacity.f64_at(t) / 100.0
}

#[test]
fn neighbouring_faces_are_blurred_apart_unless_they_overlap() {
    use filmcraft_vision::track::TrackPoint;
    let face = |x: f64, from: f64, to: f64| Track {
        kind: Kind::Face,
        label: None,
        points: (0..)
            .map(|k| from + f64::from(k) * 0.1)
            .take_while(|t| *t <= to + 1e-9)
            .map(|t| TrackPoint { t, bbox: [x, 0.4, 0.1, 0.2], corners: vec![], seen: true })
            .collect(),
        hits: 10,
        best_score: 0.9,
        pose_hits: 0,
    };
    let at = |m: &filmcraft_project::Mask, s: f64| (m.feather.f64_at(Tick::from_seconds_f64(s)), m.expansion.f64_at(Tick::from_seconds_f64(s)));
    // two faces 100 px square in a 1000 × 500 clip, side by side from 2 s to 3 s, their padded
    // ellipses 10 px apart (290–410 and 420–540); alone each one's blur would reach 16 px out
    let c = clip_redact(vec![(face(0.3, 2.0, 3.0), None), (face(0.43, 2.0, 3.0), None)]);
    let ms: Vec<filmcraft_project::Mask> = c.tracks.iter().map(|(t, _)| mask_for(t, 1, &c, 0.1).unwrap()).collect();
    for m in &ms {
        // reaching 5 px, half the gap: a soft edge 10 px wide centred on the outline
        let (f, e) = at(m, 2.5);
        assert!((f - 10.0).abs() < 0.2 && e.abs() < 1e-9, "{f} {e}");
    }
    // a sharp line between them: no blur half way; each face (300–400, 430–530) wholly hidden
    let mid = (415.0, 250.0);
    assert!(ms.iter().map(|m| cover(m, 2.5, mid)).sum::<f64>() < 0.01, "{:?}", ms.iter().map(|m| cover(m, 2.5, mid)).collect::<Vec<_>>());
    for x in [300.0, 350.0, 400.0] {
        assert!(cover(&ms[0], 2.5, (x, 250.0)) > 0.999 && cover(&ms[1], 2.5, (x + 130.0, 250.0)) > 0.999);
    }
    // alone: the full soft edge (16 px, wholly outside), not keyframed, and the middle blurred
    let alone = clip_redact(vec![(face(0.3, 2.0, 3.0), None)]);
    let m = mask_for(&alone.tracks[0].0, 1, &alone, 0.1).unwrap();
    assert_eq!(at(&m, 2.5), (100.0 * FEATHER, 50.0 * FEATHER));
    assert!(m.feather.keyframes.is_empty() && m.expansion.keyframes.is_empty());
    assert!(cover(&m, 2.5, mid) > 0.1, "{}", cover(&m, 2.5, mid));
    // far apart: as alone
    let c = clip_redact(vec![(face(0.1, 2.0, 3.0), None), (face(0.7, 2.0, 3.0), None)]);
    let m = mask_for(&c.tracks[0].0, 1, &c, 0.1).unwrap();
    assert!(m.feather.keyframes.is_empty() && at(&m, 2.5) == (16.0, 8.0));
    // shapes 30 px apart: drawn in a little (reaching 15 px); 50 px apart: room for both
    let c = clip_redact(vec![(face(0.3, 2.0, 3.0), None), (face(0.45, 2.0, 3.0), None)]);
    let m = mask_for(&c.tracks[0].0, 1, &c, 0.1).unwrap();
    assert_eq!(at(&m, 2.5), (16.0, 7.0));
    let c = clip_redact(vec![(face(0.3, 2.0, 3.0), None), (face(0.47, 2.0, 3.0), None)]);
    let m = mask_for(&c.tracks[0].0, 1, &c, 0.1).unwrap();
    assert_eq!(at(&m, 2.5), (16.0, 8.0));
    // overlapping faces truly merge, each still softly edged (half the feather, centred)
    let c = clip_redact(vec![(face(0.3, 2.0, 3.0), None), (face(0.36, 2.0, 3.0), None)]);
    let m = mask_for(&c.tracks[0].0, 1, &c, 0.1).unwrap();
    assert_eq!(at(&m, 2.5), (8.0, 0.0));
    // a neighbour there only from 3.5 s on draws the edge in from just before it shows (its
    // linger and fade) and not at 2 s
    let c = clip_redact(vec![(face(0.3, 2.0, 5.0), None), (face(0.43, 3.5, 5.0), None)]);
    let m = mask_for(&c.tracks[0].0, 1, &c, 0.1).unwrap();
    assert_eq!(at(&m, 2.0), (16.0, 8.0));
    assert!(at(&m, 3.2).0 <= 10.2 && at(&m, 4.0).0 <= 10.2, "{:?} {:?}", at(&m, 3.2), at(&m, 4.0));
    // the soft edge for each reach: wholly outside, drawn in, narrowed, never under half
    assert_eq!(soft_edge(16.0, 16.0), (16.0, 8.0));
    assert_eq!(soft_edge(12.0, 16.0), (16.0, 4.0));
    assert_eq!(soft_edge(8.0, 16.0), (16.0, 0.0));
    assert_eq!(soft_edge(5.0, 16.0), (10.0, 0.0));
    assert_eq!(soft_edge(0.0, 16.0), (8.0, 0.0));
    for (r, f) in [(f64::NAN, 16.0), (5.0, f64::NAN), (f64::INFINITY, 16.0), (-3.0, -3.0), (1e12, 1e12)] {
        let (a, b) = soft_edge(r, f);
        assert!(a.is_finite() && b.is_finite() && a >= 0.5 && b >= 0.0, "{r} {f}: {a} {b}");
    }
    // hostile neighbours: finite edges
    let mut odd = face(0.43, 2.0, 3.0);
    odd.points[3].bbox = [f64::NAN, f64::INFINITY, -1.0, f64::NAN];
    let c = clip_redact(vec![(face(0.3, 2.0, 3.0), None), (odd, None)]);
    let m = mask_for(&c.tracks[0].0, 1, &c, 0.1).unwrap();
    for k in 0..40 {
        let s = 1.8 + f64::from(k) * 0.04;
        let (f, e) = at(&m, s);
        assert!(f.is_finite() && (8.0..=16.0).contains(&f) && e.is_finite() && (0.0..=8.0).contains(&e), "{s}: {f} {e}");
    }
    assert_eq!(shape_gap(&[0.0, 0.0, 10.0, 10.0], true, &[f64::NAN, 0.0, 1.0, 1.0], true), 0.0);
    assert!((shape_gap(&[0.0, 0.0, 10.0, 10.0], false, &[20.0, 0.0, 10.0, 10.0], false) - 10.0).abs() < 1e-9);
    assert_eq!(shape_gap(&[0.0, 0.0, 10.0, 10.0], true, &[2.0, 2.0, 4.0, 4.0], true), 0.0, "one inside the other");
}

#[test]
fn a_small_faces_mask_reaches_down_to_the_least_height() {
    use filmcraft_vision::track::TrackPoint;
    let face = |h: f64| Track {
        kind: Kind::Face,
        label: None,
        points: vec![TrackPoint { t: 2.0, bbox: [0.4, 0.4, h / 2.0, h], corners: vec![], seen: true }],
        hits: 2,
        best_score: 0.9,
        pose_hits: 0,
    };
    let c = clip_redact(Vec::new());
    let bounds = |t: &Track| mask_for(t, 1, &c, 0.1).unwrap().path_at(Tick::from_seconds_f64(2.0)).bounds();
    // a face 4 % of the 500 px picture tall (20 px): its mask starts 0.1 of the face above it and
    // reaches down to MIN_FACE_MASK (45 px), as wide as padding makes it
    let (lo, hi) = bounds(&face(0.04));
    assert!((lo.y - 198.0).abs() < 1.0 && (hi.y - (198.0 + MIN_FACE_MASK * 500.0)).abs() < 1.0, "{lo:?} {hi:?}");
    assert!(((hi.x - lo.x) - 24.0).abs() < 1.0, "{lo:?} {hi:?}");
    // a larger face's mask is taller anyway: padding and CHIN only
    let (lo, hi) = bounds(&face(0.2));
    assert!(((hi.y - lo.y) - 0.2 * 500.0 * (1.2 + CHIN)).abs() < 1.0, "{lo:?} {hi:?}");
}

#[test]
fn faces_are_blurred_by_their_own_size() {
    use filmcraft_vision::track::TrackPoint;
    let track = |h: f64, x: f64| Track {
        kind: Kind::Face,
        label: None,
        points: vec![
            TrackPoint { t: 2.0, bbox: [x, 0.3, h / 2.0, h], corners: vec![], seen: true },
            TrackPoint { t: 2.1, bbox: [x, 0.3, h / 2.0, h], corners: vec![], seen: true },
        ],
        hits: 2,
        best_score: 0.9,
        pose_hits: 0,
    };
    // faces 40, 50 and 200 px tall (in a 500 px tall clip): the two small ones share an effect,
    // the large one has its own, each blurred for its band's largest face
    let c = clip_redact(vec![(track(0.08, 0.1), None), (track(0.1, 0.3), None), (track(0.4, 0.6), None)]);
    let o = Options::default();
    let fx = effects_for(&c, &o);
    assert_eq!(fx.len(), 2, "{:?}", fx.iter().map(|e| e.masks.len()).collect::<Vec<_>>());
    assert!(fx.iter().all(is_auto_redaction));
    assert_eq!(fx.iter().map(|e| e.masks.len()).collect::<Vec<_>>(), vec![2, 1]);
    let blur = |e: &EffectInstance| e.params["blurriness"].value.as_f64().unwrap();
    assert!((blur(&fx[0]) - 50.0 * BLUR_PER_HEIGHT).abs() < 1e-9, "{}", blur(&fx[0]));
    assert!((blur(&fx[1]) - 200.0 * BLUR_PER_HEIGHT).abs() < 1e-9, "{}", blur(&fx[1]));
    // σ 0.15 of the face's height at the default strength; lighter and heavier with it
    assert!((blurriness(100.0, 60.0) - 30.0).abs() < 1e-9);
    assert!(blurriness(100.0, 30.0) < blurriness(100.0, 60.0) && blurriness(100.0, 60.0) < blurriness(100.0, 100.0));
    // hostile sizes and strengths stay finite and in range
    for (size, strength) in [(f64::NAN, 60.0), (f64::INFINITY, 60.0), (100.0, f64::NAN), (-5.0, 60.0), (1e12, 1e12)] {
        let b = blurriness(size, strength);
        assert!(b.is_finite() && (6.0..=2000.0).contains(&b), "{size} {strength}: {b}");
    }
    let mut odd = track(0.1, 0.3);
    odd.points[0].bbox = [f64::NAN, f64::INFINITY, -1.0, f64::NAN];
    let c = clip_redact(vec![(odd, None)]);
    let _ = effects_for(&c, &o);
    // the mosaic too, one effect per band
    let c = clip_redact(vec![(track(0.08, 0.1), None), (track(0.4, 0.6), None)]);
    let fx = effects_for(&c, &Options { style: Style::Mosaic, ..Options::default() });
    assert_eq!(fx.len(), 2);
    assert!(fx.iter().all(|e| e.effect == "mosaic"));
    // nothing hidden: no effect
    let c = clip_redact(vec![(track(0.1, 0.3), Some(filmcraft_vision::track::Skip::TooSmall))]);
    assert!(effects_for(&c, &o).is_empty());
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
    s.redacts.detector = Some(Arc::new(move |rgba: &[u8], w: usize, h: usize, _: usize| {
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

/// The outcome says how much of the picture the masks cover, and when in the sequence: a small
/// face is a few percent; a "face" over most of the picture (what a blur filling the frame came
/// from) is reported as such, at its sequence time.
#[test]
fn coverage_reports_the_largest_mask_and_when_it_plays() {
    let (mut s, _, seq) = session();
    // the clip starts 1 s into the sequence
    let mut p = (*s.project).clone();
    p.sequence_mut(seq).unwrap().video_tracks[0].items[0].start = Tick::from_seconds_f64(1.0);
    s.project = Arc::new(p);
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    let cov = &r["redact"]["clips"][0]["coverage"];
    // a 48 px disc in 320 × 180, padded by a fifth each side, as an ellipse: about 6 %
    let largest = cov["largestMask"].as_f64().unwrap();
    assert!((0.04..0.09).contains(&largest), "{cov}");
    assert_eq!(cov["largestMaskName"], json!("Redact face 1"));
    assert_eq!(cov["mostHidden"].as_f64(), Some(largest), "one face: the most hidden is that face");
    let (media, sequence) = (cov["largestAt"]["media"].as_f64().unwrap(), cov["largestAt"]["sequence"].as_f64().unwrap());
    assert!((sequence - media - 1.0).abs() < 0.01, "{cov}");
    // a detector that boxes most of the picture
    s.redacts.detector = Some(Arc::new(|_: &[u8], _: usize, _: usize, _: usize| {
        vec![Detection { kind: Kind::Face, score: 0.95, bbox: [0.1, 0.05, 0.8, 0.9], points: Vec::new(), pose: false }]
    }));
    let r = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    let cov = &r["redact"]["clips"][0]["coverage"];
    assert!(cov["largestMask"].as_f64().unwrap() > 0.5, "{cov}");
}

#[test]
fn media_times_map_to_the_sequence_through_speed_and_reverse() {
    let (s, clip, seq) = session();
    let mut it = s.project.sequence(seq).unwrap().find_item(clip).unwrap().1.clone();
    it.start = Tick::from_seconds_f64(2.0);
    it.source_in = Tick::from_seconds_f64(1.0);
    let at = |it: &filmcraft_project::TrackItem, m: f64| timeline_at(it, Tick::from_seconds_f64(m)).seconds();
    assert!((at(&it, 1.5) - 2.5).abs() < 1e-6);
    it.speed = 2.0;
    assert!((at(&it, 2.0) - 2.5).abs() < 1e-6, "twice as fast: a media second is half a sequence second");
    it.reverse = true;
    let end = it.source_out().seconds();
    assert!((at(&it, end) - 2.0).abs() < 0.01, "reversed: the media's end plays first");
    // outside the clip: its first or last frame; hostile speeds: its start
    it.reverse = false;
    assert_eq!(timeline_at(&it, Tick::ZERO), it.start);
    assert!(timeline_at(&it, Tick::from_seconds_f64(1e6)) < it.end());
    it.speed = f64::NAN;
    assert_eq!(timeline_at(&it, Tick::from_seconds_f64(3.0)), it.start);
    it.speed = 0.0;
    assert_eq!(timeline_at(&it, Tick::from_seconds_f64(3.0)), it.start);
}

/// The session with a redaction cache of its own and the warm-disc detector counting its calls
/// (cached under an id, as a host's detector is).
fn cached_session(calls: Arc<std::sync::atomic::AtomicUsize>) -> (Session, ClipId, ItemId) {
    let (mut s, clip, seq) = session();
    s.redacts.cache = Some(Arc::new(crate::redact_cache::RedactCache::in_data_dir(&tmp_dir("redact-cache"))));
    s.redacts.detector = Some(Arc::new(move |rgba: &[u8], w: usize, h: usize, _: usize| {
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        warm_disc(rgba, w, h)
    }));
    s.redacts.detector_id = Some("warm disc v1".into());
    (s, clip, seq)
}

fn tracks_of(r: &Value) -> Value {
    r["redact"]["clips"].as_array().unwrap().iter().map(|c| c["tracks"].clone()).collect()
}

/// A second run over the same footage decodes and detects nothing: every frame's detections and
/// the vision check's answers come from the cache, and what is found is the same; `fresh` reads
/// none of it.
#[test]
fn a_second_run_takes_detections_and_verdicts_from_the_cache() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (mut s, _, _) = cached_session(calls.clone());
    let judge = Arc::new(Judge(Ok(Some(true)), Default::default()));
    s.redacts.judge = Some(judge.clone());
    let first = s.execute("clip.autoRedact", json!({"wait": true, "dryRun": true})).unwrap();
    let c1 = &first["redact"]["cache"];
    let frames = first["frames"].as_u64().unwrap();
    assert_eq!((c1["enabled"].as_bool(), c1["framesFromCache"].as_u64(), c1["framesDecoded"].as_u64()), (Some(true), Some(0), Some(frames)), "{c1}");
    let detected = calls.load(std::sync::atomic::Ordering::Relaxed);
    let asked = judge.1.load(std::sync::atomic::Ordering::Relaxed);
    assert!(detected as u64 == frames && asked >= 1, "{detected} detections, {asked} crops");
    let second = s.execute("clip.autoRedact", json!({"wait": true, "dryRun": true})).unwrap();
    let c2 = &second["redact"]["cache"];
    assert_eq!((c2["framesFromCache"].as_u64(), c2["framesDecoded"].as_u64()), (Some(frames), Some(0)), "{c2}");
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), detected, "nothing detected again");
    assert_eq!(judge.1.load(std::sync::atomic::Ordering::Relaxed), asked, "nothing asked again");
    assert_eq!(second["redact"]["verify"]["fromCache"].as_u64(), Some(asked as u64));
    assert_eq!(tracks_of(&first), tracks_of(&second), "the same tracks");
    // the real edit from the cache hides the face as before
    let r = s.execute("clip.autoRedact", json!({"wait": true})).unwrap();
    assert_eq!(r["redact"]["clips"][0]["faces"], json!(1), "{r}");
    // fresh: analysed again
    let fresh = s.execute("clip.autoRedact", json!({"wait": true, "dryRun": true, "fresh": true})).unwrap();
    assert_eq!(fresh["redact"]["cache"]["framesDecoded"].as_u64(), Some(frames));
    assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > detected);
}

/// Another window of the same clip (the next language's cut of a shot) shares the frames it
/// overlaps: they come from the cache, only the new ones are decoded.
#[test]
fn another_window_of_a_clip_shares_the_frames_it_overlaps() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (mut s, clip, seq) = cached_session(calls.clone());
    // the first window: the clip's first 30 frames
    let rate = s.project.sequence(seq).unwrap().settings.frame_rate;
    let mut p = (*s.project).clone();
    p.sequence_mut(seq).unwrap().video_tracks[0].items[0].duration = rate.tick_of(30);
    s.project = Arc::new(p);
    let first = s.execute("clip.autoRedact", json!({"wait": true, "dryRun": true, "verify": false})).unwrap();
    // the second: frames 6 to 40, placed elsewhere in the sequence
    let mut p = (*s.project).clone();
    let it = &mut p.sequence_mut(seq).unwrap().video_tracks[0].items[0];
    (it.start, it.source_in, it.duration) = (rate.tick_of(12), rate.tick_of(6), rate.tick_of(34));
    s.project = Arc::new(p);
    let second = s.execute("clip.autoRedact", json!({"wait": true, "dryRun": true, "verify": false})).unwrap();
    let c = &second["redact"]["cache"];
    let (hit, decoded) = (c["framesFromCache"].as_u64().unwrap(), c["framesDecoded"].as_u64().unwrap());
    assert!(hit >= 6 && decoded <= 6, "{c} (first run: {})", first["redact"]["cache"]);
    assert_eq!(second["redact"]["clips"][0]["faces"], json!(1));
    let _ = clip;
}

/// The named targets' looks come from the cache too; a detector without an id is not cached.
#[test]
fn looks_are_cached_and_an_unnamed_detector_is_not() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (mut s, _, _) = cached_session(calls.clone());
    s.redacts.judge = Some(Arc::new(Judge(Ok(Some(true)), Default::default())));
    let p = json!({"wait": true, "dryRun": true, "kinds": [], "targets": ["the logo"], "targetEvery": 0.5});
    let first = s.execute("clip.autoRedact", p.clone()).unwrap();
    let looks = first["redact"]["clips"][0]["looks"]["asked"].as_u64().unwrap();
    assert!(looks >= 2);
    let second = s.execute("clip.autoRedact", p).unwrap();
    assert_eq!(second["redact"]["cache"]["looksFromCache"].as_u64(), Some(looks), "{}", second["redact"]["cache"]);
    assert_eq!(tracks_of(&first), tracks_of(&second));
    // a host's detector without an id: nothing kept
    s.redacts.detector_id = None;
    s.redacts.cache = Some(Arc::new(crate::redact_cache::RedactCache::in_data_dir(&tmp_dir("redact-cache-none"))));
    for _ in 0..2 {
        let r = s.execute("clip.autoRedact", json!({"wait": true, "dryRun": true, "verify": false})).unwrap();
        assert_eq!(r["redact"]["cache"]["framesFromCache"].as_u64(), Some(0));
    }
}

#[test]
fn analysis_times_are_a_grid_of_media_time() {
    let (s, clip, seq) = session();
    let q = s.project.sequence(seq).unwrap();
    let rate = q.settings.frame_rate;
    let fd = rate.frame_duration();
    let mut it = q.find_item(clip).unwrap().1.clone();
    it.duration = rate.tick_of(30);
    let g = fd.0 * 3;
    let a = grid_times(&it, fd, 3, 30);
    // the first and last frames, and every third frame of media time between them
    assert_eq!(a.first(), Some(&it.source_in));
    assert!(a.iter().skip(1).take(a.len().saturating_sub(2)).all(|t| t.0 % g == 0), "{a:?}");
    // a window starting 4 frames later shares the grid's frames
    it.source_in = rate.tick_of(4);
    let b = grid_times(&it, fd, 3, 30);
    let shared = a.iter().filter(|t| b.contains(t)).count();
    assert!(shared >= 7, "{shared} of {} shared", a.len());
    // twice as fast: a grid twice as wide in media time; hostile speeds stay bounded
    it.speed = 2.0;
    assert!(grid_times(&it, fd, 3, 30).iter().skip(1).all(|t| t.0 % (2 * g) == 0 || *t == it.source_time_at(it.start + Tick(fd.0 * 29))));
    it.speed = f64::NAN;
    assert!(!grid_times(&it, fd, 3, 30).is_empty());
    it.speed = 1e-12;
    assert!(grid_times(&it, fd, 3, 30).len() <= MAX_FRAMES + 3);
}
