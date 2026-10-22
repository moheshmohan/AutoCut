use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use filmcraft_project::{ClipId, ItemId, SequenceSettings, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use filmcraft_vision::{Detection, Kind};
use serde_json::{Value, json};

use super::visible_centre;
use crate::Session;
use crate::media_test_util::tmp_dir;
use crate::redact::{Crop, VisionJudge};

const W: u32 = 320;
const H: u32 = 180;
const FRAMES: u32 = 30;

/// A face-sized warm disc, finely textured, still at (`x`, 90), on a blue-grey picture lit red on
/// the left and green on the right (a show's lighting).
fn picture(x: f64) -> image::RgbaImage {
    image::RgbaImage::from_fn(W, H, |px, py| {
        let (fx, fy) = (f64::from(px) + 0.5, f64::from(py) + 0.5);
        if ((fx - x).powi(2) + (fy - 90.0).powi(2)).sqrt() < 24.0 {
            let v = if (px / 2 + py / 2) % 2 == 0 { 255 } else { 120 };
            image::Rgba([v, v / 2, 0, 255])
        } else if px < W / 2 {
            image::Rgba([90, 30, 60, 255])
        } else {
            image::Rgba([30, 90, 60, 255])
        }
    })
}

/// A session with one clip of the still face on V1 of a 320 × 180 sequence, and a detector that
/// finds the warm disc (score `score`).
fn session(name: &str, score: f32) -> (Session, ClipId, ItemId, std::path::PathBuf) {
    session_at(name, score, 100.0)
}

/// [`session`] with the face at `x`.
fn session_at(name: &str, score: f32, x: f64) -> (Session, ClipId, ItemId, std::path::PathBuf) {
    let dir = tmp_dir(name);
    for k in 0..FRAMES {
        picture(x).save(dir.join(format!("shot{k:04}.png"))).unwrap();
    }
    let mut s = Session::default();
    let r = s.execute("file.importImageSequence", json!({"path": dir.join("shot0000.png").to_string_lossy()})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}")));
    let rate = s.project.item(item).unwrap().as_media().unwrap().info.video.clone().unwrap().frame_rate;
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Show", SequenceSettings { width: W, height: H, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let d = p.item(item).unwrap().duration();
    let mut v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, d), rate).unwrap();
    for e in &mut v.effects {
        filmcraft_project::resolve_auto_points(e, (W, H), (W, H));
    }
    let clip = v.id;
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    s.project = Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    s.state.selection = vec![clip];
    s.redacts.detector = Some(Arc::new(move |rgba: &[u8], w: usize, h: usize, _: usize| warm_disc(rgba, w, h, score)));
    (s, clip, seq, dir)
}

/// The warm pixels' bounding box as a face (a blurred disc has none left: its warm pixels are
/// spread out and dimmed).
fn warm_disc(rgba: &[u8], w: usize, h: usize, score: f32) -> Vec<Detection> {
    let (mut x0, mut y0, mut x1, mut y1, mut n) = (usize::MAX, usize::MAX, 0, 0, 0usize);
    for (i, p) in rgba.as_chunks::<4>().0.iter().enumerate() {
        if p[0] > 200 && p[2] < 30 && p[1] > 100 {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            n += 1;
        }
    }
    if x1 < x0 || n < 20 {
        return Vec::new();
    }
    let d = Detection { kind: Kind::Face, score, bbox: [x0 as f32, y0 as f32, (x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32], points: Vec::new(), pose: false };
    vec![d.normalized(w as f32, h as f32)]
}

/// A judge with a first answer (the grid) and a second (alone, outlined).
struct Judge {
    first: Option<bool>,
    second: Option<bool>,
    asked: AtomicUsize,
    again: AtomicUsize,
}

impl Judge {
    fn new(first: Option<bool>, second: Option<bool>) -> Arc<Self> {
        Arc::new(Self { first, second, asked: AtomicUsize::new(0), again: AtomicUsize::new(0) })
    }
}

impl VisionJudge for Judge {
    fn name(&self) -> String {
        "test judge".into()
    }
    fn judge(&self, _: Kind, crops: &[Crop]) -> Result<Vec<Option<bool>>, String> {
        self.asked.fetch_add(crops.len(), Ordering::Relaxed);
        Ok(vec![self.first; crops.len()])
    }
    fn confirm(&self, _: Kind, crop: &Crop) -> Result<Option<bool>, String> {
        // the second look is outlined in red
        assert!(crop.rgba.as_chunks::<4>().0.contains(&[255, 0, 0, 255]), "the face is outlined");
        self.again.fetch_add(1, Ordering::Relaxed);
        Ok(self.second)
    }
}

fn times() -> Value {
    json!([0.0, 0.2, 0.4, 0.6, 0.8])
}

fn unmasked(r: &Value) -> &Vec<Value> {
    r["unmasked"].as_array().unwrap_or_else(|| panic!("{r}"))
}

#[test]
fn an_unmasked_face_is_found_and_a_masked_one_is_not() {
    let (mut s, clip, _, dir) = session("privacy-check-mask", 0.95);
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    let u = unmasked(&r);
    assert_eq!(u.len(), 1, "{r}");
    assert_eq!(r["shown"], json!(1));
    let f = &u[0];
    assert_eq!((f["clip"].as_u64(), f["onSubject"].as_bool(), f["allowed"].as_bool()), (Some(clip.0), Some(false), Some(false)), "{f}");
    assert_eq!(f["verdict"], json!("unchecked"));
    // the face's box, in shares of the frame: the disc at (100, 90), 48 px across
    let b = &f["box"];
    assert!((b["x"].as_f64().unwrap() * f64::from(W) - 76.0).abs() < 3.0 && (b["h"].as_f64().unwrap() * f64::from(H) - 48.0).abs() < 3.0, "{b}");
    assert!(f["hits"].as_u64().unwrap() >= 4, "followed across the frames: {f}");
    // hidden by the redaction: nothing left to show
    let red = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    assert_eq!(red["redact"]["hidden"], json!(1), "{red}");
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert!(unmasked(&r).is_empty(), "{r}");
    assert_eq!(r["shown"], json!(0));
    // the face is not found in the blurred frames; where it still would be, the mask covers it
    assert!(r["frames"].as_u64() == Some(5), "{r}");
    // faces smaller than asked for are not counted
    let mut s2 = session("privacy-check-small", 0.95).0;
    let r = s2.execute("sequence.checkPrivacy", json!({"seconds": times(), "minFaceHeight": 0.5})).unwrap();
    assert!(unmasked(&r).is_empty() && r["tooSmall"].as_u64().unwrap() >= 5, "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_mask_over_the_face_counts_as_hidden_even_where_the_detector_still_sees_it() {
    // the detector finds the disc on the picture whatever the blur (a strong face, a weak blur)
    let (mut s, _, _, dir) = session("privacy-check-weak", 0.95);
    s.execute("clip.autoRedact", json!({"wait": true, "verify": false, "strength": 1})).unwrap();
    s.redacts.detector = Some(Arc::new(|_: &[u8], _: usize, _: usize, _: usize| {
        vec![Detection {
            kind: Kind::Face,
            score: 0.9,
            bbox: [76.0 / W as f32, 66.0 / H as f32, 48.0 / W as f32, 48.0 / H as f32],
            points: Vec::new(),
            pose: false,
        }]
    }));
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert!(unmasked(&r).is_empty(), "{r}");
    assert_eq!(r["masked"], json!(5), "{r}");
    // elsewhere in the frame, not under a mask: shown
    s.redacts.detector = Some(Arc::new(|_: &[u8], _: usize, _: usize, _: usize| {
        vec![Detection { kind: Kind::Face, score: 0.9, bbox: [0.7, 0.3, 0.12, 0.25], points: Vec::new(), pose: false }]
    }));
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert_eq!(unmasked(&r).len(), 1, "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_face_on_the_subject_stays_sharp_only_when_the_model_twice_says_it_is_not_a_person() {
    let (mut s, clip, _, dir) = session("privacy-check-robot", 0.95);
    let robot = json!({"clip": clip.0, "x": 0.1, "y": 0.1, "w": 0.5, "h": 0.8});
    let p = json!({"seconds": times(), "exclude": robot});
    // a robot's face: the grid and the second look agree
    let j = Judge::new(Some(false), Some(false));
    s.redacts.judge = Some(j.clone());
    let r = s.execute("sequence.checkPrivacy", p.clone()).unwrap();
    let f = &unmasked(&r)[0];
    assert_eq!((f["onSubject"].as_bool(), f["person"].as_bool(), f["allowed"].as_bool()), (Some(true), Some(false), Some(true)), "{r}");
    assert_eq!(r["shown"], json!(0));
    assert_eq!((j.asked.load(Ordering::Relaxed), j.again.load(Ordering::Relaxed)), (1, 1));
    // a visitor beside the robot that the grid called a robot: the second look sees a person
    s.redacts.judge = Some(Judge::new(Some(false), Some(true)));
    let r = s.execute("sequence.checkPrivacy", p.clone()).unwrap();
    assert_eq!((unmasked(&r)[0]["allowed"].as_bool(), r["shown"].as_u64()), (Some(false), Some(1)), "{r}");
    // a person in the robot's box
    s.redacts.judge = Some(Judge::new(Some(true), Some(true)));
    let r = s.execute("sequence.checkPrivacy", p.clone()).unwrap();
    assert_eq!(unmasked(&r)[0]["verdict"], json!("person"));
    assert_eq!(r["shown"], json!(1));
    // the robot's own face the grid called a person, alone a robot: unclear, not allowed
    s.redacts.judge = Some(Judge::new(Some(true), Some(false)));
    let r = s.execute("sequence.checkPrivacy", p).unwrap();
    assert_eq!((unmasked(&r)[0]["verdict"].as_str(), unmasked(&r)[0]["allowed"].as_bool()), (Some("unsure"), Some(false)), "{r}");
    // away from the subject, a face the detector is sure of shows whatever the model says
    let j = Judge::new(Some(false), Some(false));
    s.redacts.judge = Some(j.clone());
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert_eq!((unmasked(&r)[0]["allowed"].as_bool(), r["shown"].as_u64()), (Some(false), Some(1)), "{r}");
    assert_eq!(j.again.load(Ordering::Relaxed), 0, "no second look where it would not count");
    // …one it is not sure of may stay sharp
    s.redacts.detector = Some(Arc::new(|rgba: &[u8], w: usize, h: usize, _: usize| warm_disc(rgba, w, h, 0.7)));
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert_eq!(r["shown"], json!(0), "{r}");
    // without asking the model, nothing unmasked may stay sharp
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times(), "verify": false, "exclude": robot})).unwrap();
    assert_eq!((r["shown"].as_u64(), r["verify"]["state"].as_str()), (Some(1), Some("off")), "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A head the profile pass found (in profile, half hidden: the face detector missed it) is a face
/// to the check: unmasked it is shown (its own thresholds: `faceScore` does not drop it), masked by
/// the redaction it passes, and on the robot the vision check's two "not a person" leave it sharp.
#[test]
fn an_unmasked_head_from_a_pose_is_shown_and_passes_once_masked() {
    let (mut s, clip, _, dir) = session("privacy-check-pose", 0.95);
    let pose = |rgba: &[u8], w: usize, h: usize, _: usize| warm_disc(rgba, w, h, 0.45).into_iter().map(|d| Detection { pose: true, ..d }).collect();
    s.redacts.detector = Some(Arc::new(pose));
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    let u = unmasked(&r);
    assert_eq!((u.len(), r["shown"].as_u64()), (1, Some(1)), "{r}");
    assert_eq!((u[0]["fromPose"].as_bool(), u[0]["allowed"].as_bool()), (Some(true), Some(false)), "{r}");
    // the face detector's face that weak would not count
    s.redacts.detector = Some(Arc::new(|rgba: &[u8], w: usize, h: usize, _: usize| warm_disc(rgba, w, h, 0.45)));
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert!(unmasked(&r).is_empty(), "{r}");
    // on the robot, called not a person twice: sharp, as a face would be
    s.redacts.detector = Some(Arc::new(pose));
    s.redacts.judge = Some(Judge::new(Some(false), Some(false)));
    let robot = json!({"clip": clip.0, "x": 0.1, "y": 0.1, "w": 0.5, "h": 0.8});
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times(), "exclude": robot})).unwrap();
    assert_eq!((unmasked(&r)[0]["allowed"].as_bool(), r["shown"].as_u64()), (Some(true), Some(0)), "{r}");
    // a person: hidden by the redaction, then nothing is left to show
    s.redacts.judge = None;
    let red = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    assert_eq!(red["redact"]["clips"][0]["facesFromPose"], json!(1), "{red}");
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert!(unmasked(&r).is_empty(), "{r}");
    assert_eq!(r["shown"], json!(0));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A head found from a pose (0.4), cut by the frame's left edge (a third of it outside): the
/// check shows it (a head at the edge is not let off) until the redaction hides it.
#[test]
fn a_head_cut_by_the_frame_is_shown_until_it_is_masked() {
    let (mut s, _, _, dir) = session_at("privacy-check-edge", 0.95, 14.0);
    let pose = |rgba: &[u8], w: usize, h: usize, _: usize| warm_disc(rgba, w, h, 0.4).into_iter().map(|d| Detection { pose: true, ..d }).collect();
    s.redacts.detector = Some(Arc::new(pose));
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    let u = unmasked(&r);
    assert_eq!((u.len(), r["shown"].as_u64()), (1, Some(1)), "{r}");
    let b = &u[0]["box"];
    assert!(b["x"].as_f64().unwrap() < 0.01, "at the edge: {b}");
    assert_eq!((u[0]["fromPose"].as_bool(), u[0]["allowed"].as_bool()), (Some(true), Some(false)), "{r}");
    let red = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    assert_eq!(red["redact"]["clips"][0]["faces"], json!(1), "{red}");
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert!(unmasked(&r).is_empty(), "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A head whose box runs past the frame's left edge, its centre outside the picture (a pose
/// head's box does): flagged while its visible part is sharp, passed once the redaction's mask
/// covers that part (the mask is carried past the edge). Its centre was under no clip, so it
/// used to be flagged whatever the masks.
#[test]
fn a_head_with_its_centre_outside_the_frame_is_checked_where_it_shows() {
    let (mut s, _, _, dir) = session("privacy-check-outside", 0.95);
    // the detector sees the same head in every frame, blurred or not
    let head = |_: &[u8], _: usize, _: usize, _: usize| {
        vec![Detection { kind: Kind::Face, score: 0.95, bbox: [-0.05, 0.3, 0.08, 0.3], points: Vec::new(), pose: true }]
    };
    s.redacts.detector = Some(Arc::new(head));
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert_eq!((unmasked(&r).len(), r["shown"].as_u64()), (1, Some(1)), "{r}");
    assert!(unmasked(&r)[0]["clip"].is_u64(), "the clip under its visible part: {r}");
    let red = s.execute("clip.autoRedact", json!({"wait": true, "verify": false})).unwrap();
    assert_eq!(red["redact"]["clips"][0]["faces"], json!(1), "{red}");
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": times()})).unwrap();
    assert!(unmasked(&r).is_empty(), "{r}");
    assert_eq!(r["masked"].as_u64(), Some(5), "{r}");
    // the visible point: the centre when it shows, else the nearest point of the box that does
    let v = visible_centre(&[-0.05, 0.3, 0.08, 0.3], 320.0, 180.0);
    assert!((v.x - 0.5).abs() < 1e-9 && (v.y - 81.0).abs() < 1e-4, "{v:?}");
    let v = visible_centre(&[0.4, 0.3, 0.1, 0.2], 320.0, 180.0);
    assert!((v.x - 144.0).abs() < 1e-4 && (v.y - 72.0).abs() < 1e-4, "{v:?}");
    let v = visible_centre(&[0.95, 0.9, 0.1, 0.2], 320.0, 180.0);
    assert!((v.x - 319.5).abs() < 1e-9 && (v.y - 179.5).abs() < 1e-9, "{v:?}");
    // wholly outside, or not a number: the centre as it is (under no clip: flagged)
    let v = visible_centre(&[-0.5, 0.3, 0.1, 0.2], 320.0, 180.0);
    assert!((v.x - (-0.45 * 320.0)).abs() < 1e-3, "{v:?}");
    let v = visible_centre(&[f32::NAN, 0.3, 0.1, 0.2], 320.0, 180.0);
    assert!(v.x.is_nan() || v.x.is_finite());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, _, _, dir) = session("privacy-check-hostile", 0.95);
    let many: Vec<f64> = (0..=super::MAX_TIMES).map(|i| i as f64).collect();
    for p in [
        json!({}),
        json!({"seconds": []}),
        json!({"seconds": "1"}),
        json!({"seconds": [1.0, "x"]}),
        json!({"seconds": [-1.0]}),
        json!({"seconds": [1e300]}),
        json!({"seconds": many}),
        json!({"seconds": [0.0], "maxSide": 5}),
        json!({"seconds": [0.0], "faceScore": 2}),
        json!({"seconds": [0.0], "minFaceHeight": -1}),
        json!({"seconds": [0.0], "verify": "yes"}),
        json!({"seconds": [0.0], "exclude": 3}),
        json!({"seconds": [0.0], "exclude": [{"x": 0.0}]}),
        json!([1, 2]),
    ] {
        assert!(s.execute("sequence.checkPrivacy", p.clone()).is_err(), "{p}");
    }
    // times past the end are not looked at; a huge render is capped by maxSide
    let r = s.execute("sequence.checkPrivacy", json!({"seconds": [0.0, 1e6], "maxSide": 3840})).unwrap();
    assert_eq!(r["frames"], json!(1), "{r}");
    // no sequence: refused
    assert!(Session::default().execute("sequence.checkPrivacy", json!({"seconds": [0.0]})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn two_looks_make_one_answer() {
    use super::both_looks;
    // not a person only when both say so
    assert_eq!(both_looks(Some(false), Some(false), true), Some(false));
    assert_eq!(both_looks(Some(false), Some(true), true), Some(true));
    assert_eq!(both_looks(Some(false), None, false), None);
    // a person stays one away from the robot; on it, the second look must agree
    assert_eq!(both_looks(Some(true), Some(false), false), Some(true));
    assert_eq!(both_looks(Some(true), Some(false), true), None);
    assert_eq!(both_looks(Some(true), Some(true), true), Some(true));
    assert_eq!(both_looks(None, Some(false), true), None);
}
