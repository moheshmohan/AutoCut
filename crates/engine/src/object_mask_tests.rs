//! Object Mask with a stand-in cutter (what has the clicked colour, inside the box it is given)
//! on a movie of a red disc crossing a forest: the mask is the disc in every frame looked at,
//! before and after the click.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use filmcraft_frame::VideoFrame;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, FrameRequest, Generator, MediaInfo, MediaSource};
use filmcraft_project::{ClipId, Mask, MaskMode, TrackMethod};
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use super::*;
use crate::media_test_util::{make_movie_from, session_with, tmp_dir};

const W: usize = 320;
const H: usize = 180;
const FRAMES: i64 = 48;
const R: f32 = 18.0;

/// Where the disc is at media time `t`: left to right across the picture, a little down.
fn disc_at(t: Tick) -> (f32, f32) {
    let k = (t.seconds() / FrameRate::FPS_24.tick_of(FRAMES).seconds()) as f32;
    (W as f32 * (0.2 + 0.6 * k), H as f32 * (0.45 + 0.1 * k))
}

/// The forest with the disc on it.
struct Disc {
    bg: GeneratorSource,
}

impl MediaSource for Disc {
    fn info(&self) -> &MediaInfo {
        self.bg.info()
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let f = self.bg.video_frame(FrameRequest { time: req.time, scale: 1.0 })?;
        let (w, h) = (f.width as usize, f.height as usize);
        let mut px = f.to_rgba8();
        let (cx, cy) = disc_at(req.time);
        for y in 0..h {
            for x in 0..w {
                if (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) < R {
                    px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[250, 30, 40, 255]);
                }
            }
        }
        Ok(Arc::new(VideoFrame::rgba8(w as u32, h as u32, px).with_pts(f.pts)))
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.bg.audio(start, frames, sample_rate)
    }
}

/// The stand-in: what has the colour at the first point inside the box (the whole picture
/// without one); with `wrong`, every third picture's answers are its top-left corner instead.
#[derive(Default)]
struct Colour {
    pictures: AtomicUsize,
    wrong: bool,
}

struct ColourPicture {
    rgba: Vec<u8>,
    w: usize,
    h: usize,
    wrong: bool,
}

impl ObjectCutter for Colour {
    fn picture(&self, rgba: &[u8], w: usize, h: usize) -> std::result::Result<Box<dyn Cuts + '_>, String> {
        let n = self.pictures.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(ColourPicture { rgba: rgba.to_vec(), w, h, wrong: self.wrong && n % 3 == 2 }))
    }
}

impl Cuts for ColourPicture {
    fn cut(&self, prompt: &Prompt) -> std::result::Result<Vec<Matte>, String> {
        let (w, h, rgba) = (self.w, self.h, &self.rgba);
        let b = prompt.bbox.unwrap_or([0.0, 0.0, w as f32, h as f32]);
        let mut alpha = vec![0u8; w * h];
        if self.wrong {
            for y in 0..h / 4 {
                for x in 0..w / 4 {
                    alpha[y * w + x] = 255;
                }
            }
            return Ok(vec![Matte { w, h, alpha, score: 0.9 }]);
        }
        let Some(&(sx, sy)) = prompt.inside.first() else { return Ok(Vec::new()) };
        let at = |x: usize, y: usize| &rgba[(y * w + x) * 4..(y * w + x) * 4 + 3];
        let seed = at((sx as usize).min(w - 1), (sy as usize).min(h - 1)).to_vec();
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                if fx < b[0] || fy < b[1] || fx > b[2] || fy > b[3] {
                    continue;
                }
                let d: i32 = at(x, y).iter().zip(&seed).map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs()).sum();
                if d < 90 {
                    alpha[y * w + x] = 255;
                }
            }
        }
        Ok(vec![Matte { w, h, alpha, score: 0.9 }])
    }
}

fn session(name: &str, cutter: Colour) -> (Session, ClipId, std::path::PathBuf) {
    let dir = tmp_dir(name);
    let path = dir.join("disc.mp4");
    let rate = FrameRate::FPS_24;
    let bg = GeneratorSource::new(Generator::Demo(DemoScene::Forest), W as u32, H as u32, rate, rate.tick_of(FRAMES));
    make_movie_from(&path, Arc::new(Disc { bg }));
    let (mut s, _, _) = session_with(&[&path]);
    s.cutouts.object = Some(Arc::new(cutter));
    let clip = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.state.selection = vec![clip];
    (s, clip, dir)
}

fn mask(s: &Session, c: ClipId, name: &str) -> Option<Mask> {
    let (_, it) = s.active_sequence()?.find_item(c)?;
    it.effect("opacity")?.masks.iter().find(|m| m.name == name).cloned()
}

/// How far each outline's middle is from the disc's at its time, at most, and the outlines'
/// times (seconds).
fn following(m: &Mask) -> (f32, Vec<f64>) {
    let mut worst = 0f32;
    let mut times = Vec::new();
    for k in &m.path.keyframes {
        let path = m.path_at(k.time);
        let n = path.vertices.len().max(1) as f64;
        let (x, y) = path.vertices.iter().fold((0.0, 0.0), |a, v| (a.0 + v.p.x / n, a.1 + v.p.y / n));
        let (cx, cy) = disc_at(k.time);
        worst = worst.max((x as f32 - cx).hypot(y as f32 - cy));
        if std::env::var_os("OBJ_DEBUG").is_some() {
            eprintln!("{:.3}s outline middle ({x:.1}, {y:.1}) disc ({cx:.1}, {cy:.1}) {} points", k.time.seconds(), path.vertices.len());
        }
        times.push(k.time.seconds());
    }
    (worst, times)
}

fn click(s: &mut Session, at: f64, extra: Value) -> Value {
    let (x, y) = disc_at(Tick::from_seconds_f64(at));
    let mut p = json!({"point": [x / W as f32, y / H as f32], "at": at, "every": 2.0 / 24.0, "wait": true});
    if let (Some(p), Some(e)) = (p.as_object_mut(), extra.as_object()) {
        p.extend(e.clone());
    }
    s.execute(CMD, p).unwrap()
}

#[test]
fn a_click_masks_the_thing_and_follows_it_both_ways() {
    let (mut s, c, dir) = session("object-click", Colour::default());
    let before = s.history.undo.len();
    let r = click(&mut s, 1.0, json!({}));
    assert!(r["status"]["finished"] == json!(true) && r["status"]["result"]["error"].is_null(), "{r}");
    let m = mask(&s, c, "Object (1)").expect("the mask");
    assert_eq!(m.mode, MaskMode::Add, "the clip shows the thing alone");
    assert_eq!(m.track_method, TrackMethod::Object);
    let (worst, times) = following(&m);
    assert!(times.len() >= 20, "an outline every other frame: {times:?}");
    assert!(times.first().is_some_and(|t| *t < 0.1) && times.last().is_some_and(|t| *t > 1.85), "back to the start and on to the end: {times:?}");
    assert!(worst < 2.5, "each outline on the disc: {worst} px off at worst");
    // the outline is the disc's: its area
    let path = m.path_at(Tick::from_seconds_f64(1.0));
    let n = path.vertices.len();
    let area =
        0.5 * (0..n).map(|i| path.vertices[i].p.x * path.vertices[(i + 1) % n].p.y - path.vertices[(i + 1) % n].p.x * path.vertices[i].p.y).sum::<f64>().abs();
    let disc = std::f64::consts::PI * f64::from(R * R);
    assert!((area / disc - 1.0).abs() < 0.15, "the disc's area: {area:.0} of {disc:.0}");
    assert_eq!(s.state.selected_mask.map(|m| (m.clip, m.mask)), Some((c, 0)), "the new mask is selected");
    // one undo step
    assert_eq!(s.history.undo.len(), before + 1);
    s.execute("edit.undo", json!({})).unwrap();
    assert!(mask(&s, c, "Object (1)").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_second_click_adds_a_second_mask_and_a_refining_click_replaces_one_frame() {
    let (mut s, c, dir) = session("object-refine", Colour::default());
    click(&mut s, 0.5, json!({"track": "forward"}));
    let first = mask(&s, c, "Object (1)").unwrap();
    let (_, times) = following(&first);
    assert!(times.first().is_some_and(|t| (t - 0.5).abs() < 0.03), "forward only, from the click: {times:?}");
    click(&mut s, 1.0, json!({"track": "none", "name": "Ball", "mode": "subtract", "inverted": true}));
    let ball = mask(&s, c, "Ball").unwrap();
    assert_eq!((ball.path.keyframes.len(), ball.mode, ball.inverted), (1, MaskMode::Subtract, true));
    // the first mask cut again at 1 s: that frame only
    let n = first.path.keyframes.len();
    click(&mut s, 1.0, json!({"track": "none", "mask": 0}));
    let again = mask(&s, c, "Object (1)").unwrap();
    assert_eq!(again.path.keyframes.len(), n, "the frame's outline replaced, the others kept");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn frames_the_cutter_gets_wrong_keep_no_outline() {
    let (mut s, c, dir) = session("object-wrong", Colour { wrong: true, ..Default::default() });
    click(&mut s, 1.0, json!({"track": "forward"}));
    let m = mask(&s, c, "Object (1)").unwrap();
    let (worst, times) = following(&m);
    // 12 looks from 1 s: the click's answer right, then a third of them wrong
    assert!((7..=9).contains(&times.len()), "the wrong ones dropped: {times:?}");
    assert!(worst < 2.5, "and the rest on the disc: {worst}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_mask_drawn_round_the_thing_is_tracked_by_object() {
    let (mut s, c, dir) = session("object-track", Colour::default());
    let (x, y) = disc_at(Tick::ZERO);
    s.execute("masks.add", json!({"clip": c.0, "shape": "ellipse", "center": [x, y], "size": [2.6 * R, 2.4 * R]})).unwrap();
    s.set_playhead(Tick::ZERO);
    let r = s.execute("masks.track", json!({"clip": c.0, "mask": 0, "method": "object", "direction": "forward", "wait": true})).unwrap();
    assert!(r["status"]["finished"] == json!(true) && r["status"]["result"]["error"].is_null(), "{r}");
    let (_, it) = s.active_sequence().unwrap().find_item(c).unwrap();
    let m = it.effect("opacity").unwrap().masks[0].clone();
    assert_eq!(m.track_method, TrackMethod::Object);
    let (worst, times) = following(&m);
    assert!(times.len() >= 18 && times.last().is_some_and(|t| *t > 1.8), "{times:?}");
    assert!(worst < 2.5, "{worst}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, c, dir) = session("object-hostile", Colour::default());
    let many: Vec<[f64; 2]> = (0..17).map(|_| [0.5, 0.5]).collect();
    for p in [
        json!({}),
        json!({"point": [1.5, 0.5]}),
        json!({"point": "middle"}),
        json!({"point": [0.5]}),
        json!({"point": [f64::MAX, 0.5]}),
        json!({"inside": many}),
        json!({"point": [0.5, 0.5], "track": "sideways"}),
        json!({"point": [0.5, 0.5], "mode": "blurry"}),
        json!({"point": [0.5, 0.5], "at": -1.0}),
        json!({"point": [0.5, 0.5], "at": 99.0}),
        json!({"point": [0.5, 0.5], "at": 1e300}),
        json!({"point": [0.5, 0.5], "at": 2.0}),
        json!({"point": [0.5, 0.5], "mask": 99}),
        json!({"point": [0.5, 0.5], "effect": "motion"}),
        json!({"point": [0.5, 0.5], "every": 0.0}),
        json!({"point": [0.5, 0.5], "inverted": "yes"}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    assert!(mask(&s, c, "Object (1)").is_none());
    // a clip of no length (a hostile project): refused, not a panic
    let c2_id = |s: &Session| s.active_sequence().map_or(0, |q| q.video_tracks[0].items[0].id.0);
    let mut s2 = {
        let (s2, c2, _) = session("object-empty", Colour::default());
        let mut s2 = s2;
        s2.execute("masks.add", json!({"clip": c2.0, "shape": "ellipse"})).unwrap();
        // (edits refuse it: as a project file read from disk could have it)
        let seq = s2.state.active_sequence.unwrap();
        let q = Arc::make_mut(&mut s2.project).sequence_mut(seq).unwrap();
        q.find_item_mut(c2).unwrap().1.duration = Tick::ZERO;
        s2
    };
    assert!(s2.execute(CMD, json!({"point": [0.5, 0.5], "wait": true})).is_err());
    assert!(s2.execute("masks.track", json!({"clip": c2_id(&s2), "mask": 0, "method": "object", "wait": true})).is_err());
    // a click on nothing the cutter cuts out: an error, no mask outlines
    s.cutouts.object = Some(Arc::new(Nothing));
    let r = s.execute(CMD, json!({"point": [0.5, 0.5], "wait": true})).unwrap();
    assert!(r["status"]["result"]["error"].as_str().is_some_and(|e| e.contains("nothing to mask")), "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

struct Nothing;

impl ObjectCutter for Nothing {
    fn picture(&self, _: &[u8], w: usize, h: usize) -> std::result::Result<Box<dyn Cuts + '_>, String> {
        Ok(Box::new(NothingThere(w, h)))
    }
}

struct NothingThere(usize, usize);

impl Cuts for NothingThere {
    fn cut(&self, _: &Prompt) -> std::result::Result<Vec<Matte>, String> {
        Ok(vec![Matte { w: self.0, h: self.1, alpha: vec![0; self.0 * self.1], score: 0.1 }])
    }
}

#[test]
fn shapes_on_the_grid() {
    let g = Grid::new(512, 256);
    assert_eq!((g.gw, g.gh, g.k, g.w, g.h), (256, 128, 2.0, 512, 256));
    let square = [(100.0, 50.0), (200.0, 50.0), (200.0, 150.0), (100.0, 150.0)];
    let a = g.fill(&square);
    assert_eq!(a.iter().filter(|c| **c).count(), 50 * 50);
    assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    assert_eq!(iou(&a, &vec![false; a.len()]), 0.0);
    let d = g.deepest(&a);
    assert_eq!(d.len(), 1, "a square has one deepest point: {d:?}");
    assert!((d[0].0 - 150.0).abs() <= 3.0 && (d[0].1 - 100.0).abs() <= 3.0, "{d:?}");
    // a long bar: two points along it
    let bar = g.fill(&[(20.0, 100.0), (480.0, 100.0), (480.0, 130.0), (20.0, 130.0)]);
    let d = g.deepest(&bar);
    assert_eq!(d.len(), 2, "{d:?}");
    assert!((d[0].0 - d[1].0).abs() > 30.0);
    // hostile shapes
    assert!(g.fill(&[(f32::NAN, 1.0), (2.0, 2.0), (3.0, 1.0)]).iter().all(|c| !c));
    assert!(g.fill(&[(-1e9, -1e9), (1e9, -1e9), (1e9, 1e9)]).iter().any(|c| *c));
    assert!(g.deepest(&[true; 3]).is_empty());
    assert!(Grid::new(0, 0).deepest(&[false]).is_empty());
    let from = SubjectBox::new(0.5, 0.5, 0.1, 0.1).unwrap();
    let to = ObjectBox { cx: 0.6, cy: 0.5, w: 0.0, h: f64::NAN, rotation: 0.0, scale: 1.0 };
    let m = moved(&[(0.5, 0.5), (0.55, 0.5)], &from, &to);
    assert!(m.iter().all(|p| p.0.is_finite() && p.1.is_finite()), "{m:?}");
    assert!(bounds(&[(1.0, 1.0)]).is_none());
    let big = at_size(Matte { w: 2, h: 1, alpha: vec![255, 0], score: 1.0 }, 4, 2);
    assert_eq!(big.alpha, vec![255, 255, 0, 0, 255, 255, 0, 0]);
}

// ---------------------------------------------------------------------------------------------
// with MobileSAM (downloaded): measured, not run by default
// ---------------------------------------------------------------------------------------------

/// A striped ball crossing a forest behind a post (640 × 360, 3 s).
struct Ball {
    bg: GeneratorSource,
}

const BW: usize = 640;
const BH: usize = 360;
const BALL_R: f32 = 45.0;
const POST: (f32, f32) = (300.0, 330.0);

fn ball_at(t: Tick) -> (f32, f32) {
    let k = (t.seconds() / 3.0) as f32;
    (BW as f32 * (0.15 + 0.7 * k), BH as f32 * (0.5 - 0.15 * (k * std::f32::consts::PI).sin()))
}

/// Whether the ball shows at a pixel at `t`.
fn ball_shows(x: f32, y: f32, t: Tick) -> bool {
    let (cx, cy) = ball_at(t);
    (x - cx).hypot(y - cy) < BALL_R && !(POST.0..POST.1).contains(&x)
}

impl MediaSource for Ball {
    fn info(&self) -> &MediaInfo {
        self.bg.info()
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let f = self.bg.video_frame(FrameRequest { time: req.time, scale: 1.0 })?;
        let (w, h) = (f.width as usize, f.height as usize);
        let mut px = f.to_rgba8();
        let (cx, cy) = ball_at(req.time);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let c = if (POST.0..POST.1).contains(&fx) {
                    [120, 110, 100]
                } else if (fx - cx).hypot(fy - cy) < BALL_R {
                    // stripes turning with the ball
                    if (((fx - cx) * 0.8 + (fy - cy) * 0.6 + req.time.seconds() as f32 * 40.0) / 9.0).floor() as i64 % 2 == 0 {
                        [230, 200, 40]
                    } else {
                        [40, 60, 200]
                    }
                } else {
                    continue;
                };
                px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        Ok(Arc::new(VideoFrame::rgba8(w as u32, h as u32, px).with_pts(f.pts)))
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.bg.audio(start, frames, sample_rate)
    }
}

/// How much a mask's outline at each of its keyframes overlaps what `truth` says shows.
fn overlaps(m: &Mask, times: &[Tick], truth: &dyn Fn(f32, f32, Tick) -> bool, w: usize, h: usize) -> Vec<f32> {
    let g = Grid::new(w, h);
    times
        .iter()
        .map(|t| {
            let pts: Vec<(f32, f32)> = m.path_at(*t).flatten(0.5).iter().map(|v| (v.x as f32, v.y as f32)).collect();
            let a = g.fill(&pts);
            let b: Vec<bool> = (0..g.gw * g.gh).map(|i| truth(((i % g.gw) as f32 + 0.5) * g.k, ((i / g.gw) as f32 + 0.5) * g.k, *t)).collect();
            iou(&a, &b)
        })
        .collect()
}

fn summary(v: &[f32]) -> String {
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    let mean = s.iter().sum::<f32>() / s.len().max(1) as f32;
    format!("mean {mean:.3}, worst {:.3}, tenth {:.3} ({} frames)", s.first().unwrap_or(&0.0), s.get(s.len() / 10).unwrap_or(&0.0), s.len())
}

/// The click's outline as a still mask `name`, then tracked both ways by `masks.track` with
/// `method` (what masks did before Object Mask).
fn feature_tracked(s: &mut Session, c: ClipId, from: &Mask, at: Tick, method: &str) -> Mask {
    let path = from.path_at(at);
    let r = s.execute("masks.add", json!({"clip": c.0, "shape": "bezier", "path": crate::masks::path_to_json(&path)})).unwrap();
    let mi = r["mask"].as_u64().unwrap();
    for dir in ["forward", "backward"] {
        s.set_playhead(at);
        let _ = s.execute("masks.track", json!({"clip": c.0, "mask": mi, "method": method, "direction": dir, "wait": true}));
    }
    let (_, it) = s.active_sequence().unwrap().find_item(c).unwrap();
    it.effect("opacity").unwrap().masks[mi as usize].clone()
}

#[test]
#[ignore = "needs MobileSAM downloaded (FILMCRAFT_DATA_DIR naming the data directory); measures"]
fn mobilesam_follows_a_ball_behind_a_post() {
    let dir = tmp_dir("object-ball");
    let path = dir.join("ball.mp4");
    let rate = FrameRate::FPS_24;
    let bg = GeneratorSource::new(Generator::Demo(DemoScene::Forest), BW as u32, BH as u32, rate, rate.tick_of(72));
    make_movie_from(&path, Arc::new(Ball { bg }));
    let (mut s, _, _) = session_with(&[&path]);
    let c = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.state.selection = vec![c];
    let at = Tick::from_seconds_f64(0.5);
    let (x, y) = ball_at(at);
    let t0 = std::time::Instant::now();
    let r = s.execute(CMD, json!({"point": [x / BW as f32, y / BH as f32], "at": 0.5, "every": 1.0 / 24.0, "wait": true})).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    assert!(r["status"]["result"]["error"].is_null(), "{r}");
    let m = mask(&s, c, "Object (1)").unwrap();
    let times: Vec<Tick> = m.path.keyframes.iter().map(|k| k.time).collect();
    eprintln!("Object Mask: {} outlines of 72 frames in {secs:.1} s ({:.0} ms a frame)", times.len(), secs * 1000.0 / 72.0);
    let ours = overlaps(&m, &times, &ball_shows, BW, BH);
    eprintln!("  overlap with the ball as it shows: {}", summary(&ours));
    if std::env::var_os("OBJ_DEBUG").is_some() {
        for (t, v) in times.iter().zip(&ours) {
            let path = m.path_at(*t);
            let n = path.vertices.len().max(1) as f64;
            let (x, y) = path.vertices.iter().fold((0.0, 0.0), |a, v| (a.0 + v.p.x / n, a.1 + v.p.y / n));
            let b = bounds(&path.vertices.iter().map(|v| (v.p.x as f32, v.p.y as f32)).collect::<Vec<_>>()).unwrap_or_default();
            eprintln!("  {:.3}s {v:.3} middle ({x:.0}, {y:.0}) ball {:?} box {:?}", t.seconds(), ball_at(*t), b.map(|v| v.round()));
        }
    }
    let before = feature_tracked(&mut s, c, &m, at, "positionScaleRotation");
    let feat_times: Vec<Tick> = before.path.keyframes.iter().map(|k| k.time).collect();
    eprintln!("Track Mask (position, scale and rotation) from the click's outline: {} frames", feat_times.len());
    eprintln!("  overlap: {}", summary(&overlaps(&before, &feat_times, &ball_shows, BW, BH)));
    assert!(ours.iter().sum::<f32>() / ours.len().max(1) as f32 > 0.8);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A frame of the clip at media time `t` (`w` wide) with masks' outlines drawn on it.
fn drawn(s: &Session, c: ClipId, t: Tick, masks: &[(&Mask, [u8; 3])], w: usize) -> (Vec<u8>, usize, usize) {
    let (_, it) = s.active_sequence().unwrap().find_item(c).unwrap();
    let src = crate::vr360::clip_source(s, it).unwrap();
    let size = filmcraft_render::clip_size(&s.project, it).unwrap();
    let k = w as f32 / size.0 as f32;
    let f = src.video_frame(FrameRequest { time: t, scale: k }).unwrap();
    let (fw, fh) = (f.width as usize, f.height as usize);
    let (mut px, _, h) = crate::cutout::resize_rgba(&f.to_rgba8(), fw, fh, w, (fh * w / fw).max(1));
    for (m, col) in masks {
        let pts: Vec<(f32, f32)> = m.path_at(t).flatten(0.5).iter().map(|v| (v.x as f32 * k, v.y as f32 * k)).collect();
        for i in 0..pts.len() {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            let n = ((b.0 - a.0).hypot(b.1 - a.1).ceil() as usize).max(1);
            for j in 0..=n {
                let (x, y) = (a.0 + (b.0 - a.0) * j as f32 / n as f32, a.1 + (b.1 - a.1) * j as f32 / n as f32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (x, y) = (x as i64 + dx, y as i64 + dy);
                    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                        px[(y as usize * w + x as usize) * 4..(y as usize * w + x as usize) * 4 + 3].copy_from_slice(col);
                    }
                }
            }
        }
    }
    (px, w, h)
}

/// Object Mask on a real clip: `OBJECT_CLIP` (a file), `OBJECT_POINT` (`x,y`, 0..1),
/// `OBJECT_AT` (seconds; 0), `OBJECT_SECONDS` (the clip cut to that long; 6), `OBJECT_OUT` (a
/// folder for a strip of eight frames: the object mask yellow, Track Mask from the click's
/// outline cyan).
#[test]
#[ignore = "needs MobileSAM downloaded (FILMCRAFT_DATA_DIR naming the data directory) and a clip; measures"]
fn object_mask_on_a_real_clip() {
    let Some(clip) = std::env::var_os("OBJECT_CLIP").map(std::path::PathBuf::from) else { return };
    let num = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let point: Vec<f64> = std::env::var("OBJECT_POINT").unwrap_or("0.5,0.5".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    let (at, long) = (num("OBJECT_AT", 0.0), num("OBJECT_SECONDS", 6.0));
    let out = std::env::var_os("OBJECT_OUT").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let (mut s, _, _) = session_with(&[&clip]);
    let c = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.edit_sequence("cut", |q, _, _| {
        let (_, it) = q.find_item_mut(c).unwrap();
        it.duration = it.duration.min(Tick::from_seconds_f64(long));
        Ok(())
    })
    .unwrap();
    s.state.selection = vec![c];
    let name = clip.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if std::env::var_os("OBJECT_POINT").is_none() {
        // no point: the frames with a grid of tenths, to choose one
        let it = s.active_sequence().unwrap().find_item(c).unwrap().1.clone();
        let (first, last) = (it.source_time_at(it.start), it.source_time_at(it.end() - Tick(1)));
        let size = filmcraft_render::clip_size(&s.project, &it).unwrap();
        let grid = Mask::new(
            "grid",
            filmcraft_project::MaskPath {
                vertices: (0..=10)
                    .flat_map(|i| {
                        let (x, y) = (f64::from(size.0) * f64::from(i) / 10.0, f64::from(size.1) * f64::from(i) / 10.0);
                        [(x, 0.0), (x, f64::from(size.1)), (x, 0.0), (0.0, y), (f64::from(size.0), y), (0.0, y)]
                    })
                    .map(|(x, y)| filmcraft_project::MaskVertex::corner(filmcraft_geom::Vec2::new(x, y)))
                    .collect(),
                closed: true,
            },
        );
        let frames: Vec<_> = (0..8).map(|i| drawn(&s, c, Tick(first.0 + (last.0 - first.0) * i / 7), &[(&grid, [255, 255, 255])], 480)).collect();
        write_sheet(&frames, &out.join(format!("frames-{name}.png")));
        return;
    }
    let t0 = std::time::Instant::now();
    let r = s.execute(CMD, json!({"point": point, "at": at, "wait": true})).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    eprintln!("{r}");
    let m = mask(&s, c, "Object (1)").unwrap();
    let looks = r["status"]["done"].as_u64().unwrap_or(0);
    eprintln!("Object Mask: {} outlines of {looks} looks in {secs:.1} s ({:.0} ms a look)", m.path.keyframes.len(), secs * 1000.0 / looks.max(1) as f64);
    // how steady: the overlap of each outline with the next (both on one grid)
    let it = s.active_sequence().unwrap().find_item(c).unwrap().1.clone();
    let size = filmcraft_render::clip_size(&s.project, &it).unwrap();
    let g = Grid::new(size.0 as usize, size.1 as usize);
    let cells = |t: Tick| g.fill(&m.path_at(t).flatten(0.5).iter().map(|v| (v.x as f32, v.y as f32)).collect::<Vec<_>>());
    let steps: Vec<f32> = m.path.keyframes.windows(2).map(|w| iou(&cells(w[0].time), &cells(w[1].time))).collect();
    eprintln!("  overlap of each outline with the next: {}", summary(&steps));
    let media_at = it.source_time_at(it.start + Tick::from_seconds_f64(at));
    let (first, last) = (it.source_time_at(it.start), it.source_time_at(it.end() - Tick(1)));
    let before = feature_tracked(&mut s, c, &m, it.start + Tick::from_seconds_f64(at), "positionScaleRotation");
    eprintln!("Track Mask from the click's outline: {} frames", before.path.keyframes.len());
    // the strip: eight frames, two rows of four
    let w = 480usize;
    let frames: Vec<(Vec<u8>, usize, usize)> = (0..8)
        .map(|i| {
            let t = Tick(first.0 + (last.0 - first.0) * i / 7);
            drawn(&s, c, t, &[(&before, [0, 230, 255]), (&m, [255, 220, 0])], w)
        })
        .collect();
    let file = out.join(format!("object-{name}.png"));
    write_sheet(&frames, &file);
    eprintln!("strip: {} (click at {:.2} s media)", file.display(), media_at.seconds());
}

/// Eight frames as a PNG in two rows of four.
fn write_sheet(frames: &[(Vec<u8>, usize, usize)], file: &std::path::Path) {
    let w = frames.iter().map(|f| f.1).max().unwrap_or(1);
    let h = frames.iter().map(|f| f.2).max().unwrap_or(1);
    let mut strip = vec![0u8; w * 4 * h * 2 * 4];
    for (i, (px, fw, fh)) in frames.iter().enumerate() {
        let (ox, oy) = ((i % 4) * w, (i / 4) * h);
        for y in 0..*fh {
            let row = &px[y * fw * 4..(y + 1) * fw * 4];
            let at = ((oy + y) * w * 4 + ox) * 4;
            strip[at..at + fw * 4].copy_from_slice(row);
        }
    }
    let png = filmcraft_export::encode_png(strip, (w * 4) as u32, (h * 2) as u32).unwrap();
    std::fs::write(file, png).unwrap();
}
