//! Named targets for the automatic blur (`clip.autoRedact {"targets": ["DP World logo"]}`): a
//! logo, a brand, a sign the user names in words, found by the host's vision model and followed
//! between its looks.
//!
//! The vision model ([`VisionJudge::find`]: AutoCut's Qwen2.5-VL, grounding boxes) looks at a
//! clip's analysed frames every `targetEvery` seconds (1 by default) and at every shot change.
//! Each place it finds a target starts a follow, or re-seeds the follow of that target it
//! overlaps: Median Flow (`filmcraft_render::subject::SubjectTracker`) carries the box over the
//! analysed frames in between — forwards, and backwards to the previous look for a place found
//! for the first time, so the blur is on from where the target came into view. A follow ends when
//! the tracker loses it (a box it only holds is not kept, and a look that does not find the
//! place again ends it) or the model has not seen it at [`UNCONFIRMED_LOOKS`] looks in a row.
//! The follows become tracks (`filmcraft_vision::track::Kind::Target`) like faces and plates:
//! smoothed, padded, a rectangle mask each on the clip's blur.

use std::collections::VecDeque;

use filmcraft_render::scene::{Signature, difference, signature};
use filmcraft_render::subject::{SubjectBox, SubjectTracker, TrackState, downscale_rgba};
use filmcraft_vision::Kind;
use filmcraft_vision::track::{Track, TrackPoint};

use crate::redact::{Crop, VisionJudge};

/// Longest side of the frames the tracker follows targets on.
const TRACK_SIDE: usize = 384;
/// Longest side of the frame the vision model is shown.
pub const LOOK_SIDE: usize = 1280;
/// A shot change (scene signature difference) brings a look forward.
const CUT: f32 = 0.3;
/// …but not closer than this to the previous look, seconds.
const MIN_LOOK_GAP: f64 = 0.3;
/// A follow the model has not confirmed at this many looks in a row ends (it often names one
/// place a look, so a single miss is not the end).
pub const UNCONFIRMED_LOOKS: usize = 2;
/// A place found overlapping a follow's box at least this much (IoU), or centred inside it,
/// re-seeds that follow.
const SAME: f64 = 0.2;
/// Most frames kept for following a new place backwards.
const MAX_BUFFER: usize = 128;
/// Most follows at once in one clip.
const MAX_FOLLOWS: usize = 32;

/// One place a target is followed.
struct Follow {
    target: usize,
    tracker: SubjectTracker,
    /// (seconds, box `[x, y, w, h]`).
    points: Vec<(f64, [f64; 4])>,
    last: SubjectBox,
    unconfirmed: usize,
    /// The tracker has lost it and holds its last box: nothing is recorded, and a look that
    /// does not find it again ends the follow (a held box drifts onto whatever is there now).
    coasting: bool,
    /// A box from the latest look, for the next frame pushed.
    hint: Option<SubjectBox>,
}

/// What the follower made of a clip.
#[derive(Debug, Default)]
pub struct Followed {
    pub tracks: Vec<Track>,
    /// Looks asked of the vision model, and the places it found.
    pub looks: usize,
    pub found: usize,
    /// What went wrong asking (the first few).
    pub errors: Vec<String>,
}

/// Follows a clip's named targets over its analysed frames (fed in time order).
pub struct Follower<'a> {
    names: &'a [String],
    every: f64,
    vision: &'a dyn VisionJudge,
    last_look: Option<f64>,
    prev_sig: Option<Signature>,
    /// Small frames since the last look: (seconds, rgba, w, h).
    buffer: VecDeque<(f64, Vec<u8>, usize, usize)>,
    active: Vec<Follow>,
    done: Vec<Follow>,
    out: Followed,
}

fn corner(b: &SubjectBox) -> [f64; 4] {
    [b.cx - b.w / 2.0, b.cy - b.h / 2.0, b.w, b.h]
}

fn same_place(a: &SubjectBox, b: &SubjectBox) -> bool {
    a.iou(b) >= SAME || ((b.cx - a.cx).abs() <= a.w / 2.0 && (b.cy - a.cy).abs() <= a.h / 2.0)
}

/// An RGBA frame fitted inside `side` (never enlarged) as a [`Crop`].
fn picture(rgba: &[u8], w: usize, h: usize, side: usize) -> Option<Crop> {
    let (px, pw, ph) = downscale_rgba(rgba, w, h, side)?;
    Some(Crop { rgba: px, width: u32::try_from(pw).ok()?, height: u32::try_from(ph).ok()? })
}

impl<'a> Follower<'a> {
    pub fn new(names: &'a [String], every: f64, vision: &'a dyn VisionJudge) -> Self {
        let every = if every.is_finite() { every.clamp(0.1, 60.0) } else { 1.0 };
        Self { names, every, vision, last_look: None, prev_sig: None, buffer: VecDeque::new(), active: Vec::new(), done: Vec::new(), out: Followed::default() }
    }

    /// The next analysed frame (RGBA, `w` × `h`) at `t` seconds of media time.
    pub fn push(&mut self, rgba: &[u8], w: usize, h: usize, t: f64) {
        if self.names.is_empty() {
            return;
        }
        let Some((small, sw, sh)) = downscale_rgba(rgba, w, h, TRACK_SIDE) else { return };
        let sig = signature(&small, sw, sh);
        let cut = self.prev_sig.as_ref().is_some_and(|p| difference(p, &sig) > CUT);
        self.prev_sig = Some(sig);
        // a look at the first frame of every `every` seconds of media time — a grid, so two
        // windows of one clip look at the same frames (and share the redaction cache's looks) —
        // and at a cut
        let bucket = |t: f64| (t / self.every + 1e-9).floor();
        let look = match self.last_look {
            None => true,
            Some(l) => bucket(t) > bucket(l) || (cut && t - l >= MIN_LOOK_GAP),
        };
        if look {
            self.look(rgba, w, h, t, &small, sw, sh);
        }
        for f in &mut self.active {
            let hint = f.hint.take();
            let o = f.tracker.push(&small, sw, sh, t, hint);
            match (o.state, o.bbox) {
                (TrackState::Lost | TrackState::None, _) | (_, None) => f.unconfirmed = usize::MAX,
                (TrackState::Coasting, _) => f.coasting = true,
                (_, Some(b)) => {
                    f.coasting = false;
                    f.last = b;
                    f.points.push((t, corner(&b)));
                }
            }
        }
        let (ended, active): (Vec<Follow>, Vec<Follow>) = std::mem::take(&mut self.active).into_iter().partition(|f| f.unconfirmed == usize::MAX);
        self.active = active;
        self.done.extend(ended);
        if self.buffer.len() >= MAX_BUFFER {
            self.buffer.pop_front();
        }
        self.buffer.push_back((t, small, sw, sh));
    }

    /// Ask the vision model where the targets are now; re-seed or start follows.
    #[allow(clippy::too_many_arguments)]
    fn look(&mut self, rgba: &[u8], w: usize, h: usize, t: f64, small: &[u8], sw: usize, sh: usize) {
        self.last_look = Some(t);
        self.out.looks += 1;
        let found = match picture(rgba, w, h, LOOK_SIDE).map(|p| self.vision.find(&p, self.names)) {
            Some(Ok(f)) => f,
            Some(Err(e)) => {
                if self.out.errors.len() < 4 {
                    self.out.errors.push(e);
                }
                // nothing new learned: the follows go on as they were
                self.buffer.clear();
                return;
            }
            None => Vec::new(),
        };
        for f in &mut self.active {
            f.unconfirmed = if f.coasting { UNCONFIRMED_LOOKS } else { f.unconfirmed + 1 };
        }
        for (target, b) in found {
            if target >= self.names.len() || !b.iter().all(|v| v.is_finite()) {
                continue;
            }
            let Some(sb) = SubjectBox::from_corner(b[0], b[1], b[2], b[3]) else { continue };
            self.out.found += 1;
            // the same place again (under any of the targets' names: the model mixes them up)
            if let Some(f) = self.active.iter_mut().find(|f| same_place(&f.last, &sb)) {
                f.hint = Some(sb);
                f.unconfirmed = 0;
                f.coasting = false;
                continue;
            }
            if self.active.len() >= MAX_FOLLOWS {
                continue;
            }
            // a place seen for the first time: followed back to the previous look first
            let mut back = SubjectTracker::new();
            back.push(small, sw, sh, t, Some(sb));
            let mut before: Vec<(f64, [f64; 4])> = Vec::new();
            for (bt, px, bw, bh) in self.buffer.iter().rev() {
                let o = back.push(px, *bw, *bh, *bt, None);
                match (o.state, o.bbox) {
                    (TrackState::Lost | TrackState::None | TrackState::Coasting, _) | (_, None) => break,
                    (_, Some(bb)) => before.push((*bt, corner(&bb))),
                }
            }
            before.reverse();
            self.active.push(Follow { target, tracker: SubjectTracker::new(), points: before, last: sb, unconfirmed: 0, coasting: false, hint: Some(sb) });
        }
        let (ended, active): (Vec<Follow>, Vec<Follow>) = std::mem::take(&mut self.active).into_iter().partition(|f| f.unconfirmed >= UNCONFIRMED_LOOKS);
        self.active = active;
        self.done.extend(ended);
        self.buffer.clear();
    }

    /// The follows as tracks (smoothed over `smooth` seconds), and what the looks found.
    pub fn finish(mut self, smooth: f64) -> Followed {
        self.done.append(&mut self.active);
        let mut out = std::mem::take(&mut self.out);
        for f in self.done {
            let mut points: Vec<TrackPoint> = Vec::with_capacity(f.points.len());
            for (t, b) in f.points {
                if points.last().is_some_and(|p: &TrackPoint| p.t >= t) {
                    continue;
                }
                points.push(TrackPoint { t, bbox: b, corners: Vec::new(), seen: true });
            }
            if points.is_empty() {
                continue;
            }
            let label = self.names.get(f.target).cloned();
            let hits = points.len();
            let track = Track { kind: Kind::Target, label, points, hits, best_score: 1.0 };
            out.tracks.push(filmcraft_vision::track::smoothed(track, smooth));
        }
        out.tracks.sort_by(|a, b| a.start().total_cmp(&b.start()));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(x: i64, y: i64) -> u8 {
        let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (h >> 56) as u8
    }

    /// A frame of a bright, finely patterned square (a "logo") at `x` (shares of a 320 × 180
    /// frame) on a dark, faintly textured ground.
    fn frame(x: f64) -> Vec<u8> {
        let (w, h) = (320usize, 180usize);
        let left = (x - 0.06) * w as f64;
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for xx in 0..w {
                let i = (y * w + xx) * 4;
                let inside = (xx as f64 / w as f64 - x).abs() < 0.06 && (y as f64 / h as f64 - 0.5).abs() < 0.08;
                let v = if inside {
                    // the pattern moves with the square
                    let n = hash(((xx as f64 - left) / 3.0) as i64, (y / 3) as i64) / 4;
                    [210 + n / 2, 150 + n, 20, 255]
                } else {
                    let n = hash(xx as i64 / 6, y as i64 / 6) / 16;
                    [30 + n, 40 + n, 70 + n, 255]
                };
                px[i..i + 4].copy_from_slice(&v);
            }
        }
        px
    }

    struct Finder {
        looks: std::sync::Mutex<Vec<f64>>,
        /// From which look on the model sees it.
        from: usize,
    }

    impl VisionJudge for Finder {
        fn name(&self) -> String {
            "finder".into()
        }
        fn judge(&self, _: Kind, crops: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String> {
            Ok(vec![None; crops.len()])
        }
        fn find(&self, frame: &Crop, targets: &[String]) -> std::result::Result<Vec<(usize, [f64; 4])>, String> {
            let mut l = self.looks.lock().unwrap();
            l.push(0.0);
            if l.len() <= self.from || targets.is_empty() {
                return Ok(Vec::new());
            }
            // the bright square, from the pixels
            let (w, h) = (frame.width as usize, frame.height as usize);
            let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
            for (i, p) in frame.rgba.as_chunks::<4>().0.iter().enumerate() {
                if p[0] > 200 && p[2] < 40 {
                    x0 = x0.min(i % w);
                    x1 = x1.max(i % w);
                    y0 = y0.min(i / w);
                    y1 = y1.max(i / w);
                }
            }
            if x1 < x0 {
                return Ok(Vec::new());
            }
            Ok(vec![(0, [x0 as f64 / w as f64, y0 as f64 / h as f64, (x1 - x0 + 1) as f64 / w as f64, (y1 - y0 + 1) as f64 / h as f64])])
        }
    }

    #[test]
    fn a_target_is_followed_between_looks_and_back_to_where_it_came_in() {
        let names = vec!["the logo".to_string()];
        // the model sees it from its second look (t = 1 s) on
        let finder = Finder { looks: Default::default(), from: 1 };
        let mut f = Follower::new(&names, 1.0, &finder);
        for k in 0..30 {
            let t = k as f64 / 10.0;
            let x = 0.2 + 0.015 * k as f64;
            f.push(&frame(x), 320, 180, t);
        }
        let out = f.finish(0.08);
        assert_eq!(out.looks, 3, "a look a second");
        assert_eq!(out.tracks.len(), 1, "{:?}", out.tracks.iter().map(|t| (t.start(), t.end())).collect::<Vec<_>>());
        let tr = &out.tracks[0];
        assert_eq!(tr.kind, Kind::Target);
        assert_eq!(tr.label.as_deref(), Some("the logo"));
        // followed back towards the first look and forward to the end
        assert!(tr.start() <= 0.5, "starts at {}", tr.start());
        assert!(tr.end() >= 2.8, "ends at {}", tr.end());
        for p in &tr.points {
            let want = 0.2 + 0.015 * (p.t * 10.0).round();
            let cx = p.bbox[0] + p.bbox[2] / 2.0;
            assert!((cx - want).abs() < 0.03, "at {}: {cx} for {want}", p.t);
        }
    }

    /// Regression (Promobot clip, 70 s): when the logo left the picture, the tracker held its
    /// last box and the blur drifted onto the robot's LED face for three seconds. A held box is
    /// not kept, and a look that does not find the place again ends it.
    #[test]
    fn a_target_that_leaves_the_picture_is_not_followed_on() {
        let names = vec!["the logo".to_string()];
        let finder = Finder { looks: Default::default(), from: 0 };
        let mut f = Follower::new(&names, 1.0, &finder);
        let empty = frame(5.0);
        for k in 0..40 {
            let t = k as f64 / 10.0;
            // the logo until 1.45 s, then only the ground
            if t < 1.45 { f.push(&frame(0.4), 320, 180, t) } else { f.push(&empty, 320, 180, t) }
        }
        let out = f.finish(0.08);
        let end = out.tracks.iter().map(Track::end).fold(0.0, f64::max);
        assert!(end <= 2.05, "followed until {end}");
    }

    #[test]
    fn a_model_that_fails_or_sees_nothing_makes_no_tracks() {
        struct Broken;
        impl VisionJudge for Broken {
            fn name(&self) -> String {
                "broken".into()
            }
            fn judge(&self, _: Kind, c: &[Crop]) -> std::result::Result<Vec<Option<bool>>, String> {
                Ok(vec![None; c.len()])
            }
        }
        let names = vec!["x".to_string()];
        let mut f = Follower::new(&names, 0.5, &Broken);
        for k in 0..10 {
            f.push(&frame(0.5), 320, 180, k as f64 / 10.0);
        }
        // hostile frames are skipped
        f.push(&[], 0, 0, 2.0);
        f.push(&[1, 2, 3], 320, 180, f64::NAN);
        let out = f.finish(0.08);
        assert!(out.tracks.is_empty());
        assert_eq!(out.looks, 2);
        assert!(!out.errors.is_empty());
    }
}
