//! Object tracking for motion tracks (`track.create`): follow one object the user boxed through
//! a clip, frame by frame, forwards or backwards.
//!
//! Two methods ([`TrackingMethod`]):
//!
//! - **Subject** — the [`crate::subject`] tracker (Median Flow on pyramidal Lucas–Kanade, a
//!   correlation template that re-detects the object after an occlusion), seeded with the box.
//!   Position and size; for things that change shape as they move (people, animals, a walking
//!   robot).
//! - **Points** — planar tracking: Shi–Tomasi corners inside the (rotated) box followed with
//!   pyramidal Lucas–Kanade forward and back and a RANSAC similarity fit
//!   ([`crate::track::track_step`], position, scale and rotation), chained from frame to frame.
//!   Each step is checked against the object's look: a small zero-mean, unit-norm patch sampled in
//!   the object's own (rotated, scaled) frame, normalised cross-correlation. A poor match — an
//!   occluder, the object leaving the picture — marks the frame lost; the box then holds while
//!   the object's last axis-aligned look is searched for around it (a wider window the longer it
//!   is gone), at a few scales, and a hit re-acquires it (its rotation kept).
//!
//! Everything is in the analysis frame's pixels internally; observations are normalised (0..1 of
//! the frame) so callers can analyse small frames. Every input is checked: odd frame sizes, short
//! buffers and NaN boxes give a lost observation, never a panic.

use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::{TrackMethod, TrackingMethod};

use crate::subject::{self, Frame, SubjectBox, SubjectTracker, Template, TrackState};

/// Longest side of the frames the object tracker sees by default.
pub const ANALYSIS_SIDE: usize = 512;
/// A step with fewer RANSAC inliers than this is a failure.
const MIN_INLIERS: usize = 5;
/// Below this correlation with the object's look the box no longer shows it.
const LOST_NCC: f32 = 0.45;
/// A search hit at least this good is a candidate…
const CANDIDATE_NCC: f32 = 0.45;
/// …that re-acquires the object when its own look, turned and refined there, matches this well.
const REDETECT_NCC: f32 = 0.6;
/// At least this good, the look learns the object's current appearance.
const LEARN_NCC: f32 = 0.8;
const LEARN_RATE: f32 = 0.05;
/// How long a lost object's box is held, confidence decaying, before it reads as gone (seconds).
const COAST_SECONDS: f64 = 1.5;
/// The object's look: at most this many samples a side.
const LOOK_SIDE: f64 = 32.0;
/// How far the look may pull the points' pose: pixels, degrees and a share of the scale.
const REFINE_REACH: (f64, f64, f64) = (3.0, 4.0, 0.05);
/// Largest scale change accepted in one step.
const MAX_STEP_SCALE: f64 = 1.25;

/// Where the object is: the centre and size of its box as shares of the frame (the size is the
/// object's own, before its rotation), its rotation since the seed (degrees, clockwise on screen)
/// and its size relative to the seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectBox {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub rotation: f64,
    pub scale: f64,
}

/// What the tracker made of one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectObservation {
    /// The caller's time of the frame (seconds).
    pub t: f64,
    pub bbox: ObjectBox,
    /// How sure the tracker is, 0..1.
    pub confidence: f32,
    /// Not seen in this frame: the box holds where it was last seen.
    pub lost: bool,
    pub state: TrackState,
}

/// Follows one object from a seed box; feed it the frames in tracking order
/// ([`ObjectTracker::push`]; backwards for a backward track).
pub struct ObjectTracker {
    method: TrackingMethod,
    seed: SubjectBox,
    subject: SubjectTracker,
    planar: Planar,
    seeded: bool,
}

impl ObjectTracker {
    /// A tracker for the object in `seed` (normalised; the first frame pushed is the seed frame).
    pub fn new(method: TrackingMethod, seed: SubjectBox) -> Self {
        Self { method, seed, subject: SubjectTracker::new().widening_search(6.0), planar: Planar::default(), seeded: false }
    }

    /// The next frame: RGBA8, `w`×`h`, at time `t` (seconds).
    pub fn push(&mut self, rgba: &[u8], w: usize, h: usize, t: f64) -> ObjectObservation {
        let t = if t.is_finite() { t } else { 0.0 };
        match self.method {
            TrackingMethod::Subject => self.push_subject(rgba, w, h, t),
            TrackingMethod::Points => {
                let frame = Frame::new(rgba, w, h);
                self.planar.push(frame, &self.seed, t)
            }
        }
    }

    fn push_subject(&mut self, rgba: &[u8], w: usize, h: usize, t: f64) -> ObjectObservation {
        let hint = (!self.seeded).then_some(self.seed);
        let obs = self.subject.push(rgba, w, h, t, hint);
        let lost = matches!(obs.state, TrackState::Coasting | TrackState::Lost | TrackState::None);
        let Some(b) = obs.bbox else {
            return ObjectObservation { t, bbox: from_subject(&self.seed, &self.seed), confidence: 0.0, lost: true, state: obs.state };
        };
        self.seeded = true;
        ObjectObservation { t, bbox: from_subject(&b, &self.seed), confidence: obs.confidence.clamp(0.0, 1.0), lost, state: obs.state }
    }
}

fn from_subject(b: &SubjectBox, seed: &SubjectBox) -> ObjectBox {
    let ratio = (b.w * b.h) / (seed.w * seed.h).max(1e-12);
    let scale = if ratio.is_finite() && ratio > 0.0 { ratio.sqrt() } else { 1.0 };
    ObjectBox { cx: b.cx, cy: b.cy, w: b.w, h: b.h, rotation: 0.0, scale }
}

/// The planar tracker's state, in the analysis frame's pixels.
#[derive(Default)]
struct Planar {
    prev: Option<Frame>,
    started: bool,
    /// Centre, the seed's size, rotation (degrees) and scale.
    c: Vec2,
    size0: Vec2,
    angle: f64,
    scale: f64,
    /// The object's look in its own frame.
    look: Option<Template>,
    /// The last frame it was seen well in, for re-detection.
    reference: Option<Reference>,
    seen_at: Option<f64>,
    confidence: f32,
    lost: bool,
}

impl Planar {
    fn size(&self) -> Vec2 {
        self.size0 * self.scale
    }

    fn observation(&self, frame: (usize, usize), t: f64, state: TrackState) -> ObjectObservation {
        let (fw, fh) = (frame.0.max(1) as f64, frame.1.max(1) as f64);
        let s = self.size();
        let bbox = ObjectBox { cx: self.c.x / fw, cy: self.c.y / fh, w: s.x / fw, h: s.y / fh, rotation: self.angle, scale: self.scale };
        ObjectObservation { t, bbox, confidence: self.confidence.clamp(0.0, 1.0), lost: self.lost, state }
    }

    /// The box as it is, unseen: confidence decaying since the object was last seen.
    fn hold(&mut self, frame: (usize, usize), t: f64) -> ObjectObservation {
        self.lost = true;
        let since = self.seen_at.map_or(f64::INFINITY, |s| (t - s).abs());
        let state = if since <= COAST_SECONDS {
            self.confidence *= 0.85;
            TrackState::Coasting
        } else {
            self.confidence = 0.0;
            TrackState::Lost
        };
        self.observation(frame, t, state)
    }

    fn push(&mut self, frame: Option<Frame>, seed: &SubjectBox, t: f64) -> ObjectObservation {
        let Some(frame) = frame else {
            let size = self.prev.as_ref().map_or((1, 1), |p| (p.w, p.h));
            if !self.started {
                self.c = Vec2::new(seed.cx * size.0 as f64, seed.cy * size.1 as f64);
                self.size0 = Vec2::new(seed.w * size.0 as f64, seed.h * size.1 as f64);
                self.scale = 1.0;
            }
            return self.hold(size, t);
        };
        let dims = (frame.w, frame.h);
        let resized = self.prev.as_ref().is_some_and(|p| p.w != frame.w || p.h != frame.h);
        if !self.started || resized {
            // seed (or re-seed in a frame of another size, keeping the normalised pose)
            let (fw, fh) = (frame.w as f64, frame.h as f64);
            if self.started {
                if let Some(p) = &self.prev {
                    let (kx, ky) = (fw / p.w.max(1) as f64, fh / p.h.max(1) as f64);
                    self.c = Vec2::new(self.c.x * kx, self.c.y * ky);
                    self.size0 = Vec2::new(self.size0.x * kx, self.size0.y * ky);
                }
            } else {
                self.c = Vec2::new(seed.cx * fw, seed.cy * fh);
                self.size0 = Vec2::new((seed.w * fw).max(2.0), (seed.h * fh).max(2.0));
                self.angle = 0.0;
                self.scale = 1.0;
            }
            self.look = self.sample_look(&frame);
            self.reference = Some(self.reference_of(&frame));
            self.started = true;
            self.seen_at = Some(t);
            self.confidence = 1.0;
            self.lost = false;
            self.prev = Some(frame);
            return self.observation(dims, t, TrackState::Seeded);
        }
        let obs = if self.lost { self.reacquire(&frame, t) } else { self.step(&frame, t) };
        self.prev = Some(frame);
        obs
    }

    /// One tracked step from the previous frame.
    fn step(&mut self, frame: &Frame, t: f64) -> ObjectObservation {
        let dims = (frame.w, frame.h);
        let tracked = self.prev.as_ref().and_then(|prev| {
            let region = self.region(0.8);
            let st = crate::track::track_step(&prev.prep, &frame.prep, &region, TrackMethod::PositionScaleRotation)?;
            (st.inliers >= MIN_INLIERS).then_some((st.transform, st.inliers as f32 / st.tracked.max(1) as f32))
        });
        if let Some((m, inlier_share)) = tracked {
            let ds = (m.a * m.a + m.b * m.b).sqrt();
            let da = m.b.atan2(m.a).to_degrees();
            let c = m.apply(self.c);
            if ds.is_finite() && (1.0 / MAX_STEP_SCALE..=MAX_STEP_SCALE).contains(&ds) && c.x.is_finite() && c.y.is_finite() && da.is_finite() {
                let before = (self.c, self.angle, self.scale);
                self.c = c;
                self.angle = wrap(self.angle + da);
                self.scale = (self.scale * ds).clamp(1e-3, 1e3);
                let ncc = self.refine(frame, REFINE_REACH, (1.0, 1.0, 0.015));
                if ncc >= LOST_NCC {
                    if ncc >= LEARN_NCC {
                        self.learn(frame);
                    }
                    self.seen_at = Some(t);
                    self.confidence = (ncc * (0.6 + 0.4 * inlier_share.clamp(0.0, 1.0))).clamp(0.0, 1.0);
                    return self.observation(dims, t, TrackState::Tracked);
                }
                // the points followed something that does not look like the object: undo
                (self.c, self.angle, self.scale) = before;
            }
        }
        // failed: look for it around where it was
        if let Some(obs) = self.redetect(frame, t, 1.5) {
            return obs;
        }
        self.hold(dims, t)
    }

    /// While lost: search, wider the longer it has been gone.
    fn reacquire(&mut self, frame: &Frame, t: f64) -> ObjectObservation {
        let gone = self.seen_at.map_or(f64::INFINITY, |s| (t - s).abs());
        let reach = if gone.is_finite() { (1.5 + 6.0 * gone).min(8.0) } else { 8.0 };
        if let Some(obs) = self.redetect(frame, t, reach) {
            return obs;
        }
        self.hold((frame.w, frame.h), t)
    }

    /// Search for the object within `reach` box sizes of where it was, as it looked in the
    /// reference frame turned to a few angles around its last rotation (it may have turned while
    /// hidden); a good hit that also matches the object's own look, turned and refined there,
    /// re-acquires it.
    fn redetect(&mut self, frame: &Frame, t: f64, reach: f64) -> Option<ObjectObservation> {
        let look = self.look.clone()?;
        let before = (self.c, self.angle, self.scale);
        let mut found: Option<(SubjectBox, f32, f64, f64)> = None;
        for da in [0.0, -10.0, 10.0, -20.0, 20.0] {
            let angle = before.1 + da;
            let Some((tp, ext)) = self.turned_template(angle) else { continue };
            let Some(near) = SubjectBox::from_px([self.c.x - ext.x, self.c.y - ext.y, self.c.x + ext.x, self.c.y + ext.y], frame.w, frame.h) else { continue };
            if let Some((hit, score)) = subject::search(frame, &tp, &near, Some(reach))
                && found.as_ref().is_none_or(|f| score > f.1)
            {
                let k = ((hit.w * hit.h) / (near.w * near.h).max(1e-12)).sqrt();
                found = Some((hit, score, angle, k));
            }
        }
        let (hit, score, angle, k) = found?;
        if score < CANDIDATE_NCC {
            return None;
        }
        self.c = Vec2::new(hit.cx * frame.w as f64, hit.cy * frame.h as f64);
        self.angle = angle;
        if k.is_finite() && k > 0.0 {
            self.scale = (self.scale * k).clamp(1e-3, 1e3);
        }
        // the angle to within a few degrees, then refined
        let base = self.angle;
        let mut best = (f32::NEG_INFINITY, base);
        for k in -2i32..=2 {
            self.angle = base + 2.5 * k as f64;
            let v = self.look_ncc(frame, &look).unwrap_or(0.0);
            if v > best.0 {
                best = (v, self.angle);
            }
        }
        self.angle = best.1;
        let ncc = self.refine(frame, (6.0, 8.0, 0.12), (2.0, 2.0, 0.03));
        if ncc < REDETECT_NCC {
            (self.c, self.angle, self.scale) = before;
            return None;
        }
        self.lost = false;
        self.seen_at = Some(t);
        self.confidence = ncc.clamp(0.0, 1.0);
        Some(self.observation((frame.w, frame.h), t, TrackState::Redetected))
    }

    /// Pull the pose onto the object's look: a hill climb over the centre, rotation and scale
    /// (from `steps`: pixels, degrees and a share of the scale, halving) maximising the
    /// correlation, within `reach` (the same units) of where it starts. Frame-to-frame chaining
    /// drifts a little each step (Lucas–Kanade under-reads small scale changes); the look anchors
    /// it. The correlation at the final pose.
    fn refine(&mut self, frame: &Frame, reach: (f64, f64, f64), steps: (f64, f64, f64)) -> f32 {
        let Some(tp) = self.look.clone() else { return 0.0 };
        let start = (self.c, self.angle, self.scale);
        let mut best = self.look_ncc(frame, &tp).unwrap_or(0.0);
        if best < LOST_NCC * 0.6 {
            return best;
        }
        let (mut step_px, mut step_deg, mut step_scale) = steps;
        let mut evals = 0;
        for _ in 0..5 {
            let mut improved = true;
            while improved && evals < 120 {
                improved = false;
                let here = (self.c, self.angle, self.scale);
                let moves = [
                    (Vec2::new(step_px, 0.0), 0.0, 1.0),
                    (Vec2::new(-step_px, 0.0), 0.0, 1.0),
                    (Vec2::new(0.0, step_px), 0.0, 1.0),
                    (Vec2::new(0.0, -step_px), 0.0, 1.0),
                    (Vec2::ZERO, step_deg, 1.0),
                    (Vec2::ZERO, -step_deg, 1.0),
                    (Vec2::ZERO, 0.0, 1.0 + step_scale),
                    (Vec2::ZERO, 0.0, 1.0 / (1.0 + step_scale)),
                ];
                let mut pick: Option<((Vec2, f64, f64), f32)> = None;
                for (dc, da, ds) in moves {
                    let cand = (here.0 + dc, here.1 + da, here.2 * ds);
                    let d = cand.0 - start.0;
                    if d.length() > reach.0 || (cand.1 - start.1).abs() > reach.1 || (cand.2 / start.2 - 1.0).abs() > reach.2 {
                        continue;
                    }
                    (self.c, self.angle, self.scale) = cand;
                    evals += 1;
                    let v = self.look_ncc(frame, &tp).unwrap_or(0.0);
                    if v > best + 1e-4 && pick.as_ref().is_none_or(|p| v > p.1) {
                        pick = Some((cand, v));
                    }
                }
                (self.c, self.angle, self.scale) = here;
                if let Some((cand, v)) = pick {
                    (self.c, self.angle, self.scale) = cand;
                    best = v;
                    improved = true;
                }
            }
            step_px /= 2.0;
            step_deg /= 2.0;
            step_scale /= 2.0;
        }
        self.angle = wrap(self.angle);
        best
    }

    /// The object's corners (frame pixels) shrunk to `k` of its size.
    fn region(&self, k: f64) -> Vec<Vec2> {
        let m = self.object_to_frame();
        let s = self.size() * (k / 2.0);
        [Vec2::new(-s.x, -s.y), Vec2::new(s.x, -s.y), Vec2::new(s.x, s.y), Vec2::new(-s.x, s.y)].into_iter().map(|p| m.apply(p)).collect()
    }

    /// Object coordinates (pixels from its centre, unrotated) → frame pixels.
    fn object_to_frame(&self) -> Affine {
        Affine::translate(self.c.x, self.c.y).then_apply(&Affine::rotate_deg(self.angle))
    }

    /// The object's look in its own frame: `tw`×`th` samples over its rotated box, from the
    /// pyramid level just finer than the samples.
    fn patch(&self, frame: &Frame, tw: usize, th: usize) -> Option<Vec<f32>> {
        let s = self.size();
        if !(s.x > 1.0 && s.y > 1.0) || tw == 0 || th == 0 {
            return None;
        }
        let ratio = (s.x / tw as f64).min(s.y / th as f64).max(1.0);
        let level = (ratio.log2().floor().max(0.0) as usize).min(frame.prep.levels.len().saturating_sub(1));
        let g = frame.prep.levels.get(level)?;
        let k = (1u32 << level) as f64;
        let m = self.object_to_frame();
        let mut out = Vec::with_capacity(tw * th);
        for j in 0..th {
            for i in 0..tw {
                let o = Vec2::new(((i as f64 + 0.5) / tw as f64 - 0.5) * s.x, ((j as f64 + 0.5) / th as f64 - 0.5) * s.y);
                let p = m.apply(o);
                out.push(g.sample((p.x / k) as f32, (p.y / k) as f32));
            }
        }
        Some(out)
    }

    fn sample_look(&self, frame: &Frame) -> Option<Template> {
        let s = self.size();
        let (w, h) = look_size(s.x, s.y);
        let mut px = self.patch(frame, w, h)?;
        subject::normalize(&mut px).then_some(Template { w, h, px })
    }

    fn look_ncc(&self, frame: &Frame, tp: &Template) -> Option<f32> {
        let mut px = self.patch(frame, tp.w, tp.h)?;
        if !subject::normalize(&mut px) {
            return Some(0.0);
        }
        Some(subject::dot(&tp.px, &px))
    }

    /// Blend the object's current look into its look, and keep its axis-aligned look for search.
    fn learn(&mut self, frame: &Frame) {
        if let Some(tp) = self.look.as_ref()
            && let Some(mut now) = self.patch(frame, tp.w, tp.h)
            && subject::normalize(&mut now)
            && let Some(tp) = self.look.as_mut()
        {
            for (a, n) in tp.px.iter_mut().zip(&now) {
                *a = *a * (1.0 - LEARN_RATE) + n * LEARN_RATE;
            }
            subject::normalize(&mut tp.px);
        }
        self.reference = Some(self.reference_of(frame));
    }

    fn reference_of(&self, frame: &Frame) -> Reference {
        let copy = Frame { prep: crate::track::Prepared { levels: frame.prep.levels.clone() }, w: frame.w, h: frame.h };
        Reference { frame: copy, c: self.c, angle: self.angle, scale: self.scale }
    }

    /// A search template of the object as it looked in the reference frame, turned to `angle` at
    /// its current size: the axis-aligned box around it (so its corners show what was around
    /// it), and that box's half size in pixels.
    fn turned_template(&self, angle: f64) -> Option<(Template, Vec2)> {
        let r = self.reference.as_ref()?;
        let s = self.size();
        let (sn, cs) = angle.to_radians().sin_cos();
        let ext = Vec2::new((cs.abs() * s.x + sn.abs() * s.y) / 2.0, (sn.abs() * s.x + cs.abs() * s.y) / 2.0);
        if !(ext.x > 1.0 && ext.y > 1.0 && ext.x.is_finite() && ext.y.is_finite()) {
            return None;
        }
        let (tw, th) = subject::template_size(2.0 * ext.x, 2.0 * ext.y);
        let k_ref = (r.scale / self.scale.max(1e-9)).clamp(1e-3, 1e3);
        let ratio = ((2.0 * ext.x / tw as f64) * k_ref).max(1.0);
        let level = (ratio.log2().floor().max(0.0) as usize).min(r.frame.prep.levels.len().saturating_sub(1));
        let g = r.frame.prep.levels.get(level)?;
        let kl = (1u32 << level) as f64;
        // this frame's offsets from the centre → the object's own coordinates → the reference's
        let m = Affine::translate(r.c.x, r.c.y)
            .then_apply(&Affine::rotate_deg(r.angle))
            .then_apply(&Affine::scale(k_ref, k_ref))
            .then_apply(&Affine::rotate_deg(-angle));
        let mut px = Vec::with_capacity(tw * th);
        for j in 0..th {
            for i in 0..tw {
                let d = Vec2::new(((i as f64 + 0.5) / tw as f64 - 0.5) * 2.0 * ext.x, ((j as f64 + 0.5) / th as f64 - 0.5) * 2.0 * ext.y);
                let p = m.apply(d);
                px.push(g.sample((p.x / kl) as f32, (p.y / kl) as f32));
            }
        }
        subject::normalize(&mut px).then_some((Template { w: tw, h: th, px }, ext))
    }
}

/// The last frame the object was seen well in, and its pose there.
struct Reference {
    frame: Frame,
    c: Vec2,
    angle: f64,
    scale: f64,
}

/// The look's size for an object of `bw`×`bh` pixels: at most [`LOOK_SIDE`] a side (never much
/// more than the object, so it is not upsampled), at least 8.
fn look_size(bw: f64, bh: f64) -> (usize, usize) {
    let side = LOOK_SIDE.min(bw.max(bh)).max(8.0);
    let (w, h) = if bw >= bh { (side, side * bh / bw.max(1e-6)) } else { (side * bw / bh.max(1e-6), side) };
    let fit = |v: f64| if v.is_finite() { v.round().clamp(8.0, LOOK_SIDE) as usize } else { 8 };
    (fit(w), fit(h))
}

/// An angle in degrees, wrapped to -180..180.
fn wrap(a: f64) -> f64 {
    if !a.is_finite() {
        return 0.0;
    }
    let r = (a + 180.0).rem_euclid(360.0) - 180.0;
    if r.is_finite() { r } else { 0.0 }
}

#[cfg(test)]
#[path = "object_track_tests.rs"]
mod tests;
