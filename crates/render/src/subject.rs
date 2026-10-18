//! Subject tracking for Auto Reframe: follow one subject (a person, a robot, a car) through a
//! clip, frame by frame, from a box that seeds it.
//!
//! The method (textbook, implemented from the papers' descriptions):
//!
//! 1. **Seed** — the box the caller gives (the vision model's, or the user's); without one, the
//!    salient region of the frame: frequency-tuned saliency (Achanta, Hemami, Estrada &
//!    Süsstrunk, CVPR 2009: the distance in CIELAB between the frame's mean colour and a slightly
//!    blurred copy), weighted by a centre prior and a skin-tone prior (the Cb/Cr ranges of Chai &
//!    Ngan 1999) and boxed around the weighted centroid of the pixels above twice the mean.
//! 2. **Track** — Median Flow (Kalal, Mikolajczyk & Matas, ICPR 2010): Shi–Tomasi corners (and a
//!    grid where the subject has little texture) inside the inner part of the box are followed
//!    with the pyramidal Lucas–Kanade tracker of [`crate::track`] forward and back; the half with
//!    the smallest forward–backward error and the best patch correlation votes, the box moves by
//!    the median displacement and scales by the median ratio of pairwise distances. A median
//!    forward–backward error above a few pixels is a failure.
//! 3. **Check** — the box's patch is compared with the subject's appearance (a small zero-mean,
//!    unit-norm template, normalised cross-correlation); a poor match is a failure too, so the
//!    tracker does not slide onto the background or an occluder.
//! 4. **Re-detect** — after a failure the template is searched for (normalised cross-correlation
//!    over a window around the last box at a few scales, coarse to fine; the whole frame after a
//!    shot change); until it is found again the box holds where it was ("coasting") and its
//!    confidence decays. A shot change (the [`crate::scene`] signature difference, well above the
//!    recent activity) re-seeds from the template, a hint or the saliency.
//! 5. **Hints** — a box from the vision model for a frame re-seeds the tracker when it is lost or
//!    when the tracked box no longer overlaps the hint (the model knows what the subject is; the
//!    tracker only knows what it looked like).
//!
//! Everything is in the analysis frame's pixels internally; [`SubjectBox`] is normalised (0..1 of
//! the frame), so callers can analyse small frames and apply the result at any size. Every input
//! is checked: odd frame sizes, short buffers and NaN boxes give a "lost" observation, never a
//! panic. [`crate::reframe`] turns the observations into a smooth crop path.

use serde::{Deserialize, Serialize};

use crate::scene::{Signature, difference, signature};
use crate::track::{Gray, Prepared, lk};

/// Smallest analysis frame side the tracker works on.
pub const MIN_SIDE: usize = 16;
/// Largest analysis frame (pixels) it accepts: callers downscale first ([`downscale_rgba`]).
pub const MAX_PIXELS: usize = 4096 * 4096;
/// Smallest box side, as a share of the frame.
const MIN_BOX: f64 = 0.02;
/// Points tracked per frame, at most.
const MAX_POINTS: usize = 40;
/// A median forward–backward error above this (pixels of a ~384-pixel frame) is a failure.
const FB_FAIL: f32 = 2.5;
/// Below this template correlation the tracked box no longer shows the subject.
const LOST_NCC: f32 = 0.35;
/// A search hit at least this good re-acquires the subject.
const REDETECT_NCC: f32 = 0.62;
/// At least this good, the template learns the subject's current look.
const LEARN_NCC: f32 = 0.7;
/// How fast the template follows the subject's look.
const LEARN_RATE: f32 = 0.08;
/// How long a lost subject's box is held (seconds) before the observation says it is gone.
const COAST_SECONDS: f64 = 1.2;
/// Largest template side, pixels.
const TEMPLATE_SIDE: f64 = 24.0;
/// A shot change: a signature difference above this and well above the recent activity.
const CUT_DIFFERENCE: f32 = 0.28;

/// A box around the subject, normalised to the frame: its centre and size as shares of the
/// frame's width (`cx`, `w`) and height (`cy`, `h`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubjectBox {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
}

impl SubjectBox {
    /// A checked box: finite numbers, the size in `MIN_BOX..=1` and the centre inside the frame.
    pub fn new(cx: f64, cy: f64, w: f64, h: f64) -> Option<Self> {
        if !(cx.is_finite() && cy.is_finite() && w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
            return None;
        }
        Some(Self { cx: cx.clamp(0.0, 1.0), cy: cy.clamp(0.0, 1.0), w: w.clamp(MIN_BOX, 1.0), h: h.clamp(MIN_BOX, 1.0) })
    }

    /// From the top-left corner and size (as commands take boxes: `{x, y, w, h}`).
    pub fn from_corner(x: f64, y: f64, w: f64, h: f64) -> Option<Self> {
        Self::new(x + w / 2.0, y + h / 2.0, w, h)
    }

    /// A box for a centre and the share of the frame it covers (the vision model's facts), square
    /// in pixels on a `aspect` (width over height) frame.
    pub fn from_area(center: [f64; 2], area: f64, aspect: f64) -> Option<Self> {
        if !(area.is_finite() && area > 0.0 && aspect.is_finite() && aspect > 0.0) {
            return None;
        }
        let area = area.min(1.0);
        // side² = area · W · H, as shares: w = side / W, h = side / H
        let w = (area / aspect).sqrt();
        let h = (area * aspect).sqrt();
        Self::new(center[0], center[1], w, h)
    }

    /// Intersection over union with `o`.
    pub fn iou(&self, o: &SubjectBox) -> f64 {
        let ix = ((self.cx + self.w / 2.0).min(o.cx + o.w / 2.0) - (self.cx - self.w / 2.0).max(o.cx - o.w / 2.0)).max(0.0);
        let iy = ((self.cy + self.h / 2.0).min(o.cy + o.h / 2.0) - (self.cy - self.h / 2.0).max(o.cy - o.h / 2.0)).max(0.0);
        let inter = ix * iy;
        let union = self.w * self.h + o.w * o.h - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }

    /// Corners in a `w`×`h` frame's pixels: `[x0, y0, x1, y1]`.
    fn px(&self, w: usize, h: usize) -> [f64; 4] {
        let (w, h) = (w as f64, h as f64);
        [(self.cx - self.w / 2.0) * w, (self.cy - self.h / 2.0) * h, (self.cx + self.w / 2.0) * w, (self.cy + self.h / 2.0) * h]
    }

    fn from_px(b: [f64; 4], w: usize, h: usize) -> Option<Self> {
        let (fw, fh) = (w.max(1) as f64, h.max(1) as f64);
        Self::new((b[0] + b[2]) / 2.0 / fw, (b[1] + b[3]) / 2.0 / fh, (b[2] - b[0]) / fw, (b[3] - b[1]) / fh)
    }
}

/// What the tracker made of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrackState {
    /// Seeded from a hint, the caller's box or the saliency.
    Seeded,
    /// Followed from the previous frame.
    Tracked,
    /// Found again by its template after a failure or a shot change.
    Redetected,
    /// Not found: the box is held where it was last seen.
    Coasting,
    /// Gone for longer than the tracker holds a box.
    Lost,
    /// Nothing to follow (a flat frame and no hint).
    None,
}

/// One frame's observation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// The caller's time of the frame (seconds).
    pub t: f64,
    /// Where the subject is (held while coasting or lost); `None` before anything was seeded.
    pub bbox: Option<SubjectBox>,
    /// How sure the tracker is, 0..1.
    pub confidence: f32,
    pub state: TrackState,
    /// This frame starts a new shot.
    pub cut: bool,
}

/// The subject's look: a small zero-mean, unit-norm grey patch.
#[derive(Clone, Debug)]
struct Template {
    w: usize,
    h: usize,
    px: Vec<f32>,
}

/// A frame ready for tracking.
struct Frame {
    prep: Prepared,
    w: usize,
    h: usize,
}

/// Follows one subject through a clip; feed it the clip's frames in order ([`SubjectTracker::push`]).
pub struct SubjectTracker {
    prev: Option<Frame>,
    prev_sig: Option<Signature>,
    recent: Vec<f32>,
    bbox: Option<SubjectBox>,
    template: Option<Template>,
    /// When the subject was last seen for sure.
    seen_at: Option<f64>,
    confidence: f32,
}

impl Default for SubjectTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SubjectTracker {
    pub fn new() -> Self {
        Self { prev: None, prev_sig: None, recent: Vec::new(), bbox: None, template: None, seen_at: None, confidence: 0.0 }
    }

    /// The next frame: RGBA8, `w`×`h`, at time `t` (seconds), with the box the caller knows for it
    /// (`hint`: the vision model's, or the seed the user drew) when there is one.
    pub fn push(&mut self, rgba: &[u8], w: usize, h: usize, t: f64, hint: Option<SubjectBox>) -> Observation {
        let t = if t.is_finite() { t } else { self.seen_at.unwrap_or(0.0) };
        let n = w.checked_mul(h).filter(|n| (MIN_SIDE * MIN_SIDE..=MAX_PIXELS).contains(n));
        let pixels = n.and_then(|n| n.checked_mul(4)).and_then(|len| rgba.get(..len));
        let (Some(pixels), true) = (pixels, w >= MIN_SIDE && h >= MIN_SIDE) else {
            // nothing to look at: keep what we had, unsure
            return self.hold(t, false, hint);
        };
        let sig = signature(pixels, w, h);
        let size_changed = self.prev.as_ref().is_some_and(|p| p.w != w || p.h != h);
        let cut = size_changed || self.is_cut(&sig);
        let frame = Frame { prep: pyramid(Gray::from_rgba8(w, h, pixels)), w, h };
        let obs = self.observe(&frame, pixels, t, hint, cut);
        self.prev = Some(frame);
        self.prev_sig = Some(sig);
        obs
    }

    /// The box as it is, with confidence decaying since the subject was last seen.
    fn hold(&mut self, t: f64, cut: bool, hint: Option<SubjectBox>) -> Observation {
        if let Some(h) = hint {
            self.bbox = Some(h);
            self.seen_at = Some(t);
            self.confidence = 0.9;
            return Observation { t, bbox: self.bbox, confidence: self.confidence, state: TrackState::Seeded, cut };
        }
        let Some(b) = self.bbox else { return Observation { t, bbox: None, confidence: 0.0, state: TrackState::None, cut } };
        let since = self.seen_at.map_or(f64::INFINITY, |s| (t - s).abs());
        let (state, conf) = if since <= COAST_SECONDS {
            (TrackState::Coasting, (self.confidence * (1.0 - (since / COAST_SECONDS) as f32)).max(0.0))
        } else {
            (TrackState::Lost, 0.0)
        };
        Observation { t, bbox: Some(b), confidence: conf, state, cut }
    }

    /// Whether `sig` starts a new shot: much more different from the previous frame than the
    /// frames before were from each other.
    fn is_cut(&mut self, sig: &Signature) -> bool {
        let Some(prev) = &self.prev_sig else { return false };
        let d = difference(prev, sig);
        let mut around = self.recent.clone();
        around.sort_by(f32::total_cmp);
        let med = around.get(around.len() / 2).copied().unwrap_or(0.0);
        self.recent.push(d);
        if self.recent.len() > 8 {
            self.recent.remove(0);
        }
        d > CUT_DIFFERENCE && d > 2.5 * med + 0.05
    }

    fn seed(&mut self, frame: &Frame, b: SubjectBox, t: f64, confidence: f32, state: TrackState, cut: bool) -> Observation {
        self.bbox = Some(b);
        self.template = template(frame, &b).or(self.template.take());
        self.seen_at = Some(t);
        self.confidence = confidence;
        Observation { t, bbox: Some(b), confidence, state, cut }
    }

    fn observe(&mut self, frame: &Frame, rgba: &[u8], t: f64, hint: Option<SubjectBox>, cut: bool) -> Observation {
        // nothing yet: the hint, else the salient region
        let Some(current) = self.bbox else {
            if let Some(h) = hint {
                return self.seed(frame, h, t, 0.9, TrackState::Seeded, cut);
            }
            return match salient_box(rgba, frame.w, frame.h) {
                Some(b) => self.seed(frame, b, t, 0.5, TrackState::Seeded, cut),
                None => Observation { t, bbox: None, confidence: 0.0, state: TrackState::None, cut },
            };
        };
        if cut {
            // the same subject in the new shot, else what the caller knows, else what stands out
            if let Some((b, score)) = self.search(frame, &current, true).filter(|(_, s)| *s >= REDETECT_NCC) {
                return self.seed(frame, b, t, score, TrackState::Redetected, true);
            }
            if let Some(h) = hint {
                return self.seed(frame, h, t, 0.9, TrackState::Seeded, true);
            }
            return match salient_box(rgba, frame.w, frame.h) {
                Some(b) => self.seed(frame, b, t, 0.5, TrackState::Seeded, true),
                None => self.hold(t, true, None),
            };
        }
        let tracked = self.prev.as_ref().and_then(|prev| flow_step(prev, frame, &current));
        let mut result = tracked.and_then(|(b, reliability)| {
            let ncc = self.template.as_ref().and_then(|tp| match_at(frame, tp, &b)).unwrap_or(0.0);
            (ncc >= LOST_NCC).then_some((b, ncc, reliability))
        });
        // a hint that disagrees with the track wins: the model knows what the subject is
        if let Some(h) = hint {
            let agrees = result.as_ref().is_some_and(|(b, ncc, _)| b.iou(&h) >= 0.15 && *ncc >= 0.5);
            if !agrees {
                return self.seed(frame, h, t, 0.9, TrackState::Seeded, false);
            }
        }
        if let Some((b, ncc, reliability)) = result.take() {
            if ncc >= LEARN_NCC {
                self.learn(frame, &b);
            }
            let conf = (ncc * (0.5 + 0.5 * reliability)).clamp(0.0, 1.0);
            self.bbox = Some(b);
            self.seen_at = Some(t);
            self.confidence = conf.max(0.3);
            return Observation { t, bbox: Some(b), confidence: self.confidence, state: TrackState::Tracked, cut: false };
        }
        // failed: look around where it was (everywhere once it has been gone a while)
        let gone = self.seen_at.is_none_or(|s| (t - s).abs() > COAST_SECONDS);
        if let Some((b, score)) = self.search(frame, &current, gone).filter(|(_, s)| *s >= REDETECT_NCC) {
            return self.seed_keep_template(b, t, score);
        }
        self.hold(t, false, None)
    }

    /// Re-acquired by the template: the template stays (it is what was found).
    fn seed_keep_template(&mut self, b: SubjectBox, t: f64, score: f32) -> Observation {
        self.bbox = Some(b);
        self.seen_at = Some(t);
        self.confidence = score;
        Observation { t, bbox: Some(b), confidence: score, state: TrackState::Redetected, cut: false }
    }

    /// Blend the subject's current look into the template.
    fn learn(&mut self, frame: &Frame, b: &SubjectBox) {
        let Some(tp) = self.template.as_mut() else { return };
        let Some(mut now) = patch(frame, b, tp.w, tp.h) else { return };
        if !normalize(&mut now) {
            return;
        }
        for (a, n) in tp.px.iter_mut().zip(&now) {
            *a = *a * (1.0 - LEARN_RATE) + n * LEARN_RATE;
        }
        normalize(&mut tp.px);
    }

    /// Search the template around `near` (or the whole frame), at sizes around its size: the
    /// best box and its correlation.
    fn search(&self, frame: &Frame, near: &SubjectBox, everywhere: bool) -> Option<(SubjectBox, f32)> {
        let tp = self.template.as_ref()?;
        let b = near.px(frame.w, frame.h);
        let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
        let area = if everywhere {
            [0.0, 0.0, frame.w as f64, frame.h as f64]
        } else {
            let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
            [(cx - bw * 1.5).max(0.0), (cy - bh * 1.5).max(0.0), (cx + bw * 1.5).min(frame.w as f64), (cy + bh * 1.5).min(frame.h as f64)]
        };
        let scales: &[f64] = if everywhere { &[0.85, 1.0, 1.18] } else { &[0.9, 1.0, 1.1] };
        let mut best: Option<([f64; 4], f32)> = None;
        for s in scales {
            if let Some(hit) = search_scale(frame, tp, (bw * s, bh * s), area)
                && best.as_ref().is_none_or(|(_, v)| hit.1 > *v)
            {
                best = Some(hit);
            }
        }
        let (b, score) = best?;
        Some((SubjectBox::from_px(b, frame.w, frame.h)?, score))
    }
}

/// Median of `v` (sorted in place); 0 when empty.
fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Correlation of two equal-length zero-mean unit-norm vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Make `v` zero-mean and unit-norm; false when it is flat.
fn normalize(v: &mut [f32]) -> bool {
    if v.is_empty() {
        return false;
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    v.iter_mut().for_each(|x| *x -= mean);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm.is_nan() || norm <= 1e-4 {
        v.iter_mut().for_each(|x| *x = 0.0);
        return false;
    }
    v.iter_mut().for_each(|x| *x /= norm);
    true
}

/// Template size for a box of `bw`×`bh` pixels: at most [`TEMPLATE_SIDE`] (never larger than the
/// box, so a search never upsamples much), at least 6.
fn template_size(bw: f64, bh: f64) -> (usize, usize) {
    let side = TEMPLATE_SIDE.min(bw.max(bh)).max(6.0);
    let (w, h) = if bw >= bh { (side, side * bh / bw.max(1e-6)) } else { (side * bw / bh.max(1e-6), side) };
    (w.round().clamp(6.0, TEMPLATE_SIDE) as usize, h.round().clamp(6.0, TEMPLATE_SIDE) as usize)
}

/// The grey patch under `b`, resampled to `tw`×`th` from the pyramid level that is just finer
/// than the patch (so it is not aliased).
fn patch(frame: &Frame, b: &SubjectBox, tw: usize, th: usize) -> Option<Vec<f32>> {
    let p = b.px(frame.w, frame.h);
    let (bw, bh) = (p[2] - p[0], p[3] - p[1]);
    if !(bw > 0.5 && bh > 0.5) || tw == 0 || th == 0 {
        return None;
    }
    let ratio = (bw / tw as f64).min(bh / th as f64).max(1.0);
    let level = (ratio.log2().floor().max(0.0) as usize).min(frame.prep.levels.len().saturating_sub(1));
    let g = frame.prep.levels.get(level)?;
    let k = (1u32 << level) as f64;
    let mut out = Vec::with_capacity(tw * th);
    for j in 0..th {
        for i in 0..tw {
            let x = p[0] + (i as f64 + 0.5) * bw / tw as f64 - 0.5;
            let y = p[1] + (j as f64 + 0.5) * bh / th as f64 - 0.5;
            out.push(g.sample((x / k) as f32, (y / k) as f32));
        }
    }
    Some(out)
}

fn template(frame: &Frame, b: &SubjectBox) -> Option<Template> {
    let p = b.px(frame.w, frame.h);
    let (w, h) = template_size(p[2] - p[0], p[3] - p[1]);
    let mut px = patch(frame, b, w, h)?;
    normalize(&mut px).then_some(Template { w, h, px })
}

/// How well the box `b` in `frame` matches the template.
fn match_at(frame: &Frame, tp: &Template, b: &SubjectBox) -> Option<f32> {
    let mut px = patch(frame, b, tp.w, tp.h)?;
    if !normalize(&mut px) {
        return Some(0.0);
    }
    Some(dot(&tp.px, &px))
}

/// Normalised cross-correlation search of `tp` over `area` (frame pixels) for a box of `size`
/// pixels: the area is resampled so the box is the template's size, every window is scored
/// (integral images give each window's mean and variance) on a 2-pixel grid, then refined around
/// the best. Returns the best box (pixels) and its score.
fn search_scale(frame: &Frame, tp: &Template, size: (f64, f64), area: [f64; 4]) -> Option<([f64; 4], f32)> {
    let (bw, bh) = size;
    if !(bw >= 2.0 && bh >= 2.0) {
        return None;
    }
    let f = (tp.w as f64 / bw).min(tp.h as f64 / bh).clamp(1e-3, 2.0);
    let (aw, ah) = (area[2] - area[0], area[3] - area[1]);
    let (sw, sh) = ((aw * f).floor() as usize, (ah * f).floor() as usize);
    // a search image of at most 512×512: bigger areas are not worth it
    if sw < tp.w || sh < tp.h || sw > 512 || sh > 512 {
        return None;
    }
    let level = ((1.0 / f).log2().floor().max(0.0) as usize).min(frame.prep.levels.len().saturating_sub(1));
    let g = frame.prep.levels.get(level)?;
    let k = (1u32 << level) as f64;
    let mut img = Vec::with_capacity(sw * sh);
    for j in 0..sh {
        for i in 0..sw {
            let x = area[0] + (i as f64 + 0.5) / f - 0.5;
            let y = area[1] + (j as f64 + 0.5) / f - 0.5;
            img.push(g.sample((x / k) as f32, (y / k) as f32) as f64);
        }
    }
    // integral images of the values and their squares, (sw + 1) × (sh + 1)
    let iw = sw + 1;
    let mut s1 = vec![0.0f64; iw * (sh + 1)];
    let mut s2 = vec![0.0f64; iw * (sh + 1)];
    for j in 0..sh {
        let (mut r1, mut r2) = (0.0, 0.0);
        for i in 0..sw {
            let v = img.get(j * sw + i).copied().unwrap_or(0.0);
            r1 += v;
            r2 += v * v;
            let at = (j + 1) * iw + i + 1;
            let above = j * iw + i + 1;
            if let (Some(a1), Some(a2)) = (s1.get(above).copied(), s2.get(above).copied()) {
                if let Some(c) = s1.get_mut(at) {
                    *c = a1 + r1;
                }
                if let Some(c) = s2.get_mut(at) {
                    *c = a2 + r2;
                }
            }
        }
    }
    let n = (tp.w * tp.h) as f64;
    let rect = |s: &[f64], x: usize, y: usize| -> f64 {
        let g = |i: usize| s.get(i).copied().unwrap_or(0.0);
        g((y + tp.h) * iw + x + tp.w) - g(y * iw + x + tp.w) - g((y + tp.h) * iw + x) + g(y * iw + x)
    };
    let score = |x: usize, y: usize| -> f32 {
        let sum = rect(&s1, x, y);
        let sq = rect(&s2, x, y);
        let var = sq - sum * sum / n;
        if var <= 1e-9 {
            return -1.0;
        }
        let mut c = 0.0f64;
        for ty in 0..tp.h {
            let row = (y + ty) * sw + x;
            for tx in 0..tp.w {
                c += tp.px.get(ty * tp.w + tx).copied().unwrap_or(0.0) as f64 * img.get(row + tx).copied().unwrap_or(0.0);
            }
        }
        (c / var.sqrt()) as f32
    };
    let (mx, my) = (sw - tp.w, sh - tp.h);
    let mut best = (f32::NEG_INFINITY, 0usize, 0usize);
    for y in (0..=my).step_by(2) {
        for x in (0..=mx).step_by(2) {
            let v = score(x, y);
            if v > best.0 {
                best = (v, x, y);
            }
        }
    }
    let (_, bx, by) = best;
    for y in by.saturating_sub(1)..=(by + 1).min(my) {
        for x in bx.saturating_sub(1)..=(bx + 1).min(mx) {
            let v = score(x, y);
            if v > best.0 {
                best = (v, x, y);
            }
        }
    }
    let (v, x, y) = best;
    if !v.is_finite() {
        return None;
    }
    let cx = area[0] + (x as f64 + tp.w as f64 / 2.0) / f;
    let cy = area[1] + (y as f64 + tp.h as f64 / 2.0) / f;
    Some(([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], v))
}

/// Levels of an image pyramid, as [`Prepared::new`] builds them (5-tap binomial blur, 2×
/// decimation, down to 32 pixels, four levels at most) but computing only the samples the
/// decimation keeps.
fn pyramid(g: Gray) -> Prepared {
    let mut levels = vec![g];
    while levels.len() < 4 {
        let Some(last) = levels.last() else { break };
        if last.w < 32 || last.h < 32 {
            break;
        }
        let next = half(last);
        levels.push(next);
    }
    Prepared { levels }
}

/// One pyramid step: the binomial blur, sampled at the even pixels.
fn half(g: &Gray) -> Gray {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let (w, h) = (g.w, g.h);
    let (nw, nh) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
    if g.px.len() < w * h || w == 0 || h == 0 {
        return Gray { w: nw, h: nh, px: vec![0.0; nw * nh] };
    }
    let cx = |x: isize| x.clamp(0, w as isize - 1) as usize;
    let cy = |y: isize| y.clamp(0, h as isize - 1) as usize;
    // horizontal blur at the kept columns, every row
    let mut tmp = vec![0.0f32; nw * h];
    for y in 0..h {
        let row = &g.px[y * w..(y + 1) * w];
        for x in 0..nw {
            let c = 2 * x as isize;
            tmp[y * nw + x] = K.iter().enumerate().map(|(i, k)| k * row[cx(c + i as isize - 2)]).sum();
        }
    }
    // vertical blur at the kept rows
    let mut px = vec![0.0f32; nw * nh];
    for y in 0..nh {
        let c = 2 * y as isize;
        for x in 0..nw {
            px[y * nw + x] = K.iter().enumerate().map(|(i, k)| k * tmp[cy(c + i as isize - 2) * nw + x]).sum();
        }
    }
    Gray { w: nw, h: nh, px }
}

/// Shi–Tomasi corners in the rectangle `r` (pixels; the same response as
/// [`crate::track::features`]: the smaller eigenvalue of the gradient structure tensor over a
/// 7×7 window, above 1 % of the strongest, 3×3 maxima, `min_dist` apart), strongest first; the
/// window sums come from integral images, so the cost is the rectangle's area.
fn corners(g: &Gray, r: [f64; 4], max: usize, min_dist: f32) -> Vec<[f32; 2]> {
    const RAD: i64 = 3;
    let margin = 9i64;
    let (w, h) = (g.w as i64, g.h as i64);
    if g.px.len() < g.w * g.h || w < 2 * margin + 3 || h < 2 * margin + 3 {
        return Vec::new();
    }
    let x0 = (r[0].floor() as i64).max(margin);
    let y0 = (r[1].floor() as i64).max(margin);
    let x1 = (r[2].ceil() as i64).min(w - 1 - margin);
    let y1 = (r[3].ceil() as i64).min(h - 1 - margin);
    if x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }
    // gradients over the rectangle and the window's reach around it
    let (gx0, gy0) = (x0 - RAD, y0 - RAD);
    let (gw, gh) = ((x1 - x0 + 1 + 2 * RAD) as usize, (y1 - y0 + 1 + 2 * RAD) as usize);
    let at = |x: i64, y: i64| g.px[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize];
    let iw = gw + 1;
    let mut sa = vec![0.0f64; iw * (gh + 1)];
    let mut sb = vec![0.0f64; iw * (gh + 1)];
    let mut sc = vec![0.0f64; iw * (gh + 1)];
    for j in 0..gh {
        let (mut ra, mut rb, mut rc) = (0.0, 0.0, 0.0);
        let y = gy0 + j as i64;
        for i in 0..gw {
            let x = gx0 + i as i64;
            let ix = ((at(x + 1, y) - at(x - 1, y)) * 0.5) as f64;
            let iy = ((at(x, y + 1) - at(x, y - 1)) * 0.5) as f64;
            ra += ix * ix;
            rb += ix * iy;
            rc += iy * iy;
            let o = (j + 1) * iw + i + 1;
            sa[o] = sa[o - iw] + ra;
            sb[o] = sb[o - iw] + rb;
            sc[o] = sc[o - iw] + rc;
        }
    }
    let (bw, bh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
    let win = (2 * RAD + 1) as usize;
    let sum = |s: &[f64], i: usize, j: usize| s[(j + win) * iw + i + win] - s[j * iw + i + win] - s[(j + win) * iw + i] + s[j * iw + i];
    let mut resp = vec![0.0f32; bw * bh];
    let mut best = 0.0f32;
    for j in 0..bh {
        for i in 0..bw {
            let (a, b, c) = (sum(&sa, i, j), sum(&sb, i, j), sum(&sc, i, j));
            let tr = (a + c) * 0.5;
            let v = (tr - ((a - c) * (a - c) * 0.25 + b * b).sqrt()) as f32;
            resp[j * bw + i] = v;
            best = best.max(v);
        }
    }
    if best <= 1e-9 {
        return Vec::new();
    }
    let thr = best * 0.01;
    let mut cand: Vec<(f32, i64, i64)> = Vec::new();
    for j in 1..bh.saturating_sub(1) {
        for i in 1..bw.saturating_sub(1) {
            let v = resp[j * bw + i];
            if v < thr {
                continue;
            }
            let peak = (j - 1..=j + 1).all(|jj| (i - 1..=i + 1).all(|ii| (ii == i && jj == j) || resp[jj * bw + ii] <= v));
            if peak {
                cand.push((v, x0 + i as i64, y0 + j as i64));
            }
        }
    }
    cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let md2 = min_dist * min_dist;
    let mut out: Vec<[f32; 2]> = Vec::new();
    for (_, x, y) in cand {
        let p = [x as f32, y as f32];
        if out.iter().all(|q| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) >= md2) {
            out.push(p);
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

/// Correlation of the 7×7 patches around `p` in `a` and `q` in `b` (Median Flow's second vote).
fn point_ncc(a: &Gray, b: &Gray, p: [f32; 2], q: [f32; 2]) -> f32 {
    let mut pa = Vec::with_capacity(49);
    let mut pb = Vec::with_capacity(49);
    for dy in -3..=3 {
        for dx in -3..=3 {
            pa.push(a.sample(p[0] + dx as f32, p[1] + dy as f32));
            pb.push(b.sample(q[0] + dx as f32, q[1] + dy as f32));
        }
    }
    if normalize(&mut pa) && normalize(&mut pb) { dot(&pa, &pb) } else { 0.0 }
}

/// One Median Flow step: where the box `b` of `prev` went in `next`, and the share of its points
/// that tracked reliably. `None` is a failure.
fn flow_step(prev: &Frame, next: &Frame, b: &SubjectBox) -> Option<(SubjectBox, f32)> {
    if prev.w != next.w || prev.h != next.h {
        return None;
    }
    let p = b.px(prev.w, prev.h);
    let (bw, bh) = (p[2] - p[0], p[3] - p[1]);
    // the inner part of the box: mostly the subject, not the background around it
    let inner = [p[0] + bw * 0.15, p[1] + bh * 0.15, p[2] - bw * 0.15, p[3] - bh * 0.15];
    let min_dist = ((inner[2] - inner[0]).min(inner[3] - inner[1]) / 10.0).clamp(2.0, 12.0) as f32;
    let mut pts = corners(prev.prep.levels.first()?, inner, MAX_POINTS, min_dist);
    if pts.len() < 12 {
        // little texture: a grid too (flat points fail in Lucas–Kanade and drop out)
        for j in 0..6 {
            for i in 0..6 {
                let x = inner[0] + (inner[2] - inner[0]) * (i as f64 + 0.5) / 6.0;
                let y = inner[1] + (inner[3] - inner[1]) * (j as f64 + 0.5) / 6.0;
                pts.push([x as f32, y as f32]);
            }
        }
    }
    let (a, bl) = (&prev.prep.levels, &next.prep.levels);
    let (ga, gb) = (a.first()?, bl.first()?);
    // (p, q, forward–backward error, patch correlation)
    let mut tracks: Vec<([f32; 2], [f32; 2], f32, f32)> = pts
        .iter()
        .filter_map(|&p| {
            let q = lk(a, bl, p)?;
            let back = lk(bl, a, q)?;
            let fb = ((back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2)).sqrt();
            fb.is_finite().then(|| (p, q, fb, point_ncc(ga, gb, p, q)))
        })
        .collect();
    if tracks.len() < 4 {
        return None;
    }
    let scale_px = (prev.w.max(prev.h) as f32 / 384.0).max(1.0);
    let mut fbs: Vec<f32> = tracks.iter().map(|t| t.2).collect();
    let med_fb = median(&mut fbs);
    if med_fb > FB_FAIL * scale_px {
        return None;
    }
    let reliable = tracks.iter().filter(|t| t.2 < 1.0 * scale_px).count() as f32 / tracks.len() as f32;
    let mut nccs: Vec<f32> = tracks.iter().map(|t| t.3).collect();
    let med_ncc = median(&mut nccs);
    let best: Vec<([f32; 2], [f32; 2], f32, f32)> = tracks.iter().copied().filter(|t| t.2 <= med_fb && t.3 >= med_ncc).collect();
    if best.len() >= 4 {
        tracks = best;
    } else {
        tracks.retain(|t| t.2 <= med_fb);
    }
    if tracks.len() < 3 {
        return None;
    }
    let mut ratios = Vec::new();
    let pick: Vec<&([f32; 2], [f32; 2], f32, f32)> = tracks.iter().take(30).collect();
    for (i, s) in pick.iter().enumerate() {
        for e in pick.iter().skip(i + 1) {
            let d0 = ((s.0[0] - e.0[0]).powi(2) + (s.0[1] - e.0[1]).powi(2)).sqrt();
            let d1 = ((s.1[0] - e.1[0]).powi(2) + (s.1[1] - e.1[1]).powi(2)).sqrt();
            if d0 > 4.0 && d1.is_finite() {
                ratios.push(d1 / d0);
            }
        }
    }
    let scale = if ratios.is_empty() { 1.0 } else { (median(&mut ratios) as f64).clamp(0.85, 1.18) };
    // each point's vote for the new centre, q − s·(p − c): the median of the displacements alone
    // drifts when the subject grows and its points are not spread evenly around the centre
    let (ox, oy) = ((p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0);
    let mut vx: Vec<f32> = tracks.iter().map(|t| (t.1[0] as f64 - scale * (t.0[0] as f64 - ox)) as f32).collect();
    let mut vy: Vec<f32> = tracks.iter().map(|t| (t.1[1] as f64 - scale * (t.0[1] as f64 - oy)) as f32).collect();
    let (cx, cy) = (median(&mut vx) as f64, median(&mut vy) as f64);
    let (nw, nh) = (bw * scale, bh * scale);
    let moved = SubjectBox::from_px([cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0], next.w, next.h)?;
    Some((moved, reliable))
}

/// sRGB (0..255) → CIELAB (D65).
fn lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let lin = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (r, g, b) = (lin(r), lin(g), lin(b));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f32| if t > 0.008_856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Skin tones in Cb/Cr (Chai & Ngan 1999).
fn skin(r: u8, g: u8, b: u8) -> bool {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    (77.0..=127.0).contains(&cb) && (133.0..=173.0).contains(&cr)
}

/// The salient region of an RGBA8 frame (frequency-tuned saliency with a centre and a skin-tone
/// prior, see the module docs), or `None` when nothing stands out (a flat frame).
pub fn salient_box(rgba: &[u8], w: usize, h: usize) -> Option<SubjectBox> {
    if w < 2 || h < 2 || rgba.len() < w.checked_mul(h)?.checked_mul(4)? {
        return None;
    }
    // a grid of at most 64 cells across, each the average of its pixels
    let gw = w.min(64);
    let gh = ((h as f64 * gw as f64 / w as f64).round() as usize).clamp(1, 64);
    let mut acc = vec![[0.0f32; 4]; gw * gh];
    let mut skin_n = vec![0u32; gw * gh];
    let mut cnt = vec![0u32; gw * gh];
    let (sx, sy) = ((w / 256).max(1), (h / 256).max(1));
    for y in (0..h).step_by(sy) {
        let cy = (y * gh / h).min(gh - 1);
        for x in (0..w).step_by(sx) {
            let cx = (x * gw / w).min(gw - 1);
            let i = (y * w + x) * 4;
            let Some(px) = rgba.get(i..i + 3) else { continue };
            let (r, g, b) = (px[0], px[1], px[2]);
            let l = lab(r, g, b);
            let c = cy * gw + cx;
            if let Some(a) = acc.get_mut(c) {
                a[0] += l[0];
                a[1] += l[1];
                a[2] += l[2];
            }
            if skin(r, g, b)
                && let Some(s) = skin_n.get_mut(c)
            {
                *s += 1;
            }
            if let Some(n) = cnt.get_mut(c) {
                *n += 1;
            }
        }
    }
    let cells: Vec<[f32; 3]> =
        acc.iter().zip(&cnt).map(|(a, n)| if *n > 0 { [a[0] / *n as f32, a[1] / *n as f32, a[2] / *n as f32] } else { [0.0; 3] }).collect();
    // a slight blur (binomial 3×3) against noise and texture, as in the paper
    let at = |x: i64, y: i64| -> [f32; 3] {
        let x = x.clamp(0, gw as i64 - 1) as usize;
        let y = y.clamp(0, gh as i64 - 1) as usize;
        cells.get(y * gw + x).copied().unwrap_or([0.0; 3])
    };
    let mut mean = [0.0f32; 3];
    for c in &cells {
        mean[0] += c[0];
        mean[1] += c[1];
        mean[2] += c[2];
    }
    let n = cells.len().max(1) as f32;
    let mean = [mean[0] / n, mean[1] / n, mean[2] / n];
    let mut sal = vec![0.0f32; gw * gh];
    let mut peak = 0.0f32;
    for y in 0..gh {
        for x in 0..gw {
            let mut b = [0.0f32; 3];
            for (dy, wy) in [(-1i64, 1.0f32), (0, 2.0), (1, 1.0)] {
                for (dx, wx) in [(-1i64, 1.0f32), (0, 2.0), (1, 1.0)] {
                    let v = at(x as i64 + dx, y as i64 + dy);
                    for (o, c) in b.iter_mut().zip(v) {
                        *o += c * wx * wy / 16.0;
                    }
                }
            }
            let d = ((b[0] - mean[0]).powi(2) + (b[1] - mean[1]).powi(2) + (b[2] - mean[2]).powi(2)).sqrt();
            peak = peak.max(d);
            let (u, v) = ((x as f32 + 0.5) / gw as f32 - 0.5, (y as f32 + 0.5) / gh as f32 - 0.5);
            let centre = (-(u * u + v * v) / (2.0 * 0.3 * 0.3)).exp();
            let i = y * gw + x;
            let skin_share = skin_n.get(i).copied().unwrap_or(0) as f32 / cnt.get(i).copied().unwrap_or(0).max(1) as f32;
            if let Some(s) = sal.get_mut(i) {
                *s = d * centre * (1.0 + 0.6 * skin_share);
            }
        }
    }
    // nothing stands out by more than a just-noticeable difference or so
    if peak < 4.0 {
        return None;
    }
    let avg = sal.iter().sum::<f32>() / sal.len().max(1) as f32;
    let thr = 2.0 * avg;
    let (mut sw, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
    for (i, s) in sal.iter().enumerate() {
        if *s >= thr && *s > 0.0 {
            let (x, y) = ((i % gw) as f64 + 0.5, (i / gw) as f64 + 0.5);
            sw += *s as f64;
            mx += *s as f64 * x;
            my += *s as f64 * y;
        }
    }
    if sw <= 0.0 {
        return None;
    }
    let (mx, my) = (mx / sw, my / sw);
    let (mut vx, mut vy) = (0.0f64, 0.0f64);
    for (i, s) in sal.iter().enumerate() {
        if *s >= thr && *s > 0.0 {
            let (x, y) = ((i % gw) as f64 + 0.5, (i / gw) as f64 + 0.5);
            vx += *s as f64 * (x - mx).powi(2);
            vy += *s as f64 * (y - my).powi(2);
        }
    }
    let (sdx, sdy) = ((vx / sw).sqrt() / gw as f64, (vy / sw).sqrt() / gh as f64);
    SubjectBox::new(mx / gw as f64, my / gh as f64, (3.0 * sdx).clamp(0.15, 0.85), (3.0 * sdy).clamp(0.15, 0.85))
}

/// An RGBA8 frame scaled down (area average) so its longest side is at most `max_side`; the frame
/// itself when it is small enough. `None` for a buffer shorter than `w`×`h`.
pub fn downscale_rgba(rgba: &[u8], w: usize, h: usize, max_side: usize) -> Option<(Vec<u8>, usize, usize)> {
    let len = w.checked_mul(h)?.checked_mul(4)?;
    let src = rgba.get(..len)?;
    if w == 0 || h == 0 {
        return None;
    }
    let max_side = max_side.max(MIN_SIDE);
    if w.max(h) <= max_side {
        return Some((src.to_vec(), w, h));
    }
    let k = max_side as f64 / w.max(h) as f64;
    let (nw, nh) = (((w as f64 * k).round() as usize).max(1), ((h as f64 * k).round() as usize).max(1));
    let mut out = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        let (y0, y1) = (y * h / nh, ((y + 1) * h / nh).max(y * h / nh + 1).min(h));
        for x in 0..nw {
            let (x0, x1) = (x * w / nw, ((x + 1) * w / nw).max(x * w / nw + 1).min(w));
            let mut sum = [0u32; 4];
            let mut n = 0u32;
            for yy in y0..y1 {
                let row = yy * w;
                for xx in x0..x1 {
                    let i = (row + xx) * 4;
                    if let Some(p) = src.get(i..i + 4) {
                        for (o, c) in sum.iter_mut().zip(p) {
                            *o += *c as u32;
                        }
                        n += 1;
                    }
                }
            }
            let o = (y * nw + x) * 4;
            if let Some(d) = out.get_mut(o..o + 4) {
                for (o, c) in d.iter_mut().zip(sum) {
                    *o = (c / n.max(1)) as u8;
                }
            }
        }
    }
    Some((out, nw, nh))
}

#[cfg(test)]
#[path = "subject_tests.rs"]
mod tests;
