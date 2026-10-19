//! From per-frame detections to tracks: linking (overlap, then nearness for fast motion), short
//! gaps filled, sizes and positions smoothed, and the rules that decide which tracks are hidden
//! (a face large enough to be recognised, a plate large enough to read, seen often enough or
//! clearly enough, not inside an excluded box).
//!
//! Everything here is in *normalised picture coordinates*: `x` and `w` are shares of the
//! picture's width, `y` and `h` of its height, times are seconds of media time. Plain Rust, no
//! model, so it is tested with synthetic detections and builds everywhere.

use serde::{Deserialize, Serialize};

use crate::post::iou;
use crate::{Detection, Kind};

/// Linking and smoothing settings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkParams {
    /// Longest stretch, seconds, a track may go unseen and still continue (the gap is filled).
    pub max_gap: f64,
    /// Overlap with the track's predicted box that links a detection to it.
    pub min_iou: f32,
    /// Distance between centres, in box sizes, that links a detection when the boxes do not
    /// overlap enough (fast motion), if the sizes are within a factor of two.
    pub max_jump: f64,
    /// Gaussian smoothing of the boxes over time, seconds (σ); 0: none.
    pub smooth: f64,
}

impl Default for LinkParams {
    fn default() -> Self {
        Self { max_gap: 0.7, min_iou: 0.2, max_jump: 0.8, smooth: 0.08 }
    }
}

/// Which tracks are hidden.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeepParams {
    /// A face is hidden when its box is at least this share of the picture's height at some
    /// point (smaller faces are not recognisable).
    pub min_face_height: f64,
    /// A plate is hidden when its box is at least this share of the picture's width at some
    /// point (smaller plates cannot be read).
    pub min_plate_width: f64,
    /// Detections a track needs…
    pub min_hits: usize,
    /// …unless one of them scored at least this.
    pub sure_score: f32,
}

impl Default for KeepParams {
    fn default() -> Self {
        Self { min_face_height: 0.05, min_plate_width: 0.025, min_hits: 2, sure_score: 0.9 }
    }
}

/// A box `[x, y, w, h]` (top-left corner and size), normalised.
pub type BoxN = [f64; 4];

/// One moment of a track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackPoint {
    /// Media time, seconds.
    pub t: f64,
    pub bbox: BoxN,
    /// A plate's corners (clockwise from the top left), normalised; empty for faces.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corners: Vec<[f64; 2]>,
    /// Detected at this moment (false: filled in across a gap).
    pub seen: bool,
}

/// One face or plate followed over time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub kind: Kind,
    /// Its moments in time order, one per analysed frame from its first sighting to its last.
    pub points: Vec<TrackPoint>,
    /// Detections linked to it.
    pub hits: usize,
    pub best_score: f32,
}

impl Track {
    pub fn start(&self) -> f64 {
        self.points.first().map_or(0.0, |p| p.t)
    }
    pub fn end(&self) -> f64 {
        self.points.last().map_or(0.0, |p| p.t)
    }
    /// The largest box height (faces) or width (plates) over the track.
    pub fn max_size(&self) -> f64 {
        self.points.iter().map(|p| if self.kind == Kind::Face { p.bbox[3] } else { p.bbox[2] }).fold(0.0, f64::max)
    }
    /// The moment nearest `t`.
    pub fn at(&self, t: f64) -> Option<&TrackPoint> {
        self.points.iter().min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
    }
}

/// The detections of one analysed frame, normalised to its picture.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// Media time, seconds.
    pub t: f64,
    pub detections: Vec<Detection>,
}

fn to_box(b: &[f32; 4]) -> BoxN {
    [f64::from(b[0]), f64::from(b[1]), f64::from(b[2]), f64::from(b[3])]
}

fn iou64(a: &BoxN, b: &BoxN) -> f64 {
    let f = |x: &BoxN| [x[0] as f32, x[1] as f32, x[2] as f32, x[3] as f32];
    f64::from(iou(&f(a), &f(b)))
}

/// A track being built.
struct Open {
    kind: Kind,
    /// (frame index, box, corners, score) of each sighting.
    seen: Vec<(usize, BoxN, Vec<[f64; 2]>, f32)>,
}

impl Open {
    /// Where the track's box should be at time `t`: the last box moved by the last velocity
    /// (from the last two sightings), half of it so a jittery pair does not throw it off.
    fn predict(&self, frames: &[Frame], t: f64) -> Option<BoxN> {
        let (i1, b1, _, _) = self.seen.last()?;
        let t1 = frames.get(*i1)?.t;
        let Some((i0, b0, _, _)) = self.seen.len().checked_sub(2).and_then(|k| self.seen.get(k)) else { return Some(*b1) };
        let t0 = frames.get(*i0)?.t;
        let dt = t1 - t0;
        if dt.is_nan() || dt <= 1e-6 {
            return Some(*b1);
        }
        let k = 0.5 * (t - t1) / dt;
        Some([b1[0] + (b1[0] - b0[0]) * k, b1[1] + (b1[1] - b0[1]) * k, b1[2], b1[3]])
    }
}

/// How well detection `d` continues a track predicted at `p`: its overlap, or failing that a
/// nearness score below `min_iou` when the centres are within `max_jump` box sizes and the sizes
/// within a factor of two; `None` when it does not.
fn affinity(p: &BoxN, d: &BoxN, lp: &LinkParams) -> Option<f64> {
    let o = iou64(p, d);
    if o >= f64::from(lp.min_iou) {
        return Some(o);
    }
    let (w, h) = ((p[2] + d[2]) / 2.0, (p[3] + d[3]) / 2.0);
    if !(w > 0.0 && h > 0.0) {
        return None;
    }
    let ratio = (p[2] * p[3]) / (d[2] * d[3]).max(1e-12);
    if !(0.25..=4.0).contains(&ratio) {
        return None;
    }
    let (dx, dy) = ((p[0] + p[2] / 2.0 - d[0] - d[2] / 2.0) / w, (p[1] + p[3] / 2.0 - d[1] - d[3] / 2.0) / h);
    let dist = (dx * dx + dy * dy).sqrt();
    (dist <= lp.max_jump).then(|| f64::from(lp.min_iou) * (1.0 - dist / lp.max_jump).max(0.0) * 0.99)
}

/// Link the detections of `frames` (in time order, normalised) into tracks, one kind at a time.
/// A detection joins the open track it continues best (greedy, best pairs first); one that
/// continues none starts a track; a track unseen for longer than `max_gap` is closed. Each track
/// then gets a point at every analysed frame from its first sighting to its last (gaps filled
/// by straight lines) and is smoothed.
pub fn link(frames: &[Frame], lp: &LinkParams) -> Vec<Track> {
    let mut done: Vec<Open> = Vec::new();
    let mut open: Vec<Open> = Vec::new();
    for (fi, f) in frames.iter().enumerate() {
        // close the tracks unseen for too long
        let (keep, closed): (Vec<Open>, Vec<Open>) =
            open.into_iter().partition(|o| o.seen.last().and_then(|(i, ..)| frames.get(*i)).is_some_and(|last| f.t - last.t <= lp.max_gap + 1e-9));
        open = keep;
        done.extend(closed);
        let dets: Vec<(BoxN, Vec<[f64; 2]>, f32, Kind)> = f
            .detections
            .iter()
            .filter(|d| d.bbox.iter().all(|v| v.is_finite()) && d.bbox[2] > 0.0 && d.bbox[3] > 0.0)
            .map(|d| (to_box(&d.bbox), d.points.iter().map(|p| [f64::from(p[0]), f64::from(p[1])]).collect(), d.score, d.kind))
            .collect();
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (oi, o) in open.iter().enumerate() {
            let Some(p) = o.predict(frames, f.t) else { continue };
            for (di, d) in dets.iter().enumerate() {
                if d.3 != o.kind {
                    continue;
                }
                if let Some(a) = affinity(&p, &d.0, lp) {
                    pairs.push((a, oi, di));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut used_o, mut used_d) = (vec![false; open.len()], vec![false; dets.len()]);
        for (_, oi, di) in pairs {
            if used_o.get(oi).copied().unwrap_or(true) || used_d.get(di).copied().unwrap_or(true) {
                continue;
            }
            used_o[oi] = true;
            used_d[di] = true;
            if let (Some(o), Some(d)) = (open.get_mut(oi), dets.get(di)) {
                o.seen.push((fi, d.0, d.1.clone(), d.2));
            }
        }
        for (di, d) in dets.into_iter().enumerate() {
            if !used_d.get(di).copied().unwrap_or(true) {
                open.push(Open { kind: d.3, seen: vec![(fi, d.0, d.1, d.2)] });
            }
        }
    }
    done.extend(open);
    let mut tracks: Vec<Track> = done.into_iter().filter_map(|o| fill(frames, &o)).map(|t| smooth(t, lp.smooth)).collect();
    tracks.sort_by(|a, b| a.start().total_cmp(&b.start()));
    tracks
}

/// A point at every analysed frame from the first sighting to the last.
fn fill(frames: &[Frame], o: &Open) -> Option<Track> {
    let (first, ..) = o.seen.first()?;
    let (last, ..) = o.seen.last()?;
    let mut points = Vec::with_capacity(last.saturating_sub(*first) + 1);
    let mut k = 0usize;
    for fi in *first..=*last {
        let t = frames.get(fi)?.t;
        while o.seen.get(k + 1).is_some_and(|s| s.0 <= fi) {
            k += 1;
        }
        let a = o.seen.get(k)?;
        if a.0 == fi {
            points.push(TrackPoint { t, bbox: a.1, corners: a.2.clone(), seen: true });
            continue;
        }
        // between sighting k and k + 1: a straight line
        let b = o.seen.get(k + 1).unwrap_or(a);
        let (ta, tb) = (frames.get(a.0)?.t, frames.get(b.0)?.t);
        let u = if tb - ta > 1e-9 { ((t - ta) / (tb - ta)).clamp(0.0, 1.0) } else { 0.0 };
        let lerp = |x: f64, y: f64| x + (y - x) * u;
        let bbox = [lerp(a.1[0], b.1[0]), lerp(a.1[1], b.1[1]), lerp(a.1[2], b.1[2]), lerp(a.1[3], b.1[3])];
        let corners = if a.2.len() == b.2.len() { a.2.iter().zip(&b.2).map(|(p, q)| [lerp(p[0], q[0]), lerp(p[1], q[1])]).collect() } else { a.2.clone() };
        points.push(TrackPoint { t, bbox, corners, seen: false });
    }
    let best_score = o.seen.iter().map(|s| s.3).fold(0.0f32, f32::max);
    Some(Track { kind: o.kind, points, hits: o.seen.len(), best_score })
}

/// Gaussian-weighted local linear smoothing of the box centres, sizes and corners over time
/// (σ = `sigma` seconds): each value is the value at its moment of the straight line fitted to
/// its neighbours, so jitter goes while steady motion keeps its place, at the ends of a track
/// too (a plain weighted mean would pull the ends inwards).
fn smooth(mut t: Track, sigma: f64) -> Track {
    if sigma.is_nan() || sigma <= 0.0 || t.points.len() < 3 {
        return t;
    }
    let src = t.points.clone();
    let reach = 3.0 * sigma;
    // the values smoothed: centre x, centre y, w, h, then the corners' coordinates
    let values = |q: &TrackPoint| -> Vec<f64> {
        let mut v = vec![q.bbox[0] + q.bbox[2] / 2.0, q.bbox[1] + q.bbox[3] / 2.0, q.bbox[2], q.bbox[3]];
        v.extend(q.corners.iter().flat_map(|c| [c[0], c[1]]));
        v
    };
    for (i, p) in t.points.iter_mut().enumerate() {
        let n = 4 + 2 * p.corners.len();
        let (mut s0, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0f64);
        let (mut t0, mut t1) = (vec![0.0f64; n], vec![0.0f64; n]);
        // the neighbours within 3σ, on both sides
        let lo = src.get(..i).unwrap_or(&[]).iter().rposition(|q| p.t - q.t > reach).map_or(0, |k| k + 1);
        let hi = src.get(i..).unwrap_or(&[]).iter().position(|q| q.t - p.t > reach).map_or(src.len(), |k| i + k);
        for q in src.get(lo..hi).unwrap_or(&[]) {
            let v = values(q);
            if v.len() != n {
                continue;
            }
            let d = q.t - p.t;
            let w = (-0.5 * (d / sigma) * (d / sigma)).exp();
            s0 += w;
            s1 += w * d;
            s2 += w * d * d;
            for ((a, b), x) in t0.iter_mut().zip(t1.iter_mut()).zip(&v) {
                *a += w * x;
                *b += w * d * x;
            }
        }
        if s0.is_nan() || s0 <= 0.0 {
            continue;
        }
        let den = s0 * s2 - s1 * s1;
        let fit: Vec<f64> = t0.iter().zip(&t1).map(|(a, b)| if den.abs() > 1e-12 * s0 * s0 { (s2 * a - s1 * b) / den } else { a / s0 }).collect();
        if fit.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let (cx, cy, w, h) = (fit[0], fit[1], fit[2].max(0.0), fit[3].max(0.0));
        p.bbox = [cx - w / 2.0, cy - h / 2.0, w, h];
        p.corners = fit[4..].as_chunks::<2>().0.to_vec();
    }
    t
}

/// An area never hidden: a track whose box centre lies inside it (at `t`, if given, within a
/// second of it; else at any time) is left alone — the subject a video is about (a robot's face).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Exclude {
    pub t: Option<f64>,
    pub bbox: BoxN,
}

/// How far, seconds, an [`Exclude`] with a time reaches.
pub const EXCLUDE_REACH: f64 = 1.0;

impl Exclude {
    fn covers(&self, p: &TrackPoint) -> bool {
        let (cx, cy) = (p.bbox[0] + p.bbox[2] / 2.0, p.bbox[1] + p.bbox[3] / 2.0);
        let b = &self.bbox;
        cx >= b[0] && cx <= b[0] + b[2] && cy >= b[1] && cy <= b[1] + b[3]
    }

    /// The exclusion applies to `track`.
    pub fn applies(&self, track: &Track) -> bool {
        match self.t {
            Some(t) => track.at(t).is_some_and(|p| (p.t - t).abs() <= EXCLUDE_REACH && self.covers(p)),
            None => track.points.iter().any(|p| self.covers(p)),
        }
    }
}

/// Why a track is not hidden ([`verdict`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Skip {
    /// A face never as tall as `min_face_height` (or a plate never as wide as `min_plate_width`).
    TooSmall,
    /// Seen fewer than `min_hits` times and never surely.
    TooBrief,
    /// Inside an excluded box.
    Excluded,
    /// The vision model said it is not a real person's face.
    NotAPerson,
}

/// Whether `track` is hidden under `kp` and `exclude`: `None` hides it.
pub fn verdict(track: &Track, kp: &KeepParams, exclude: &[Exclude]) -> Option<Skip> {
    let big = match track.kind {
        Kind::Face => track.max_size() >= kp.min_face_height,
        Kind::Plate => track.max_size() >= kp.min_plate_width,
    };
    if !big {
        return Some(Skip::TooSmall);
    }
    if track.hits < kp.min_hits.max(1) && track.best_score < kp.sure_score {
        return Some(Skip::TooBrief);
    }
    if exclude.iter().any(|e| e.applies(track)) {
        return Some(Skip::Excluded);
    }
    None
}

/// `b` grown by `padding` of its size on every side (0.2: a fifth of its width left and right,
/// a fifth of its height above and below).
pub fn padded(b: &BoxN, padding: f64) -> BoxN {
    let p = if padding.is_finite() { padding.clamp(0.0, 2.0) } else { 0.0 };
    [b[0] - b[2] * p, b[1] - b[3] * p, b[2] * (1.0 + 2.0 * p), b[3] * (1.0 + 2.0 * p)]
}

/// Corners pushed out from their centroid by `padding` of the box size (a plate's quad).
pub fn padded_corners(c: &[[f64; 2]], b: &BoxN, padding: f64) -> Vec<[f64; 2]> {
    let p = if padding.is_finite() { padding.clamp(0.0, 2.0) } else { 0.0 };
    let n = c.len().max(1) as f64;
    let (mx, my) = (c.iter().map(|q| q[0]).sum::<f64>() / n, c.iter().map(|q| q[1]).sum::<f64>() / n);
    c.iter()
        .map(|q| {
            let (dx, dy) = (q[0] - mx, q[1] - my);
            // grow by a share of the box (not of the distance), so thin quads widen too
            let (sx, sy) = (dx.signum() * b[2] * p, dy.signum() * b[3] * p);
            [q[0] + sx, q[1] + sy]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: f32, y: f32, s: f32, score: f32) -> Detection {
        Detection { kind: Kind::Face, score, bbox: [x, y, s, s], points: Vec::new() }
    }

    fn frames(n: usize, fps: f64, at: impl Fn(usize) -> Vec<Detection>) -> Vec<Frame> {
        (0..n).map(|i| Frame { t: i as f64 / fps, detections: at(i) }).collect()
    }

    #[test]
    fn two_faces_crossing_paths_keep_their_tracks() {
        // one face moves right, the other left; they pass at frame 10
        let f = frames(21, 10.0, |i| {
            let u = i as f32 / 20.0;
            vec![face(0.1 + 0.6 * u, 0.2, 0.1, 0.9), face(0.7 - 0.6 * u, 0.5, 0.1, 0.8)]
        });
        let t = link(&f, &LinkParams { smooth: 0.0, ..LinkParams::default() });
        assert_eq!(t.len(), 2);
        for tr in &t {
            assert_eq!(tr.points.len(), 21);
            assert_eq!(tr.hits, 21);
            let ys: Vec<f64> = tr.points.iter().map(|p| p.bbox[1]).collect();
            assert!(ys.iter().all(|y| (y - ys[0]).abs() < 1e-6), "a track jumped between faces: {ys:?}");
        }
    }

    #[test]
    fn short_gaps_are_filled_and_long_ones_split() {
        // seen at frames 0–4, missed 5–7, seen 8–12 (10 fps: a 0.3 s gap)
        let f = frames(13, 10.0, |i| if (5..8).contains(&i) { vec![] } else { vec![face(0.1 + 0.02 * i as f32, 0.3, 0.1, 0.9)] });
        let t = link(&f, &LinkParams { smooth: 0.0, ..LinkParams::default() });
        assert_eq!(t.len(), 1);
        let tr = &t[0];
        assert_eq!(tr.points.len(), 13);
        assert_eq!(tr.hits, 10);
        assert!(!tr.points[6].seen);
        assert!((tr.points[6].bbox[0] - (0.1 + 0.02 * 6.0)).abs() < 1e-6, "filled on the straight line");
        // a 1.5 s gap closes the track
        let f = frames(30, 10.0, |i| if (5..20).contains(&i) { vec![] } else { vec![face(0.3, 0.3, 0.1, 0.9)] });
        assert_eq!(link(&f, &LinkParams::default()).len(), 2);
    }

    #[test]
    fn fast_motion_links_by_nearness() {
        // a face moving three quarters of its width every frame (overlapping too little)
        let f = frames(10, 10.0, |i| vec![face(0.05 + 0.045 * i as f32, 0.4, 0.06, 0.9)]);
        let t = link(&f, &LinkParams { smooth: 0.0, ..LinkParams::default() });
        assert_eq!(t.len(), 1, "{t:?}");
        // but a far jump starts another track
        let f = frames(2, 10.0, |i| vec![face(0.05 + 0.5 * i as f32, 0.4, 0.06, 0.9)]);
        assert_eq!(link(&f, &LinkParams::default()).len(), 2);
    }

    #[test]
    fn smoothing_removes_jitter_without_lag() {
        let f = frames(40, 30.0, |i| vec![face(0.4 + if i % 2 == 0 { 0.01 } else { -0.01 }, 0.4, 0.1, 0.9)]);
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1);
        let mid = &t[0].points[20];
        assert!((mid.bbox[0] - 0.4).abs() < 0.003, "{}", mid.bbox[0]);
        // a steady motion keeps its place (no lag)
        let f = frames(40, 30.0, |i| vec![face(0.1 + 0.01 * i as f32, 0.4, 0.1, 0.9)]);
        let t = link(&f, &LinkParams::default());
        for (i, p) in t[0].points.iter().enumerate() {
            assert!((p.bbox[0] - (0.1 + 0.01 * i as f64)).abs() < 1e-3, "frame {i}: {}", p.bbox[0]);
        }
    }

    #[test]
    fn thresholds_decide_what_is_hidden() {
        let kp = KeepParams::default();
        let mk = |h: f64, hits: usize, score: f32| Track {
            kind: Kind::Face,
            points: vec![TrackPoint { t: 1.0, bbox: [0.4, 0.4, h * 0.6, h], corners: vec![], seen: true }],
            hits,
            best_score: score,
        };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[]), None);
        assert_eq!(verdict(&mk(0.03, 5, 0.99), &kp, &[]), Some(Skip::TooSmall));
        assert_eq!(verdict(&mk(0.2, 1, 0.7), &kp, &[]), Some(Skip::TooBrief));
        assert_eq!(verdict(&mk(0.2, 1, 0.95), &kp, &[]), None, "one sure sighting is enough");
        // an excluded box at the track's time, and one that is too far away in time
        let ex = Exclude { t: Some(1.2), bbox: [0.3, 0.3, 0.4, 0.4] };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[ex]), Some(Skip::Excluded));
        let late = Exclude { t: Some(5.0), ..ex };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[late]), None);
        let elsewhere = Exclude { t: None, bbox: [0.0, 0.0, 0.1, 0.1] };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[elsewhere]), None);
        // plates by width
        let mut p = mk(0.02, 3, 0.8);
        p.kind = Kind::Plate;
        p.points[0].bbox = [0.1, 0.1, 0.03, 0.01];
        assert_eq!(verdict(&p, &kp, &[]), None);
        p.points[0].bbox[2] = 0.01;
        assert_eq!(verdict(&p, &kp, &[]), Some(Skip::TooSmall));
    }

    #[test]
    fn padding_grows_boxes_and_quads() {
        let p = padded(&[0.4, 0.4, 0.1, 0.2], 0.2);
        for (a, b) in p.iter().zip([0.38, 0.36, 0.14, 0.28]) {
            assert!((a - b).abs() < 1e-12, "{p:?}");
        }
        let p = padded(&[0.4, 0.4, 0.1, 0.2], f64::NAN);
        assert_eq!(p, [0.4, 0.4, 0.1, 0.2]);
        let q = padded_corners(&[[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]], &[0.0, 0.0, 1.0, 0.5], 0.1);
        assert_eq!(q[0], [-0.1, -0.05]);
        assert_eq!(q[2], [1.1, 0.55]);
    }

    #[test]
    fn hostile_detections_never_panic() {
        let f = vec![
            Frame { t: 0.0, detections: vec![face(f32::NAN, 0.0, 0.1, 0.9), face(0.0, 0.0, 0.0, 0.9), face(f32::INFINITY, 1.0, 1.0, 1.0)] },
            Frame { t: f64::NAN, detections: vec![face(0.1, 0.1, 0.1, f32::NAN)] },
            Frame { t: -1.0, detections: vec![face(0.1, 0.1, 0.1, 0.9)] },
        ];
        let _ = link(&f, &LinkParams::default());
        let _ = link(&[], &LinkParams::default());
    }
}
