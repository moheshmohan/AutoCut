//! From per-frame detections to tracks: linking (overlap, then nearness for fast motion, the
//! boxes compared a little grown), short gaps filled (longer ones after an unsure sighting: a
//! head cut by the frame, a pose the network is unsure of), one frame's size errors removed,
//! sizes and positions smoothed (sizes more), and the rules that decide which tracks are hidden
//! (a face large enough to be recognised, within a tolerance of the size rule, a plate large
//! enough to read, seen often enough or clearly enough, not inside an excluded box).
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
    /// A head found from a pose ([`Detection::pose`]) scoring under this is a weak sighting: it
    /// may continue a track (matched after the other detections, ByteTrack's second
    /// association) but never starts one. A profile head the pose network is unsure of in some
    /// frames keeps its track and its mask instead of flickering into one-sighting tracks.
    #[serde(default)]
    pub weak_below: f32,
    /// Boxes are compared grown by this share of their size on every side: a head boxed as
    /// tightly as a face (the profile pass's heads since 2026-10-22) and moving its own width
    /// between two analysed frames in a crowd still continues its track, as it did when heads
    /// were boxed larger.
    #[serde(default = "grow")]
    pub grow: f64,
    /// Gaussian smoothing of the boxes' sizes, seconds (σ; at least `smooth`): sizes change
    /// slowly, and a face seen by the face detector in one frame and from a pose in the next
    /// must not breathe. A size more than [`SIZE_SPIKE`] times its neighbours' is one sighting's
    /// error and takes theirs. 0: sizes as seen (neither smoothed apart nor despiked).
    #[serde(default = "smooth_size")]
    pub smooth_size: f64,
    /// Hysteresis for the unsure sightings ([`unsure`]: a head cut by the frame's edge, or found
    /// from a pose scoring under [`UNSURE_POSE`]): a track last seen unsurely continues across
    /// a stretch this long unseen (the gap filled, its mask kept on), where a sure one stops after
    /// `max_gap`. Such a head is lost and found again from frame to frame (2026-10-22: most of
    /// the faces the privacy check still saw on the Ameca clip were people cut by the frame whose
    /// pose came and went): its mask neither starts on one sighting (a track needs two,
    /// [`KeepParams::min_hits`]) nor drops on one miss. At most `max_gap` when smaller (0: off).
    #[serde(default = "unsure_gap")]
    pub unsure_gap: f64,
    /// Camera-motion compensation: the tracks a frame's first association leaves unmatched are
    /// predicted moved with the picture — the median shift, in that frame, of the tracks that
    /// matched and were seen in the frame before (else the frame before's shift) — and matched
    /// again. In a fast pan a small face moves two of its widths between analysed frames
    /// (2026-10-22, OnRobot at 86.4 s: 0.05 of the frame in 0.1 s) and would start a new track
    /// at every frame, each too brief to hide, its mask left behind where it was.
    #[serde(default = "yes")]
    pub pan: bool,
}

fn yes() -> bool {
    true
}

fn smooth_size() -> f64 {
    0.2
}

fn unsure_gap() -> f64 {
    1.5
}

fn grow() -> f64 {
    0.15
}

impl Default for LinkParams {
    fn default() -> Self {
        Self {
            max_gap: 0.7,
            min_iou: 0.2,
            max_jump: 0.8,
            smooth: 0.08,
            weak_below: WEAK_POSE,
            grow: grow(),
            smooth_size: smooth_size(),
            unsure_gap: unsure_gap(),
            pan: true,
        }
    }
}

/// [`LinkParams::weak_below`] by default.
pub const WEAK_POSE: f32 = 0.35;

/// A head found from a pose scoring under this is an unsure sighting ([`unsure`]).
pub const UNSURE_POSE: f32 = 0.5;

/// A box this close to the picture's edge (a share of it) is at the edge ([`at_edge`]): a face
/// cut by the frame.
pub const EDGE_MARGIN: f64 = 0.02;

/// Whether a sighting is unsure: its box at the frame's edge ([`EDGE_MARGIN`]: the face is cut
/// by the frame, and its detectors are least sure there), or a head found from a pose scoring
/// under [`UNSURE_POSE`]. Its track gets [`LinkParams::unsure_gap`].
pub fn unsure(bbox: &BoxN, score: f32, pose: bool) -> bool {
    // a score that is not a number is unsure too
    at_edge(bbox, EDGE_MARGIN) || (pose && !matches!(score.partial_cmp(&UNSURE_POSE), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)))
}

/// A sighting whose box is this many times larger (or smaller) than the middle of its track's
/// sightings within [`SPIKE_REACH`] seconds is one frame's error (a wrong keypoint blew a head up):
/// it keeps its centre and takes their size.
pub const SIZE_SPIKE: f64 = 1.6;
/// How far, seconds, the sightings a size is compared with reach on each side.
pub const SPIKE_REACH: f64 = 0.5;

/// A face counts as at least [`KeepParams::min_face_height`] tall when its largest box is within
/// this share of it: detectors measure one face some 10 % differently at different picture
/// sizes (2026-10-22: a visitor at 4.9 % of the frame in the redaction's frames was 5.4 % in the
/// privacy check's), and a face that is at the limit by one measure must be hidden by both.
pub const SIZE_TOLERANCE: f64 = 0.85;

/// Whether box `b` (normalised) reaches within `margin` of the picture's edge or past it: a head
/// cut by the frame, whose detections are the least sure.
pub fn at_edge(b: &BoxN, margin: f64) -> bool {
    b.iter().all(|v| v.is_finite()) && (b[0] < margin || b[1] < margin || b[0] + b[2] > 1.0 - margin || b[1] + b[3] > 1.0 - margin)
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
    /// What a [`Kind::Target`] track follows (the user's description).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Its moments in time order, one per analysed frame from its first sighting to its last.
    pub points: Vec<TrackPoint>,
    /// Detections linked to it.
    pub hits: usize,
    pub best_score: f32,
    /// Of `hits`, the sightings found from a person's pose ([`Detection::pose`]: a head in
    /// profile or half hidden), not by the face detector.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub pose_hits: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

impl Track {
    pub fn start(&self) -> f64 {
        self.points.first().map_or(0.0, |p| p.t)
    }
    pub fn end(&self) -> f64 {
        self.points.last().map_or(0.0, |p| p.t)
    }
    /// The largest box height (faces, targets) or width (plates) over the track.
    pub fn max_size(&self) -> f64 {
        self.points.iter().map(|p| if self.kind == Kind::Plate { p.bbox[2] } else { p.bbox[3] }).fold(0.0, f64::max)
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
    /// Sightings found from a pose.
    pose: usize,
    /// The last sighting was unsure ([`unsure`]).
    unsure: bool,
}

impl Open {
    /// Where the track's box should be at time `t`: the last box moved by the last velocity
    /// (from the last two sightings), half of it so a jittery pair does not throw it off, for at
    /// most `horizon` seconds (a track continued across a long gap is not thrown far off).
    fn predict(&self, frames: &[Frame], t: f64, horizon: f64) -> Option<BoxN> {
        let (i1, b1, _, _) = self.seen.last()?;
        let t1 = frames.get(*i1)?.t;
        let Some((i0, b0, _, _)) = self.seen.len().checked_sub(2).and_then(|k| self.seen.get(k)) else { return Some(*b1) };
        let t0 = frames.get(*i0)?.t;
        let dt = t1 - t0;
        if dt.is_nan() || dt <= 1e-6 {
            return Some(*b1);
        }
        let ahead = if horizon.is_finite() { (t - t1).min(horizon.max(0.0)) } else { t - t1 };
        let k = 0.5 * ahead / dt;
        Some([b1[0] + (b1[0] - b0[0]) * k, b1[1] + (b1[1] - b0[1]) * k, b1[2], b1[3]])
    }
}

/// How well detection `d` continues a track predicted at `p`: its overlap, or failing that a
/// nearness score below `min_iou` when the centres are within `max_jump` box sizes and the sizes
/// within a factor of two; `None` when it does not.
fn affinity(p: &BoxN, d: &BoxN, lp: &LinkParams) -> Option<f64> {
    let (p, d) = (&padded(p, lp.grow), &padded(d, lp.grow));
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

/// One frame's detection as the linking holds it: box, corners, score, kind, found from a pose.
type Det = (BoxN, Vec<[f64; 2]>, f32, Kind, bool);

/// Greedy assignment of `pairs` (affinity, open track, detection) not yet `used`: the sure
/// detections first, best pairs first, then the weak sightings to the tracks still unmatched
/// (ByteTrack's two associations). The (track, detection) pairs made.
fn assign(mut pairs: Vec<(f64, usize, usize)>, dets: &[Det], used_o: &mut [bool], used_d: &mut [bool], lp: &LinkParams) -> Vec<(usize, usize)> {
    let weak = |d: &Det| d.4 && d.2 < lp.weak_below;
    pairs.sort_by(|a, b| {
        let (wa, wb) = (dets.get(a.2).is_some_and(weak), dets.get(b.2).is_some_and(weak));
        wa.cmp(&wb).then(b.0.total_cmp(&a.0))
    });
    let mut made = Vec::new();
    for (_, oi, di) in pairs {
        let (Some(uo), Some(ud)) = (used_o.get(oi).copied(), used_d.get(di).copied()) else { continue };
        if uo || ud {
            continue;
        }
        if let (Some(a), Some(b)) = (used_o.get_mut(oi), used_d.get_mut(di)) {
            (*a, *b) = (true, true);
        }
        made.push((oi, di));
    }
    made
}

/// The middle of `v` (0 when empty).
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Link the detections of `frames` (in time order, normalised) into tracks, one kind at a time.
/// A detection joins the open track it continues best (greedy, best pairs first; then, with
/// `pan`, the tracks left over predicted moved with the picture); one that continues none starts
/// a track; a track unseen for longer than `max_gap` (`unsure_gap` after an unsure sighting) is
/// closed. Each track then gets a point at every analysed frame from its first sighting to its
/// last (gaps filled by straight lines) and is smoothed.
pub fn link(frames: &[Frame], lp: &LinkParams) -> Vec<Track> {
    let mut done: Vec<Open> = Vec::new();
    let mut open: Vec<Open> = Vec::new();
    // the picture's motion: its cumulative shift at each frame, and the last frame's shift
    let mut cum: Vec<[f64; 2]> = Vec::with_capacity(frames.len());
    let mut last_pan: Option<[f64; 2]> = None;
    for (fi, f) in frames.iter().enumerate() {
        // close the tracks unseen for too long (longer for one last seen unsurely: hysteresis)
        let unsure_gap = if lp.unsure_gap.is_finite() { lp.unsure_gap.max(lp.max_gap) } else { lp.max_gap };
        let (keep, closed): (Vec<Open>, Vec<Open>) = open.into_iter().partition(|o| {
            let gap = if o.unsure { unsure_gap } else { lp.max_gap };
            o.seen.last().and_then(|(i, ..)| frames.get(*i)).is_some_and(|last| f.t - last.t <= gap + 1e-9)
        });
        open = keep;
        done.extend(closed);
        let dets: Vec<Det> = f
            .detections
            .iter()
            .filter(|d| d.bbox.iter().all(|v| v.is_finite()) && d.bbox[2] > 0.0 && d.bbox[3] > 0.0)
            // a plate keeps its corners; a face's landmarks are not followed (a head found from a pose has
            // other points than the face detector's face, and one track may hold both)
            .map(|d| {
                (
                    to_box(&d.bbox),
                    if d.kind == Kind::Plate { d.points.iter().map(|p| [f64::from(p[0]), f64::from(p[1])]).collect() } else { Vec::new() },
                    d.score,
                    d.kind,
                    d.pose,
                )
            })
            .collect();
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (oi, o) in open.iter().enumerate() {
            let Some(p) = o.predict(frames, f.t, lp.max_gap) else { continue };
            for (di, d) in dets.iter().enumerate() {
                if d.3 != o.kind {
                    continue;
                }
                if let Some(a) = affinity(&p, &d.0, lp) {
                    pairs.push((a, oi, di));
                }
            }
        }
        let (mut used_o, mut used_d) = (vec![false; open.len()], vec![false; dets.len()]);
        let mut made = assign(pairs, &dets, &mut used_o, &mut used_d, lp);
        // the picture's shift this frame: the median of the matched tracks seen in the frame
        // before, else the last frame's if it was measured (it is used only for the tracks the
        // first association leaves over, which would otherwise end)
        let centre = |b: &BoxN| [b[0] + b[2] / 2.0, b[1] + b[3] / 2.0];
        let shifts: Vec<[f64; 2]> = made
            .iter()
            .filter_map(|(oi, di)| {
                let (i, b, ..) = open.get(*oi)?.seen.last()?;
                let d = dets.get(*di)?;
                (*i + 1 == fi).then(|| {
                    let (a, c) = (centre(b), centre(&d.0));
                    [c[0] - a[0], c[1] - a[1]]
                })
            })
            .filter(|s| s.iter().all(|v| v.is_finite()))
            .collect();
        let pan = if !shifts.is_empty() { Some([median(shifts.iter().map(|s| s[0]).collect()), median(shifts.iter().map(|s| s[1]).collect())]) } else { None };
        let step = pan.or(last_pan).unwrap_or([0.0, 0.0]);
        last_pan = pan;
        let here = cum.last().map_or([0.0, 0.0], |c| [c[0] + step[0], c[1] + step[1]]);
        cum.push(here);
        // the tracks left over, predicted moved with the picture since they were last seen
        if lp.pan && used_o.iter().any(|u| !u) && used_d.iter().any(|u| !u) {
            let mut again: Vec<(f64, usize, usize)> = Vec::new();
            for (oi, o) in open.iter().enumerate().filter(|(oi, _)| !used_o.get(*oi).copied().unwrap_or(true)) {
                let Some((i, b, ..)) = o.seen.last() else { continue };
                let Some(then) = cum.get(*i) else { continue };
                let p = [b[0] + here[0] - then[0], b[1] + here[1] - then[1], b[2], b[3]];
                for (di, d) in dets.iter().enumerate().filter(|(di, d)| d.3 == o.kind && !used_d.get(*di).copied().unwrap_or(true)) {
                    if let Some(a) = affinity(&p, &d.0, lp) {
                        again.push((a, oi, di));
                    }
                }
            }
            made.extend(assign(again, &dets, &mut used_o, &mut used_d, lp));
        }
        for (oi, di) in made {
            if let (Some(o), Some(d)) = (open.get_mut(oi), dets.get(di)) {
                o.seen.push((fi, d.0, d.1.clone(), d.2));
                o.pose += usize::from(d.4);
                o.unsure = unsure(&d.0, d.2, d.4);
            }
        }
        for (di, d) in dets.into_iter().enumerate() {
            if !used_d.get(di).copied().unwrap_or(true) && !(d.4 && d.2 < lp.weak_below) {
                let unsure = unsure(&d.0, d.2, d.4);
                open.push(Open { kind: d.3, seen: vec![(fi, d.0, d.1, d.2)], pose: usize::from(d.4), unsure });
            }
        }
    }
    done.extend(open);
    let mut tracks: Vec<Track> = done
        .into_iter()
        .filter_map(|o| fill(frames, &o))
        .map(|t| if lp.smooth_size > 0.0 { despike(t) } else { t })
        .map(|t| smooth(t, lp.smooth, lp.smooth_size.max(lp.smooth), SMOOTH_REACH))
        .collect();
    tracks.sort_by(|a, b| a.start().total_cmp(&b.start()));
    tracks
}

/// Each sighting whose size is [`SIZE_SPIKE`] times off the middle of the track's sightings
/// within [`SPIKE_REACH`] seconds (at least two others) keeps its centre and takes their middle
/// size: one frame's wrong keypoint does not blow a mask up.
fn despike(mut t: Track) -> Track {
    let seen: Vec<(f64, f64, f64)> = t.points.iter().filter(|p| p.seen).map(|p| (p.t, p.bbox[2], p.bbox[3])).collect();
    for p in t.points.iter_mut().filter(|p| p.seen) {
        let mut ws: Vec<f64> = Vec::new();
        let mut hs: Vec<f64> = Vec::new();
        for (tt, w, h) in &seen {
            if (tt - p.t).abs() <= SPIKE_REACH && (tt - p.t).abs() > 1e-9 {
                ws.push(*w);
                hs.push(*h);
            }
        }
        if hs.len() < 2 {
            continue;
        }
        ws.sort_by(f64::total_cmp);
        hs.sort_by(f64::total_cmp);
        let (mw, mh) = (ws.get(ws.len() / 2).copied().unwrap_or(0.0), hs.get(hs.len() / 2).copied().unwrap_or(0.0));
        if !(mh > 0.0 && mw > 0.0) {
            continue;
        }
        let r = p.bbox[3] / mh;
        if r > SIZE_SPIKE || r < 1.0 / SIZE_SPIKE {
            let (cx, cy) = (p.bbox[0] + p.bbox[2] / 2.0, p.bbox[1] + p.bbox[3] / 2.0);
            p.bbox = [cx - mw / 2.0, cy - mh / 2.0, mw, mh];
        }
    }
    t
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
    Some(Track { kind: o.kind, label: None, points, hits: o.seen.len(), best_score, pose_hits: o.pose })
}

/// [`link`]'s smoothing for a track made elsewhere (a followed target).
pub fn smoothed(t: Track, sigma: f64) -> Track {
    smooth(t, sigma, sigma, f64::INFINITY)
}

/// How far, a share of its box's size, smoothing may move a sighting's centre: a detector's
/// jitter is a few hundredths of a face, while a whip pan or a turn of the head bends the path
/// faster than the smoothing follows (2026-10-22, OnRobot at 86.5 s: smoothed masks a third of a
/// face off the faces they were made for).
pub const SMOOTH_REACH: f64 = 0.15;

/// Gaussian-weighted local linear smoothing of the box centres and corners (σ = `sigma`
/// seconds) and sizes (σ = `sigma_size`) over time: each value is the value at its moment of the
/// straight line fitted to its neighbours, so jitter goes while steady motion keeps its place, at
/// the ends of a track too (a plain weighted mean would pull the ends inwards). A sighting's
/// centre moves at most `reach` of its box's size.
fn smooth(t: Track, sigma: f64, sigma_size: f64, reach: f64) -> Track {
    let raw: Vec<(bool, BoxN)> = t.points.iter().map(|p| (p.seen, p.bbox)).collect();
    let sized = smooth_with(t.clone(), sigma_size);
    let mut t = smooth_with(t, sigma);
    if sigma_size.is_finite() && sigma_size > sigma {
        for (p, q) in t.points.iter_mut().zip(&sized.points) {
            let (cx, cy) = (p.bbox[0] + p.bbox[2] / 2.0, p.bbox[1] + p.bbox[3] / 2.0);
            let (w, h) = (q.bbox[2], q.bbox[3]);
            p.bbox = [cx - w / 2.0, cy - h / 2.0, w, h];
        }
    }
    if reach.is_finite() && reach >= 0.0 {
        for (p, (_, b)) in t.points.iter_mut().zip(&raw).filter(|(_, (seen, _))| *seen) {
            let (rx, ry) = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0);
            let (cx, cy) = (p.bbox[0] + p.bbox[2] / 2.0, p.bbox[1] + p.bbox[3] / 2.0);
            let (mx, my) = (reach * b[2].abs(), reach * b[3].abs());
            if !(rx.is_finite() && ry.is_finite() && mx.is_finite() && my.is_finite()) {
                continue;
            }
            let (nx, ny) = (cx.clamp(rx - mx, rx + mx), cy.clamp(ry - my, ry + my));
            p.bbox[0] += nx - cx;
            p.bbox[1] += ny - cy;
        }
    }
    t
}

fn smooth_with(mut t: Track, sigma: f64) -> Track {
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

/// The subject a video is about (a robot's face): a track whose box centre lies inside the area
/// (at `t`, if given, within a second of it; else at any time) is the subject's, left alone
/// ([`Skip::Excluded`]). The engine shows such a track to its vision model when it has one, and
/// hides it if the model sees a person: a subject's box is coarse, and visitors crowd around a
/// robot (2026-10-21: two visitors next to Ameca were left sharp).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Exclude {
    pub t: Option<f64>,
    pub bbox: BoxN,
}

/// How far, seconds, an [`Exclude`] with a time reaches.
pub const EXCLUDE_REACH: f64 = 1.0;

impl Exclude {
    fn covers(&self, p: &TrackPoint) -> bool {
        self.contains(p.bbox[0] + p.bbox[2] / 2.0, p.bbox[1] + p.bbox[3] / 2.0)
    }

    /// The exclusion applies to `track`.
    pub fn applies(&self, track: &Track) -> bool {
        match self.t {
            Some(t) => track.at(t).is_some_and(|p| (p.t - t).abs() <= EXCLUDE_REACH && self.covers(p)),
            None => track.points.iter().any(|p| self.covers(p)),
        }
    }

    /// `track`'s box centre is inside the area at any of its moments, whatever the exclusion's
    /// time: the track may be on the subject (a robot keeps its place in a shot), which is where
    /// a vision model's "not a person" is believed for a face the detector is sure of.
    pub fn near(&self, track: &Track) -> bool {
        track.points.iter().any(|p| self.covers(p))
    }

    /// Whether point `(x, y)` (normalised) is inside the area.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let b = &self.bbox;
        x >= b[0] && x <= b[0] + b[2] && y >= b[1] && y <= b[1] + b[3]
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
    /// Inside an excluded box (the subject's), with no vision model to ask.
    Excluded,
    /// The vision model said it is not a real person's face.
    NotAPerson,
    /// The vision model said it is not a vehicle's number plate (a sign, a label on a machine).
    NotAPlate,
}

/// Whether `track` is hidden under `kp` and `exclude`: `None` hides it.
pub fn verdict(track: &Track, kp: &KeepParams, exclude: &[Exclude]) -> Option<Skip> {
    let big = match track.kind {
        Kind::Face => track.max_size() >= kp.min_face_height * SIZE_TOLERANCE,
        Kind::Plate => track.max_size() >= kp.min_plate_width,
        // what the user named is hidden at any size, and is not checked against exclusions
        Kind::Target => return None,
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
        Detection { kind: Kind::Face, score, bbox: [x, y, s, s], points: Vec::new(), pose: false }
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
    fn a_face_turning_to_profile_stays_one_track() {
        // YuNet sees the face for 6 frames; the visitor turns away into profile and only the pose
        // pass finds the head (a slightly larger box, its own points) for 8 more, missing 2 of them
        let head = |x: f32| Detection {
            kind: Kind::Face,
            score: 0.7,
            bbox: [x - 0.01, 0.29, 0.12, 0.15],
            points: vec![[x, 0.33], [x + 0.03, 0.32], [x + 0.08, 0.34]],
            pose: true,
        };
        let f = frames(16, 10.0, |i| {
            let x = 0.2 + 0.01 * i as f32;
            match i {
                0..6 => vec![face(x, 0.3, 0.1, 0.9)],
                9 | 10 => vec![],
                _ => vec![head(x)],
            }
        });
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1, "{t:?}");
        let tr = &t[0];
        assert_eq!((tr.hits, tr.pose_hits), (14, 8));
        assert_eq!(tr.points.len(), 16);
        assert!(!tr.points[9].seen && tr.points[11].seen);
        // a plate keeps its corners; the faces' points are not followed
        assert!(tr.points.iter().all(|p| p.corners.is_empty()));
        // its best score is the face detector's
        assert!((tr.best_score - 0.9).abs() < 1e-6);
        // a pose head never seen twice is too brief to hide unless sure (and pose heads never are)
        let once = frames(3, 10.0, |i| if i == 1 { vec![head(0.5)] } else { vec![] });
        let t = link(&once, &LinkParams::default());
        assert_eq!(verdict(&t[0], &KeepParams::default(), &[]), Some(Skip::TooBrief));
    }

    #[test]
    fn weak_pose_sightings_continue_tracks_but_never_start_them() {
        let head = |x: f32, score: f32| Detection { kind: Kind::Face, score, bbox: [x, 0.3, 0.1, 0.13], points: Vec::new(), pose: true };
        // a profile head seen surely once, then only weakly: one track, every sighting counted
        let f = frames(8, 10.0, |i| vec![head(0.2 + 0.005 * i as f32, if i == 0 { 0.6 } else { 0.28 })]);
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1, "{t:?}");
        assert_eq!((t[0].hits, t[0].pose_hits), (8, 8));
        assert_eq!(verdict(&t[0], &KeepParams::default(), &[]), None);
        // weak sightings alone: nothing (three weak sightings of a robot's gripper were a head
        // half the frame tall, 2026-10-22)
        let f = frames(8, 10.0, |i| vec![head(0.2 + 0.005 * i as f32, 0.28)]);
        assert!(link(&f, &LinkParams::default()).is_empty());
        // a weak sighting next to a sure one goes to the track the sure one does not take
        let f = frames(2, 10.0, |i| if i == 0 { vec![head(0.2, 0.6), head(0.6, 0.6)] } else { vec![head(0.21, 0.28), head(0.205, 0.7)] });
        let t = link(&f, &LinkParams::default());
        let mut hits: Vec<usize> = t.iter().map(|tr| tr.hits).collect();
        hits.sort_unstable();
        assert_eq!(hits, vec![1, 2], "{t:?}");
        let a = t.iter().find(|tr| tr.hits == 2).unwrap();
        assert!((a.points[1].bbox[0] - 0.205).abs() < 1e-6, "the sure sighting continues it: {a:?}");
        // with weak_below 0 (a face detector's own linking), every detection starts a track
        let f = frames(3, 10.0, |i| vec![head(0.2 + 0.3 * i as f32, 0.28)]);
        assert_eq!(link(&f, &LinkParams { weak_below: 0.0, ..LinkParams::default() }).len(), 3);
    }

    #[test]
    fn an_unsure_head_neither_starts_nor_drops_its_track_on_one_frame() {
        let head = |x: f32, score: f32| Detection { kind: Kind::Face, score, bbox: [x, 0.4, 0.06, 0.15], points: Vec::new(), pose: true };
        // a visitor cut by the left edge (x < 0.02), found from the pose in bursts with gaps of
        // 1.0–1.2 s between (longer than `max_gap`): one track, the gaps filled
        let seen = |i: usize| matches!(i, 0..=2 | 13..=14 | 25..=27 | 38);
        let f = frames(40, 10.0, |i| if seen(i) { vec![head(-0.01 + 0.0005 * i as f32, 0.6)] } else { vec![] });
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1, "{t:?}");
        assert_eq!((t[0].points.len(), t[0].hits), (39, 9));
        assert!(t[0].points.iter().all(|p| p.bbox.iter().all(|v| v.is_finite())));
        assert_eq!(verdict(&t[0], &KeepParams::default(), &[]), None);
        // the same head well inside the frame and sure (0.8): the gaps split it
        let f = frames(40, 10.0, |i| if seen(i) { vec![head(0.4 + 0.0005 * i as f32, 0.8)] } else { vec![] });
        assert_eq!(link(&f, &LinkParams::default()).len(), 4);
        // inside the frame but unsure (a pose scoring 0.4): one track again
        let f = frames(40, 10.0, |i| if seen(i) { vec![head(0.4 + 0.0005 * i as f32, 0.4)] } else { vec![] });
        assert_eq!(link(&f, &LinkParams::default()).len(), 1);
        // without the hysteresis (the privacy check's linking), the edge head splits too
        let f = frames(40, 10.0, |i| if seen(i) { vec![head(-0.01 + 0.0005 * i as f32, 0.6)] } else { vec![] });
        assert_eq!(link(&f, &LinkParams { unsure_gap: 0.0, ..LinkParams::default() }).len(), 4);
        // a gap longer than `unsure_gap` closes it
        let f = frames(40, 10.0, |i| if matches!(i, 0..=2 | 20..=22) { vec![head(-0.01, 0.6)] } else { vec![] });
        assert_eq!(link(&f, &LinkParams::default()).len(), 2);
        // one unsure sighting starts no mask: too brief (a pose head is never sure)
        let f = frames(10, 10.0, |i| if i == 4 { vec![head(-0.01, 0.6)] } else { vec![] });
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1);
        assert_eq!(verdict(&t[0], &KeepParams::default(), &[]), Some(Skip::TooBrief));
        // a head continued across a long gap is predicted at most `max_gap` ahead: moving 0.02 a
        // frame, then found 1.2 s later having stopped 0.02 further on, it continues the track
        // (predicted 1.2 s ahead it would be 0.13 further on, too far)
        let f = frames(20, 10.0, |i| match i {
            0 => vec![head(-0.01, 0.6)],
            1 => vec![head(0.01, 0.6)],
            13 => vec![head(0.03, 0.6)],
            _ => vec![],
        });
        assert_eq!(link(&f, &LinkParams::default()).len(), 1);
        // hostile settings do not panic
        for g in [f64::NAN, f64::INFINITY, -1.0] {
            let _ = link(&f, &LinkParams { unsure_gap: g, ..LinkParams::default() });
        }
        assert!(unsure(&[0.4, 0.4, 0.1, 0.1], f32::NAN, true) && !unsure(&[0.4, 0.4, 0.1, 0.1], 0.3, false));
        assert!(unsure(&[0.95, 0.4, 0.1, 0.1], 0.99, false) && !unsure(&[0.4, 0.4, 0.1, 0.1], 0.6, true));
    }

    #[test]
    fn a_size_spike_is_one_frames_error_and_sizes_are_steadier_than_places() {
        // a head 0.1 tall moving right; in frame 5 a wrong keypoint makes it 0.18 tall
        let f = frames(11, 10.0, |i| {
            let s = if i == 5 { 0.18 } else { 0.1 };
            let cx = 0.2 + 0.01 * i as f32;
            vec![face(cx - s / 2.0, 0.4 - s / 2.0, s, 0.9)]
        });
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1, "{t:?}");
        let p = &t[0].points[5];
        assert!((p.bbox[3] - 0.1).abs() < 0.01, "the spike takes its neighbours' size: {:?}", p.bbox);
        assert!((p.bbox[0] + p.bbox[2] / 2.0 - 0.25).abs() < 0.005, "and keeps its place: {:?}", p.bbox);
        // the face detector's face and the pose's head of one person alternate (sizes 0.09 and
        // 0.11): the mask does not breathe
        let f = frames(30, 10.0, |i| {
            let s = if i % 2 == 0 { 0.09 } else { 0.11 };
            vec![face(0.3 - s / 2.0, 0.4 - s / 2.0, s, 0.9)]
        });
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1);
        for p in &t[0].points[5..25] {
            assert!((p.bbox[3] - 0.1).abs() < 0.003, "{:?}", p.bbox);
        }
        let plain = link(&f, &LinkParams { smooth_size: 0.0, ..LinkParams::default() });
        let swing =
            |t: &Track| t.points[5..25].iter().map(|p| p.bbox[3]).fold(0.0f64, f64::max) - t.points[5..25].iter().map(|p| p.bbox[3]).fold(1.0f64, f64::min);
        assert!(swing(&t[0]) < swing(&plain[0]) / 2.0, "{} vs {}", swing(&t[0]), swing(&plain[0]));
        // a face growing steadily (walking up to the camera) keeps its size: no lag
        let f = frames(30, 10.0, |i| {
            let s = 0.05 + 0.005 * i as f32;
            vec![face(0.3 - s / 2.0, 0.4 - s / 2.0, s, 0.9)]
        });
        let t = link(&f, &LinkParams::default());
        for (i, p) in t[0].points.iter().enumerate() {
            assert!((p.bbox[3] - (0.05 + 0.005 * i as f64)).abs() < 2e-3, "frame {i}: {:?}", p.bbox);
        }
    }

    #[test]
    fn boxes_at_the_frames_edge() {
        assert!(at_edge(&[0.005, 0.4, 0.1, 0.1], 0.01) && at_edge(&[0.9, 0.4, 0.2, 0.1], 0.01) && at_edge(&[0.4, 0.95, 0.1, 0.1], 0.01));
        assert!(at_edge(&[-0.05, 0.4, 0.1, 0.1], 0.01) && at_edge(&[0.4, -0.01, 0.1, 0.1], 0.0));
        assert!(!at_edge(&[0.4, 0.4, 0.1, 0.1], 0.01) && !at_edge(&[f64::NAN, 0.4, 0.1, 0.1], 0.01));
    }

    #[test]
    fn a_small_head_moving_its_width_a_frame_keeps_its_track() {
        // a head 0.032 wide moving 0.031 a frame, a little down: no overlap, about its width
        // apart (a visitor in a crowd, 2026-10-22)
        let head = |i: usize| Detection {
            kind: Kind::Face,
            score: 0.8,
            bbox: [0.044 + 0.031 * i as f32, 0.32 + 0.023 * i as f32, 0.032, 0.078],
            points: Vec::new(),
            pose: true,
        };
        let f = frames(6, 10.0, |i| vec![head(i)]);
        assert_eq!(link(&f, &LinkParams::default()).len(), 1);
        // compared as they are (the privacy check's linking), it breaks up
        assert!(link(&f, &LinkParams { grow: 0.0, ..LinkParams::default() }).len() > 1);
    }

    #[test]
    fn a_fast_pan_keeps_small_faces_on_their_tracks() {
        // the camera pans: everything moves 0.05 of the frame a frame (OnRobot, 2026-10-22); a
        // large face (0.12 wide) still overlaps itself, three small ones (0.025 wide) jump two
        // of their widths
        let small = |x: f32, y: f32| Detection { kind: Kind::Face, score: 0.8, bbox: [x, y, 0.025, 0.06], points: Vec::new(), pose: false };
        let f = frames(8, 10.0, |i| {
            let dx = 0.05 * i as f32;
            vec![face(0.1 + dx, 0.2, 0.12, 0.9), small(0.05 + dx, 0.5), small(0.15 + dx, 0.55), small(0.25 + dx, 0.5)]
        });
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 4, "{:?}", t.iter().map(|t| (t.start(), t.hits)).collect::<Vec<_>>());
        assert!(t.iter().all(|t| t.hits == 8));
        // without the compensation each small face is a new track at nearly every frame
        assert!(link(&f, &LinkParams { pan: false, ..LinkParams::default() }).len() > 10);
        // a pan followed after a missed frame: the small faces missed at frame 3 continue
        let f = frames(8, 10.0, |i| {
            let dx = 0.05 * i as f32;
            let mut v = vec![face(0.1 + dx, 0.2, 0.12, 0.9), face(0.4 + dx, 0.2, 0.12, 0.9)];
            if i != 3 {
                v.extend([small(0.05 + dx, 0.5), small(0.25 + dx, 0.5)]);
            }
            v
        });
        assert_eq!(link(&f, &LinkParams::default()).len(), 4);
        // people moving on their own (no pan) are not thrown together: two faces crossing keep
        // their tracks (`two_faces_crossing_paths_keep_their_tracks`), and a still face next to a
        // moving one keeps its own
        let f = frames(10, 10.0, |i| vec![face(0.1 + 0.03 * i as f32, 0.2, 0.1, 0.9), face(0.6, 0.2, 0.1, 0.9)]);
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 2);
        assert!(t.iter().all(|t| t.hits == 10));
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
    fn smoothing_never_moves_a_sighting_far() {
        // a face still, then whipped right half its width a frame and straight back, then
        // still: the smoothing would round the turn off by a quarter of the face
        let x = |i: usize| 0.3 + 0.05 * (3 - (i as i32 - 13).abs()).max(0) as f32;
        let f = frames(27, 10.0, |i| vec![face(x(i), 0.4, 0.1, 0.9)]);
        let t = link(&f, &LinkParams::default());
        assert_eq!(t.len(), 1);
        for (i, p) in t[0].points.iter().enumerate() {
            assert!((p.bbox[0] - f64::from(x(i))).abs() <= SMOOTH_REACH * 0.1 + 1e-9, "frame {i}: {} vs {}", p.bbox[0], x(i));
        }
        // the unclamped smoothing of the same path is further off at the turn
        let seen = (0..27).map(|i| (i, [f64::from(x(i)), 0.4, 0.1, 0.1], Vec::new(), 0.9)).collect();
        let t0 = fill(&f, &Open { kind: Kind::Face, seen, pose: 0, unsure: false }).unwrap();
        let free = smooth(t0.clone(), 0.08, 0.2, f64::INFINITY);
        assert!(free.points.iter().enumerate().any(|(i, p)| (p.bbox[0] - f64::from(x(i))).abs() > SMOOTH_REACH * 0.1));
        // hostile reaches change nothing and do not panic
        for r in [f64::NAN, -1.0] {
            assert_eq!(smooth(t0.clone(), 0.08, 0.2, r), smooth(t0.clone(), 0.08, 0.2, f64::INFINITY));
        }
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
            label: None,
            points: vec![TrackPoint { t: 1.0, bbox: [0.4, 0.4, h * 0.6, h], corners: vec![], seen: true }],
            hits,
            best_score: score,
            pose_hits: 0,
        };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[]), None);
        assert_eq!(verdict(&mk(0.03, 5, 0.99), &kp, &[]), Some(Skip::TooSmall));
        // a face just under the limit by one measure is hidden ([`SIZE_TOLERANCE`]), one well
        // under is not
        assert_eq!(verdict(&mk(0.046, 5, 0.8), &kp, &[]), None);
        assert_eq!(verdict(&mk(0.04, 5, 0.8), &kp, &[]), Some(Skip::TooSmall));
        assert_eq!(verdict(&mk(0.2, 1, 0.7), &kp, &[]), Some(Skip::TooBrief));
        assert_eq!(verdict(&mk(0.2, 1, 0.95), &kp, &[]), None, "one sure sighting is enough");
        // an excluded box at the track's time, and one that is too far away in time
        let ex = Exclude { t: Some(1.2), bbox: [0.3, 0.3, 0.4, 0.4] };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[ex]), Some(Skip::Excluded));
        let late = Exclude { t: Some(5.0), ..ex };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[late]), None);
        // …which is still on the subject's place, at any time
        assert!(late.near(&mk(0.2, 5, 0.8)) && ex.near(&mk(0.2, 5, 0.8)));
        let elsewhere = Exclude { t: None, bbox: [0.0, 0.0, 0.1, 0.1] };
        assert_eq!(verdict(&mk(0.2, 5, 0.8), &kp, &[elsewhere]), None);
        assert!(!elsewhere.near(&mk(0.2, 5, 0.8)));
        assert!(elsewhere.contains(0.05, 0.1) && !elsewhere.contains(0.05, 0.11) && !elsewhere.contains(f64::NAN, 0.0));
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
