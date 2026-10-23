//! From a person's pose to a face to hide: the head box from the face keypoints, whether the
//! face shows at all, and which heads the face detector has already found.
//!
//! YuNet finds frontal faces; a face in profile or half hidden behind something it misses (two
//! visitors in profile behind a robot arm stayed sharp in a reel, 2026-10-22). The person
//! detector and the pose network find those people anyway, and their nose, eyes and ears say
//! where the head is. Plain Rust, no model: tested with synthetic poses, builds everywhere.
//!
//! **Which heads.** A face is hidden when it is at least partly visible: the nose or an eye is
//! found surely ([`HeadParams::face`], 0.5), or less surely (`face_low`, 0.25: a face half hidden
//! behind a robot arm scores 0.35–0.5) on a pose that does not face away. A head seen from behind
//! identifies nobody and is left alone ([`View::Back`]); the pose network places a back's
//! "eyes" and "ears" mirrored, its left on the picture's left ([`faces_away`]). A head scored
//! under [`crate::track::WEAK_POSE`] (0.35; its person or its face that unsure) is a weak
//! sighting: it continues a track, never starts one.
//!
//! **The box.** Its size comes from the distances between the keypoints found, each divided by
//! that distance's share of a head's width (ear to ear 1.05, eye to eye 0.41, nose to ear 0.79,
//! eye to ear 0.56, nose to eye 0.31; calibrated against the faces YuNet also found of the same
//! people in the OnRobot and Ameca clips, 2026-10-22): each is only shortened by the head
//! turning, so the largest is kept — of the sure keypoints' (≥ 0.5) when two face keypoints are
//! sure; else the middle estimate, unsure keypoints far from the others dropped (a wrong guess
//! blew a head up to four faces tall). Three face keypoints or more cap it at 3.5 times their
//! spread; with fewer than two, the shoulders' distance over 2.6 measures it (their width varies
//! with posture: only then). The box is the face as the face detector boxes it ([`FACE_BOX`]):
//! from the front 0.88 head widths wide and 1.14 tall, turned (the turn shortens the distances
//! that measure the width) 0.98 × 1.34, centred between the outermost face keypoints (the ears
//! for a frontal face, ear and nose in profile), its centre 0.13 widths below the eyes, grown to
//! keep the nose and eyes inside with a margin. Calibrated against YuNet's faces of the same
//! people on 174 frames of the OnRobot and Ameca clips (808 pairs, 2026-10-22): a median 1.05
//! times their height and 1.06 their width (90 % under 1.40); the earlier box, a head width wide
//! and 1.3 tall (hair and ears), was 1.17 and 1.26, and its masks were larger and heavier than
//! the face detector's.

use serde::{Deserialize, Serialize};

use crate::{Detection, Kind};

/// COCO's keypoints used here.
pub const NOSE: usize = 0;
pub const LEFT_EYE: usize = 1;
pub const RIGHT_EYE: usize = 2;
pub const LEFT_EAR: usize = 3;
pub const RIGHT_EAR: usize = 4;
pub const LEFT_SHOULDER: usize = 5;
pub const RIGHT_SHOULDER: usize = 6;
/// Keypoints of a COCO pose.
pub const KEYPOINTS: usize = 17;

/// The highest score a head found from a pose gets as a face detection: under the engine's
/// "sure face" (0.85) and the tracker's "seen once surely" (0.9), so such a face is always seen
/// twice before it is hidden and the vision check's "not a person" (a robot) is believed.
pub const HEAD_SCORE_MAX: f32 = 0.84;

/// A face box in head widths (the width the keypoints measure): its width, height, and how far
/// its centre is below the eyes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceShape {
    pub width: f32,
    pub height: f32,
    pub below_eyes: f32,
}

/// The face boxes of a head seen from the front (nose and both eyes sure) and turned (profile,
/// three quarters: the keypoints' distances shorten as the head turns, so a turned face is
/// larger for the width they measure).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBoxes {
    pub front: FaceShape,
    pub turned: FaceShape,
}

/// The face box of a head, calibrated against YuNet's faces of the same people (2026-10-22: 812
/// pairs on 174 frames of the OnRobot and Ameca clips, every 0.1 s): YuNet's faces are a median
/// 0.84 × 1.09 head widths from the front, 0.94 × 1.28 turned, their centres 0.13 widths below
/// the eyes. A little more than the median, so a head box is about the face detector's box and
/// a little more (the mask's padding and feather do the rest).
pub const FACE_BOX: FaceBoxes =
    FaceBoxes { front: FaceShape { width: 0.88, height: 1.14, below_eyes: 0.13 }, turned: FaceShape { width: 0.98, height: 1.34, below_eyes: 0.14 } };

/// One keypoint, in the picture's pixels, with its confidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// A person found by the person detector, with its pose (picture pixels).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Person {
    /// `[x, y, w, h]`.
    pub bbox: [f32; 4],
    pub score: f32,
    /// The 17 COCO keypoints (empty when the pose was not looked at).
    pub keypoints: Vec<Keypoint>,
}

/// The thresholds of the pose pass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadParams {
    /// A keypoint counts when its score is at least this.
    pub keypoint: f32,
    /// The face shows (partly) when the nose or an eye scores at least this…
    pub face: f32,
    /// …or at least this, and the pose does not face away ([`faces_away`]). Such a head scores
    /// under [`crate::track::WEAK_POSE`] when this unsure: it may continue a track, never start one.
    pub face_low: f32,
    /// Keypoints at least this sure measure the head (lower ones only when there are none).
    pub sure: f32,
}

impl Default for HeadParams {
    fn default() -> Self {
        Self { keypoint: 0.3, face: 0.5, face_low: 0.25, sure: 0.5 }
    }
}

/// How a head is seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum View {
    /// Nose and both eyes found.
    Front,
    /// The face partly visible: the nose or an eye, not all three.
    Profile,
    /// No nose or eye: the back of the head, which identifies nobody.
    Back,
}

/// A head found from a pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head {
    /// `[x, y, w, h]`, picture pixels.
    pub bbox: [f32; 4],
    pub view: View,
    /// The best of the nose's and eyes' scores.
    pub face_score: f32,
    /// The head's width estimated from the keypoints, picture pixels.
    pub width: f32,
}

impl Head {
    /// Whether the face shows (front or profile).
    pub fn shows_face(&self) -> bool {
        self.view != View::Back
    }
}

/// Keypoint `i` of `kps` when it scores at least `min` and is finite.
fn found(kps: &[Keypoint], i: usize, min: f32) -> Option<Keypoint> {
    kps.get(i).copied().filter(|k| k.score >= min && k.x.is_finite() && k.y.is_finite() && k.score.is_finite())
}

fn dist(a: Keypoint, b: Keypoint) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// The head of a person with a pose, or `None` when no face or ear keypoint was found.
pub fn head_of(p: &Person, hp: &HeadParams) -> Option<Head> {
    let kps = &p.keypoints;
    let (pw, ph) = (p.bbox[2].max(0.0), p.bbox[3].max(0.0));
    if !(p.bbox.iter().all(|v| v.is_finite()) && pw > 0.0 && ph > 0.0) {
        return None;
    }
    // a keypoint outside the person's box grown by a quarter is a wrong one
    let near = |q: &Keypoint| {
        let (gx, gy) = (pw * 0.25, ph * 0.25);
        q.x >= p.bbox[0] - gx && q.x <= p.bbox[0] + pw + gx && q.y >= p.bbox[1] - gy && q.y <= p.bbox[1] + ph + gy
    };
    let at = |i: usize, min: f32| found(kps, i, min).filter(near);
    let face_ids = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR];
    if face_ids.iter().all(|i| at(*i, hp.keypoint).is_none()) {
        return None;
    }
    // the sure keypoints measure and place the head; when fewer than two face keypoints are
    // sure, every one found does
    let sure = face_ids.iter().filter(|i| at(**i, hp.sure).is_some()).count() >= 2;
    let min = if sure { hp.sure } else { hp.keypoint };
    let k = |i: usize| at(i, min);
    let mut face_kps: Vec<Keypoint> = face_ids.into_iter().filter_map(k).collect();
    // a face keypoint's score, where it is a plausible one
    let score = |i: usize| found(kps, i, 0.0).filter(near).map_or(0.0, |k| k.score);
    let face_score = score(NOSE).max(score(LEFT_EYE)).max(score(RIGHT_EYE));
    let seen = |i: usize| score(i) >= hp.face;
    let view = if face_score >= hp.face || (face_score >= hp.face_low && !faces_away(p, hp)) {
        if seen(NOSE) && seen(LEFT_EYE) && seen(RIGHT_EYE) { View::Front } else { View::Profile }
    } else {
        View::Back
    };
    // the head's width: each pair's distance over its share of a head's width (each only shrinks
    // as the head turns, so the largest is kept; measured against YuNet's faces of the same
    // people, 2026-10-22)
    let pairs: [(usize, usize, f32); 8] = [
        (LEFT_EAR, RIGHT_EAR, 1.05),
        (LEFT_EYE, RIGHT_EYE, 0.41),
        (NOSE, LEFT_EAR, 0.79),
        (NOSE, RIGHT_EAR, 0.79),
        (LEFT_EYE, LEFT_EAR, 0.56),
        (RIGHT_EYE, RIGHT_EAR, 0.56),
        // nose tip to an eye: mostly the drop from the eyes, kept in profile
        (NOSE, LEFT_EYE, 0.31),
        (NOSE, RIGHT_EYE, 0.31),
    ];
    let estimates: Vec<f32> = pairs.iter().filter_map(|&(a, b, share)| Some(dist(k(a)?, k(b)?) / share)).collect();
    let mut width = if sure {
        estimates.iter().copied().fold(0.0f32, f32::max)
    } else {
        // unsure keypoints scatter: the middle estimate, not the largest
        let mut e = estimates.clone();
        e.sort_by(f32::total_cmp);
        e.get(e.len() / 2).copied().unwrap_or(0.0)
    };
    if width <= 0.0 {
        // no two face keypoints: the shoulders (their width varies with posture: only then), else
        // a share of the person's box
        width = match (at(LEFT_SHOULDER, hp.keypoint), at(RIGHT_SHOULDER, hp.keypoint)) {
            (Some(a), Some(b)) => dist(a, b) / 2.6,
            _ => 0.0,
        };
        if width <= 0.0 {
            width = (pw * 0.3).min(ph * 0.25);
        }
    }
    if face_kps.len() >= 3 {
        // three face keypoints or more span a good part of the face: a head is at most 3.5
        // times as wide as their spread
        let (sx0, sx1) = face_kps.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), q| (a.min(q.x), b.max(q.x)));
        let (sy0, sy1) = face_kps.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), q| (a.min(q.y), b.max(q.y)));
        let spread = (sx1 - sx0).max(sy1 - sy0);
        if spread > 0.0 {
            width = width.min(3.5 * spread);
        }
    }
    if !sure && face_kps.len() >= 3 && width > 0.0 {
        // unsure keypoints far from the others (more than a head's width from their middle) are
        // wrong guesses: they neither widen nor move the head
        let mid = |f: fn(&Keypoint) -> f32| {
            let mut v: Vec<f32> = face_kps.iter().map(f).collect();
            v.sort_by(f32::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let (mx, my) = (mid(|q| q.x), mid(|q| q.y));
        face_kps.retain(|q| ((q.x - mx).powi(2) + (q.y - my).powi(2)).sqrt() <= width);
        if face_kps.is_empty() {
            return None;
        }
    }
    // never wider than the person (a wrong keypoint far away), never narrower than the face's
    // own keypoints
    let (x0, x1) = face_kps.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), q| (a.min(q.x), b.max(q.x)));
    let (y0, y1) = face_kps.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), q| (a.min(q.y), b.max(q.y)));
    let spread = (x1 - x0).max(y1 - y0);
    width = width.min(pw.max(ph * 0.5) * 0.9).max(spread * 1.1);
    if !(width.is_finite() && width > 0.0) {
        return None;
    }
    let cx = (x0 + x1) / 2.0;
    let eyes: Vec<Keypoint> = [LEFT_EYE, RIGHT_EYE].into_iter().filter_map(k).collect();
    let ears: Vec<Keypoint> = [LEFT_EAR, RIGHT_EAR].into_iter().filter_map(k).collect();
    let eye_y = if !eyes.is_empty() {
        eyes.iter().map(|q| q.y).sum::<f32>() / eyes.len() as f32
    } else if let Some(n) = k(NOSE) {
        n.y - 0.3 * width
    } else {
        ears.iter().map(|q| q.y).sum::<f32>() / ears.len().max(1) as f32 - 0.05 * width
    };
    // the face as the face detector boxes it (FACE_BOX: calibrated against YuNet's faces of the
    // same people)
    let fb = if view == View::Front { FACE_BOX.front } else { FACE_BOX.turned };
    let cy = eye_y + fb.below_eyes * width;
    let (bw, bh) = (fb.width * width, fb.height * width);
    let mut b = [cx - bw / 2.0, cy - bh / 2.0, bw, bh];
    // and around the nose and eyes, with a margin (a tilted head, a face in profile: the box is
    // centred between the nose and the ear)
    let m = 0.1 * width;
    for q in [NOSE, LEFT_EYE, RIGHT_EYE].into_iter().filter_map(k).filter(|q| face_kps.contains(q)) {
        let (ux0, uy0) = (b[0].min(q.x - m), b[1].min(q.y - m));
        let (ux1, uy1) = ((b[0] + b[2]).max(q.x + m), (b[1] + b[3]).max(q.y + m));
        b = [ux0, uy0, ux1 - ux0, uy1 - uy0];
    }
    b.iter().all(|v| v.is_finite()).then_some(Head { bbox: b, view, face_score, width })
}

/// Whether the pose faces away from the camera: its left eyes, ears or shoulders on the picture's
/// left of the right ones (a person facing the camera shows their left on the picture's right),
/// by at least a tenth (eyes), 15 % (ears) or 30 % (shoulders) of the person's width, and none of
/// the pairs saying the opposite as clearly.
pub fn faces_away(p: &Person, hp: &HeadParams) -> bool {
    let pw = p.bbox[2].max(1.0);
    let lr = |l: usize, r: usize| -> Option<f32> {
        let (a, b) = (found(&p.keypoints, l, hp.keypoint)?, found(&p.keypoints, r, hp.keypoint)?);
        Some((a.x - b.x) / pw)
    };
    let cues = [(lr(LEFT_EYE, RIGHT_EYE), 0.1), (lr(LEFT_EAR, RIGHT_EAR), 0.15), (lr(LEFT_SHOULDER, RIGHT_SHOULDER), 0.3)];
    let away = cues.iter().any(|(v, t)| v.is_some_and(|v| v <= -t));
    let toward = cues.iter().any(|(v, t)| v.is_some_and(|v| v >= *t));
    away && !toward
}

fn inside(b: &[f32; 4], x: f32, y: f32) -> bool {
    x >= b[0] && x <= b[0] + b[2] && y >= b[1] && y <= b[1] + b[3]
}

/// Whether a face detector's box `face` is the face of head `head` (both `[x, y, w, h]` in the
/// same units): the face's centre inside the head box grown by a tenth, and their heights within
/// a factor of three.
pub fn same_face(head: &[f32; 4], face: &[f32; 4]) -> bool {
    if !head.iter().chain(face).all(|v| v.is_finite()) || head[2] <= 0.0 || head[3] <= 0.0 || face[3] <= 0.0 {
        return false;
    }
    let (fx, fy) = (face[0] + face[2] / 2.0, face[1] + face[3] / 2.0);
    let (gx, gy) = (head[2] * 0.1, head[3] * 0.1);
    let inside = fx >= head[0] - gx && fx <= head[0] + head[2] + gx && fy >= head[1] - gy && fy <= head[1] + head[3] + gy;
    let ratio = head[3] / face[3];
    inside && (1.0 / 3.0..=3.0).contains(&ratio)
}

/// The faces to add for `people` in a `w` × `h` picture (their boxes in its pixels): the head of
/// each person whose face shows and is not one of `faces` (the face detector's, normalised), as
/// face detections normalised to the picture, marked [`Detection::pose`]. Their score is the
/// worse of the person's and the face keypoints', at most [`HEAD_SCORE_MAX`].
pub fn candidates(people: &[Person], faces: &[Detection], hp: &HeadParams, w: f32, h: f32) -> Vec<Detection> {
    let (w, h) = (w.max(1.0), h.max(1.0));
    let known: Vec<[f32; 4]> =
        faces.iter().filter(|f| f.kind == Kind::Face && !f.pose).map(|f| [f.bbox[0] * w, f.bbox[1] * h, f.bbox[2] * w, f.bbox[3] * h]).collect();
    let heads: Vec<(&Person, Head)> = people.iter().filter_map(|p| Some((p, head_of(p, hp).filter(Head::shows_face)?))).collect();
    // each face of the face detector is one head's, the nearest one it can be (a neighbour's face
    // next to a head box must not hide that head: once the neighbour is blurred, it shows)
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for (hi, (_, hd)) in heads.iter().enumerate() {
        for (fi, f) in known.iter().enumerate() {
            if same_face(&hd.bbox, f) {
                let (dx, dy) = (hd.bbox[0] + hd.bbox[2] / 2.0 - f[0] - f[2] / 2.0, hd.bbox[1] + hd.bbox[3] / 2.0 - f[1] - f[3] / 2.0);
                pairs.push(((dx * dx + dy * dy).sqrt() / hd.bbox[2].max(1e-6), hi, fi));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut head_found, mut face_used) = (vec![false; heads.len()], vec![false; known.len()]);
    for (_, hi, fi) in pairs {
        if head_found.get(hi).copied().unwrap_or(true) || face_used.get(fi).copied().unwrap_or(true) {
            continue;
        }
        head_found[hi] = true;
        face_used[fi] = true;
    }
    let mut out: Vec<Detection> = Vec::new();
    for ((p, head), found_already) in heads.into_iter().zip(head_found) {
        if found_already {
            continue;
        }
        let score = p.score.min(head.face_score).clamp(0.0, HEAD_SCORE_MAX);
        let b = head.bbox;
        let points = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
            .into_iter()
            .filter_map(|i| found(&p.keypoints, i, hp.keypoint))
            .filter(|q| inside(&b, q.x, q.y))
            .map(|q| [q.x / w, q.y / h])
            .collect();
        out.push(Detection { kind: Kind::Face, score, bbox: [b[0] / w, b[1] / h, b[2] / w, b[3] / h], points, pose: true });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kp(x: f32, y: f32, score: f32) -> Keypoint {
        Keypoint { x, y, score }
    }

    /// A person whose head is `width` px wide at `(cx, eye_y)`, seen at `yaw` degrees (0: facing
    /// the camera, 90: right profile, 180: from behind), standing in a box below.
    fn person(cx: f32, eye_y: f32, width: f32, yaw: f32) -> Person {
        let r = width / 2.0;
        let a = yaw.to_radians();
        // a head as a circle seen from above: a point at angle t (0 = straight ahead) shows at
        // x = sin(t − yaw)·r and is visible when cos(t − yaw) > −0.2
        let at = |t: f32, depth: f32| {
            let d = t.to_radians() - a;
            (cx + d.sin() * r * depth, d.cos())
        };
        let mut kps = vec![Keypoint::default(); KEYPOINTS];
        let mut put = |i: usize, t: f32, depth: f32, dy: f32| {
            let (x, facing) = at(t, depth);
            let score = if facing > 0.1 {
                0.9
            } else if facing > -0.2 {
                0.55
            } else {
                0.1
            };
            kps[i] = kp(x, eye_y + dy * width, score);
        };
        put(NOSE, 0.0, 1.15, 0.3);
        put(LEFT_EYE, 30.0, 1.0, 0.0);
        put(RIGHT_EYE, -30.0, 1.0, 0.0);
        put(LEFT_EAR, 95.0, 0.98, 0.05);
        put(RIGHT_EAR, -95.0, 0.98, 0.05);
        kps[LEFT_SHOULDER] = kp(cx + 1.2 * width * a.cos().abs().max(0.2), eye_y + 1.6 * width, 0.9);
        kps[RIGHT_SHOULDER] = kp(cx - 1.2 * width * a.cos().abs().max(0.2), eye_y + 1.6 * width, 0.9);
        Person { bbox: [cx - 1.5 * width, eye_y - 0.8 * width, 3.0 * width, 7.0 * width], score: 0.9, keypoints: kps }
    }

    fn centre(b: &[f32; 4]) -> (f32, f32) {
        (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
    }

    #[test]
    fn a_frontal_head_is_a_box_round_the_face() {
        let p = person(500.0, 300.0, 100.0, 0.0);
        let h = head_of(&p, &HeadParams::default()).unwrap();
        assert_eq!(h.view, View::Front);
        let (cx, cy) = centre(&h.bbox);
        assert!((cx - 500.0).abs() < 5.0, "{:?}", h.bbox);
        // a little below the eyes, about a head wide and 1.3 tall
        assert!(cy > 300.0 && cy < 330.0, "{:?}", h.bbox);
        assert!((h.bbox[2] - 110.0).abs() < 25.0 && (h.bbox[3] / h.bbox[2] - 1.3).abs() < 0.25, "{:?}", h.bbox);
        // it covers the eyes, the nose and the ears
        for i in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR] {
            let q = p.keypoints[i];
            assert!(q.x >= h.bbox[0] && q.x <= h.bbox[0] + h.bbox[2] && q.y >= h.bbox[1] && q.y <= h.bbox[1] + h.bbox[3]);
        }
    }

    #[test]
    fn a_profile_keeps_its_size_and_covers_nose_to_ear() {
        let front = head_of(&person(500.0, 300.0, 100.0, 0.0), &HeadParams::default()).unwrap();
        let p = person(500.0, 300.0, 100.0, 80.0);
        let side = head_of(&p, &HeadParams::default()).unwrap();
        assert_eq!(side.view, View::Profile);
        // the turned head is measured from nose to ear: about as large as the frontal one
        assert!((side.bbox[2] / front.bbox[2] - 1.0).abs() < 0.35, "{:?} vs {:?}", side.bbox, front.bbox);
        let (nose, ear) = (p.keypoints[NOSE], p.keypoints[LEFT_EAR]);
        for q in [nose, ear] {
            assert!(q.x >= side.bbox[0] && q.x <= side.bbox[0] + side.bbox[2], "{q:?} {:?}", side.bbox);
        }
        // centred between them, not on the back of the head
        let (cx, _) = centre(&side.bbox);
        assert!((cx - (nose.x + ear.x) / 2.0).abs() < 0.25 * side.bbox[2]);
    }

    #[test]
    fn the_back_of_a_head_is_not_a_face() {
        let p = person(500.0, 300.0, 100.0, 180.0);
        let h = head_of(&p, &HeadParams::default()).unwrap();
        assert_eq!(h.view, View::Back);
        assert!(!h.shows_face());
        assert!(candidates(&[p], &[], &HeadParams::default(), 1920.0, 1080.0).is_empty());
        // an ear and the back of the head, three quarters from behind: still not a face
        let p = person(500.0, 300.0, 100.0, 140.0);
        assert!(!head_of(&p, &HeadParams::default()).unwrap().shows_face());
        // a profile is
        assert!(head_of(&person(500.0, 300.0, 100.0, -85.0), &HeadParams::default()).unwrap().shows_face());
    }

    #[test]
    fn a_half_hidden_face_counts_unless_the_pose_faces_away() {
        // a face half hidden behind a robot arm: its keypoints only fairly sure
        let mut p = person(500.0, 300.0, 100.0, 0.0);
        for i in [NOSE, LEFT_EYE, RIGHT_EYE] {
            p.keypoints[i].score = 0.42;
        }
        let hp = HeadParams::default();
        assert!(!faces_away(&p, &hp));
        assert!(head_of(&p, &hp).unwrap().shows_face());
        // the same scores on a pose seen from behind (its left on the picture's left): the back of
        // a head
        let mut back = p.clone();
        for (l, r) in [(LEFT_EYE, RIGHT_EYE), (LEFT_EAR, RIGHT_EAR), (LEFT_SHOULDER, RIGHT_SHOULDER)] {
            let (a, b) = (back.keypoints[l].x, back.keypoints[r].x);
            back.keypoints[l].x = b;
            back.keypoints[r].x = a;
        }
        assert!(faces_away(&back, &hp));
        assert_eq!(head_of(&back, &hp).unwrap().view, View::Back);
        // a sure face is a face whichever way the shoulders say
        for i in [NOSE, LEFT_EYE, RIGHT_EYE] {
            back.keypoints[i].score = 0.8;
        }
        assert!(head_of(&back, &hp).unwrap().shows_face());
        // too unsure: not a face
        for i in [NOSE, LEFT_EYE, RIGHT_EYE] {
            p.keypoints[i].score = 0.2;
        }
        assert!(!head_of(&p, &hp).unwrap().shows_face());
    }

    #[test]
    fn unsure_keypoints_do_not_blow_the_head_up() {
        // sure eyes and nose; an unsure ear far off (a wrong guess): the sure keypoints measure
        let mut p = person(500.0, 300.0, 100.0, 0.0);
        p.keypoints[LEFT_EAR] = kp(700.0, 320.0, 0.35);
        let h = head_of(&p, &HeadParams::default()).unwrap();
        assert!(h.bbox[2] < 160.0, "{:?}", h.bbox);
        // nothing sure: the middle of the estimates, not the largest
        let mut q = person(500.0, 300.0, 100.0, 0.0);
        for k in q.keypoints.iter_mut().take(5) {
            k.score = 0.4;
        }
        q.keypoints[LEFT_EAR] = kp(720.0, 320.0, 0.4);
        let h = head_of(&q, &HeadParams::default()).unwrap();
        assert!(h.bbox[3] < 2.5 * 130.0, "{:?}", h.bbox);
    }

    #[test]
    fn no_face_keypoints_is_no_head_and_hostile_poses_are_refused() {
        let mut p = person(500.0, 300.0, 100.0, 0.0);
        for k in p.keypoints.iter_mut().take(5) {
            k.score = 0.05;
        }
        assert!(head_of(&p, &HeadParams::default()).is_none());
        let mut p = person(500.0, 300.0, 100.0, 0.0);
        p.keypoints[NOSE].x = f32::NAN;
        p.keypoints[LEFT_EYE].score = f32::INFINITY;
        let h = head_of(&p, &HeadParams::default());
        assert!(h.is_none_or(|h| h.bbox.iter().all(|v| v.is_finite())));
        let p = Person { bbox: [f32::NAN, 0.0, f32::INFINITY, 1.0], score: 0.9, keypoints: person(1.0, 1.0, 1.0, 0.0).keypoints };
        assert!(head_of(&p, &HeadParams::default()).is_none());
        assert!(head_of(&Person { bbox: [0.0; 4], score: 1.0, keypoints: Vec::new() }, &HeadParams::default()).is_none());
        // a wrong ear far away does not make the head wider than the person
        let mut p = person(500.0, 300.0, 100.0, 0.0);
        p.keypoints[LEFT_EAR].x = 5000.0;
        let h = head_of(&p, &HeadParams::default()).unwrap();
        assert!(h.bbox[2] < 200.0 && h.bbox[0] > 300.0, "{:?}", h.bbox);
    }

    #[test]
    fn heads_the_face_detector_found_are_not_added_again() {
        let (w, h) = (1920.0, 1080.0);
        let a = person(500.0, 300.0, 100.0, 0.0);
        let b = person(1200.0, 320.0, 90.0, 85.0);
        let hp = HeadParams::default();
        let all = candidates(&[a.clone(), b.clone()], &[], &hp, w, h);
        assert_eq!(all.len(), 2);
        assert!(all.iter().all(|d| d.pose && d.kind == Kind::Face && d.score <= HEAD_SCORE_MAX && d.points.len() >= 3));
        // YuNet found a's face (a box a little smaller than the head's): only b is added
        let ha = head_of(&a, &hp).unwrap().bbox;
        let face = Detection {
            kind: Kind::Face,
            score: 0.95,
            bbox: [(ha[0] + 10.0) / w, (ha[1] + 15.0) / h, (ha[2] - 20.0) / w, (ha[3] - 30.0) / h],
            points: Vec::new(),
            pose: false,
        };
        let added = candidates(&[a.clone(), b.clone()], std::slice::from_ref(&face), &hp, w, h);
        assert_eq!(added.len(), 1);
        let hb = head_of(&b, &hp).unwrap().bbox;
        assert!((added[0].bbox[0] - hb[0] / w).abs() < 1e-6);
        // a face elsewhere, or of a very different size, is not a's
        let far = Detection { bbox: [0.9, 0.9, 0.05, 0.05], ..face.clone() };
        let tiny = Detection { bbox: [ha[0] / w + 0.02, ha[1] / h + 0.05, 8.0 / w, 8.0 / h], ..face.clone() };
        assert_eq!(candidates(std::slice::from_ref(&a), &[far, tiny], &hp, w, h).len(), 1);
        // a pose head among the faces is not a detector's face
        let pose_face = Detection { pose: true, ..face };
        assert_eq!(candidates(&[a], &[pose_face], &hp, w, h).len(), 1);
    }

    #[test]
    fn a_neighbours_face_does_not_hide_a_head() {
        // two visitors one above the other, their head boxes touching; the face detector found
        // only the upper one's face, its centre inside both head boxes
        let (w, h) = (1920.0, 1080.0);
        let hp = HeadParams::default();
        let upper = person(500.0, 300.0, 100.0, 0.0);
        let lower = person(520.0, 380.0, 100.0, 80.0);
        let (hu, hl) = (head_of(&upper, &hp).unwrap().bbox, head_of(&lower, &hp).unwrap().bbox);
        let face_px = [hu[0] + 10.0, hu[1] + 30.0, hu[2] - 20.0, hu[3] - 30.0];
        let fc = (face_px[0] + face_px[2] / 2.0, face_px[1] + face_px[3] / 2.0);
        assert!(same_face(&hu, &face_px));
        // (the setup: the face would pass for the lower head too)
        let lower_too = same_face(&hl, &face_px);
        assert!(lower_too, "lower {hl:?} face centre {fc:?}");
        let face =
            Detection { kind: Kind::Face, score: 0.9, bbox: [face_px[0] / w, face_px[1] / h, face_px[2] / w, face_px[3] / h], points: Vec::new(), pose: false };
        let added = candidates(&[lower.clone(), upper.clone()], std::slice::from_ref(&face), &hp, w, h);
        assert_eq!(added.len(), 1, "lower {hl:?} upper {hu:?} face centre {fc:?} (inside the lower head too: {lower_too})");
        assert!((added[0].bbox[1] * h - hl[1]).abs() < 1.0, "the lower head is added");
        // two faces found: nothing to add
        let lower_face = Detection { bbox: [(hl[0] + 10.0) / w, (hl[1] + 30.0) / h, (hl[2] - 20.0) / w, (hl[3] - 30.0) / h], ..face.clone() };
        assert!(candidates(&[lower, upper], &[face, lower_face], &hp, w, h).is_empty());
    }

    /// The head box is the face detector's box of that face ([`FACE_BOX`]): from the front a
    /// face box 0.88 × 1.14 head widths, its centre 0.13 widths below the eyes and between the
    /// ears; turned 0.98 × 1.34 (the turn shortens the distances that measure the width); the
    /// nose and eyes always inside with a margin. (Against YuNet's faces of the same people on
    /// real frames: `head_boxes_match_the_face_detectors`, ignored.)
    #[test]
    fn head_boxes_are_the_size_of_the_face_detectors_faces() {
        let hp = HeadParams::default();
        for width in [40.0f32, 100.0, 400.0] {
            let p = person(800.0, 300.0, width, 0.0);
            let h = head_of(&p, &hp).unwrap();
            assert_eq!(h.view, View::Front);
            let w = h.width;
            assert!(w > 0.0 && (w / width - 1.0).abs() < 0.35, "the width measured: {w} for {width}");
            let fb = FACE_BOX.front;
            assert!((h.bbox[2] - fb.width * w).abs() < 1e-3 * w, "{:?}", h.bbox);
            assert!((h.bbox[3] - fb.height * w).abs() < 1e-3 * w, "{:?}", h.bbox);
            let (cx, cy) = centre(&h.bbox);
            assert!((cx - 800.0).abs() < 0.02 * w && (cy - (300.0 + fb.below_eyes * w)).abs() < 1e-3 * w, "{:?}", h.bbox);
            // a face box about as tall as wide × 1.3, smaller than the old head box (1.3 widths
            // tall, a width wide)
            assert!((h.bbox[3] / h.bbox[2] - 1.3).abs() < 0.05);
            assert!(h.bbox[3] < 1.3 * w && h.bbox[2] < w);
        }
        // turned: the turned shape, still round the nose and the eye
        let p = person(800.0, 300.0, 100.0, 80.0);
        let h = head_of(&p, &hp).unwrap();
        assert_eq!(h.view, View::Profile);
        assert!(h.bbox[3] >= FACE_BOX.turned.height * h.width - 1e-3, "{:?} {}", h.bbox, h.width);
        for i in [NOSE, LEFT_EYE] {
            let q = p.keypoints[i];
            if q.score >= hp.sure {
                assert!(q.x > h.bbox[0] && q.x < h.bbox[0] + h.bbox[2] && q.y > h.bbox[1] && q.y < h.bbox[1] + h.bbox[3], "{i}: {q:?} {:?}", h.bbox);
            }
        }
        // the turned face is larger per measured width than the frontal one, and both are
        // finite
        const { assert!(FACE_BOX.turned.width > FACE_BOX.front.width && FACE_BOX.turned.height > FACE_BOX.front.height) };
        for f in [FACE_BOX.front, FACE_BOX.turned] {
            assert!([f.width, f.height, f.below_eyes].iter().all(|v| v.is_finite() && *v > 0.0 && *v < 2.0));
        }
    }

    #[test]
    fn same_face_needs_the_centre_inside_and_a_similar_size() {
        let head = [100.0, 100.0, 100.0, 130.0];
        assert!(same_face(&head, &[120.0, 120.0, 60.0, 80.0]));
        assert!(!same_face(&head, &[400.0, 120.0, 60.0, 80.0]));
        assert!(!same_face(&head, &[140.0, 150.0, 10.0, 10.0]));
        assert!(!same_face(&head, &[f32::NAN, 120.0, 60.0, 80.0]));
        assert!(!same_face(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]));
    }
}
