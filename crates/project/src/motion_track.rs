//! Motion tracks (CapCut-style object tracking): one object followed through a clip's media, kept
//! in the project as a named [`MotionTrack`], and what follows it ([`Attachment`]s).
//!
//! A track belongs to a source clip: its [`TrackSample`]s are in that clip's **media time** and in
//! shares of its **source picture** (0..1 of the width and height, before Motion), so they stay
//! right however the clip is trimmed, moved, sped up or transformed. The engine
//! (`filmcraft_engine::motion_track`) computes them (subject tracking or KLT points + RANSAC) and
//! maps them through the clip — its speed, reverse, frame hold and Motion — into sequence pixels
//! when it writes an attachment's keyframes.
//!
//! **Attachments are baked, and stay linked.** `track.attach` writes ordinary keyframes on the
//! target (a clip's Motion, a graphic layer's position, an effect's point parameter or a mask
//! path), so the renderer, Effect Controls, the graph editor, interchange and the web build need
//! nothing new. The [`Attachment`] record keeps the link: refining the track re-writes every
//! attachment's keyframes from the new samples (one undo step), and `track.rebake` does the same
//! after the clips were trimmed or moved.
//!
//! Everything here is read from project files, so it is untrusted: [`MotionTrack::sanitize`]
//! drops non-finite and out-of-order samples and caps the counts; every query also copes with
//! what is left (empty tracks, NaN, unsorted).

use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};

use crate::mask::MaskPath;
use crate::{ClipId, ItemId};

/// Most samples one track keeps (one a frame: 55 minutes at 60 fps).
pub const MAX_SAMPLES: usize = 200_000;
/// Most attachments one track drives.
pub const MAX_ATTACHMENTS: usize = 256;
/// Longest track name, characters.
pub const MAX_NAME_CHARS: usize = 200;
/// Smallest box side, as a share of the picture.
pub const MIN_BOX: f64 = 0.005;

/// A motion track's id (from the project's id counter).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MotionTrackId(pub u64);

/// How a track follows its object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrackingMethod {
    /// The whole object (Median Flow with template re-detection): a person, an animal, a robot
    /// walking. Position and size; no rotation.
    #[default]
    Subject,
    /// A flat surface (KLT corner points and a RANSAC similarity): a screen, a sign, a logo, a
    /// face of a box. Position, size and rotation.
    Points,
}

impl TrackingMethod {
    pub const ALL: [TrackingMethod; 2] = [TrackingMethod::Subject, TrackingMethod::Points];
    pub fn name(self) -> &'static str {
        match self {
            TrackingMethod::Subject => "subject",
            TrackingMethod::Points => "points",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "subject" | "object" => Some(TrackingMethod::Subject),
            "points" | "planar" | "point" | "klt" => Some(TrackingMethod::Points),
            _ => None,
        }
    }
}

/// Which way a track runs from the frame it was seeded on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrackDirection {
    Forward,
    Backward,
    #[default]
    Both,
}

impl TrackDirection {
    pub fn name(self) -> &'static str {
        match self {
            TrackDirection::Forward => "forward",
            TrackDirection::Backward => "backward",
            TrackDirection::Both => "both",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "forward" | "forwards" => Some(TrackDirection::Forward),
            "backward" | "backwards" | "back" => Some(TrackDirection::Backward),
            "both" => Some(TrackDirection::Both),
            _ => None,
        }
    }
    pub fn forward(self) -> bool {
        self != TrackDirection::Backward
    }
    pub fn backward(self) -> bool {
        self != TrackDirection::Forward
    }
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}
fn one() -> f64 {
    1.0
}
fn is_one(v: &f64) -> bool {
    *v == 1.0
}
fn full() -> f32 {
    1.0
}
fn is_full(v: &f32) -> bool {
    *v == 1.0
}

/// Where the object is in one frame of the source: the centre and size of its box as shares of
/// the source picture (`cx`, `w` of its width; `cy`, `h` of its height), its rotation since the
/// seed (degrees, clockwise on screen as Motion's) and its size relative to the seed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackSample {
    /// Media time of the frame.
    pub t: Tick,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub rotation: f64,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub scale: f64,
    /// How sure the tracker was, 0..1.
    #[serde(default = "full", skip_serializing_if = "is_full")]
    pub confidence: f32,
    /// Not seen in this frame (the box holds where it was last seen).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lost: bool,
}

impl TrackSample {
    /// A seed box from its top-left corner and size (shares of the picture), at media time `t`:
    /// `None` for non-finite numbers, a size at or below zero, or a box wholly outside the picture.
    /// A box partly outside is cut to the picture.
    pub fn from_corner(t: Tick, x: f64, y: f64, w: f64, h: f64) -> Option<Self> {
        if ![x, y, w, h].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
            return None;
        }
        let (x0, y0, x1, y1) = (x.max(0.0), y.max(0.0), (x + w).min(1.0), (y + h).min(1.0));
        if x1 - x0 < MIN_BOX || y1 - y0 < MIN_BOX {
            return None;
        }
        Some(TrackSample { t, cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, w: x1 - x0, h: y1 - y0, rotation: 0.0, scale: 1.0, confidence: 1.0, lost: false })
    }
    /// Every number finite, the size positive, the confidence 0..1.
    pub fn is_valid(&self) -> bool {
        [self.cx, self.cy, self.w, self.h, self.rotation, self.scale].iter().all(|v| v.is_finite())
            && self.w > 0.0
            && self.h > 0.0
            && self.scale > 0.0
            && self.confidence.is_finite()
    }
    pub fn pose(&self) -> Pose {
        Pose {
            cx: self.cx,
            cy: self.cy,
            w: self.w,
            h: self.h,
            rotation: self.rotation,
            scale: self.scale,
            confidence: self.confidence.clamp(0.0, 1.0),
            lost: self.lost,
        }
    }
}

/// The object at one moment (between samples: interpolated).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub rotation: f64,
    pub scale: f64,
    pub confidence: f32,
    pub lost: bool,
}

/// What an attachment takes from the object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FollowMode {
    #[default]
    Position,
    PositionScale,
    PositionScaleRotation,
}

impl FollowMode {
    pub const ALL: [FollowMode; 3] = [FollowMode::Position, FollowMode::PositionScale, FollowMode::PositionScaleRotation];
    pub fn name(self) -> &'static str {
        match self {
            FollowMode::Position => "position",
            FollowMode::PositionScale => "positionScale",
            FollowMode::PositionScaleRotation => "positionScaleRotation",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name().eq_ignore_ascii_case(s.trim()))
    }
    pub fn scale(self) -> bool {
        self != FollowMode::Position
    }
    pub fn rotation(self) -> bool {
        self == FollowMode::PositionScaleRotation
    }
}

/// What follows a track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum AttachTarget {
    /// A clip's Motion (a sticker, a picture, a graphic clip as a whole) on any video track.
    Clip { clip: ClipId },
    /// One layer of a graphic clip (by its [`crate::graphic_design::LayerExtra::uid`]): its
    /// Position, Scale and Rotation.
    GraphicLayer { clip: ClipId, layer: u64 },
    /// A point parameter of an effect on a clip (`effect`: its index in the clip's effects), e.g.
    /// a blur's or a lens flare's centre; in that clip's pixels.
    EffectParam { clip: ClipId, effect: usize, param: String },
    /// A mask path of an effect on a clip; `base` is the path at the moment it was attached (clip
    /// pixels), moved, scaled and turned with the object.
    Mask { clip: ClipId, effect: usize, mask: usize, base: MaskPath },
}

impl AttachTarget {
    pub fn clip(&self) -> ClipId {
        match self {
            AttachTarget::Clip { clip }
            | AttachTarget::GraphicLayer { clip, .. }
            | AttachTarget::EffectParam { clip, .. }
            | AttachTarget::Mask { clip, .. } => *clip,
        }
    }
    pub fn kind_name(&self) -> &'static str {
        match self {
            AttachTarget::Clip { .. } => "clip",
            AttachTarget::GraphicLayer { .. } => "graphicLayer",
            AttachTarget::EffectParam { .. } => "effectParam",
            AttachTarget::Mask { .. } => "mask",
        }
    }
}

fn hundred() -> f64 {
    100.0
}

/// A thing following a track, and how ([`crate::motion_track`] module docs: baked, linked).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attachment {
    pub id: u64,
    pub target: AttachTarget,
    pub follow: FollowMode,
    /// Where the attached point sits from the object's centre, sequence pixels, at [`Self::at`]
    /// (with scale and rotation following, it turns and grows with the object).
    pub offset: [f64; 2],
    /// Sequence time the attachment was made at (informational: the reference below is what
    /// counts).
    pub at: Tick,
    /// The object as shown at [`Self::at`]: its scale (1 = as tracked at the seed, times the
    /// tracked clip's Motion scale) and rotation (degrees). Scale and rotation follow relative to
    /// these.
    #[serde(default = "one")]
    pub ref_scale: f64,
    #[serde(default)]
    pub ref_rotation: f64,
    /// The target's own scale (percent) and rotation (degrees) at [`Self::at`].
    #[serde(default = "hundred")]
    pub base_scale: f64,
    #[serde(default)]
    pub base_rotation: f64,
}

/// One object followed through a clip ([`crate::motion_track`] module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionTrack {
    pub id: MotionTrackId,
    pub name: String,
    /// The media item followed: the samples are in its media time.
    pub item: ItemId,
    /// The timeline clip the track was made on and its sequence, which place attachments in
    /// sequence time.
    pub sequence: ItemId,
    pub clip: ClipId,
    pub method: TrackingMethod,
    pub direction: TrackDirection,
    /// The box it was seeded with, at the media time it was drawn on.
    pub seed: TrackSample,
    /// One a tracked frame, by media time.
    #[serde(default)]
    pub samples: Vec<TrackSample>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,
}

impl MotionTrack {
    /// Make a track read from a file safe to use: samples finite, sorted by time and unique, at
    /// most [`MAX_SAMPLES`]; at most [`MAX_ATTACHMENTS`] attachments; the name on one line.
    pub fn sanitize(&mut self) {
        self.samples.retain(TrackSample::is_valid);
        self.samples.truncate(MAX_SAMPLES);
        self.samples.sort_by_key(|s| s.t);
        self.samples.dedup_by_key(|s| s.t);
        for s in &mut self.samples {
            s.confidence = s.confidence.clamp(0.0, 1.0);
        }
        self.attachments.truncate(MAX_ATTACHMENTS);
        self.attachments.retain(|a| {
            a.offset.iter().all(|v| v.is_finite())
                && [a.base_scale, a.base_rotation, a.ref_rotation].iter().all(|v| v.is_finite())
                && a.ref_scale.is_finite()
                && a.ref_scale > 0.0
        });
        self.name = self.name.chars().filter(|c| !c.is_control()).take(MAX_NAME_CHARS).collect();
        if !self.seed.is_valid()
            && let Some(first) = self.samples.first()
        {
            self.seed = *first;
        }
    }

    /// First and last sample times.
    pub fn range(&self) -> Option<(Tick, Tick)> {
        Some((self.samples.first()?.t, self.samples.last()?.t))
    }

    /// The object at media time `t`: interpolated between the samples around it (the position and
    /// size linearly, the scale evenly on a log scale, the rotation the short way round); held at
    /// the first or last sample outside them. Lost where the nearer sample is. `None` without
    /// usable samples.
    pub fn pose_at(&self, t: Tick) -> Option<Pose> {
        let s = &self.samples;
        let i = s.partition_point(|x| x.t <= t);
        let after = s.get(i).filter(|x| x.is_valid());
        let before = i.checked_sub(1).and_then(|k| s.get(k)).filter(|x| x.is_valid());
        match (before, after) {
            (Some(a), Some(b)) => {
                let span = (b.t - a.t).0 as f64;
                let u = if span > 0.0 { ((t - a.t).0 as f64 / span).clamp(0.0, 1.0) } else { 0.0 };
                let lerp = |x: f64, y: f64| x + (y - x) * u;
                let mut dr = (b.rotation - a.rotation) % 360.0;
                if dr > 180.0 {
                    dr -= 360.0;
                } else if dr < -180.0 {
                    dr += 360.0;
                }
                let scale = (a.scale.ln() + (b.scale.ln() - a.scale.ln()) * u).exp();
                Some(Pose {
                    cx: lerp(a.cx, b.cx),
                    cy: lerp(a.cy, b.cy),
                    w: lerp(a.w, b.w),
                    h: lerp(a.h, b.h),
                    rotation: a.rotation + dr * u,
                    scale: if scale.is_finite() && scale > 0.0 { scale } else { 1.0 },
                    confidence: (a.confidence + (b.confidence - a.confidence) * u as f32).clamp(0.0, 1.0),
                    lost: if u < 0.5 { a.lost } else { b.lost },
                })
            }
            (Some(a), None) => Some(a.pose()),
            (None, Some(b)) => Some(b.pose()),
            (None, None) => s.iter().find(|x| x.is_valid()).map(TrackSample::pose),
        }
    }

    /// The stretches (first and last lost sample's media time) where the object was not seen.
    pub fn lost_spans(&self) -> Vec<(Tick, Tick)> {
        let mut out: Vec<(Tick, Tick)> = Vec::new();
        let mut open: Option<(Tick, Tick)> = None;
        for s in &self.samples {
            match (&mut open, s.lost) {
                (Some(span), true) => span.1 = s.t,
                (None, true) => open = Some((s.t, s.t)),
                (Some(_), false) => out.extend(open.take()),
                (None, false) => {}
            }
        }
        out.extend(open);
        out
    }

    /// Share (0..1) of the samples where the object was seen.
    pub fn seen_share(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().filter(|s| !s.lost).count() as f64 / self.samples.len() as f64
    }

    /// Replace the samples from `from` on (`forward`) or up to `from` (backward), inclusive,
    /// with `new`; the samples on the other side stay. The result is sorted and capped.
    pub fn splice(&mut self, from: Tick, forward: bool, new: Vec<TrackSample>) {
        self.samples.retain(|s| if forward { s.t < from } else { s.t > from });
        self.samples.extend(new.into_iter().filter(TrackSample::is_valid));
        self.samples.sort_by_key(|s| s.t);
        self.samples.dedup_by_key(|s| s.t);
        self.samples.truncate(MAX_SAMPLES);
    }

    /// The next attachment id.
    pub fn next_attachment_id(&self) -> u64 {
        self.attachments.iter().map(|a| a.id).max().map_or(1, |m| m.saturating_add(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t: i64, cx: f64, lost: bool) -> TrackSample {
        TrackSample { t: Tick(t), cx, cy: 0.5, w: 0.1, h: 0.2, rotation: t as f64, scale: 1.0, confidence: 0.9, lost }
    }

    fn track(samples: Vec<TrackSample>) -> MotionTrack {
        MotionTrack {
            id: MotionTrackId(7),
            name: "Track 1".into(),
            item: ItemId(1),
            sequence: ItemId(2),
            clip: ClipId(3),
            method: TrackingMethod::Subject,
            direction: TrackDirection::Both,
            seed: samples.first().copied().unwrap_or_else(|| sample(0, 0.5, false)),
            samples,
            attachments: Vec::new(),
        }
    }

    #[test]
    fn poses_interpolate_and_hold_at_the_ends() {
        let t = track(vec![sample(0, 0.2, false), sample(100, 0.4, false), sample(200, 0.6, true)]);
        let p = t.pose_at(Tick(50)).unwrap();
        assert!((p.cx - 0.3).abs() < 1e-12 && (p.rotation - 50.0).abs() < 1e-9 && !p.lost);
        assert_eq!(t.pose_at(Tick(-10)).unwrap().cx, 0.2);
        assert_eq!(t.pose_at(Tick(900)).unwrap().cx, 0.6);
        assert!(t.pose_at(Tick(190)).unwrap().lost);
        assert_eq!(t.lost_spans(), vec![(Tick(200), Tick(200))]);
        assert!(track(Vec::new()).pose_at(Tick(0)).is_none());
    }

    #[test]
    fn rotation_interpolates_the_short_way() {
        let mut a = sample(0, 0.5, false);
        let mut b = sample(100, 0.5, false);
        a.rotation = 170.0;
        b.rotation = -170.0;
        let p = track(vec![a, b]).pose_at(Tick(50)).unwrap();
        assert!((p.rotation - 180.0).abs() < 1e-9, "{}", p.rotation);
    }

    #[test]
    fn hostile_tracks_are_sanitized() {
        let mut bad = sample(5, f64::NAN, false);
        bad.scale = -1.0;
        let mut t = track(vec![sample(100, 0.5, false), bad, sample(0, 0.1, false), sample(0, 0.2, false)]);
        t.seed.w = f64::INFINITY;
        t.name = "a\nb".into();
        t.sanitize();
        assert_eq!(t.samples.iter().map(|s| s.t.0).collect::<Vec<_>>(), vec![0, 100]);
        assert!(t.seed.is_valid());
        assert_eq!(t.name, "ab");
        // a box: no NaN, no zero size, not wholly outside; cut to the picture when partly outside
        assert!(TrackSample::from_corner(Tick(0), f64::NAN, 0.0, 0.1, 0.1).is_none());
        assert!(TrackSample::from_corner(Tick(0), 0.1, 0.1, 0.0, 0.1).is_none());
        assert!(TrackSample::from_corner(Tick(0), 1.2, 0.1, 0.3, 0.1).is_none());
        let s = TrackSample::from_corner(Tick(0), -0.1, 0.5, 0.3, 0.2).unwrap();
        assert!((s.cx - 0.1).abs() < 1e-12 && (s.w - 0.2).abs() < 1e-12);
    }

    #[test]
    fn splice_keeps_the_other_side() {
        let mut t = track((0..10).map(|k| sample(k * 10, 0.1, false)).collect());
        t.splice(Tick(50), true, vec![sample(50, 0.9, false), sample(60, 0.9, false)]);
        assert_eq!(t.samples.len(), 7);
        assert_eq!(t.samples.last().unwrap().cx, 0.9);
        assert_eq!(t.samples.first().unwrap().cx, 0.1);
        t.splice(Tick(20), false, vec![sample(20, 0.5, false)]);
        assert_eq!(t.samples.first().unwrap().t, Tick(20));
        assert_eq!(t.samples.first().unwrap().cx, 0.5);
    }

    #[test]
    fn serde_round_trip_and_defaults() {
        let mut t = track(vec![sample(0, 0.2, false), sample(10, 0.3, true)]);
        t.attachments.push(Attachment {
            id: 1,
            target: AttachTarget::Mask { clip: ClipId(3), effect: 2, mask: 0, base: MaskPath::rect(0.0, 0.0, 10.0, 10.0) },
            follow: FollowMode::PositionScale,
            offset: [1.0, -2.0],
            at: Tick(5),
            ref_scale: 1.0,
            ref_rotation: 0.0,
            base_scale: 100.0,
            base_rotation: 0.0,
        });
        let v = serde_json::to_value(&t).unwrap();
        assert_eq!(v["attachments"][0]["target"]["kind"], "mask");
        assert_eq!(v["method"], "subject");
        let back: MotionTrack = serde_json::from_value(v).unwrap();
        assert_eq!(back, t);
        // a minimal sample reads with its defaults
        let s: TrackSample = serde_json::from_str(r#"{"t":0,"cx":0.5,"cy":0.5,"w":0.1,"h":0.1}"#).unwrap();
        assert_eq!((s.scale, s.rotation, s.confidence, s.lost), (1.0, 0.0, 1.0, false));
    }
}
