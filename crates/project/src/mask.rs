//! Effect masks (Premiere's per-effect masks: ellipse, 4-point polygon or rounded rectangle,
//! free-draw Bézier).
//!
//! Every mask is a closed cubic Bézier path in **clip pixel space** (full-resolution source
//! pixels, the space of effect point parameters). An ellipse is four smooth vertices with circular
//! tangents, a polygon is vertices with zero-length tangents and a rounded corner is a pair of
//! vertices joined by a circular-arc cubic ([`MaskPath::with_rounded_corners`]), so all the tools
//! produce the same [`MaskPath`] and the same rasterizer, keyframe interpolation and tracker apply
//! to each.
//!
//! A mask has a keyframable *Mask Path* (vertex-wise interpolation when the vertex counts match,
//! hold otherwise), *Feather* (falloff width in pixels, centred on the edge), *Opacity* (%),
//! *Expansion* (pixels, grows/shrinks the shape by signed distance), *Inverted*, and a combine
//! [`MaskMode`] with the masks above it. Coverage is computed by `filmcraft_render::mask`.

use filmcraft_geom::{Affine, Vec2};
use serde::{Deserialize, Serialize};

use crate::effect::{ParamDef, ParamKind};
use crate::keyframe::{Param, ParamValue};

/// Circle tangent length for a 4-segment cubic approximation of a quarter circle.
pub const KAPPA: f64 = 0.552_284_749_830_793_4;

/// One path vertex with its incoming/outgoing Bézier tangents (relative to `p`; zero = corner).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MaskVertex {
    pub p: Vec2,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub t_in: Vec2,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub t_out: Vec2,
}

fn is_zero(v: &Vec2) -> bool {
    v.x == 0.0 && v.y == 0.0
}

impl MaskVertex {
    pub fn corner(p: Vec2) -> Self {
        Self { p, t_in: Vec2::ZERO, t_out: Vec2::ZERO }
    }
    /// A smooth vertex with mirrored tangents (`t_out = t`, `t_in = -t`).
    pub fn smooth(p: Vec2, t: Vec2) -> Self {
        Self { p, t_in: t * -1.0, t_out: t }
    }
    pub fn is_corner(&self) -> bool {
        is_zero(&self.t_in) && is_zero(&self.t_out)
    }
}

/// A closed cubic Bézier path (an open path is drawn but still filled as if closed).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MaskPath {
    pub vertices: Vec<MaskVertex>,
    #[serde(default = "yes")]
    pub closed: bool,
}

fn yes() -> bool {
    true
}

impl MaskPath {
    /// An axis-aligned ellipse (four smooth vertices: top, right, bottom, left).
    pub fn ellipse(center: Vec2, radius: Vec2) -> Self {
        let (rx, ry) = (radius.x, radius.y);
        let k = KAPPA;
        let v = vec![
            MaskVertex::smooth(center + Vec2::new(0.0, -ry), Vec2::new(rx * k, 0.0)),
            MaskVertex::smooth(center + Vec2::new(rx, 0.0), Vec2::new(0.0, ry * k)),
            MaskVertex::smooth(center + Vec2::new(0.0, ry), Vec2::new(-rx * k, 0.0)),
            MaskVertex::smooth(center + Vec2::new(-rx, 0.0), Vec2::new(0.0, -ry * k)),
        ];
        Self { vertices: v, closed: true }
    }
    /// A polygon through `points` (corner vertices).
    pub fn polygon(points: &[Vec2]) -> Self {
        Self { vertices: points.iter().map(|p| MaskVertex::corner(*p)).collect(), closed: true }
    }
    /// An axis-aligned rectangle (Premiere's 4-point polygon mask default).
    pub fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self::polygon(&[Vec2::new(x0, y0), Vec2::new(x1, y0), Vec2::new(x1, y1), Vec2::new(x0, y1)])
    }

    /// An axis-aligned rectangle with corners rounded by `radius` (Premiere 26's rounded
    /// rectangle mask). `radius` ≤ 0 (or not finite) gives the plain 4-point [`MaskPath::rect`].
    pub fn rounded_rect(x0: f64, y0: f64, x1: f64, y1: f64, radius: f64) -> Self {
        let r = Self::rect(x0, y0, x1, y1);
        if radius.is_finite() && radius > 0.0 { r.with_rounded_corners(radius, &[0, 1, 2, 3]) } else { r }
    }

    /// Indices of the sharp corners [`MaskPath::with_rounded_corners`] can round: corner vertices
    /// of a closed path whose two segments are straight lines.
    pub fn roundable_corners(&self) -> Vec<usize> {
        let n = self.vertices.len();
        if !self.closed || n < 3 {
            return Vec::new();
        }
        (0..n)
            .filter(|&i| {
                let (Some(prev), Some(v), Some(next)) = (self.vertices.get((i + n - 1) % n), self.vertices.get(i), self.vertices.get((i + 1) % n)) else {
                    return false;
                };
                v.is_corner() && is_zero(&prev.t_out) && is_zero(&next.t_in)
            })
            .collect()
    }

    /// The path with each listed corner replaced by a circular fillet of `radius`: two vertices on
    /// the adjacent edges joined by a cubic arc (handles `4/3·tan(φ/4)·r` for a turn of φ, so a
    /// right angle gets the [`KAPPA`] quarter circle). The fillet is shrunk so it never takes more
    /// than half of either edge. Every listed roundable corner ([`MaskPath::roundable_corners`])
    /// becomes two vertices, coincident when its geometry is degenerate (a straight or reversing
    /// corner, a zero-length edge), so applying the same list to each keyframe of an animated path
    /// keeps the vertex counts equal. Other indices (out of range, curved, repeated) are left alone.
    pub fn with_rounded_corners(&self, radius: f64, corners: &[usize]) -> Self {
        let n = self.vertices.len();
        let mut round = vec![false; n];
        for &i in corners {
            if let Some(r) = round.get_mut(i) {
                *r = true;
            }
        }
        let roundable = self.roundable_corners();
        round.iter_mut().enumerate().for_each(|(i, r)| *r &= roundable.binary_search(&i).is_ok());
        let radius = if radius.is_finite() { radius.clamp(0.0, 1e7) } else { 0.0 };
        let mut vertices = Vec::with_capacity(n.saturating_mul(2));
        for (i, v) in self.vertices.iter().enumerate() {
            if !round.get(i).copied().unwrap_or(false) {
                vertices.push(*v);
                continue;
            }
            let (Some(prev), Some(next)) = (self.vertices.get((i + n - 1) % n), self.vertices.get((i + 1) % n)) else {
                vertices.push(*v);
                continue;
            };
            let (a, b) = (v.p - prev.p, next.p - v.p);
            let (la, lb) = (a.length(), b.length());
            let fillet = (la > 1e-9 && lb > 1e-9 && radius > 0.0).then(|| (a * (1.0 / la), b * (1.0 / lb))).and_then(|(u, w)| {
                // turn angle φ between the incoming and outgoing directions
                let phi = (u.x * w.x + u.y * w.y).clamp(-1.0, 1.0).acos();
                if !(1e-6..std::f64::consts::PI - 1e-6).contains(&phi) {
                    return None;
                }
                let tan_half = (phi / 2.0).tan();
                let d = (radius * tan_half).min(la / 2.0).min(lb / 2.0);
                let r = d / tan_half;
                let h = 4.0 / 3.0 * (phi / 4.0).tan() * r;
                Some((u, w, d, h))
            });
            match fillet {
                Some((u, w, d, h)) => {
                    vertices.push(MaskVertex { p: v.p - u * d, t_in: Vec2::ZERO, t_out: u * h });
                    vertices.push(MaskVertex { p: v.p + w * d, t_in: w * -h, t_out: Vec2::ZERO });
                }
                None => {
                    vertices.push(MaskVertex::corner(v.p));
                    vertices.push(MaskVertex::corner(v.p));
                }
            }
        }
        Self { vertices, closed: self.closed }
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// The path with every point and tangent mapped by `m` (tangents by its linear part).
    pub fn transformed(&self, m: &Affine) -> Self {
        let lin = Affine { e: 0.0, f: 0.0, ..*m };
        Self {
            vertices: self.vertices.iter().map(|v| MaskVertex { p: m.apply(v.p), t_in: lin.apply(v.t_in), t_out: lin.apply(v.t_out) }).collect(),
            closed: self.closed,
        }
    }

    /// Mean of the vertex positions.
    pub fn centroid(&self) -> Vec2 {
        if self.vertices.is_empty() {
            return Vec2::ZERO;
        }
        let s = self.vertices.iter().fold(Vec2::ZERO, |a, v| a + v.p);
        s * (1.0 / self.vertices.len() as f64)
    }

    /// Bounding box of the control polygon (contains the curve): (min, max).
    pub fn bounds(&self) -> (Vec2, Vec2) {
        let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for v in &self.vertices {
            for q in [v.p, v.p + v.t_in, v.p + v.t_out] {
                lo = Vec2::new(lo.x.min(q.x), lo.y.min(q.y));
                hi = Vec2::new(hi.x.max(q.x), hi.y.max(q.y));
            }
        }
        (lo, hi)
    }

    /// Cubic segments `(p0, c1, c2, p3)`; a closed path includes the last→first segment.
    pub fn segments(&self) -> Vec<[Vec2; 4]> {
        let n = self.vertices.len();
        if n < 2 {
            return Vec::new();
        }
        let count = if self.closed { n } else { n - 1 };
        (0..count)
            .map(|i| {
                let a = &self.vertices[i];
                let b = &self.vertices[(i + 1) % n];
                [a.p, a.p + a.t_out, b.p + b.t_in, b.p]
            })
            .collect()
    }

    /// Flatten to a polygon (closed implicitly) whose chords deviate from the curve by at most
    /// `tol` (same units as the path). Corner-only segments emit just their end point.
    pub fn flatten(&self, tol: f64) -> Vec<Vec2> {
        let mut out = Vec::new();
        let n = self.vertices.len();
        if n == 0 {
            return out;
        }
        out.push(self.vertices[0].p);
        // an open path is filled as if closed: flatten its closing segment too
        let segs = if self.closed || n < 2 {
            self.segments()
        } else {
            let mut s = self.segments();
            let (a, b) = (&self.vertices[n - 1], &self.vertices[0]);
            s.push([a.p, a.p + a.t_out, b.p + b.t_in, b.p]);
            s
        };
        let tol = tol.max(1e-4);
        for (i, [p0, c1, c2, p3]) in segs.iter().enumerate() {
            let last = i + 1 == segs.len();
            let straight = (*c1 - *p0).length() < 1e-12 && (*c2 - *p3).length() < 1e-12;
            if !straight {
                // Wang's bound on the number of chords for a cubic.
                let d1 = (*p0 - *c1 * 2.0 + *c2).length();
                let d2 = (*c1 - *c2 * 2.0 + *p3).length();
                let m = d1.max(d2);
                let steps = ((0.75 * m / tol).sqrt().ceil() as usize).clamp(1, 256);
                for k in 1..steps {
                    let t = k as f64 / steps as f64;
                    out.push(cubic(*p0, *c1, *c2, *p3, t));
                }
            }
            if !last {
                out.push(*p3);
            }
        }
        out
    }

    /// Flattened components for keyframe interpolation: `[x, y, in.x, in.y, out.x, out.y]` per vertex.
    pub fn components(&self) -> Vec<f64> {
        self.vertices.iter().flat_map(|v| [v.p.x, v.p.y, v.t_in.x, v.t_in.y, v.t_out.x, v.t_out.y]).collect()
    }
    /// Inverse of [`MaskPath::components`] (`closed` is taken from `self`).
    pub fn with_components(&self, c: &[f64]) -> Self {
        Self {
            vertices: c
                .as_chunks::<6>()
                .0
                .iter()
                .map(|k| MaskVertex { p: Vec2::new(k[0], k[1]), t_in: Vec2::new(k[2], k[3]), t_out: Vec2::new(k[4], k[5]) })
                .collect(),
            closed: self.closed,
        }
    }
}

/// Most tips of a star / sides of a regular polygon preset.
pub const MAX_PRESET_POINTS: usize = 100;

/// The corner a diagonal split keeps ([`MaskPath::diagonal_split`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitCorner {
    #[default]
    TopRight,
    TopLeft,
    BottomLeft,
    BottomRight,
}

impl SplitCorner {
    pub const ALL: [SplitCorner; 4] = [SplitCorner::TopRight, SplitCorner::TopLeft, SplitCorner::BottomLeft, SplitCorner::BottomRight];
    pub fn name(self) -> &'static str {
        match self {
            SplitCorner::TopRight => "topRight",
            SplitCorner::TopLeft => "topLeft",
            SplitCorner::BottomLeft => "bottomLeft",
            SplitCorner::BottomRight => "bottomRight",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        let s = s.to_ascii_lowercase().replace(['-', '_', ' '], "");
        Self::ALL.into_iter().find(|c| c.name().to_ascii_lowercase() == s)
    }
}

/// Shape presets (`masks.add` `shape`). Each is an ordinary path — corner vertices, or smooth
/// vertices with Bézier handles for the heart — edited, keyframed, feathered and tracked like any
/// other mask. Counts and ratios are clamped to sensible ranges, so any input gives a valid path.
impl MaskPath {
    /// A star of `points` tips (3–[`MAX_PRESET_POINTS`]) on the ellipse of `radius`, its inner
    /// vertices at `inner` (0.05–1) of that radius, the first tip straight up: `2·points` corners.
    pub fn star(center: Vec2, radius: Vec2, points: usize, inner: f64) -> Self {
        let n = points.clamp(3, MAX_PRESET_POINTS);
        let inner = if inner.is_finite() { inner.clamp(0.05, 1.0) } else { 0.5 };
        let step = std::f64::consts::PI / n as f64;
        let pts: Vec<Vec2> = (0..2 * n)
            .map(|k| {
                let a = -std::f64::consts::FRAC_PI_2 + k as f64 * step;
                let r = if k % 2 == 0 { 1.0 } else { inner };
                center + Vec2::new(radius.x * r * a.cos(), radius.y * r * a.sin())
            })
            .collect();
        Self::polygon(&pts)
    }

    /// A regular polygon of `sides` (3–[`MAX_PRESET_POINTS`]) inscribed in the ellipse of
    /// `radius`, the first vertex straight up.
    pub fn regular_polygon(center: Vec2, radius: Vec2, sides: usize) -> Self {
        let n = sides.clamp(3, MAX_PRESET_POINTS);
        let step = std::f64::consts::TAU / n as f64;
        let pts: Vec<Vec2> = (0..n)
            .map(|k| {
                let a = -std::f64::consts::FRAC_PI_2 + k as f64 * step;
                center + Vec2::new(radius.x * a.cos(), radius.y * a.sin())
            })
            .collect();
        Self::polygon(&pts)
    }

    /// A triangle filling the box: apex at the top centre, base along the bottom.
    pub fn triangle(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self::polygon(&[Vec2::new((x0 + x1) / 2.0, y0), Vec2::new(x1, y1), Vec2::new(x0, y1)])
    }

    /// A heart filling the box `center ± radius`: two round lobes (smooth vertices with Bézier
    /// handles) meeting in a cusp at the top and a point at the bottom; six vertices.
    pub fn heart(center: Vec2, radius: Vec2) -> Self {
        // (point, in handle, out handle) in a unit box (-1..1, y down), drawn for this preset
        const UNIT: [[(f64, f64); 3]; 6] = [
            [(0.0, -0.55), (-0.15, -0.33), (0.15, -0.33)],
            [(0.5, -1.0), (-0.28, 0.0), (0.28, 0.0)],
            [(1.0, -0.35), (0.0, -0.36), (0.0, 0.36)],
            [(0.0, 1.0), (0.38, -0.4), (-0.38, -0.4)],
            [(-1.0, -0.35), (0.0, 0.36), (0.0, -0.36)],
            [(-0.5, -1.0), (-0.28, 0.0), (0.28, 0.0)],
        ];
        let s = |(x, y): (f64, f64)| Vec2::new(x * radius.x, y * radius.y);
        Self { vertices: UNIT.iter().map(|[p, i, o]| MaskVertex { p: center + s(*p), t_in: s(*i), t_out: s(*o) }).collect(), closed: true }
    }

    /// A plus sign filling the box, its arms `thickness` (0.05–0.95) of the box wide: 12 corners.
    pub fn cross(center: Vec2, radius: Vec2, thickness: f64) -> Self {
        let t = if thickness.is_finite() { thickness.clamp(0.05, 0.95) } else { 1.0 / 3.0 };
        let (rx, ry) = (radius.x, radius.y);
        let (hx, hy) = (rx * t, ry * t);
        let pts = [(-hx, -ry), (hx, -ry), (hx, -hy), (rx, -hy), (rx, hy), (hx, hy), (hx, ry), (-hx, ry), (-hx, hy), (-rx, hy), (-rx, -hy), (-hx, -hy)];
        Self::polygon(&pts.map(|(x, y)| center + Vec2::new(x, y)))
    }

    /// An arrow pointing right, filling the box: a shaft 0.7 of the box high and a head over the
    /// right 40%: 7 corners.
    pub fn arrow(center: Vec2, radius: Vec2) -> Self {
        let (rx, ry) = (radius.x, radius.y);
        let (shaft, head) = (0.35 * ry, 0.2 * rx);
        let pts = [(-rx, -shaft), (head, -shaft), (head, -ry), (rx, 0.0), (head, ry), (head, shaft), (-rx, shaft)];
        Self::polygon(&pts.map(|(x, y)| center + Vec2::new(x, y)))
    }

    /// The picture area between cinematic bars on a `w`×`h` frame: a rectangle of the frame's
    /// width at `aspect` (width / height; e.g. 2.39), centred — or of its height when `aspect` is
    /// narrower than the frame (bars at the sides). Edges without bars reach 10% past the frame so
    /// feathering never darkens them. Masking Opacity with it cuts the bars out; inverted, an
    /// effect fills the bars. None when the frame or aspect is not a positive finite size.
    pub fn letterbox(w: f64, h: f64, aspect: f64) -> Option<Self> {
        if !(w.is_finite() && h.is_finite() && aspect.is_finite() && w > 0.0 && h > 0.0 && aspect > 0.0) {
            return None;
        }
        let m = 0.1 * w.max(h);
        Some(if aspect >= w / h {
            let ph = w / aspect;
            let y0 = (h - ph) / 2.0;
            Self::rect(-m, y0, w + m, y0 + ph)
        } else {
            let pw = h * aspect;
            let x0 = (w - pw) / 2.0;
            Self::rect(x0, -m, x0 + pw, h + m)
        })
    }

    /// Half of a `w`×`h` frame cut along a diagonal: the half holding `corner`. Edges other than
    /// the diagonal reach 10% past the frame, so only the diagonal is feathered. None when the
    /// frame is not a positive finite size.
    pub fn diagonal_split(w: f64, h: f64, corner: SplitCorner) -> Option<Self> {
        if !(w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0) {
            return None;
        }
        let m = 0.1 * w.max(h);
        let rect = [Vec2::new(-m, -m), Vec2::new(w + m, -m), Vec2::new(w + m, h + m), Vec2::new(-m, h + m)];
        // the split line through two opposite frame corners, and the corner whose half is kept
        let (a, b, keep) = match corner {
            SplitCorner::TopRight => (Vec2::new(0.0, 0.0), Vec2::new(w, h), Vec2::new(w, 0.0)),
            SplitCorner::BottomLeft => (Vec2::new(0.0, 0.0), Vec2::new(w, h), Vec2::new(0.0, h)),
            SplitCorner::TopLeft => (Vec2::new(w, 0.0), Vec2::new(0.0, h), Vec2::new(0.0, 0.0)),
            SplitCorner::BottomRight => (Vec2::new(w, 0.0), Vec2::new(0.0, h), Vec2::new(w, h)),
        };
        let raw = |p: Vec2| (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        let sign = raw(keep).signum();
        let side = |p: Vec2| raw(p) * sign;
        // Sutherland–Hodgman against one half-plane (side ≥ 0 is kept)
        let mut out = Vec::with_capacity(6);
        for (p, q) in rect.iter().zip(rect.iter().cycle().skip(1)) {
            let (sp, sq) = (side(*p), side(*q));
            if sp >= 0.0 {
                out.push(*p);
            }
            if (sp < 0.0 && sq > 0.0) || (sp > 0.0 && sq < 0.0) {
                let t = sp / (sp - sq);
                out.push(*p + (*q - *p) * t);
            }
        }
        (out.len() >= 3).then(|| Self::polygon(&out))
    }
}

/// Point on a cubic Bézier at `t`.
pub fn cubic(p0: Vec2, c1: Vec2, c2: Vec2, p3: Vec2, t: f64) -> Vec2 {
    let u = 1.0 - t;
    p0 * (u * u * u) + c1 * (3.0 * u * u * t) + c2 * (3.0 * u * t * t) + p3 * (t * t * t)
}

/// How a mask combines with the result of the masks above it (After Effects' mask modes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskMode {
    /// Ignored (kept for the user to switch back on).
    None,
    /// Union: `a + m − a·m`.
    #[default]
    Add,
    /// `a · (1 − m)`.
    Subtract,
    /// `a · m`.
    Intersect,
    /// `max(a, m)`.
    Lighten,
    /// `min(a, m)`.
    Darken,
    /// `a + m − 2·a·m`.
    Difference,
}

impl MaskMode {
    pub const ALL: [MaskMode; 7] =
        [MaskMode::None, MaskMode::Add, MaskMode::Subtract, MaskMode::Intersect, MaskMode::Lighten, MaskMode::Darken, MaskMode::Difference];
    pub fn label(self) -> &'static str {
        match self {
            MaskMode::None => "None",
            MaskMode::Add => "Add",
            MaskMode::Subtract => "Subtract",
            MaskMode::Intersect => "Intersect",
            MaskMode::Lighten => "Lighten",
            MaskMode::Darken => "Darken",
            MaskMode::Difference => "Difference",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        let s = s.to_ascii_lowercase();
        Self::ALL.into_iter().find(|m| m.label().to_ascii_lowercase() == s)
    }
    /// Shader / wire index.
    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|m| *m == self).unwrap_or(1) as u32
    }
    /// Combine accumulated coverage `a` with this mask's coverage `m`.
    pub fn combine(self, a: f32, m: f32) -> f32 {
        match self {
            MaskMode::None => a,
            MaskMode::Add => a + m - a * m,
            MaskMode::Subtract => a * (1.0 - m),
            MaskMode::Intersect => a * m,
            MaskMode::Lighten => a.max(m),
            MaskMode::Darken => a.min(m),
            MaskMode::Difference => a + m - 2.0 * a * m,
        }
    }
    /// Starting value when this is the first active mask: additive modes start from nothing,
    /// subtractive ones from the full frame.
    pub fn start(self) -> f32 {
        match self {
            MaskMode::Subtract | MaskMode::Intersect | MaskMode::Darken => 1.0,
            _ => 0.0,
        }
    }
}

/// Mask tracking method (Premiere's wrench menu next to the tracking buttons).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackMethod {
    Position,
    PositionRotation,
    #[default]
    PositionScaleRotation,
    /// The thing the mask outlines, cut out again in every frame it is tracked to (Object Mask:
    /// the segmenter, told where the thing went by a box tracker); the mask takes its shape.
    Object,
}

impl TrackMethod {
    pub const ALL: [TrackMethod; 4] = [TrackMethod::Position, TrackMethod::PositionRotation, TrackMethod::PositionScaleRotation, TrackMethod::Object];
    pub fn label(self) -> &'static str {
        match self {
            TrackMethod::Position => "Position",
            TrackMethod::PositionRotation => "Position & Rotation",
            TrackMethod::PositionScaleRotation => "Position, Scale & Rotation",
            TrackMethod::Object => "Object",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace([' ', '_', '-', '&', ','], "").as_str() {
            "position" => Some(TrackMethod::Position),
            "positionrotation" => Some(TrackMethod::PositionRotation),
            "positionscalerotation" => Some(TrackMethod::PositionScaleRotation),
            "object" => Some(TrackMethod::Object),
            _ => None,
        }
    }
}

/// A mask on an effect instance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub name: String,
    /// `ParamValue::Path`.
    pub path: Param,
    /// Feather width in clip pixels (≥ 0).
    pub feather: Param,
    /// 0..100 %.
    pub opacity: Param,
    /// Clip pixels (negative shrinks).
    pub expansion: Param,
    #[serde(default)]
    pub inverted: bool,
    #[serde(default)]
    pub mode: MaskMode,
    #[serde(default)]
    pub track_method: TrackMethod,
}

/// Mask parameter ids (keyframe commands address them with `"mask": n`).
pub const MASK_PARAMS: [&str; 4] = ["path", "feather", "opacity", "expansion"];

impl Mask {
    pub fn new(name: impl Into<String>, path: MaskPath) -> Self {
        Self {
            name: name.into(),
            path: Param::new(ParamValue::Path(path)),
            feather: Param::new(ParamValue::Float(10.0)),
            opacity: Param::new(ParamValue::Float(100.0)),
            expansion: Param::new(ParamValue::Float(0.0)),
            inverted: false,
            mode: MaskMode::Add,
            track_method: TrackMethod::default(),
        }
    }
    pub fn param(&self, id: &str) -> Option<&Param> {
        match id {
            "path" => Some(&self.path),
            "feather" => Some(&self.feather),
            "opacity" => Some(&self.opacity),
            "expansion" => Some(&self.expansion),
            _ => None,
        }
    }
    pub fn param_mut(&mut self, id: &str) -> Option<&mut Param> {
        match id {
            "path" => Some(&mut self.path),
            "feather" => Some(&mut self.feather),
            "opacity" => Some(&mut self.opacity),
            "expansion" => Some(&mut self.expansion),
            _ => None,
        }
    }
    pub fn path_at(&self, t: filmcraft_time::Tick) -> MaskPath {
        match self.path.value_at(t) {
            ParamValue::Path(p) => p,
            _ => MaskPath::default(),
        }
    }
    pub fn is_animated(&self) -> bool {
        MASK_PARAMS.iter().any(|p| self.param(p).is_some_and(Param::is_animated))
    }
    /// Keyframe times of every mask parameter (for the keyframe lane / presets).
    pub fn params_mut(&mut self) -> [&mut Param; 4] {
        [&mut self.path, &mut self.feather, &mut self.opacity, &mut self.expansion]
    }
}

/// Parameter schemas of the mask rows in Effect Controls (and the MCP schema).
pub fn mask_param_defs() -> &'static [ParamDef] {
    use std::sync::OnceLock;
    static DEFS: OnceLock<Vec<ParamDef>> = OnceLock::new();
    DEFS.get_or_init(|| {
        let fl = |id, label, def, (min, max): (f64, f64), (smin, smax): (f64, f64), unit| ParamDef {
            id,
            label,
            kind: ParamKind::Float { min, max, soft_min: smin, soft_max: smax, unit, decimals: 1 },
            default: ParamValue::Float(def),
            animatable: true,
            group: None,
        };
        vec![
            ParamDef { id: "path", label: "Mask Path", kind: ParamKind::Path, default: ParamValue::Path(MaskPath::default()), animatable: true, group: None },
            fl("feather", "Mask Feather", 10.0, (0.0, 10_000.0), (0.0, 500.0), ""),
            fl("opacity", "Mask Opacity", 100.0, (0.0, 100.0), (0.0, 100.0), " %"),
            fl("expansion", "Mask Expansion", 0.0, (-10_000.0, 10_000.0), (-500.0, 500.0), ""),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_time::Tick;

    #[test]
    fn ellipse_flattens_onto_the_ellipse() {
        let p = MaskPath::ellipse(Vec2::new(100.0, 50.0), Vec2::new(40.0, 20.0));
        let pts = p.flatten(0.05);
        assert!(pts.len() > 16, "{}", pts.len());
        for q in &pts {
            let e = ((q.x - 100.0) / 40.0).powi(2) + ((q.y - 50.0) / 20.0).powi(2);
            assert!((e - 1.0).abs() < 0.01, "{q:?} off the ellipse: {e}");
        }
    }

    #[test]
    fn polygon_flattens_to_its_corners() {
        let p = MaskPath::rect(0.0, 0.0, 10.0, 5.0);
        assert_eq!(p.flatten(0.1), vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 5.0), Vec2::new(0.0, 5.0)]);
    }

    #[test]
    fn path_keyframes_interpolate_vertexwise() {
        let a = MaskPath::rect(0.0, 0.0, 10.0, 10.0);
        let b = a.transformed(&Affine::translate(100.0, 20.0));
        let mut m = Mask::new("Mask (1)", a.clone());
        m.path.put_keyframe(Tick(0), ParamValue::Path(a));
        m.path.put_keyframe(Tick(100), ParamValue::Path(b));
        let mid = m.path_at(Tick(50));
        assert_eq!(mid.vertices[0].p, Vec2::new(50.0, 10.0));
        assert_eq!(mid.vertices[2].p, Vec2::new(60.0, 20.0));
        // different vertex counts hold the earlier path
        m.path.put_keyframe(Tick(200), ParamValue::Path(MaskPath::ellipse(Vec2::ZERO, Vec2::new(1.0, 1.0))));
        m.path.put_keyframe(Tick(300), ParamValue::Path(MaskPath::polygon(&[Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)])));
        assert_eq!(m.path_at(Tick(250)).len(), 4);
    }

    #[test]
    fn modes_combine() {
        assert_eq!(MaskMode::Add.combine(0.0, 0.5), 0.5);
        assert_eq!(MaskMode::Subtract.combine(1.0, 0.25), 0.75);
        assert_eq!(MaskMode::Intersect.combine(0.5, 0.5), 0.25);
        assert_eq!(MaskMode::Difference.combine(1.0, 1.0), 0.0);
        assert_eq!(MaskMode::from_name("intersect"), Some(MaskMode::Intersect));
        assert_eq!(TrackMethod::from_name("Position & Rotation"), Some(TrackMethod::PositionRotation));
    }

    #[test]
    fn mask_serde_round_trip() {
        let mut m = Mask::new("Mask (1)", MaskPath::ellipse(Vec2::new(5.0, 5.0), Vec2::new(3.0, 2.0)));
        m.inverted = true;
        m.mode = MaskMode::Subtract;
        m.feather.put_keyframe(Tick(10), ParamValue::Float(4.0));
        let s = serde_json::to_string(&m).unwrap();
        let back: Mask = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn rounded_rect_corners_are_quarter_circles() {
        let p = MaskPath::rounded_rect(0.0, 0.0, 100.0, 60.0, 20.0);
        assert_eq!(p.len(), 8);
        // top-left corner: a quarter circle of radius 20 about (20, 20); the edges stay straight
        let pts = p.flatten(0.01);
        let mut on_arc = 0;
        for q in &pts {
            let (cx, cy) = (q.x.clamp(20.0, 80.0), q.y.clamp(20.0, 40.0));
            let d = ((q.x - cx).powi(2) + (q.y - cy).powi(2)).sqrt();
            if (q.x - cx).abs() > 1e-9 && (q.y - cy).abs() > 1e-9 {
                on_arc += 1;
                assert!((d - 20.0).abs() < 0.03, "{q:?} is {d} from its corner centre");
            }
            assert!(q.x >= -1e-9 && q.x <= 100.0 + 1e-9 && q.y >= -1e-9 && q.y <= 60.0 + 1e-9, "{q:?} outside the rectangle");
        }
        assert!(on_arc >= 8, "the corners are curved: {on_arc} arc points");
        // the sharp corner itself is cut away
        assert!(pts.iter().all(|q| q.x + q.y > 5.0), "the corner at (0, 0) is rounded off");
        // radius 0 is the plain rectangle
        assert_eq!(MaskPath::rounded_rect(0.0, 0.0, 10.0, 5.0, 0.0), MaskPath::rect(0.0, 0.0, 10.0, 5.0));
    }

    #[test]
    fn rounding_clamps_to_half_an_edge_and_ignores_curves_and_hostile_input() {
        // a radius larger than the shape gives a stadium: fillets meet at the edges' midpoints
        let p = MaskPath::rounded_rect(0.0, 0.0, 100.0, 40.0, 1e9);
        assert_eq!(p.len(), 8);
        for v in &p.vertices {
            assert!(v.p.x.is_finite() && v.p.y.is_finite());
        }
        assert!(p.vertices.iter().any(|v| (v.p - Vec2::new(0.0, 20.0)).length() < 1e-9), "{:?}", p.vertices);
        // a triangle: the 60° turns get fillets tangent to both edges, within the triangle
        let tri = MaskPath::polygon(&[Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0), Vec2::new(50.0, 86.6)]);
        let r = tri.with_rounded_corners(10.0, &tri.roundable_corners());
        assert_eq!(r.len(), 6);
        assert!(r.vertices.iter().all(|v| v.p.y >= -1e-9));
        // an ellipse has no sharp corners; NaN / negative radii and bad indices change nothing harmful
        let e = MaskPath::ellipse(Vec2::new(50.0, 50.0), Vec2::new(30.0, 20.0));
        assert!(e.roundable_corners().is_empty());
        assert_eq!(e.with_rounded_corners(10.0, &[0, 1, 2, 3]), e);
        let sq = MaskPath::rect(0.0, 0.0, 10.0, 10.0);
        assert_eq!(sq.with_rounded_corners(5.0, &[usize::MAX, 99]), sq);
        for bad in [f64::NAN, f64::INFINITY, -3.0, 0.0] {
            let q = sq.with_rounded_corners(bad, &[0]);
            assert_eq!(q.len(), 5, "a listed corner always splits in two");
            assert!(q.vertices.iter().all(|v| v.is_corner() && v.p.x.is_finite()));
        }
        // a collinear vertex splits into two coincident corners, keeping vertex counts stable
        let line = MaskPath::polygon(&[Vec2::new(0.0, 0.0), Vec2::new(5.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(5.0, 5.0)]);
        let q = line.with_rounded_corners(2.0, &[1]);
        assert_eq!(q.len(), 5);
        assert_eq!(q.vertices[1].p, q.vertices[2].p);
    }

    /// Nonzero winding of a flattened path around `p`.
    fn inside(path: &MaskPath, p: Vec2) -> bool {
        let poly = path.flatten(0.05);
        let mut wn = 0i32;
        for (a, b) in poly.iter().zip(poly.iter().cycle().skip(1)) {
            let cross = (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
            if a.y <= p.y && b.y > p.y && cross > 0.0 {
                wn += 1;
            } else if a.y > p.y && b.y <= p.y && cross < 0.0 {
                wn -= 1;
            }
        }
        wn != 0
    }

    fn close(a: Vec2, b: Vec2) -> bool {
        (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9
    }

    #[test]
    fn star_alternates_outer_and_inner_vertices() {
        let c = Vec2::new(500.0, 400.0);
        let s = MaskPath::star(c, Vec2::new(100.0, 100.0), 5, 0.4);
        assert_eq!(s.len(), 10);
        assert!(s.vertices.iter().all(MaskVertex::is_corner));
        assert!(close(s.vertices[0].p, Vec2::new(500.0, 300.0)), "first tip straight up: {:?}", s.vertices[0].p);
        for (k, v) in s.vertices.iter().enumerate() {
            let r = (v.p - c).length();
            let want = if k % 2 == 0 { 100.0 } else { 40.0 };
            assert!((r - want).abs() < 1e-9, "vertex {k}: radius {r}");
        }
        assert!(inside(&s, c) && !inside(&s, Vec2::new(500.0, 255.0)) && !inside(&s, Vec2::new(560.0, 330.0)), "a notch between tips is outside");
        // counts and ratios are clamped; non-finite ratios fall back
        assert_eq!(MaskPath::star(c, Vec2::new(10.0, 10.0), 0, f64::NAN).len(), 6);
        assert_eq!(MaskPath::star(c, Vec2::new(10.0, 10.0), usize::MAX, 9.0).len(), 2 * MAX_PRESET_POINTS);
        assert_eq!(MaskPath::regular_polygon(c, Vec2::new(10.0, 10.0), 6).len(), 6);
        assert_eq!(MaskPath::regular_polygon(c, Vec2::new(10.0, 10.0), 1).len(), 3);
    }

    #[test]
    fn heart_is_symmetric_with_a_cusp_and_a_point() {
        let c = Vec2::new(0.0, 0.0);
        let h = MaskPath::heart(c, Vec2::new(100.0, 100.0));
        assert_eq!(h.len(), 6);
        // mirror image about the vertical axis: vertex k mirrors vertex (6 - k) % 6
        for k in 0..6 {
            let (a, b) = (h.vertices[k], h.vertices[(6 - k) % 6]);
            assert!(close(a.p, Vec2::new(-b.p.x, b.p.y)), "vertex {k}");
            assert!(close(a.t_in, Vec2::new(-b.t_out.x, b.t_out.y)), "vertex {k} handles");
        }
        // the cusp and the bottom point are corners with broken handles, the lobes smooth
        assert!(h.vertices[0].t_in != h.vertices[0].t_out * -1.0 && h.vertices[3].t_in != h.vertices[3].t_out * -1.0);
        assert_eq!(h.vertices[1].t_in, h.vertices[1].t_out * -1.0);
        let (lo, hi) = {
            let poly = h.flatten(0.05);
            poly.iter().fold((Vec2::new(f64::MAX, f64::MAX), Vec2::new(f64::MIN, f64::MIN)), |(lo, hi), p| {
                (Vec2::new(lo.x.min(p.x), lo.y.min(p.y)), Vec2::new(hi.x.max(p.x), hi.y.max(p.y)))
            })
        };
        assert!((lo.x + 100.0).abs() < 1.0 && (hi.x - 100.0).abs() < 1.0 && (lo.y + 100.0).abs() < 1.0 && (hi.y - 100.0).abs() < 1.0, "{lo:?} {hi:?}");
        assert!(inside(&h, c) && inside(&h, Vec2::new(-50.0, -80.0)) && inside(&h, Vec2::new(0.0, 80.0)));
        assert!(!inside(&h, Vec2::new(0.0, -80.0)), "the notch between the lobes is outside");
        assert!(!inside(&h, Vec2::new(90.0, 80.0)), "beside the point is outside");
    }

    #[test]
    fn letterbox_keeps_the_picture_between_the_bars() {
        let p = MaskPath::letterbox(1920.0, 1080.0, 2.39).unwrap();
        let (lo, hi) = p.bounds();
        let ph = 1920.0 / 2.39;
        assert!((hi.y - lo.y - ph).abs() < 1e-9 && ((lo.y + hi.y) / 2.0 - 540.0).abs() < 1e-9, "{lo:?} {hi:?}");
        assert!(lo.x < 0.0 && hi.x > 1920.0, "no bars at the sides: past the frame");
        assert!(inside(&p, Vec2::new(960.0, 540.0)) && !inside(&p, Vec2::new(960.0, 50.0)) && !inside(&p, Vec2::new(960.0, 1030.0)));
        // narrower than the frame: bars at the sides (a 4:3 pillarbox in 16:9)
        let p = MaskPath::letterbox(1920.0, 1080.0, 4.0 / 3.0).unwrap();
        let (lo, hi) = p.bounds();
        assert!((hi.x - lo.x - 1440.0).abs() < 1e-9 && lo.y < 0.0 && hi.y > 1080.0, "{lo:?} {hi:?}");
        // vertical frames get bars too
        assert!(MaskPath::letterbox(1080.0, 1920.0, 1.0).is_some());
        for (w, h, a) in [(0.0, 1080.0, 2.39), (1920.0, -1.0, 2.39), (1920.0, 1080.0, 0.0), (1920.0, 1080.0, f64::NAN), (f64::INFINITY, 1.0, 1.0)] {
            assert!(MaskPath::letterbox(w, h, a).is_none(), "{w} {h} {a}");
        }
    }

    #[test]
    fn diagonal_split_keeps_the_named_corner() {
        let (w, h) = (1920.0, 1080.0);
        for corner in SplitCorner::ALL {
            let p = MaskPath::diagonal_split(w, h, corner).unwrap();
            assert!(p.len() >= 3 && p.vertices.iter().all(MaskVertex::is_corner));
            let at = |x: f64, y: f64| inside(&p, Vec2::new(x * w, y * h));
            // the kept corner's side and the opposite corner's (the other two corners lie on the
            // split line)
            let (kept, opposite) = match corner {
                SplitCorner::TopRight => (at(0.9, 0.1), at(0.1, 0.9)),
                SplitCorner::TopLeft => (at(0.1, 0.1), at(0.9, 0.9)),
                SplitCorner::BottomLeft => (at(0.1, 0.9), at(0.9, 0.1)),
                SplitCorner::BottomRight => (at(0.9, 0.9), at(0.1, 0.1)),
            };
            assert!(kept && !opposite, "{corner:?}");
            // past the frame on the kept side is covered (no feathered frame edge)
            let outside_frame = match corner {
                SplitCorner::TopRight => at(1.05, -0.05),
                SplitCorner::TopLeft => at(-0.05, -0.05),
                SplitCorner::BottomLeft => at(-0.05, 1.05),
                SplitCorner::BottomRight => at(1.05, 1.05),
            };
            assert!(outside_frame, "{corner:?}");
            assert_eq!(SplitCorner::from_name(corner.name()), Some(corner));
        }
        assert_eq!(SplitCorner::from_name("bottom-right"), Some(SplitCorner::BottomRight));
        assert!(MaskPath::diagonal_split(0.0, 1.0, SplitCorner::TopLeft).is_none());
    }

    #[test]
    fn triangle_cross_and_arrow_fill_their_box() {
        let c = Vec2::new(100.0, 100.0);
        let t = MaskPath::triangle(0.0, 0.0, 200.0, 200.0);
        assert_eq!(t.len(), 3);
        assert!(inside(&t, Vec2::new(100.0, 150.0)) && !inside(&t, Vec2::new(10.0, 10.0)));
        let x = MaskPath::cross(c, Vec2::new(100.0, 100.0), 0.3);
        assert_eq!(x.len(), 12);
        assert!(inside(&x, c) && inside(&x, Vec2::new(100.0, 10.0)) && !inside(&x, Vec2::new(10.0, 10.0)));
        assert_eq!(MaskPath::cross(c, Vec2::new(1.0, 1.0), f64::INFINITY).len(), 12);
        let a = MaskPath::arrow(c, Vec2::new(100.0, 50.0));
        assert_eq!(a.len(), 7);
        assert!(inside(&a, Vec2::new(190.0, 100.0)) && inside(&a, Vec2::new(10.0, 100.0)) && !inside(&a, Vec2::new(10.0, 60.0)));
    }
}
