//! 360° video: the virtual camera of `vr_reframe` (the CPU reference; the GPU compositor runs
//! the same maths in WGSL).
//!
//! A 360° picture ([`filmcraft_media::Spherical`]: equirectangular, or dual fisheye straight from
//! an Insta360 camera) is a picture of every direction around the camera. The virtual camera
//! looks one way ([`camera_rotation`]: yaw turns right, pitch looks up, roll turns the picture
//! clockwise) and makes a flat picture of what it sees — a rectilinear view of `fov` degrees
//! across, or, with `distance` towards 1, the camera pulled back inside the sphere until the view
//! is stereographic (a "tiny planet" when it looks straight down) — or a whole equirectangular
//! sphere, turned (horizon levelling, re-centring before a 360° export).
//!
//! Per output pixel: the ray it sees ([`View::ray`]), turned into the world, then the source
//! pixel that saw that direction ([`SourceModel::locate`]): longitude and latitude for an
//! equirectangular picture; for fisheye lenses the angle θ from each lens's axis and its bearing
//! φ, the image circle's radius × the lens's distortion polynomial of θ / half-FOV. Where both
//! lenses of a dual fisheye see a direction, they are cross-faded over a band of
//! [`BLEND_DEGREES`] either side of the line half way between their axes. Bilinear sampling,
//! wrapping across the ±180° seam of a whole-sphere equirectangular picture.
//!
//! Directions: +Z forward, +X right, +Y up (`filmcraft_media::spherical`).

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use filmcraft_media::Spherical;
use filmcraft_media::spherical::{FisheyeLens, Projection, StereoLayout};
use filmcraft_project::{EffectInstance, ParamValue, TrackItem};
use filmcraft_time::Tick;
use rayon::prelude::*;

use crate::image::Image;

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

/// The effect id of the virtual camera.
pub const REFRAME: &str = filmcraft_project::vr::REFRAME;
/// Half the width of the band over which two lenses are cross-faded, degrees.
pub const BLEND_DEGREES: f64 = 4.0;
/// Most supersampling taps a side for a view that shrinks its source a lot.
pub const MAX_TAPS: usize = 4;

#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn mvec(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
/// mᵀ · v (the inverse of a rotation).
#[inline]
pub fn mtvec(m: &M3, v: V3) -> V3 {
    [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2], m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2], m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]]
}
pub fn mmul(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}
pub fn transpose(m: &M3) -> M3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}
fn normalize(v: V3) -> V3 {
    let n = dot(v, v).sqrt();
    if n > 0.0 && n.is_finite() { [v[0] / n, v[1] / n, v[2] / n] } else { [0.0, 0.0, 1.0] }
}

/// The direction of longitude `lon`, latitude `lat` (radians; 0, 0 is forward).
#[inline]
pub fn dir(lon: f64, lat: f64) -> V3 {
    [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()]
}
/// Longitude and latitude of a unit direction.
#[inline]
pub fn lonlat(d: V3) -> (f64, f64) {
    (d[0].atan2(d[2]), d[1].clamp(-1.0, 1.0).asin())
}

/// Camera (or lens) to world: yaw about +Y (turning right), then pitch (looking up), then roll
/// about the view axis (the picture turns clockwise), degrees.
pub fn camera_rotation(yaw: f64, pitch: f64, roll: f64) -> M3 {
    let fin = |v: f64| if v.is_finite() { v.to_radians() } else { 0.0 };
    let (sy, cy) = fin(yaw).sin_cos();
    let (sp, cp) = (-fin(pitch)).sin_cos();
    let (sr, cr) = fin(roll).sin_cos();
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
    let rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];
    mmul(&ry, &mmul(&rx, &rz))
}

/// Yaw, pitch and roll (degrees) of a camera rotation ([`camera_rotation`]'s inverse; at the poles
/// the yaw takes all of the turn).
pub fn rotation_angles(m: &M3) -> (f64, f64, f64) {
    // forward = R·(0,0,1) = third column; up = second column
    let f = [m[0][2], m[1][2], m[2][2]];
    let (yaw, pitch) = lonlat(f);
    // R = R0 · Rz(roll) with R0 the level camera of this yaw and pitch: the camera's up is
    // −sin(roll) · R0's right + cos(roll) · R0's up
    let u = [m[0][1], m[1][1], m[2][1]];
    let level = camera_rotation(yaw.to_degrees(), pitch.to_degrees(), 0.0);
    let (right0, up0) = ([level[0][0], level[1][0], level[2][0]], [level[0][1], level[1][1], level[2][1]]);
    let roll = (-dot(u, right0)).atan2(dot(u, up0));
    (yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees())
}

/// What the virtual camera makes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Output {
    /// A flat picture: `tan_x`, `tan_y` are the tangents of half the field of view across and down;
    /// `distance` (0..1) pulls the camera back from the sphere's centre (0 rectilinear, 1
    /// stereographic).
    Flat { tan_x: f64, tan_y: f64, distance: f64 },
    /// The whole sphere, equirectangular.
    Sphere,
}

/// A virtual camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Camera to world.
    pub rotation: M3,
    pub output: Output,
}

impl View {
    /// A flat view `fov` degrees across of a `w` × `h` frame.
    pub fn flat(yaw: f64, pitch: f64, roll: f64, fov: f64, distance: f64, w: u32, h: u32) -> Self {
        let fov = if fov.is_finite() { fov.clamp(1.0, 179.0) } else { 90.0 };
        let tan_x = (fov.to_radians() / 2.0).tan();
        let tan_y = tan_x * f64::from(h.max(1)) / f64::from(w.max(1));
        let distance = if distance.is_finite() { distance.clamp(0.0, 1.0) } else { 0.0 };
        View { rotation: camera_rotation(yaw, pitch, roll), output: Output::Flat { tan_x, tan_y, distance } }
    }

    /// The whole sphere, turned by yaw, pitch and roll.
    pub fn sphere(yaw: f64, pitch: f64, roll: f64) -> Self {
        View { rotation: camera_rotation(yaw, pitch, roll), output: Output::Sphere }
    }

    /// The world direction the output point (`u`, `v`) sees, both 0..1 across and down the frame.
    #[inline]
    pub fn ray(&self, u: f64, v: f64) -> V3 {
        let c = match self.output {
            Output::Flat { tan_x, tan_y, distance } => {
                let n = normalize([(2.0 * u - 1.0) * tan_x, (1.0 - 2.0 * v) * tan_y, 1.0]);
                if distance <= 0.0 {
                    n
                } else {
                    // from (0, 0, −d) along n to the unit sphere (inside it, there is always a hit)
                    let b = -distance * n[2];
                    let t = -b + (b * b - (distance * distance - 1.0)).max(0.0).sqrt();
                    normalize([t * n[0], t * n[1], t * n[2] - distance])
                }
            }
            Output::Sphere => dir((2.0 * u - 1.0) * PI, (1.0 - 2.0 * v) * FRAC_PI_2),
        };
        mvec(&self.rotation, c)
    }

    /// Where a world direction lands in the output (`u`, `v` 0..1), if the camera sees it
    /// (rectilinear views: in front of the camera; any direction lands somewhere on the sphere).
    pub fn project(&self, d: V3) -> Option<(f64, f64)> {
        let c = mtvec(&self.rotation, d);
        match self.output {
            Output::Flat { tan_x, tan_y, distance } => {
                // the ray through the camera at (0, 0, −distance) and the sphere point c
                let p = [c[0], c[1], c[2] + distance];
                if p[2] <= 1e-9 {
                    return None;
                }
                let (u, v) = (p[0] / p[2] / tan_x, p[1] / p[2] / tan_y);
                Some(((u + 1.0) / 2.0, (1.0 - v) / 2.0))
            }
            Output::Sphere => {
                let (lon, lat) = lonlat(c);
                Some((lon / TAU + 0.5, 0.5 - lat / PI))
            }
        }
    }
}

/// One fisheye lens in the pixels of the picture being sampled.
#[derive(Clone, Debug, PartialEq)]
pub struct LensPx {
    /// The lens picture's rectangle: x0, y0, x1, y1.
    pub rect: [f64; 4],
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
    /// Radians.
    pub half_fov: f64,
    pub poly: Vec<f64>,
    /// Lens to world.
    pub rotation: M3,
}

impl LensPx {
    fn new(l: &FisheyeLens, w: f64, h: f64) -> Self {
        let l = l.sanitized();
        let (x0, y0, lw, lh) = (l.rect[0] * w, l.rect[1] * h, l.rect[2] * w, l.rect[3] * h);
        LensPx {
            rect: [x0, y0, x0 + lw, y0 + lh],
            cx: x0 + l.cx * lw,
            cy: y0 + l.cy * lh,
            radius: l.radius * lw,
            half_fov: l.half_fov.to_radians(),
            poly: l.poly,
            rotation: camera_rotation(l.yaw, l.pitch, l.roll),
        }
    }

    /// The angle from this lens's axis to `d`, and where the lens saw it (picture pixels), if it
    /// is inside the image circle.
    #[inline]
    pub fn locate(&self, d: V3) -> (f64, Option<(f64, f64)>) {
        let l = mtvec(&self.rotation, d);
        let theta = l[2].clamp(-1.0, 1.0).acos();
        if theta > self.half_fov {
            return (theta, None);
        }
        let t = theta / self.half_fov;
        let r = self.radius * radial(&self.poly, t);
        let s = (l[0] * l[0] + l[1] * l[1]).sqrt();
        let (c, sn) = if s > 1e-12 { (l[0] / s, l[1] / s) } else { (1.0, 0.0) };
        let (x, y) = (self.cx + r * c, self.cy - r * sn);
        let inside = x >= self.rect[0] && x <= self.rect[2] && y >= self.rect[1] && y <= self.rect[3];
        (theta, inside.then_some((x, y)))
    }
}

/// The lens model's distance from the centre as a share of the image circle's radius, at
/// t = θ / half-FOV (equidistant without a polynomial).
#[inline]
pub fn radial(poly: &[f64], t: f64) -> f64 {
    if poly.is_empty() {
        return t;
    }
    let mut acc = 0.0;
    let mut p = t;
    for c in poly {
        acc += c * p;
        p *= t;
    }
    acc
}

/// A source picture's geometry in its own pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceModel {
    /// The view rectangle (x0, y0, w, h: the left eye of a stereoscopic frame), the captured view
    /// in radians across and down, and whether it wraps (all 360°).
    Equirect {
        rect: [f64; 4],
        span: (f64, f64),
        wrap: bool,
    },
    DualFisheye {
        lenses: Vec<LensPx>,
        blend: f64,
    },
}

/// Where the source saw one direction: up to two pixels with their weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Located {
    Missed,
    One(f64, f64),
    /// Equirectangular: wrap horizontally within the view.
    Wrapped(f64, f64),
    Two((f64, f64, f64), (f64, f64, f64)),
}

impl SourceModel {
    /// The model of a picture of `size` described by `s` (sanitised first).
    pub fn new(s: &Spherical, size: (usize, usize)) -> Self {
        let s = s.sanitized();
        let (w, h) = (size.0 as f64, size.1 as f64);
        match &s.projection {
            Projection::Equirectangular => {
                let rect = match s.layout {
                    StereoLayout::Mono => [0.0, 0.0, w, h],
                    StereoLayout::OverUnder => [0.0, 0.0, w, (h / 2.0).floor().max(1.0)],
                    StereoLayout::SideBySide => [0.0, 0.0, (w / 2.0).floor().max(1.0), h],
                };
                let span = (s.captured[0].to_radians(), s.captured[1].to_radians());
                SourceModel::Equirect { rect, span, wrap: s.captured[0] >= 359.999 }
            }
            Projection::DualFisheye { lenses } => {
                SourceModel::DualFisheye { lenses: lenses.iter().map(|l| LensPx::new(l, w, h)).collect(), blend: BLEND_DEGREES.to_radians() }
            }
        }
    }

    /// Where the source saw direction `d`.
    #[inline]
    pub fn locate(&self, d: V3) -> Located {
        match self {
            SourceModel::Equirect { rect, span, wrap } => {
                let (lon, lat) = lonlat(d);
                let x = rect[0] + (lon / span.0 + 0.5) * rect[2];
                let y = rect[1] + (0.5 - lat / span.1) * rect[3];
                if *wrap {
                    return Located::Wrapped(x, y.clamp(rect[1], rect[1] + rect[3]));
                }
                if x < rect[0] || x > rect[0] + rect[2] || y < rect[1] || y > rect[1] + rect[3] {
                    return Located::Missed;
                }
                Located::One(x, y)
            }
            SourceModel::DualFisheye { lenses, blend } => {
                let mut hits = [(0.0, None), (0.0, None)];
                for (k, l) in lenses.iter().take(2).enumerate() {
                    hits[k] = l.locate(d);
                }
                match (hits[0], hits[1], lenses.len()) {
                    ((_, Some((x, y))), _, 1) => Located::One(x, y),
                    ((t0, Some(a)), (t1, Some(b)), _) => {
                        // nearer the first lens's axis: its picture, cross-faded over the band
                        let w0 = smoothstep(-*blend, *blend, t1 - t0);
                        if w0 >= 1.0 {
                            Located::One(a.0, a.1)
                        } else if w0 <= 0.0 {
                            Located::One(b.0, b.1)
                        } else {
                            Located::Two((a.0, a.1, w0), (b.0, b.1, 1.0 - w0))
                        }
                    }
                    ((_, Some((x, y))), _, _) | (_, (_, Some((x, y))), _) => Located::One(x, y),
                    _ => Located::Missed,
                }
            }
        }
    }

    /// Source pixels per radian around the middle of the picture (how sharp it is).
    pub fn pixels_per_radian(&self) -> f64 {
        match self {
            SourceModel::Equirect { rect, span, .. } => rect[2] / span.0.max(1e-6),
            SourceModel::DualFisheye { lenses, .. } => {
                lenses.iter().map(|l| l.radius * radial_slope(&l.poly) / l.half_fov.max(1e-6)).fold(0.0, f64::max).max(1e-6)
            }
        }
    }
}

/// The slope of [`radial`] at the centre.
fn radial_slope(poly: &[f64]) -> f64 {
    match poly.first() {
        None => 1.0,
        Some(c) if *c > 0.0 => *c,
        _ => 1.0,
    }
}

#[inline]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Output pixels per radian at the middle of a view `out_w` pixels across.
pub fn view_pixels_per_radian(view: &View, out_w: usize) -> f64 {
    match view.output {
        Output::Flat { tan_x, distance, .. } => out_w as f64 / (2.0 * tan_x.max(1e-6)) * (1.0 + distance),
        Output::Sphere => out_w as f64 / TAU,
    }
}

/// How much of its size the source is needed at to make a view `out_w` pixels across (≤ 1).
pub fn source_scale(model: &SourceModel, view: &View, out_w: usize) -> f64 {
    (view_pixels_per_radian(view, out_w) / model.pixels_per_radian()).clamp(1.0 / 64.0, 1.0)
}

/// Supersampling taps a side: how many source pixels one output pixel covers (capped).
pub fn taps(model: &SourceModel, view: &View, out_w: usize) -> usize {
    let r = model.pixels_per_radian() / view_pixels_per_radian(view, out_w).max(1e-6);
    if r > 1.25 { (r.ceil() as usize).clamp(1, MAX_TAPS) } else { 1 }
}

/// Bilinear sample of premultiplied pixels at (x, y) (pixel centres at +0.5), clamped to the
/// picture, or wrapping horizontally inside `wrap` (x0, x1).
#[inline]
fn sample(img: &Image, x: f64, y: f64, wrap: Option<(f64, f64)>) -> [f32; 4] {
    let (w, h) = (img.w as isize, img.h as isize);
    if w <= 0 || h <= 0 {
        return [0.0; 4];
    }
    let fx = x - 0.5;
    let fy = (y - 0.5).clamp(0.0, (h - 1) as f64);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = ((fx - x0) as f32, (fy - y0) as f32);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let wx = |xx: isize| -> usize {
        match wrap {
            Some((a, b)) => {
                let (a, b) = (a as isize, (b as isize).max(a as isize + 1));
                (a + (xx - a).rem_euclid(b - a)).clamp(0, w - 1) as usize
            }
            None => xx.clamp(0, w - 1) as usize,
        }
    };
    let g = |xx: isize, yy: isize| img.get(wx(xx), yy.clamp(0, h - 1) as usize);
    let (a, b, c, d) = (g(x0, y0), g(x0 + 1, y0), g(x0, y0 + 1), g(x0 + 1, y0 + 1));
    let mut o = [0.0; 4];
    for k in 0..4 {
        let top = a[k] + (b[k] - a[k]) * tx;
        let bot = c[k] + (d[k] - c[k]) * tx;
        o[k] = top + (bot - top) * ty;
    }
    o
}

/// The source's colour for direction `d` (transparent where it saw nothing).
#[inline]
pub fn sample_dir(src: &Image, model: &SourceModel, d: V3) -> [f32; 4] {
    match model.locate(d) {
        Located::Missed => [0.0; 4],
        Located::One(x, y) => sample(src, x, y, None),
        Located::Wrapped(x, y) => {
            let (x0, x1) = match model {
                SourceModel::Equirect { rect, .. } => (rect[0], rect[0] + rect[2]),
                SourceModel::DualFisheye { .. } => (0.0, src.w as f64),
            };
            sample(src, x, y, Some((x0, x1)))
        }
        Located::Two((xa, ya, wa), (xb, yb, wb)) => {
            let (a, b) = (sample(src, xa, ya, None), sample(src, xb, yb, None));
            let (wa, wb) = (wa as f32, wb as f32);
            [a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb]
        }
    }
}

/// The view `out_w` × `out_h` of a 360° source picture (premultiplied linear), `taps` × `taps`
/// rays a pixel.
pub fn reframe(src: &Image, model: &SourceModel, view: &View, out_w: usize, out_h: usize, taps: usize) -> Image {
    let mut out = Image::new(out_w, out_h);
    if out_w == 0 || out_h == 0 || src.w == 0 || src.h == 0 {
        return out;
    }
    let n = taps.clamp(1, MAX_TAPS);
    let inv = 1.0 / (n * n) as f32;
    let (fw, fh) = (out_w as f64, out_h as f64);
    out.px.par_chunks_mut(out_w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..out_w {
            let mut acc = [0.0f32; 4];
            for j in 0..n {
                for i in 0..n {
                    let u = (x as f64 + (i as f64 + 0.5) / n as f64) / fw;
                    let v = (y as f64 + (j as f64 + 0.5) / n as f64) / fh;
                    let c = sample_dir(src, model, view.ray(u, v));
                    for k in 0..4 {
                        acc[k] += c[k];
                    }
                }
            }
            row[x * 4..x * 4 + 4].copy_from_slice(&acc.map(|c| c * inv));
        }
    });
    out
}

/// How well the two lenses of a dual-fisheye picture agree where they overlap: the mean luma
/// difference (linear light) between what each saw of directions around the great circle half
/// way between their axes, and a few degrees either side, leaving out the 30 % largest differences
/// (near things, which the lenses see from different places, and exposure flare). `None` when the
/// picture is not dual fisheye or the lenses share no direction.
pub fn seam_error(src: &Image, model: &SourceModel) -> Option<f64> {
    let SourceModel::DualFisheye { lenses, .. } = model else { return None };
    let (a, b) = (lenses.first()?, lenses.get(1)?);
    // the plane half way between the axes
    let za = mvec(&a.rotation, [0.0, 0.0, 1.0]);
    let zb = mvec(&b.rotation, [0.0, 0.0, 1.0]);
    let n = normalize([za[0] - zb[0], za[1] - zb[1], za[2] - zb[2]]);
    // two directions spanning the plane
    let helper = if n[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let e1 = normalize(cross(helper, n));
    let e2 = cross(n, e1);
    let luma = |c: [f32; 4]| f64::from(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]);
    let mut errs = Vec::with_capacity(SEAM_RAYS * 9 * 5);
    for k in 0..SEAM_RAYS {
        let around = k as f64 / SEAM_RAYS as f64 * TAU;
        let (s, c) = around.sin_cos();
        let inplane = [c * e1[0] + s * e2[0], c * e1[1] + s * e2[1], c * e1[2] + s * e2[2]];
        // keep away from the axis the two lenses are held on (the stick)
        if inplane[1].abs() > 0.94 {
            continue;
        }
        for off in [-3.0f64, -1.5, 0.0, 1.5, 3.0] {
            let (so, co) = off.to_radians().sin_cos();
            let d = [co * inplane[0] + so * n[0], co * inplane[1] + so * n[1], co * inplane[2] + so * n[2]];
            if let ((_, Some(pa)), (_, Some(pb))) = (a.locate(d), b.locate(d)) {
                errs.push((luma(sample(src, pa.0, pa.1, None)) - luma(sample(src, pb.0, pb.1, None))).abs());
            }
        }
    }
    if errs.len() < 32 {
        return None;
    }
    errs.sort_by(f64::total_cmp);
    let keep = errs.len() * 7 / 10;
    Some(errs.get(..keep)?.iter().sum::<f64>() / keep.max(1) as f64)
}

/// Directions sampled around the seam.
const SEAM_RAYS: usize = 1440;

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// What [`align_lenses`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct Alignment {
    pub spherical: Spherical,
    /// [`seam_error`] before and after.
    pub before: f64,
    pub after: f64,
}

/// Lines a dual-fisheye picture's lenses up with each other: the half field of view both share,
/// and the first lens's yaw, pitch and roll against the second's, moved (by a few degrees at
/// most, in halving steps) to where their pictures agree best along the seam ([`seam_error`],
/// summed over `frames`, every one a picture of the same size). Returns the description unchanged
/// (with equal errors) when nothing better is found, and `None` when it is not dual fisheye.
pub fn align_lenses(frames: &[Image], s: &Spherical) -> Option<Alignment> {
    let first = frames.first()?;
    let size = (first.w, first.h);
    let score = |s: &Spherical| -> Option<f64> {
        let model = SourceModel::new(s, size);
        let mut total = 0.0;
        for f in frames.iter().filter(|f| (f.w, f.h) == size) {
            total += seam_error(f, &model)?;
        }
        Some(total / frames.len() as f64)
    };
    let base = s.sanitized();
    if !matches!(&base.projection, Projection::DualFisheye { lenses } if lenses.len() == 2) {
        return None;
    }
    let before = score(&base)?;
    // [half_fov, yaw, pitch, roll] offsets of the first lens; half_fov moves both lenses
    let apply = |x: &[f64; 4]| -> Spherical {
        let mut t = base.clone();
        if let Projection::DualFisheye { lenses } = &mut t.projection {
            for l in lenses.iter_mut() {
                l.half_fov += x[0];
            }
            if let Some(l) = lenses.first_mut() {
                l.yaw += x[1];
                l.pitch += x[2];
                l.roll += x[3];
            }
        }
        t
    };
    let limits = [4.0, 3.0, 3.0, 3.0];
    let mut x = [0.0f64; 4];
    let mut best = before;
    let mut step = [1.0, 0.5, 0.5, 0.5];
    for _ in 0..7 {
        let mut moved = true;
        let mut rounds = 0;
        while moved && rounds < 8 {
            moved = false;
            rounds += 1;
            for i in 0..4 {
                for dir in [1.0, -1.0] {
                    let mut y = x;
                    y[i] = (y[i] + dir * step[i]).clamp(-limits[i], limits[i]);
                    if let Some(e) = score(&apply(&y))
                        && e < best - 1e-7
                    {
                        best = e;
                        x = y;
                        moved = true;
                    }
                }
            }
        }
        step = step.map(|v| v / 2.0);
    }
    Some(Alignment { spherical: apply(&x).sanitized(), before, after: best })
}

/// The turn a clip's stabilisation (`clip.stabilize360`: the `stab_yaw` / `stab_pitch` /
/// `stab_roll` keyframes) gives its camera's rays at media time `t`, before they are looked up in
/// the frame; none without it.
pub fn stabilization(e: &EffectInstance, t: Tick) -> M3 {
    let f = |id: &str| {
        let v = e.f64_at(id, t);
        if v.is_finite() { v } else { 0.0 }
    };
    camera_rotation(f("stab_yaw"), f("stab_pitch"), f("stab_roll"))
}

/// The virtual camera of a clip at media time `t`: the `vr_reframe` effect's view (its
/// stabilisation applied) and its frame size ([`filmcraft_project::vr::reframe_size`]).
pub fn clip_view(e: &EffectInstance, t: Tick) -> Option<(View, (u32, u32))> {
    let size = filmcraft_project::vr::effect_size(e)?;
    let f = |id: &str, d: f64| {
        let v = e.f64_at(id, t);
        if v.is_finite() { v } else { d }
    };
    let (yaw, pitch, roll) = (f("yaw", 0.0), f("pitch", 0.0), f("roll", 0.0));
    let sphere = matches!(e.param("output").map(|p| &p.value), Some(ParamValue::Choice(1)));
    let mut view = if sphere { View::sphere(yaw, pitch, roll) } else { View::flat(yaw, pitch, roll, f("fov", 90.0), f("distance", 0.0), size.0, size.1) };
    view.rotation = mmul(&stabilization(e, t), &view.rotation);
    Some((view, size))
}

/// A clip's enabled virtual camera, its view at media time `t` and frame size.
pub fn item_view(item: &TrackItem, t: Tick) -> Option<(View, (u32, u32))> {
    clip_view(filmcraft_project::vr::reframe_of(item)?, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: V3, b: V3) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
    }

    #[test]
    fn rotation_conventions() {
        // yaw right, pitch up, roll clockwise
        assert!(close(mvec(&camera_rotation(90.0, 0.0, 0.0), [0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]));
        assert!(close(mvec(&camera_rotation(0.0, 90.0, 0.0), [0.0, 0.0, 1.0]), [0.0, 1.0, 0.0]));
        // rolling clockwise: the camera's up turns left, so the world's up looks turned right
        let up = mvec(&camera_rotation(0.0, 0.0, 30.0), [0.0, 1.0, 0.0]);
        assert!(up[0] < 0.0);
        for (y, p, r) in [(30.0, 10.0, 5.0), (-120.0, -40.0, 20.0), (170.0, 60.0, -15.0)] {
            let (a, b, c) = rotation_angles(&camera_rotation(y, p, r));
            assert!((a - y).abs() < 1e-6 && (b - p).abs() < 1e-6 && (c - r).abs() < 1e-6, "{y} {p} {r} → {a} {b} {c}");
        }
    }

    #[test]
    fn views_and_their_inverse() {
        for v in [View::flat(40.0, -20.0, 10.0, 100.0, 0.0, 1920, 1080), View::flat(0.0, -90.0, 0.0, 120.0, 1.0, 1080, 1080), View::sphere(10.0, 5.0, 0.0)] {
            for (u, w) in [(0.5, 0.5), (0.1, 0.8), (0.9, 0.2)] {
                let d = v.ray(u, w);
                assert!((dot(d, d) - 1.0).abs() < 1e-9);
                let (pu, pv) = v.project(d).unwrap();
                assert!((pu - u).abs() < 1e-9 && (pv - w).abs() < 1e-9, "{u} {w} → {pu} {pv}");
            }
        }
        // the middle of a view looks where the camera points
        assert!(close(View::flat(90.0, 0.0, 0.0, 90.0, 0.0, 100, 100).ray(0.5, 0.5), [1.0, 0.0, 0.0]));
        // behind a rectilinear camera there is nothing to see
        assert!(View::flat(0.0, 0.0, 0.0, 90.0, 0.0, 100, 100).project([0.0, 0.0, -1.0]).is_none());
    }

    /// An equirectangular picture whose pixel colour encodes its own longitude and latitude.
    fn coded_equirect(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                img.px[i] = (x as f32 + 0.5) / w as f32;
                img.px[i + 1] = (y as f32 + 0.5) / h as f32;
                img.px[i + 3] = 1.0;
            }
        }
        img
    }

    #[test]
    fn a_flat_view_of_an_equirect_looks_where_it_points() {
        let src = coded_equirect(720, 360);
        let model = SourceModel::new(&Spherical::equirect(), (720, 360));
        let view = View::flat(90.0, 30.0, 0.0, 60.0, 0.0, 64, 36);
        let out = reframe(&src, &model, &view, 64, 36, 1);
        let c = out.get(32, 18);
        // longitude 90° is three quarters across, latitude 30° a third of the way down
        assert!((c[0] - 0.75).abs() < 0.01, "{c:?}");
        assert!((c[1] - 1.0 / 3.0).abs() < 0.01, "{c:?}");
        assert!(c[3] > 0.99);
    }

    #[test]
    fn the_seam_wraps() {
        let src = coded_equirect(360, 180);
        let model = SourceModel::new(&Spherical::equirect(), (360, 180));
        // looking backwards: the ±180° seam is in the middle of the view
        let out = reframe(&src, &model, &View::flat(180.0, 0.0, 0.0, 40.0, 0.0, 40, 20), 40, 20, 1);
        let (l, r) = (out.get(18, 10)[0], out.get(21, 10)[0]);
        assert!(l > 0.9 && r < 0.1, "{l} {r}");
        assert!((0..40).all(|x| out.get(x, 10)[3] > 0.99));
    }

    fn two_lenses() -> Spherical {
        let lens = |k: usize, yaw: f64| FisheyeLens {
            rect: [k as f64 * 0.5, 0.0, 0.5, 1.0],
            cx: 0.5,
            cy: 0.5,
            radius: 0.5,
            half_fov: 100.0,
            poly: vec![],
            yaw,
            pitch: 0.0,
            roll: 0.0,
        };
        Spherical::dual_fisheye(vec![lens(0, 180.0), lens(1, 0.0)], "test")
    }

    /// A dual-fisheye picture made from a coded equirect through the lens model itself.
    fn dual_from(src: &Image, model_src: &SourceModel, s: &Spherical, w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        let Projection::DualFisheye { lenses } = &s.projection else { panic!() };
        let lenses: Vec<LensPx> = lenses.iter().map(|l| LensPx::new(l, w as f64, h as f64)).collect();
        for y in 0..h {
            for x in 0..w {
                let l = &lenses[if x < w / 2 { 0 } else { 1 }];
                let (px, py) = (x as f64 + 0.5 - l.cx, l.cy - (y as f64 + 0.5));
                let r = (px * px + py * py).sqrt();
                if r > l.radius {
                    continue;
                }
                let theta = r / l.radius * l.half_fov;
                let phi = py.atan2(px);
                let local = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()];
                let c = sample_dir(src, model_src, mvec(&l.rotation, local));
                img.px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&c);
            }
        }
        img
    }

    #[test]
    fn dual_fisheye_reprojects_to_the_same_sphere() {
        let eq = coded_equirect(720, 360);
        let eq_model = SourceModel::new(&Spherical::equirect(), (720, 360));
        let s = two_lenses();
        let dual = dual_from(&eq, &eq_model, &s, 800, 400);
        let model = SourceModel::new(&s, (800, 400));
        // forward, sideways (in the overlap) and backward views match the equirect's
        for yaw in [0.0, 90.0, -90.0, 180.0, 45.0] {
            let view = View::flat(yaw, 10.0, 0.0, 50.0, 0.0, 32, 18);
            let a = reframe(&dual, &model, &view, 32, 18, 1);
            let b = reframe(&eq, &eq_model, &view, 32, 18, 1);
            let worst =
                a.px.chunks(4)
                    .zip(b.px.chunks(4))
                    .filter(|(p, _)| p[3] > 0.99)
                    .map(|(p, q)| ((p[1] - q[1]).abs()).max(if (p[0] - q[0]).abs() > 0.5 { 0.0 } else { (p[0] - q[0]).abs() }))
                    .fold(0.0f32, f32::max);
            assert!(worst < 0.02, "yaw {yaw}: {worst}");
            assert!(a.px.chunks(4).all(|p| p[3] > 0.99), "yaw {yaw}: a hole");
        }
    }

    /// A picture with texture everywhere (so a misaligned seam shows): colour bands by direction.
    fn textured_equirect(w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (lon, lat) = ((x as f64 + 0.5) / w as f64 * TAU, (y as f64 + 0.5) / h as f64 * PI);
                let v = 0.5 + 0.25 * (lon * 9.0).sin() * (lat * 7.0).cos() + 0.2 * (lon * 23.0 + lat * 3.0).sin();
                let i = (y * w + x) * 4;
                img.px[i..i + 4].copy_from_slice(&[v as f32, (1.0 - v) as f32 * 0.5, 0.3, 1.0]);
            }
        }
        img
    }

    #[test]
    fn lens_alignment_recovers_a_misaligned_lens() {
        let eq = textured_equirect(1440, 720);
        let eq_model = SourceModel::new(&Spherical::equirect(), (1440, 720));
        // the camera: the back lens turned by 1° yaw and 0.6° pitch, both lenses 97.5° half-FOV
        let mut truth = two_lenses();
        if let Projection::DualFisheye { lenses } = &mut truth.projection {
            for l in lenses.iter_mut() {
                l.half_fov = 97.5;
            }
            lenses[0].yaw += 1.0;
            lenses[0].pitch -= 0.6;
        }
        let dual = dual_from(&eq, &eq_model, &truth, 1600, 800);
        // what the metadata says: no misalignment, 100°
        let start = two_lenses();
        let a = align_lenses(std::slice::from_ref(&dual), &start).unwrap();
        assert!(a.after < a.before * 0.5, "{} → {}", a.before, a.after);
        let Projection::DualFisheye { lenses } = &a.spherical.projection else { panic!() };
        assert!((lenses[1].half_fov - 97.5).abs() < 0.6, "{lenses:?}");
        assert!((lenses[0].yaw - 181.0).abs() < 0.5 && (lenses[0].pitch + 0.6).abs() < 0.5, "{lenses:?}");
        // an equirect has no lenses to align
        assert!(align_lenses(&[eq], &Spherical::equirect()).is_none());
        assert!(align_lenses(&[], &start).is_none());
    }

    #[test]
    fn hostile_views_never_panic() {
        let src = coded_equirect(16, 8);
        for s in [Spherical::equirect(), two_lenses()] {
            let model = SourceModel::new(&s, (16, 8));
            for v in [
                View::flat(f64::NAN, f64::INFINITY, -1e300, f64::NAN, f64::NAN, 0, 0),
                View::flat(1e9, 1e9, 1e9, 1e9, 1e9, 1, 1),
                View::sphere(f64::NAN, 0.0, 0.0),
            ] {
                let o = reframe(&src, &model, &v, 5, 3, 99);
                assert!(o.px.iter().all(|c| c.is_finite()));
            }
            let _ = reframe(&Image::new(0, 0), &model, &View::sphere(0.0, 0.0, 0.0), 4, 4, 1);
            assert!(source_scale(&model, &View::sphere(0.0, 0.0, 0.0), 0) > 0.0);
        }
        let empty = SourceModel::new(&Spherical::dual_fisheye(vec![], ""), (16, 8));
        assert_eq!(empty.locate([0.0, 0.0, 1.0]), Located::Missed);
    }
}
