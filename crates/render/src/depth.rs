//! The depth effects ([`filmcraft_project::effect::PHOTO_3D`], [`DEPTH_BLUR`]): a picture moved in
//! depth, the near parts further than the far ones, and blurred by how far each part is.
//!
//! **Where the depth comes from.** The host installs a [`DepthModel`] ([`set_model`]: the engine's,
//! Depth Anything V2 Small); the clip's own picture (its raw frame, at most [`ANALYSIS_SIDE`] a
//! side) is given to it, and its answer kept for the picture's pixels ([`CACHE`] answers: a photo
//! is looked at once). Without a model the effects leave the picture as it is.
//!
//! **The 3D photo.** A part of nearness `n` (0 the farthest, 1 the nearest) moves by
//! `(n − pivot) · V`, where `V` is the nearest part's travel plus its push away from the middle
//! (`(p − c) · push`): the still depth stays, nearer parts follow the travel and farther ones go
//! the other way. Each output pixel looks back along `V` from the nearest depth to the farthest
//! (a pixel's worth a step, at most [`MAX_STEPS`]) for the first surface whose moved place it is
//! ([`source_of`]), so a near thing covers what is behind it; what it uncovers is the farther
//! picture stretched. The depth is first spread outwards by [`HOLD`] of its width (each value the
//! largest around it), so an outline moves whole with its thing. Zoom to Fill enlarges the
//! picture just enough that the largest move the effect's keyframes ask for shows no edge.
//!
//! **Depth blur.** Each part's blur is a disc whose radius grows from nothing inside the depth of
//! field to the full Blur half a depth beyond it (in front too, unless Blur the Foreground is
//! off). A pixel gathers the discs that reach it ([`blur_cpu`]: every pixel within the largest
//! radius, or 256 taps on a golden-angle spiral beyond 256 pixels' area), each weighted by the
//! area it stands for over its disc's — so a sharp pixel keeps its colour, a blurred one is its
//! disc's average — and a farther part's disc only within the pixel's own: a sharp thing in front
//! is not smeared over by the blurred background behind it.
//!
//! Both run on the GPU when the host installs one ([`set_gpu`], `filmcraft_gpu::depthfx`), the
//! same arithmetic; this CPU code is the reference and the fallback.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use filmcraft_project::EffectInstance;
use filmcraft_project::effect::{DEPTH_BLUR, PHOTO_3D};
use rayon::prelude::*;

use crate::effects::{FxCtx, b, f};
use crate::image::Image;

/// The long side of the picture the depth is made from, at most.
pub const ANALYSIS_SIDE: usize = 1036;
/// How far the depth is spread outwards for the 3D photo, a share of its width.
pub const HOLD: f32 = 0.004;
/// Steps along a pixel's look back, at least and at most.
pub const MIN_STEPS: u32 = 4;
pub const MAX_STEPS: u32 = 96;
/// The most taps a blurred pixel gathers.
pub const MAX_TAPS: usize = 256;
/// A part this much farther than the pixel counts as behind it.
pub const BEHIND: f32 = 0.02;
/// Depth answers kept (a photo each).
const CACHED: usize = 16;

/// How near each part of a picture is: `near` row by row, 0 the farthest and 1 the nearest,
/// covering the whole picture at its own size.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthField {
    pub w: usize,
    pub h: usize,
    pub near: Vec<f32>,
}

impl DepthField {
    /// A field of `w` × `h` values (each made 0..1; non-finite ones 0), or `None` when the sizes
    /// and the values disagree.
    pub fn new(w: usize, h: usize, mut near: Vec<f32>) -> Option<Self> {
        if w == 0 || h == 0 || w.checked_mul(h) != Some(near.len()) {
            return None;
        }
        near.iter_mut().for_each(|v| *v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
        Some(Self { w, h, near })
    }

    /// The nearness at `(u, v)`, shares of the picture (corners aligned, clamped), bilinear.
    pub fn at(&self, u: f32, v: f32) -> f32 {
        let x = (if u.is_finite() { u.clamp(0.0, 1.0) } else { 0.5 }) * (self.w.max(1) - 1) as f32;
        let y = (if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 }) * (self.h.max(1) - 1) as f32;
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w.max(1) - 1), (y0 + 1).min(self.h.max(1) - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |x: usize, y: usize| self.near.get(y * self.w + x).copied().unwrap_or(0.0);
        let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * fx;
        let bottom = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * fx;
        top + (bottom - top) * fy
    }

    /// Each value the largest within `r` samples (a square), the 3D photo's held outlines.
    pub fn held(&self, r: usize) -> DepthField {
        if r == 0 {
            return self.clone();
        }
        let (w, h) = (self.w, self.h);
        let mut rows = vec![0f32; w * h];
        rows.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = (x.saturating_sub(r)..=(x + r).min(w - 1)).map(|xx| self.near.get(y * w + xx).copied().unwrap_or(0.0)).fold(0.0, f32::max);
            }
        });
        let mut out = vec![0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = (y.saturating_sub(r)..=(y + r).min(h - 1)).map(|yy| rows.get(yy * w + x).copied().unwrap_or(0.0)).fold(0.0, f32::max);
            }
        });
        DepthField { w, h, near: out }
    }
}

/// Makes depth from pictures (installed by the host: a depth network).
pub trait DepthModel: Send + Sync {
    /// How near each part of an sRGB RGBA picture (`w` × `h`) is; `key` names its pixels (for a
    /// cache of the model's own). `None` when it cannot say (no network downloaded, a failure).
    fn depth(&self, key: u64, rgba: &[u8], w: usize, h: usize) -> Option<DepthField>;
}

static MODEL: OnceLock<Arc<dyn DepthModel>> = OnceLock::new();

/// Install the depth model (once; later calls are ignored).
pub fn set_model(m: Arc<dyn DepthModel>) {
    let _ = MODEL.set(m);
}

/// Whether a depth model is installed.
pub fn has_model() -> bool {
    MODEL.get().is_some()
}

/// The depth effects on the GPU (installed by the host); `None` to work on the CPU.
pub trait DepthGpu: Send + Sync {
    /// [`parallax_cpu`] on the GPU.
    fn parallax(&self, img: &Image, held: &DepthField, p: &Parallax) -> Option<Image>;
    /// [`blur_cpu`] on the GPU: each pixel's nearness and blur radius, and the taps.
    fn blur(&self, img: &Image, nr: &[[f32; 2]], taps: &[Tap]) -> Option<Image>;
}

static GPU: OnceLock<Arc<dyn DepthGpu>> = OnceLock::new();

/// Install the GPU's depth effects (once; later calls are ignored).
pub fn set_gpu(g: Arc<dyn DepthGpu>) {
    let _ = GPU.set(g);
}

/// Whether the depth effects run on the GPU.
pub fn on_gpu() -> bool {
    GPU.get().is_some()
}

// ---------------------------------------------------------------------------------------------
// the 3D photo
// ---------------------------------------------------------------------------------------------

/// The 3D photo's parameters at a time, in working pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parallax {
    /// The nearest part's move past the still depth.
    pub travel: [f32; 2],
    /// The nearest part's growth past the still depth (a share of its distance from the middle).
    pub push: f32,
    /// The depth that stays where it is.
    pub pivot: f32,
    /// The picture enlarged about its middle (≥ 1): Zoom to Fill.
    pub zoom: f32,
}

/// A float parameter's value and its keyframes' values, as shares (`/ 100`), finite ones only.
fn values(e: &EffectInstance, id: &str) -> Vec<f32> {
    e.param(id)
        .map(|p| {
            std::iter::once(&p.value)
                .chain(p.keyframes.iter().map(|k| &k.value))
                .filter_map(|v| v.as_f64())
                .map(|v| v as f32 / 100.0)
                .filter(|v| v.is_finite())
                .collect()
        })
        .unwrap_or_default()
}

/// A share parameter at the effect's time, kept within `±limit` (0 when not finite).
fn share(e: &EffectInstance, id: &str, cx: &FxCtx, limit: f32) -> f32 {
    let v = f(e, id, cx) / 100.0;
    if v.is_finite() { v.clamp(-limit, limit) } else { 0.0 }
}

impl Parallax {
    /// The parameters of `e` at `cx`'s time for a `w` × `h` working picture.
    pub fn eval(e: &EffectInstance, cx: &FxCtx, w: usize, h: usize) -> Self {
        let wf = w.max(1) as f32;
        let travel = [share(e, "travel_x", cx, 0.2) * wf, share(e, "travel_y", cx, 0.2) * wf];
        let push = share(e, "push", cx, 0.5);
        let pivot = share(e, "pivot", cx, 1.0).clamp(0.0, 1.0);
        let zoom = if e.param("fill").is_none() || b(e, "fill") { Self::fill(e, pivot, w, h) } else { 1.0 };
        Self { travel, push, pivot, zoom }
    }

    /// The zoom that keeps the edges out of view over all of the effect's keyframes: a part at
    /// the edge moves outwards by `(1 + grow)` and across by up to `k · travel` (`k` the largest
    /// distance from the still depth).
    fn fill(e: &EffectInstance, pivot: f32, w: usize, h: usize) -> f32 {
        let k = pivot.max(1.0 - pivot);
        let most = |id: &str| values(e, id).iter().map(|v| v.abs().min(0.2)).fold(0.0, f32::max);
        let grow = values(e, "push").iter().map(|p| p.clamp(-0.5, 0.5)).map(|p| (pivot * p).max((pivot - 1.0) * p)).fold(0.0, f32::max);
        let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
        // the move across in pixels against half the picture
        let zx = (1.0 + grow) / (1.0 - k * most("travel_x") * wf / (wf * 0.5)).max(0.2);
        let zy = (1.0 + grow) / (1.0 - k * most("travel_y") * wf / (hf * 0.5)).max(0.2);
        zx.max(zy).clamp(1.0, 5.0)
    }

    /// Whether it moves nothing.
    pub fn is_identity(&self) -> bool {
        self.travel.iter().all(|v| v.abs() < 1e-3) && self.push.abs() < 1e-6 && (self.zoom - 1.0).abs() < 1e-6
    }
}

/// Where in the picture working pixel `(x, y)` of a `w` × `h` picture shows: the nearest surface
/// of `held` whose moved place it is.
pub fn source_of(x: f32, y: f32, w: usize, h: usize, held: &DepthField, p: &Parallax) -> (f32, f32) {
    let (cx, cy) = ((w.max(1) - 1) as f32 * 0.5, (h.max(1) - 1) as f32 * 0.5);
    let zoom = if p.zoom.is_finite() && p.zoom > 0.0 { p.zoom } else { 1.0 };
    let (qx, qy) = (cx + (x - cx) / zoom, cy + (y - cy) / zoom);
    let (vx, vy) = (p.travel[0] + (qx - cx) * p.push, p.travel[1] + (qy - cy) * p.push);
    let len = (vx * vx + vy * vy).sqrt();
    if !(len.is_finite() && len > 1e-3) {
        return (qx, qy);
    }
    let n = (len.ceil() as u32).clamp(MIN_STEPS, MAX_STEPS);
    let (su, sv) = (1.0 / (w.max(2) - 1) as f32, 1.0 / (h.max(2) - 1) as f32);
    let at = |s: f32| {
        let (px, py) = (qx - (s - p.pivot) * vx, qy - (s - p.pivot) * vy);
        held.at(px * su, py * sv) - s
    };
    let (mut ps, mut pf) = (1.0f32, at(1.0));
    let mut hit = 1.0f32;
    if pf < 0.0 {
        hit = 0.0;
        for i in 1..=n {
            let s = 1.0 - i as f32 / n as f32;
            let f1 = at(s);
            if f1 >= 0.0 {
                // the crossing between the last two samples (pf < 0 ≤ f1)
                hit = ps + (s - ps) * (pf / (pf - f1));
                break;
            }
            (ps, pf) = (s, f1);
        }
    }
    (qx - (hit - p.pivot) * vx, qy - (hit - p.pivot) * vy)
}

/// The picture's colour at `(x, y)` (working pixels, clamped to the picture), bilinear.
fn sample(img: &Image, x: f32, y: f32) -> [f32; 4] {
    let (mx, my) = ((img.w.max(1) - 1) as f32, (img.h.max(1) - 1) as f32);
    let (x, y) = (if x.is_finite() { x.clamp(0.0, mx) } else { 0.0 }, if y.is_finite() { y.clamp(0.0, my) } else { 0.0 });
    let (x0, y0) = (x.floor() as isize, y.floor() as isize);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let (a, b, c, d) = (img.get_clamped(x0, y0), img.get_clamped(x0 + 1, y0), img.get_clamped(x0, y0 + 1), img.get_clamped(x0 + 1, y0 + 1));
    std::array::from_fn(|k| {
        let top = a[k] + (b[k] - a[k]) * fx;
        top + (c[k] + (d[k] - c[k]) * fx - top) * fy
    })
}

/// The 3D photo on the CPU (the reference).
pub fn parallax_cpu(img: &Image, held: &DepthField, p: &Parallax) -> Image {
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(w, h);
    out.px.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for (x, o) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (sx, sy) = source_of(x as f32, y as f32, w, h, held, p);
            o.copy_from_slice(&sample(img, sx, sy));
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// depth blur
// ---------------------------------------------------------------------------------------------

/// Depth Blur's parameters at a time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurOp {
    /// The sharp depth, its depth of field, and the largest radius (working pixels).
    pub focus: f32,
    pub range: f32,
    pub radius: f32,
    /// Whether what is nearer than the focus blurs too.
    pub near: bool,
}

impl BlurOp {
    pub fn eval(e: &EffectInstance, cx: &FxCtx, w: usize) -> Self {
        let focus = share(e, "focus", cx, 1.0).clamp(0.0, 1.0);
        let range = share(e, "range", cx, 1.0).clamp(0.0, 1.0);
        // (a share of the width: at most a tenth)
        let radius = share(e, "blur", cx, 0.1).max(0.0) * w as f32;
        Self { focus, range, radius, near: e.param("near").is_none() || b(e, "near") }
    }

    /// The blur's radius for nearness `n`.
    pub fn radius_at(&self, n: f32) -> f32 {
        if !self.near && n > self.focus {
            return 0.0;
        }
        self.radius * (((n - self.focus).abs() - self.range * 0.5) * 2.0).clamp(0.0, 1.0)
    }
}

/// One tap of the blur: its offset (whole pixels), its distance and the area it stands for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tap {
    pub dx: i32,
    pub dy: i32,
    pub dist: f32,
    pub area: f32,
}

/// The taps within `radius`: every pixel of the disc when it has at most [`MAX_TAPS`], else
/// [`MAX_TAPS`] on a golden-angle spiral, each standing for an equal share of the disc. The
/// centre is not among them.
pub fn taps(radius: f32) -> Vec<Tap> {
    if !(radius.is_finite() && radius >= 0.5) {
        return Vec::new();
    }
    let r = radius.min(4096.0);
    let disc = std::f32::consts::PI * r * r;
    if disc <= MAX_TAPS as f32 {
        let ri = r.ceil() as i32;
        let mut v = Vec::new();
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                if (dx, dy) != (0, 0) && d <= r {
                    v.push(Tap { dx, dy, dist: d, area: 1.0 });
                }
            }
        }
        return v;
    }
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    let area = disc / MAX_TAPS as f32;
    (0..MAX_TAPS)
        .map(|k| {
            let rho = r * ((k as f32 + 0.5) / MAX_TAPS as f32).sqrt();
            let a = k as f32 * golden;
            let (dx, dy) = ((rho * a.cos()).round() as i32, (rho * a.sin()).round() as i32);
            Tap { dx, dy, dist: ((dx * dx + dy * dy) as f32).sqrt(), area }
        })
        .filter(|t| (t.dx, t.dy) != (0, 0))
        .collect()
}

/// Each working pixel's nearness and blur radius.
pub fn near_and_radius(field: &DepthField, op: &BlurOp, w: usize, h: usize) -> Vec<[f32; 2]> {
    let (su, sv) = (1.0 / (w.max(2) - 1) as f32, 1.0 / (h.max(2) - 1) as f32);
    let mut v = vec![[0f32; 2]; w * h];
    v.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let n = field.at(x as f32 * su, y as f32 * sv);
            *o = [n, op.radius_at(n)];
        }
    });
    v
}

/// A disc of radius `r`'s share for one pixel's area: the weight a tap of `area` gets.
fn weight(area: f32, r: f32) -> f32 {
    area / (std::f32::consts::PI * r * r).max(1.0)
}

/// Depth Blur on the CPU (the reference).
pub fn blur_cpu(img: &Image, nr: &[[f32; 2]], taps: &[Tap]) -> Image {
    let (w, h) = (img.w, img.h);
    let mut out = Image::new(w, h);
    if nr.len() != w * h {
        return img.clone();
    }
    out.px.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for (x, o) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let [np, rp] = nr.get(y * w + x).copied().unwrap_or([0.0, 0.0]);
            let own = weight(1.0, rp);
            let c = img.get(x, y);
            let (mut acc, mut sum) = ([c[0] * own, c[1] * own, c[2] * own, c[3] * own], own);
            for t in taps {
                let (qx, qy) = (x as i64 + i64::from(t.dx), y as i64 + i64::from(t.dy));
                if qx < 0 || qy < 0 || qx >= w as i64 || qy >= h as i64 {
                    continue;
                }
                let (qx, qy) = (qx as usize, qy as usize);
                let [nq, rq] = nr.get(qy * w + qx).copied().unwrap_or([0.0, 0.0]);
                // its disc does not reach here, or it is behind and this pixel's own does not reach it
                if t.dist > rq + 0.5 || (nq < np - BEHIND && t.dist > rp + 0.5) {
                    continue;
                }
                let wt = weight(t.area, rq);
                let c = img.get(qx, qy);
                for k in 0..4 {
                    acc[k] += c[k] * wt;
                }
                sum += wt;
            }
            for k in 0..4 {
                o[k] = acc[k] / sum;
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// the picture's depth
// ---------------------------------------------------------------------------------------------

/// A picture's depth as the effects use it.
pub struct Cached {
    pub field: DepthField,
    /// Spread outwards by [`HOLD`] (the 3D photo's).
    pub held: DepthField,
}

/// The depth answers kept, newest last: the picture's key and its depth.
static CACHE: Mutex<Vec<(u64, Arc<Cached>)>> = Mutex::new(Vec::new());
/// One picture's depth made at a time (a frame's threads ask for the same picture).
static MAKING: Mutex<()> = Mutex::new(());

/// FNV-1a of a picture's size and pixels.
pub fn picture_key(rgba: &[u8], w: usize, h: usize) -> u64 {
    let mut k: u64 = 0xcbf2_9ce4_8422_2325;
    for b in (w as u64).to_le_bytes().iter().chain(&(h as u64).to_le_bytes()).chain(rgba) {
        k = (k ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    k
}

/// A premultiplied linear picture as sRGB RGBA bytes, area-averaged down to at most
/// [`ANALYSIS_SIDE`] a side.
fn analysis_bytes(img: &Image) -> (Vec<u8>, usize, usize) {
    let (w, h) = (img.w.max(1), img.h.max(1));
    let s = (ANALYSIS_SIDE as f32 / w.max(h) as f32).min(1.0);
    let (ow, oh) = (((w as f32 * s).round() as usize).max(1), ((h as f32 * s).round() as usize).max(1));
    let mut out = vec![0u8; ow * oh * 4];
    out.par_chunks_mut(ow * 4).enumerate().for_each(|(y, row)| {
        let (y0, y1) = (y * h / oh, ((y + 1) * h / oh).max(y * h / oh + 1).min(h));
        for x in 0..ow {
            let (x0, x1) = (x * w / ow, ((x + 1) * w / ow).max(x * w / ow + 1).min(w));
            let mut acc = [0f32; 4];
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let p = img.get(xx.min(img.w.saturating_sub(1)), yy.min(img.h.saturating_sub(1)));
                    for k in 0..4 {
                        acc[k] += p[k];
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)).max(1) as f32;
            let a = acc[3] / n;
            for k in 0..3 {
                let c = if a > 1e-6 { acc[k] / n / a } else { 0.0 };
                let v = filmcraft_color::linear_to_srgb(if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 });
                row[x * 4 + k] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            row[x * 4 + 3] = 255;
        }
    });
    (out, ow, oh)
}

/// The depth of the picture `img` shows: the clip's raw frame when the effect knows it and it has
/// the picture's shape, else `img` itself. `None` without a model or when it cannot say.
pub fn depth_for(img: &Image, cx: &FxCtx) -> Option<Arc<Cached>> {
    let model = MODEL.get()?;
    let raw = cx.env.and_then(|env| {
        let (sw, sh) = env.source_size();
        let s = (ANALYSIS_SIDE as f32 / sw.max(sh).max(1) as f32).min(1.0);
        env.source_frame(0.0, s)
    });
    let same_shape = |p: &Image| p.w > 0 && p.h > 0 && img.h > 0 && ((p.w as f32 / p.h as f32) / (img.w as f32 / img.h as f32) - 1.0).abs() < 0.02;
    let (rgba, w, h) = match raw.filter(same_shape) {
        Some(p) => analysis_bytes(&p),
        None => analysis_bytes(img),
    };
    let key = picture_key(&rgba, w, h);
    let find = |key: u64| CACHE.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, c)| c.clone());
    if let Some(c) = find(key) {
        return Some(c);
    }
    let _making = MAKING.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(c) = find(key) {
        return Some(c);
    }
    let field = model.depth(key, &rgba, w, h)?;
    let field = DepthField::new(field.w, field.h, field.near)?;
    let held = field.held(((field.w as f32 * HOLD).round() as usize).max(1));
    let c = Arc::new(Cached { field, held });
    let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    if cache.len() >= CACHED {
        cache.remove(0);
    }
    cache.push((key, c.clone()));
    Some(c)
}

/// Apply a depth effect (`false` for other effects). Without depth the picture stays as it is.
pub fn apply(img: &mut Image, e: &EffectInstance, cx: &FxCtx) -> bool {
    match e.effect.as_str() {
        PHOTO_3D => {
            let p = Parallax::eval(e, cx, img.w, img.h);
            if p.is_identity() || img.w < 2 || img.h < 2 {
                return true;
            }
            let Some(d) = depth_for(img, cx) else { return true };
            *img = GPU.get().and_then(|g| g.parallax(img, &d.held, &p)).unwrap_or_else(|| parallax_cpu(img, &d.held, &p));
        }
        DEPTH_BLUR => {
            let op = BlurOp::eval(e, cx, img.w);
            let t = taps(op.radius);
            if t.is_empty() || img.w < 2 || img.h < 2 {
                return true;
            }
            let Some(d) = depth_for(img, cx) else { return true };
            let nr = near_and_radius(&d.field, &op, img.w, img.h);
            *img = GPU.get().and_then(|g| g.blur(img, &nr, &t)).unwrap_or_else(|| blur_cpu(img, &nr, &t));
        }
        _ => return false,
    }
    true
}

#[cfg(test)]
#[path = "depth_tests.rs"]
mod tests;
