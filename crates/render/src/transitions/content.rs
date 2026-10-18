//! Content-aware transitions (roadmap C3): what a transition learns from the two pictures it
//! joins, computed once per transition and cached.
//!
//! The flow transitions and the Content-Aware Wipe
//! ([`filmcraft_project::flow_transitions`]) read a [`Field`]: four floats per cell of a grid at
//! most [`FIELD_MAX`] cells across, laid over the frame. The timeline computes it from the two
//! clips at the transition's midpoint (the *anchors*: the outgoing clip just before it leaves, the
//! incoming one just after it arrives, in the order the transition plays) with [`analyse`], keeps
//! it in [`cached`] and hands the same field to the CPU reference and to the GPU (an
//! `Rgba32Float` texture in the shader stage's map binding), so both warp along the same flow.
//! A caller with no anchors ([`for_frames`]) analyses the two pictures it has.
//!
//! - **Flow** (`.xy` forward, `.zw` backward, frame pixels): the dense Lucas–Kanade flow both ways
//!   between the anchors ([`crate::flow::flow_pair`]), resampled onto the grid, weighted by how
//!   well each direction undoes the other (forward–backward consistency) and smoothed by
//!   normalised convolution (Flow Smoothness), with a coarse estimate filling what nothing
//!   trusted is near (content entering or leaving at the edges); limited to a third of the frame,
//!   and scaled down where it would bend faster than a warp can follow without folding, so shots
//!   that share nothing get a gentle, tear-free warp.
//! - **Seams** (`.x`: when the wipe reaches the cell, 0..1): seam carving (Avidan & Shamir 2007)
//!   on the gradient energy of both anchors. `K` vertical seams (in the wipe's direction), each
//!   8-connected, each at or past the one before it and within Seam Freedom of its place in a
//!   straight wipe, minimise the energy they cross (dynamic programming, as in the paper) plus a
//!   pull towards the straight line; the time a cell is reached is interpolated between the seams
//!   on its side. So each row is revealed from its leading edge as a prefix that only grows (the
//!   wipe covers the frame monotonically and its edge stays one connected line), and the edge
//!   bends round what stands out instead of slicing through it. With no protection, or pictures
//!   with nothing salient, the seams sit on their straight lines: a plain wipe.

use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use filmcraft_project::flow_transitions::{CONTENT_AWARE_WIPE, flow_preset};
use filmcraft_project::{EffectInstance, ParamKind, ParamValue};
use filmcraft_time::Tick;
use rayon::prelude::*;

use crate::image::Image;

/// Widest (or tallest) field grid, in cells.
pub const FIELD_MAX: usize = 320;
/// Fields kept by [`cached`].
const CACHE: usize = 12;
/// Longest flow vector, as a share of the frame's larger side.
const MAX_FLOW: f32 = 1.0 / 3.0;
/// Steepest change of the flow across a few cells (pixels of flow per pixel): a warp at full
/// strength (each picture moved at most all the way along) then never folds over itself.
const STEEPEST: f32 = 0.9;

/// What a content-aware transition reads: `w × h` cells of four floats, rows from the top, laid
/// evenly over the frame (see the module docs for what the floats mean).
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[f32; 4]>,
    /// Identifies the pictures and parameters it was computed from (the GPU stage caches its
    /// texture by it).
    pub key: u64,
}

impl Field {
    /// Cell (x, y), clamped to the grid.
    #[inline]
    pub fn at(&self, x: i64, y: i64) -> [f32; 4] {
        if self.w == 0 || self.h == 0 {
            return [0.0; 4];
        }
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        self.px.get(y * self.w + x).copied().unwrap_or([0.0; 4])
    }

    /// The host's `getField`: the field at frame position (`x`, `y`) (y down), bilinear between
    /// cell centres and clamped at the edges; `ratio` is cells per frame pixel in each direction
    /// (`Host::inv.zw`). Operation for operation the prelude's.
    #[inline]
    pub fn sample(&self, x: f32, y: f32, ratio: (f32, f32)) -> [f32; 4] {
        if self.w == 0 || self.h == 0 {
            return [0.0; 4];
        }
        let cx = (x * ratio.0 - 0.5).clamp(0.0, self.w as f32 - 1.0);
        let cy = (y * ratio.1 - 0.5).clamp(0.0, self.h as f32 - 1.0);
        let (cx, cy) = (if cx.is_finite() { cx } else { 0.0 }, if cy.is_finite() { cy } else { 0.0 });
        let (x0, y0) = (cx.floor() as i64, cy.floor() as i64);
        let (x1, y1) = ((x0 + 1).min(self.w as i64 - 1), (y0 + 1).min(self.h as i64 - 1));
        let (kx, ky) = (cx - x0 as f32, cy - y0 as f32);
        let (a, b, c, d) = (self.at(x0, y0), self.at(x1, y0), self.at(x0, y1), self.at(x1, y1));
        let mut o = [0.0; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * kx;
            let bot = c[k] + (d[k] - c[k]) * kx;
            o[k] = top + (bot - top) * ky;
        }
        o
    }
}

/// The field grid of a `w × h` frame: the frame divided by the smallest whole number that brings
/// both sides to at most [`FIELD_MAX`] cells.
pub fn field_size(w: usize, h: usize) -> (usize, usize) {
    let (w, h) = (w.max(1), h.max(1));
    let s = w.max(h).div_ceil(FIELD_MAX).max(1);
    (w.div_ceil(s).max(1), h.div_ceil(s).max(1))
}

/// Cells per frame pixel of the field of a `w × h` frame (the shaders' `Host::inv.zw`).
pub fn field_ratio(w: usize, h: usize) -> (f32, f32) {
    let (fw, fh) = field_size(w, h);
    (fw as f32 / w.max(1) as f32, fh as f32 / h.max(1) as f32)
}

/// Whether transition `id` reads a [`Field`].
pub fn reads_content(id: &str) -> bool {
    filmcraft_project::flow_transitions::reads_content(id)
}

/// A numeric parameter (instance value, else the definition's default), clamped to its range;
/// non-finite values give the default.
fn num(e: &EffectInstance, id: &str) -> f32 {
    let def = e.def().and_then(|d| d.param(id));
    let fallback = def.and_then(|p| p.default.as_f64()).filter(|v| v.is_finite()).unwrap_or(0.0);
    let v = e.f64_at(id, Tick::ZERO);
    let v = if v.is_finite() { v } else { fallback };
    let v = match def.map(|p| &p.kind) {
        Some(ParamKind::Float { min, max, .. }) if min <= max => v.clamp(*min, *max),
        _ => v.clamp(-1e6, 1e6),
    };
    v as f32
}

fn choice(e: &EffectInstance, id: &str) -> u32 {
    let v = e.param(id).map(|p| p.value.clone()).or_else(|| e.def().and_then(|d| d.param(id)).map(|p| p.default.clone()));
    match v {
        Some(ParamValue::Choice(c)) => c,
        _ => 0,
    }
}

/// What a transition computes from its pictures.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Analysis {
    /// Flow both ways, smoothed by `smooth` (0..1).
    Flow { smooth: f32 },
    /// Seams travelling in `dir` (the `direction` choice), avoiding energy by `protect` (0..1)
    /// within `freedom` (0..1).
    Seams { dir: u32, protect: f32, freedom: f32 },
}

fn analysis(e: &EffectInstance) -> Option<Analysis> {
    if flow_preset(&e.effect).is_some() {
        return Some(Analysis::Flow { smooth: num(e, "smoothness") / 100.0 });
    }
    (e.effect == CONTENT_AWARE_WIPE).then(|| Analysis::Seams {
        dir: choice(e, "direction").min(3),
        protect: num(e, "protect") / 100.0,
        freedom: num(e, "freedom") / 100.0,
    })
}

/// A hash of what transition `e`'s field depends on besides its pictures (its kind and analysis
/// parameters). None when `e` reads no field.
pub fn params_key(e: &EffectInstance) -> Option<u64> {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    match analysis(e)? {
        Analysis::Flow { smooth } => (0u8, smooth.to_bits()).hash(&mut s),
        Analysis::Seams { dir, protect, freedom } => (1u8, dir, protect.to_bits(), freedom.to_bits()).hash(&mut s),
    }
    Some(s.finish())
}

/// The field of transition `e` between anchors `a` (outgoing) and `b` (incoming), computed now.
/// None when `e` reads no field, or the anchors are empty, of different sizes or broken.
pub fn analyse(e: &EffectInstance, a: &Image, b: &Image) -> Option<Field> {
    let n = a.w.checked_mul(a.h)?.checked_mul(4)?;
    if (a.w, a.h) != (b.w, b.h) || a.w == 0 || a.h == 0 || a.px.len() != n || b.px.len() != n {
        return None;
    }
    let mut f = match analysis(e)? {
        Analysis::Flow { smooth } => flow_field(a, b, smooth)?,
        Analysis::Seams { dir, protect, freedom } => seam_field(a, b, dir, protect, freedom),
    };
    // nothing non-finite reaches a shader
    for c in &mut f.px {
        for v in c.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }
    Some(f)
}

/// The field cached under `key`, computed by `compute` the first time it is asked for (a
/// computation that panics counts as no field, and no field is remembered too). The last
/// [`CACHE`] keys are kept.
///
/// The computation runs outside the lock and nobody waits for anybody else's: a rayon worker
/// waiting here could be the very thread computing that field further up its own stack (an
/// export renders frames as rayon tasks, and the flow's own parallel loops steal them). Threads
/// that miss together each compute the field, which is deterministic, and the first one stored
/// is the one everybody gets.
pub fn cached(key: u64, compute: impl FnOnce() -> Option<Field>) -> Option<Arc<Field>> {
    static FIELDS: OnceLock<Mutex<Vec<(u64, Option<Arc<Field>>)>>> = OnceLock::new();
    let fields = FIELDS.get_or_init(Default::default);
    let lookup = |m: &mut Vec<(u64, Option<Arc<Field>>)>| {
        let i = m.iter().position(|(k, _)| *k == key)?;
        let hit = m.remove(i);
        let field = hit.1.clone();
        m.push(hit);
        Some(field)
    };
    if let Some(hit) = lookup(&mut fields.lock().unwrap_or_else(PoisonError::into_inner)) {
        return hit;
    }
    let field = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(compute)) {
        Ok(f) => f.map(|f| Arc::new(Field { key, ..f })),
        // the transition then renders without its analysis (a straight wipe, a dissolve)
        Err(_) => None,
    };
    let mut m = fields.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(hit) = lookup(&mut m) {
        return hit;
    }
    m.push((key, field.clone()));
    if m.len() > CACHE {
        m.remove(0);
    }
    field
}

/// A fingerprint of an image: its size and every value (rows hashed in parallel, combined in
/// order).
fn fingerprint(img: &Image, s: &mut impl Hasher) {
    (img.w, img.h, img.px.len()).hash(s);
    let row = (img.w * 4).max(1);
    let rows: Vec<u64> = img
        .px
        .par_chunks(row)
        .map(|r| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for v in r {
                h = (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        })
        .collect();
    rows.hash(s);
}

/// The field of transition `e` with the two pictures it is given as its anchors (no timeline:
/// a still pair, a preview), cached by their content and `e`'s analysis parameters. None when
/// `e` reads no field.
pub fn for_frames(e: &EffectInstance, a: &Image, b: &Image) -> Option<Arc<Field>> {
    let pk = params_key(e)?;
    let mut s = std::collections::hash_map::DefaultHasher::new();
    ("frames", pk).hash(&mut s);
    fingerprint(a, &mut s);
    fingerprint(b, &mut s);
    cached(s.finish(), || analyse(e, a, b))
}

// ---------------------------------------------------------------------------------------------
// Flow

/// Separable Gaussian blur (σ in cells) of a 4-channel grid, clamped at the edges.
fn blur4(px: &mut [[f32; 4]], w: usize, h: usize, sigma: f32) {
    if sigma.is_nan() || sigma <= 0.3 || w == 0 || h == 0 || px.len() != w * h {
        return;
    }
    let r = ((sigma * 3.0).ceil() as usize).clamp(1, 96);
    let kernel: Vec<f32> = (0..=r).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let norm = kernel[0] + 2.0 * kernel[1..].iter().sum::<f32>();
    let pass = |src: &[[f32; 4]], horizontal: bool| -> Vec<[f32; 4]> {
        (0..w * h)
            .into_par_iter()
            .map(|i| {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let mut acc = [0.0f32; 4];
                for (j, &k) in kernel.iter().enumerate() {
                    for sgn in if j == 0 { [1i64, 0] } else { [1, -1] } {
                        if sgn == 0 {
                            continue;
                        }
                        let o = j as i64 * sgn;
                        let (sx, sy) = if horizontal { ((x + o).clamp(0, w as i64 - 1), y) } else { (x, (y + o).clamp(0, h as i64 - 1)) };
                        let v = src.get(sy as usize * w + sx as usize).copied().unwrap_or([0.0; 4]);
                        for c in 0..4 {
                            acc[c] += v[c] * k;
                        }
                    }
                }
                acc.map(|v| v / norm)
            })
            .collect()
    };
    let tmp = pass(px, true);
    let out = pass(&tmp, false);
    px.copy_from_slice(&out);
}

fn flow_field(a: &Image, b: &Image, smooth: f32) -> Option<Field> {
    let pair = crate::flow::flow_pair(a, b)?;
    let (fw, fh) = field_size(a.w, a.h);
    let (sx, sy) = (a.w as f32 / fw as f32, a.h as f32 / fh as f32);
    // forward–backward consistency: a vector the other direction's flow undoes is trusted, one it
    // does not (content leaving the frame, shots that share nothing) is filled in from the
    // trusted ones around it (normalised convolution)
    let tau = (0.004 * a.w.max(a.h) as f32).max(1.0);
    let trust = |e: f32| {
        let q = e / tau;
        let t = (-q * q).exp();
        if t.is_finite() { t } else { 0.0 }
    };
    let cells: Vec<([f32; 4], [f32; 4])> = (0..fw * fh)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (((i % fw) as f32 + 0.5) * sx, ((i / fw) as f32 + 0.5) * sy);
            let (f, g) = pair.at(x, y);
            let back = pair.at(x + f[0], y + f[1]).1;
            let fore = pair.at(x + g[0], y + g[1]).0;
            let (tf, tg) = (trust((f[0] + back[0]).hypot(f[1] + back[1])), trust((g[0] + fore[0]).hypot(g[1] + fore[1])));
            ([f[0] * tf, f[1] * tf, g[0] * tg, g[1] * tg], [tf, tf, tg, tg])
        })
        .collect();
    let (mut px, mut weight): (Vec<[f32; 4]>, Vec<[f32; 4]>) = cells.into_iter().unzip();
    let smooth = if smooth.is_finite() { smooth.clamp(0.0, 1.0) } else { 0.3 };
    let sigma = (smooth * fw.min(fh) as f32 * 0.08).max(1.0);
    // a coarse estimate from further away fills what nothing trusted is near (content entering
    // or leaving at the edges): the fine one where it is trusted, the coarse one where not
    let (mut far_px, mut far_w) = (px.clone(), weight.clone());
    let far = (fw.min(fh) as f32 * 0.25).max(sigma * 2.0);
    blur4(&mut far_px, fw, fh, far);
    blur4(&mut far_w, fw, fh, far);
    blur4(&mut px, fw, fh, sigma);
    blur4(&mut weight, fw, fh, sigma);
    const COARSE: f32 = 0.1;
    for ((c, w), (fc, fw_)) in px.iter_mut().zip(&weight).zip(far_px.iter().zip(&far_w)) {
        for k in 0..4 {
            // (with nothing trusted anywhere the flow fades to none)
            c[k] = (c[k] + COARSE * fc[k]) / (w[k] + COARSE * fw_[k]).max(0.02);
        }
    }
    let max = MAX_FLOW * a.w.max(a.h) as f32;
    for c in &mut px {
        for v in [0usize, 2] {
            let (x, y) = (c[v], c[v + 1]);
            let len = (x * x + y * y).sqrt();
            if !len.is_finite() {
                c[v] = 0.0;
                c[v + 1] = 0.0;
            } else if len > max {
                c[v] = x * max / len;
                c[v + 1] = y * max / len;
            }
        }
    }
    // a warp folds over itself (and looks torn) where the flow changes faster than the picture
    // it moves: keep its steepest change under STEEPEST pixels per pixel, scaling the whole
    // field down if need be (what two shots that share nothing get: a gentle warp)
    let steep = steepest(&px, fw, fh, sx, sy);
    if steep > STEEPEST {
        let k = STEEPEST / steep;
        px.iter_mut().flatten().for_each(|v| *v *= k);
    }
    Some(Field { w: fw, h: fh, px, key: 0 })
}

/// The flow's steepest change (frame pixels of flow per frame pixel) across a span of a few
/// cells (`sx`, `sy` frame pixels each), away from the frame's edges: what bends the picture
/// visibly. A one-cell wobble of the estimate is a fraction of a pixel and counts for little,
/// and at the edges, where content enters and leaves, the flow is a guess either way.
fn steepest(px: &[[f32; 4]], w: usize, h: usize, sx: f32, sy: f32) -> f32 {
    let span = ((w.min(h) as f32 * 0.04).round() as usize).max(2);
    let margin = ((w.min(h) as f32 * 0.08).round() as usize).max(span);
    // (the estimate's own noise wobbles the picture by a fraction of a pixel: measured on the
    // flow smoothed over the span)
    let mut smooth = px.to_vec();
    blur4(&mut smooth, w, h, span as f32);
    let at = |x: usize, y: usize| smooth.get(y * w + x).copied().unwrap_or([0.0; 4]);
    let mut worst = 0.0f32;
    for y in margin.min(h)..h.saturating_sub(margin) {
        for x in margin.min(w)..w.saturating_sub(margin) {
            let c = at(x, y);
            for (n, d) in [((x + span < w).then(|| at(x + span, y)), sx), ((y + span < h).then(|| at(x, y + span)), sy)] {
                let Some(n) = n else { continue };
                for k in 0..4 {
                    let g = (n[k] - c[k]).abs() / (d * span as f32).max(1e-6);
                    if g.is_finite() {
                        worst = worst.max(g);
                    }
                }
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------------------------
// Seams

/// Luma of `img` box-reduced onto a `w × h` grid (each cell the mean of the frame pixels whose
/// centres fall in it).
fn grid_luma(img: &Image, w: usize, h: usize) -> Vec<f32> {
    let (sx, sy) = (img.w as f32 / w as f32, img.h as f32 / h as f32);
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (gx, gy) = (i % w, i / w);
            let (x0, x1) = (((gx as f32 * sx) as usize).min(img.w - 1), (((gx + 1) as f32 * sx).ceil() as usize).clamp(1, img.w));
            let (y0, y1) = (((gy as f32 * sy) as usize).min(img.h - 1), (((gy + 1) as f32 * sy).ceil() as usize).clamp(1, img.h));
            let (mut sum, mut n) = (0.0f32, 0u32);
            for y in y0..y1.max(y0 + 1) {
                for x in x0..x1.max(x0 + 1) {
                    let p = img.get(x.min(img.w - 1), y.min(img.h - 1));
                    let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                    if l.is_finite() {
                        // square root lifts the shadows, as the flow's luma does
                        sum += l.clamp(0.0, 16.0).sqrt();
                        n += 1;
                    }
                }
            }
            if n > 0 { sum / n as f32 } else { 0.0 }
        })
        .collect()
}

/// Gradient energy of the two anchors on the field grid (the larger of the two at each cell),
/// spread over a few cells so the inside of an object counts as well as its outline, scaled so
/// its 98th percentile is 1. All zeros when neither picture shows anything salient.
fn energy(a: &Image, b: &Image, w: usize, h: usize) -> Vec<f32> {
    let (la, lb) = (grid_luma(a, w, h), grid_luma(b, w, h));
    let at = |g: &[f32], x: i64, y: i64| g.get(y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize).copied().unwrap_or(0.0);
    let grad = |g: &[f32], x: i64, y: i64| {
        let gx = (at(g, x + 1, y) - at(g, x - 1, y)) * 0.5;
        let gy = (at(g, x, y + 1) - at(g, x, y - 1)) * 0.5;
        (gx * gx + gy * gy).sqrt()
    };
    let raw: Vec<[f32; 4]> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            let e = grad(&la, x, y).max(grad(&lb, x, y));
            [if e.is_finite() { e } else { 0.0 }, 0.0, 0.0, 0.0]
        })
        .collect();
    let mut spread = raw;
    blur4(&mut spread, w, h, (w.min(h) as f32 * 0.02).max(1.0));
    let mut e: Vec<f32> = spread.iter().map(|c| c[0]).collect();
    let mut sorted = e.clone();
    sorted.sort_by(f32::total_cmp);
    let p98 = sorted.get(sorted.len() * 98 / 100).copied().unwrap_or(0.0);
    // below this the pictures are flat: noise, not subjects
    if p98.is_nan() || p98 <= 0.004 {
        return vec![0.0; w * h];
    }
    for v in &mut e {
        *v = (*v / p98).clamp(0.0, 1.0);
    }
    e
}

/// Number of seams of a wipe `len` cells long.
fn seam_count(len: usize) -> usize {
    (len / 4).clamp(2, 64)
}

/// The seams of a wipe across a `cw × ch` grid (travel along +x) through `energy` (row-major in
/// that grid): `K + 1` boundaries, each a column position per row; the first is the leading edge
/// (−½) and the last the trailing one (`cw − ½`). Each interior seam is 8-connected, at or right
/// of the one before it, and within `freedom` of its straight-wipe column.
fn seams(energy: &[f32], cw: usize, ch: usize, protect: f32, freedom: f32) -> Vec<Vec<f32>> {
    let k_n = seam_count(cw);
    let band = (freedom.clamp(0.0, 1.0) * 0.15 * cw as f32).max(0.0);
    // the pull towards the straight line: small, so a seam bends for anything salient but
    // stays put over flat ground
    const PULL: f32 = 0.02;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(k_n + 1);
    out.push(vec![-0.5; ch]);
    let mut prev: Vec<i64> = vec![i64::MIN; ch];
    let e_at = |x: i64, y: usize| energy.get(y * cw + x as usize).copied().unwrap_or(0.0);
    for k in 1..k_n {
        let nominal = k as f32 * cw as f32 / k_n as f32 - 0.5;
        let lo = ((nominal - band).ceil() as i64).clamp(0, cw as i64 - 1);
        let hi = ((nominal + band).floor() as i64).clamp(lo, cw as i64 - 1);
        let left = |y: usize| prev.get(y).copied().unwrap_or(i64::MIN).max(lo).min(hi);
        let width = (hi - lo + 1) as usize;
        let cost = |x: i64, y: usize| {
            let dev = (x as f32 - nominal) / band.max(1.0);
            protect * e_at(x, y) + PULL * dev * dev
        };
        // dynamic programming down the rows (columns lo..=hi, from `left(y)`)
        let mut acc = vec![f32::INFINITY; width];
        let mut back = vec![0i8; width * ch];
        for x in left(0)..=hi {
            acc[(x - lo) as usize] = cost(x, 0);
        }
        for y in 1..ch {
            let mut next = vec![f32::INFINITY; width];
            for x in left(y)..=hi {
                let mut best = (f32::INFINITY, 0i8);
                // prefer straight down, then left, then right
                for d in [0i8, -1, 1] {
                    let px = x + d as i64;
                    if px < left(y - 1) || px > hi {
                        continue;
                    }
                    let v = acc[(px - lo) as usize];
                    if v < best.0 {
                        best = (v, d);
                    }
                }
                let i = (x - lo) as usize;
                next[i] = best.0 + cost(x, y);
                back[y * width + i] = best.1;
            }
            acc = next;
        }
        // the cheapest end, nearest its straight column on a tie
        let mut end = left(ch - 1);
        for x in left(ch - 1)..=hi {
            let (v, e) = (acc[(x - lo) as usize], acc[(end - lo) as usize]);
            if v < e || (v == e && (x as f32 - nominal).abs() < (end as f32 - nominal).abs()) {
                end = x;
            }
        }
        let mut seam = vec![0i64; ch];
        if acc.get((end - lo) as usize).is_some_and(|v| v.is_finite()) {
            let mut x = end;
            for y in (0..ch).rev() {
                seam[y] = x;
                if y > 0 {
                    let d = back.get(y * width + (x - lo) as usize).copied().unwrap_or(0);
                    x = (x + d as i64).clamp(lo, hi);
                }
            }
        } else {
            // (unreachable by construction) hug the seam before: connected and in order
            for (y, s) in seam.iter_mut().enumerate() {
                *s = left(y);
            }
        }
        out.push(seam.iter().map(|&x| x as f32).collect());
        prev = seam;
    }
    out.push(vec![cw as f32 - 0.5; ch]);
    out
}

/// When each cell of row `v` is reached (0..1), interpolated between the seams on either side.
fn arrival(seams: &[Vec<f32>], cw: usize, v: usize) -> Vec<f32> {
    let k_n = seams.len().saturating_sub(1).max(1);
    let pos = |k: usize| seams.get(k).and_then(|s| s.get(v)).copied().unwrap_or(cw as f32 - 0.5);
    let mut k = 0usize;
    (0..cw)
        .map(|u| {
            let u = u as f32;
            while k + 1 < k_n && u >= pos(k + 1) {
                k += 1;
            }
            let (s0, s1) = (pos(k), pos(k + 1));
            let f = if s1 > s0 { ((u - s0) / (s1 - s0)).clamp(0.0, 1.0) } else { 0.0 };
            ((k as f32 + f) / k_n as f32).clamp(0.0, 1.0)
        })
        .collect()
}

/// Map canonical (u along the wipe, v across it) to field (x, y) for direction `dir`.
fn oriented(dir: u32, fw: usize, fh: usize, u: usize, v: usize) -> (usize, usize) {
    match dir {
        // From North: travels down
        0 => (v, u),
        // From East: travels left
        1 => (fw.saturating_sub(1).saturating_sub(u), v),
        // From South: travels up
        2 => (v, fh.saturating_sub(1).saturating_sub(u)),
        // From West: travels right
        _ => (u, v),
    }
}

fn seam_field(a: &Image, b: &Image, dir: u32, protect: f32, freedom: f32) -> Field {
    let (fw, fh) = field_size(a.w, a.h);
    let protect = if protect.is_finite() { protect.clamp(0.0, 1.0) } else { 0.0 };
    let freedom = if freedom.is_finite() { freedom.clamp(0.0, 1.0) } else { 0.0 };
    let e = if protect > 0.0 && freedom > 0.0 { energy(a, b, fw, fh) } else { vec![0.0; fw * fh] };
    let vertical = matches!(dir, 0 | 2);
    let (cw, ch) = if vertical { (fh, fw) } else { (fw, fh) };
    // the energy in the canonical grid
    let mut ce = vec![0.0f32; cw * ch];
    for v in 0..ch {
        for u in 0..cw {
            let (x, y) = oriented(dir, fw, fh, u, v);
            ce[v * cw + u] = e.get(y * fw + x).copied().unwrap_or(0.0);
        }
    }
    let s = seams(&ce, cw, ch, protect, freedom);
    let mut px = vec![[0.0f32; 4]; fw * fh];
    for v in 0..ch {
        for (u, t) in arrival(&s, cw, v).into_iter().enumerate() {
            let (x, y) = oriented(dir, fw, fh, u, v);
            if let Some(c) = px.get_mut(y * fw + x) {
                c[0] = t;
            }
        }
    }
    Field { w: fw, h: fh, px, key: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_grids_fit_the_frame() {
        assert_eq!(field_size(1920, 1080), (320, 180));
        assert_eq!(field_size(128, 72), (128, 72));
        assert_eq!(field_size(1080, 1920), (180, 320));
        assert_eq!(field_size(0, 0), (1, 1));
        assert_eq!(field_size(1, 100_000), (1, 320));
        let (rx, ry) = field_ratio(1920, 1080);
        assert_eq!((rx, ry), (320.0 / 1920.0, 180.0 / 1080.0));
    }

    #[test]
    fn seams_are_connected_ordered_and_bounded() {
        // a bright block in the middle of flat ground
        let (cw, ch) = (64usize, 40usize);
        let e: Vec<f32> = (0..cw * ch).map(|i| if (24..40).contains(&(i % cw)) && (12..28).contains(&(i / cw)) { 1.0 } else { 0.0 }).collect();
        let s = seams(&e, cw, ch, 1.0, 1.0);
        assert_eq!(s.len(), seam_count(cw) + 1);
        for k in 1..s.len() {
            for y in 0..ch {
                assert!(s[k][y] >= s[k - 1][y], "seam {k} row {y} behind the one before");
                if y > 0 && k < s.len() - 1 {
                    assert!((s[k][y] - s[k][y - 1]).abs() <= 1.0, "seam {k} breaks at row {y}");
                }
            }
        }
        // no seam crosses the block's inside while it could go round
        let inside = |k: usize| (12..28).filter(|&y| (25.0..39.0).contains(&s[k][y])).count();
        assert!((1..s.len() - 1).all(|k| inside(k) == 0), "{:?}", (1..s.len() - 1).map(inside).collect::<Vec<_>>());
        // flat ground: straight seams
        let flat = seams(&vec![0.0; cw * ch], cw, ch, 1.0, 1.0);
        for seam in &flat[1..flat.len() - 1] {
            assert!(seam.iter().all(|&x| x == seam[0]));
        }
    }

    #[test]
    fn hostile_anchors_give_no_field_or_a_sane_one() {
        let e = filmcraft_project::find_effect("flow_morph").unwrap().instance();
        let w = filmcraft_project::find_effect(CONTENT_AWARE_WIPE).unwrap().instance();
        let a = Image::filled(8, 6, [0.5, 0.5, 0.5, 1.0]);
        assert!(analyse(&e, &a, &Image::new(4, 6)).is_none());
        assert!(analyse(&e, &Image::new(0, 0), &Image::new(0, 0)).is_none());
        assert!(analyse(&e, &Image { w: 8, h: 6, px: vec![0.0; 3] }, &a).is_none());
        let cross = filmcraft_project::find_effect("cross_dissolve").unwrap().instance();
        assert!(analyse(&cross, &a, &a).is_none() && params_key(&cross).is_none());
        let mut nan = a.clone();
        nan.px[5] = f32::NAN;
        nan.px[9] = f32::INFINITY;
        for t in [&e, &w] {
            let f = analyse(t, &nan, &a).unwrap();
            assert!(f.px.iter().flatten().all(|v| v.is_finite()), "{}", t.effect);
            assert_eq!((f.w, f.h), field_size(8, 6));
        }
        for (fw, fh) in [(1, 1), (1, 7), (7, 1), (2, 2)] {
            let p = Image::filled(fw, fh, [0.2, 0.3, 0.4, 1.0]);
            assert!(analyse(&w, &p, &p).is_some());
            assert!(analyse(&e, &p, &p).is_some());
        }
    }

    #[test]
    fn the_cache_computes_once_per_key() {
        let mut calls = 0;
        let key = 0x00c3_f10e_u64;
        for _ in 0..3 {
            let f = cached(key, || {
                calls += 1;
                Some(Field { w: 1, h: 1, px: vec![[1.0; 4]], key: 0 })
            });
            assert_eq!(f.map(|f| (f.key, f.px[0][0])), Some((key, 1.0)));
        }
        assert_eq!(calls, 1);
        // a panicking analysis is no field, not a crash
        assert!(cached(key ^ 1, || panic!("broken analysis")).is_none());
    }
}
