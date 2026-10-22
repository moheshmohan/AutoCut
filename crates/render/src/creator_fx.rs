//! The Creator effects (`filmcraft_project::effect::CREATOR_EFFECTS`): evaluated parameters and
//! the CPU reference of each, which the GPU stage (`filmcraft-gpu`, `fx.wgsl`) reproduces pixel
//! for pixel.
//!
//! As for the other GPU effects ([`crate::gpufx`]), [`eval`] turns an effect instance at a time
//! into plain numbers — every one finite and inside its range, whatever the project says (NaN,
//! huge or negative values fall back to the default or the nearest limit) — and
//! [`CreatorOp::apply`] is the reference arithmetic. Noise is an integer hash ([`pcg`], 32-bit, the
//! same in WGSL) of the pixel, the frame time and the Random Seed, so a frame renders the same on
//! every run, at every resolution of the same size and on either device. An amount of zero is a
//! no-op (the picture is returned untouched, super-whites and all).

use filmcraft_color::{linear_to_srgb, srgb_to_linear};
use filmcraft_project::EffectInstance;
use rayon::prelude::*;

use crate::effects::FxCtx;
use crate::image::Image;
use crate::vfx::{bv, chv, cv, fv, pv};

/// One evaluated Creator effect.
#[derive(Clone, Debug, PartialEq)]
pub enum CreatorOp {
    /// Luma-weighted value-noise grain on display-encoded colour.
    FilmGrain { amount: f32, size: f32, response: f32, color: bool, seed: u32 },
    /// Tape look: wobble, a rolling tracking band, chroma smeared along the line, noise.
    Vhs { amount: f32, bleed: f32, tracking: f32, noise: f32, wobble: f32, sat: f32, phase: f32, band_y: f32, band_h: f32, frame: u32, seed: u32 },
    /// Mirror segments around a centre (`rot` radians, `zoom` > 0).
    Kaleidoscope { amount: f32, segments: f32, rot: f32, cx: f32, cy: f32, zoom: f32 },
    /// Block displacement, scanline jitter and an RGB split; `amount` already holds the rhythm.
    Glitch { amount: f32, block: f32, split: f32, jitter: f32, step: u32, seed: u32 },
    /// `steps` taps towards the centre (zoom) or around it (spin).
    ZoomBlur { amount: f32, cx: f32, cy: f32, spin: bool, steps: u32 },
    /// `steps` weighted taps along (dx, dy), centred or trailing.
    MotionBlur { dx: f32, dy: f32, steps: u32, trail: bool, falloff: f32 },
    /// Scanlines, flicker, vignette, a grid and corner brackets in `color` (display-encoded).
    TechHud {
        amount: f32,
        color: [f32; 3],
        scan: f32,
        spacing: f32,
        grid: f32,
        grid_size: f32,
        brackets: f32,
        vignette: f32,
        tint: f32,
        flicker: f32,
        line: f32,
    },
    /// Red and blue sampled apart: radially by `k` of the distance, or by (dx, dy).
    Chromatic { radial: bool, k: f32, dx: f32, dy: f32, cx: f32, cy: f32 },
    /// A soft band of light (linear `color`) across direction (nx, ny) at `s0` from the centre.
    LightSweep { intensity: f32, color: [f32; 3], nx: f32, ny: f32, s0: f32, half: f32, inner: f32 },
    /// Luma mapped from the shadow to the highlight colour (display-encoded).
    Duotone { amount: f32, shadows: [f32; 3], highlights: [f32; 3], contrast: f32 },
    /// Dots on a grid rotated by (cs, sn), sized by the picture under each cell's centre.
    Halftone { amount: f32, size: f32, cs: f32, sn: f32, color: bool, ink: [f32; 3], paper: [f32; 3] },
}

// ------------------------------------------------------------------ shared arithmetic (= fx.wgsl)

/// PCG-style 32-bit integer hash (`fx.wgsl` `pcg`).
#[inline]
pub fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

#[inline]
pub fn hash3(x: u32, y: u32, z: u32) -> u32 {
    pcg(x ^ pcg(y ^ pcg(z)))
}

/// Uniform in [0, 1) from three integers, exact in f32 (24 bits).
#[inline]
pub fn h01(x: u32, y: u32, z: u32) -> f32 {
    (hash3(x, y, z) >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `bitcast<u32>(i32(floor(v)))`.
#[inline]
fn ifloor(v: f32) -> u32 {
    (v.floor() as i32) as u32
}

#[inline]
fn fract(v: f32) -> f32 {
    v - v.floor()
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn enc(c: [f32; 3]) -> [f32; 3] {
    [linear_to_srgb(c[0].max(0.0)), linear_to_srgb(c[1].max(0.0)), linear_to_srgb(c[2].max(0.0))]
}

#[inline]
fn dec(c: [f32; 3]) -> [f32; 3] {
    [srgb_to_linear(c[0].clamp(0.0, 1.0)), srgb_to_linear(c[1].clamp(0.0, 1.0)), srgb_to_linear(c[2].clamp(0.0, 1.0))]
}

#[inline]
fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[inline]
fn straight(p: [f32; 4]) -> [f32; 3] {
    if p[3] <= 1e-6 { [0.0; 3] } else { [p[0] / p[3], p[1] / p[3], p[2] / p[3]] }
}

#[inline]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

#[inline]
fn premul(c: [f32; 3], a: f32) -> [f32; 4] {
    [c[0] * a, c[1] * a, c[2] * a, a]
}

/// Bilinear value noise in [0, 1) at `(qx, qy)` lattice units (`fx.wgsl` `value_noise`).
#[inline]
fn value_noise(qx: f32, qy: f32, seed: u32) -> f32 {
    let (fx0, fy0) = (qx.floor(), qy.floor());
    let (ix, iy) = (ifloor(qx), ifloor(qy));
    let (fx, fy) = (qx - fx0, qy - fy0);
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let a = h01(ix, iy, seed);
    let b = h01(ix.wrapping_add(1), iy, seed);
    let c = h01(ix, iy.wrapping_add(1), seed);
    let d = h01(ix.wrapping_add(1), iy.wrapping_add(1), seed);
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    top + (bot - top) * sy
}

/// Seed channel offsets of colour grain.
const GRAIN_CHANNEL: [u32; 3] = [0, 0x9E37_79B9, 0x7F4A_7C15];

// ------------------------------------------------------------------ evaluation

/// `v` if finite, clamped to `lo..=hi`; else `def`.
#[inline]
fn num(v: f32, lo: f32, hi: f32, def: f32) -> f32 {
    if v.is_finite() { v.clamp(lo, hi) } else { def }
}

/// A hash of the exact frame time: animated noise changes every frame, at any frame rate.
fn frame_hash(cx: &FxCtx) -> u32 {
    let t = cx.t.0 as u64;
    pcg((t as u32) ^ pcg((t >> 32) as u32))
}

fn colour3(c: [f32; 4], def: [f32; 3]) -> [f32; 3] {
    let mut o = def;
    for (k, v) in o.iter_mut().enumerate() {
        if let Some(x) = c.get(k).filter(|x| x.is_finite()) {
            *v = x.clamp(0.0, 1.0);
        }
    }
    o
}

/// Clip seconds as a bounded, finite phase (animation repeats after 1000 s, keeping f32 precise).
fn phase(cx: &FxCtx) -> f32 {
    if cx.seconds.is_finite() { cx.seconds.rem_euclid(1000.0) as f32 } else { 0.0 }
}

/// Evaluate Creator effect `e` at `cx.t` for a `w`×`h` working image (None: not a Creator effect).
pub fn eval(e: &EffectInstance, cx: &FxCtx, w: usize, h: usize) -> Option<CreatorOp> {
    let ps = if cx.px_scale.is_finite() && cx.px_scale > 0.0 { cx.px_scale.min(64.0) } else { 1.0 };
    let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
    let pct = |k: &str, def: f32| num(fv(e, k, cx) / 100.0, 0.0, 1.0, def);
    let seed = || num(fv(e, "seed", cx), 0.0, 10_000.0, 0.0) as u32;
    let size_img = Image { w, h, px: Vec::new() };
    let centre = |k: &str| {
        let p = pv(e, k, cx, &size_img);
        let x = if p.x.is_finite() { p.x.clamp(-1e6, 1e6) as f32 } else { wf * 0.5 };
        let y = if p.y.is_finite() { p.y.clamp(-1e6, 1e6) as f32 } else { hf * 0.5 };
        (x, y)
    };
    let radians = |k: &str| {
        let d = fv(e, k, cx);
        if d.is_finite() { (d.rem_euclid(360.0)).to_radians() } else { 0.0 }
    };
    Some(match e.effect.as_str() {
        "cr_film_grain" => {
            let base = seed();
            CreatorOp::FilmGrain {
                amount: pct("amount", 0.35),
                size: (num(fv(e, "size", cx), 0.5, 20.0, 1.5) * ps).max(0.25),
                response: pct("response", 0.7),
                color: bv(e, "color"),
                seed: if bv(e, "animate") { pcg(base ^ frame_hash(cx)) } else { pcg(base) },
            }
        }
        "cr_vhs" => {
            let s = seed();
            let ph = phase(cx);
            let start = h01(s, 17, 0);
            CreatorOp::Vhs {
                amount: pct("amount", 0.7),
                bleed: num(fv(e, "bleed", cx), 0.0, 200.0, 6.0) * ps,
                tracking: pct("tracking", 0.4),
                noise: pct("noise", 0.25),
                wobble: num(fv(e, "wobble", cx), 0.0, 100.0, 3.0) * ps,
                sat: num(fv(e, "saturation", cx) / 100.0, 0.0, 2.0, 0.8),
                phase: ph,
                band_y: (fract(ph * 0.17 + start) * 1.3 - 0.15) * hf,
                band_h: (hf * 0.05).max(1.0),
                frame: frame_hash(cx),
                seed: s,
            }
        }
        "cr_kaleidoscope" => {
            let (x, y) = centre("center");
            CreatorOp::Kaleidoscope {
                amount: pct("amount", 1.0),
                segments: num(fv(e, "segments", cx), 2.0, 64.0, 6.0).round(),
                rot: radians("rotation"),
                cx: x,
                cy: y,
                zoom: num(fv(e, "zoom", cx) / 100.0, 0.1, 10.0, 1.0),
            }
        }
        "cr_glitch" => {
            let s = seed();
            let rate = num(fv(e, "rate", cx), 0.0, 60.0, 12.0);
            let secs = if cx.seconds.is_finite() { cx.seconds.clamp(0.0, 1e7) } else { 0.0 };
            let step = if rate > 0.0 { (secs * rate as f64).floor() as u32 } else { 0 };
            let envelope = if chv(e, "rhythm") == 1 {
                let r = h01(step, 0xB5, s);
                if r < 0.4 {
                    1.0
                } else if r < 0.65 {
                    0.35
                } else {
                    0.0
                }
            } else {
                1.0
            };
            CreatorOp::Glitch {
                amount: pct("amount", 0.5) * envelope,
                block: (num(fv(e, "block", cx), 2.0, 2000.0, 48.0) * ps).max(1.0),
                split: num(fv(e, "split", cx), 0.0, 500.0, 12.0) * ps,
                jitter: pct("jitter", 0.4),
                step,
                seed: s,
            }
        }
        "cr_zoom_blur" => {
            let (x, y) = centre("center");
            let amount = pct("amount", 0.25);
            let spin = chv(e, "kind") == 1;
            // farthest corner from the centre
            let far = [(0.0, 0.0), (wf, 0.0), (0.0, hf), (wf, hf)].iter().map(|(px, py)| ((px - x).powi(2) + (py - y).powi(2)).sqrt()).fold(0.0f32, f32::max);
            let far = if far.is_finite() { far.min(1e5) } else { 0.0 };
            let travel = if spin { far * amount * std::f32::consts::FRAC_PI_6 } else { far * amount * 0.5 };
            CreatorOp::ZoomBlur { amount, cx: x, cy: y, spin, steps: ((travel / 1.5).ceil() as u32).clamp(2, 64) }
        }
        "cr_motion_blur" => {
            let len = num(fv(e, "length", cx), 0.0, 1000.0, 40.0) * ps;
            let dir = radians("direction");
            CreatorOp::MotionBlur {
                dx: dir.sin() * len,
                dy: -dir.cos() * len,
                steps: (len.ceil() as u32).clamp(2, 64),
                trail: chv(e, "shutter") == 1,
                falloff: pct("falloff", 0.0),
            }
        }
        "cr_tech_hud" => CreatorOp::TechHud {
            amount: pct("amount", 1.0),
            color: colour3(cv(e, "color", cx), [0.25, 0.85, 1.0]),
            scan: pct("scanlines", 0.35),
            spacing: (num(fv(e, "spacing", cx), 1.0, 64.0, 4.0) * ps).max(1.0),
            grid: pct("grid", 0.12),
            grid_size: (num(fv(e, "grid_size", cx), 8.0, 4000.0, 96.0) * ps).max(2.0),
            brackets: pct("brackets", 0.8),
            vignette: pct("vignette", 0.45),
            tint: pct("tint", 0.15),
            flicker: (h01(frame_hash(cx), 0x77, 0) - 0.5) * pct("flicker", 0.1) * 0.12,
            line: ps.max(0.75),
        },
        "cr_chromatic_aberration" => {
            let (x, y) = centre("center");
            let amt = num(fv(e, "amount", cx), 0.0, 400.0, 8.0) * ps;
            let a = radians("angle");
            let half_diag = (wf * wf + hf * hf).sqrt() * 0.5;
            CreatorOp::Chromatic { radial: chv(e, "kind") != 1, k: amt / half_diag.max(1.0), dx: a.cos() * amt, dy: a.sin() * amt, cx: x, cy: y }
        }
        "cr_light_sweep" => {
            let a = radians("angle");
            let (nx, ny) = (a.cos(), a.sin());
            let ext = (nx.abs() * wf + ny.abs() * hf) * 0.5;
            let cycle = num(fv(e, "cycle", cx), 0.0, 60.0, 0.0);
            let pos = if cycle > 0.01 { fract(phase(cx) / cycle) * 1.4 - 0.2 } else { num(fv(e, "position", cx) / 100.0, -1.0, 2.0, 0.5) };
            let half = (pct("width", 0.12) * ext * 2.0).max(1.0) * 0.5;
            CreatorOp::LightSweep {
                intensity: num(fv(e, "intensity", cx) / 100.0, 0.0, 4.0, 0.8),
                color: dec(colour3(cv(e, "color", cx), [1.0, 0.97, 0.9])),
                nx,
                ny,
                s0: (pos * 2.0 - 1.0) * ext,
                half,
                inner: half * (1.0 - pct("softness", 0.6)),
            }
        }
        "cr_duotone" => CreatorOp::Duotone {
            amount: pct("amount", 1.0),
            shadows: colour3(cv(e, "shadows", cx), [0.10, 0.05, 0.35]),
            highlights: colour3(cv(e, "highlights", cx), [1.0, 0.75, 0.30]),
            contrast: num(fv(e, "contrast", cx) / 100.0, -1.0, 1.0, 0.0),
        },
        "cr_halftone" => {
            let a = radians("angle");
            CreatorOp::Halftone {
                amount: pct("amount", 1.0),
                size: (num(fv(e, "size", cx), 2.0, 400.0, 10.0) * ps).max(1.0),
                cs: a.cos(),
                sn: a.sin(),
                color: chv(e, "mode") == 1,
                ink: colour3(cv(e, "ink", cx), [0.05, 0.05, 0.08]),
                paper: colour3(cv(e, "paper", cx), [0.97, 0.95, 0.90]),
            }
        }
        _ => return None,
    })
}

// ------------------------------------------------------------------ the reference

/// Run `f` for every pixel of a copy of `img` (pixel coordinates), writing `img`.
fn per_pixel(img: &mut Image, f: impl Fn(&Image, usize, usize) -> [f32; 4] + Sync) {
    let src = img.clone();
    let w = img.w;
    img.px.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let o = f(&src, x, y);
            row[x * 4..x * 4 + 4].copy_from_slice(&o);
        }
    });
}

impl CreatorOp {
    /// Whether every number is finite (always, from [`eval`]): the GPU takes the op.
    pub fn gpu_ok(&self) -> bool {
        let fin = |v: &[f32]| v.iter().all(|x| x.is_finite());
        match self {
            CreatorOp::FilmGrain { amount, size, response, .. } => fin(&[*amount, *size, *response]),
            CreatorOp::Vhs { amount, bleed, tracking, noise, wobble, sat, phase, band_y, band_h, .. } => {
                fin(&[*amount, *bleed, *tracking, *noise, *wobble, *sat, *phase, *band_y, *band_h])
            }
            CreatorOp::Kaleidoscope { amount, segments, rot, cx, cy, zoom } => {
                fin(&[*amount, *segments, *rot, *cx, *cy, *zoom]) && *zoom > 0.0 && *segments >= 1.0
            }
            CreatorOp::Glitch { amount, block, split, jitter, .. } => fin(&[*amount, *block, *split, *jitter]) && *block > 0.0,
            CreatorOp::ZoomBlur { amount, cx, cy, steps, .. } => fin(&[*amount, *cx, *cy]) && (2..=64).contains(steps),
            CreatorOp::MotionBlur { dx, dy, steps, falloff, .. } => fin(&[*dx, *dy, *falloff]) && (2..=64).contains(steps),
            CreatorOp::TechHud { amount, color, scan, spacing, grid, grid_size, brackets, vignette, tint, flicker, line } => {
                fin(color)
                    && fin(&[*amount, *scan, *spacing, *grid, *grid_size, *brackets, *vignette, *tint, *flicker, *line])
                    && *spacing > 0.0
                    && *grid_size > 0.0
                    && *line > 0.0
            }
            CreatorOp::Chromatic { k, dx, dy, cx, cy, .. } => fin(&[*k, *dx, *dy, *cx, *cy]),
            CreatorOp::LightSweep { intensity, color, nx, ny, s0, half, inner } => fin(color) && fin(&[*intensity, *nx, *ny, *s0, *half, *inner]),
            CreatorOp::Duotone { amount, shadows, highlights, contrast } => fin(shadows) && fin(highlights) && fin(&[*amount, *contrast]),
            CreatorOp::Halftone { amount, size, cs, sn, ink, paper, .. } => fin(ink) && fin(paper) && fin(&[*amount, *size, *cs, *sn]) && *size > 0.0,
        }
    }

    /// Whether the op leaves every picture as it is (an amount of zero): no pass at all. (Only
    /// meaningful for a [`gpu_ok`](Self::gpu_ok) op: [`eval`] never makes a non-finite one.)
    pub fn is_identity(&self) -> bool {
        match self {
            CreatorOp::FilmGrain { amount, .. }
            | CreatorOp::Vhs { amount, .. }
            | CreatorOp::Kaleidoscope { amount, .. }
            | CreatorOp::Glitch { amount, .. }
            | CreatorOp::ZoomBlur { amount, .. }
            | CreatorOp::TechHud { amount, .. }
            | CreatorOp::Duotone { amount, .. }
            | CreatorOp::Halftone { amount, .. } => *amount <= 0.0,
            CreatorOp::MotionBlur { dx, dy, .. } => dx.abs() + dy.abs() < 0.5,
            CreatorOp::Chromatic { radial: true, k, .. } => *k == 0.0,
            CreatorOp::Chromatic { dx, dy, .. } => dx.abs() + dy.abs() == 0.0,
            CreatorOp::LightSweep { intensity, .. } => *intensity <= 0.0,
        }
    }

    /// The CPU reference.
    pub fn apply(&self, img: &mut Image) {
        if img.w == 0 || img.h == 0 || self.is_identity() || !self.gpu_ok() {
            return;
        }
        let (wf, hf) = (img.w as f32, img.h as f32);
        match *self {
            CreatorOp::FilmGrain { amount, size, response, color, seed } => per_pixel(img, |s, x, y| {
                let o = s.get(x, y);
                if o[3] <= 1e-6 {
                    return o;
                }
                let (qx, qy) = ((x as f32 + 0.5) / size, (y as f32 + 0.5) / size);
                let e = enc(straight(o));
                let l = luma(e).clamp(0.0, 1.0);
                let k = amount * 0.5 * (1.0 - response + response * 4.0 * l * (1.0 - l));
                let g0 = value_noise(qx, qy, seed) - 0.5;
                let g =
                    if color { [g0, value_noise(qx, qy, seed ^ GRAIN_CHANNEL[1]) - 0.5, value_noise(qx, qy, seed ^ GRAIN_CHANNEL[2]) - 0.5] } else { [g0; 3] };
                premul(dec([e[0] + g[0] * k, e[1] + g[1] * k, e[2] + g[2] * k]), o[3])
            }),
            CreatorOp::Vhs { amount, bleed, tracking, noise, wobble, sat, phase, band_y, band_h, frame, seed } => per_pixel(img, |s, x, y| {
                let (pcx, pcy) = (x as f32 + 0.5, y as f32 + 0.5);
                let row = y as u32;
                let rowh = h01(row, frame, seed);
                let wob = wobble * (0.65 * (pcy * 0.031 + phase * 6.0).sin() + 0.35 * (rowh * 2.0 - 1.0));
                let tl = tracking * (1.0 - (pcy - band_y).abs() / band_h).clamp(0.0, 1.0);
                let sx = pcx - (wob + tl * (rowh - 0.5) * 0.08 * wf);
                let s0 = s.sample_bilinear_clamped(sx, pcy);
                let e0 = enc(straight(s0));
                let yl = luma(e0);
                let mut chroma = [0.0f32; 3];
                for k in 0..8 {
                    let xk = sx - bleed * (k as f32 / 7.0) - bleed * 0.25;
                    let ek = enc(straight(s.sample_bilinear_clamped(xk, pcy)));
                    let lk = luma(ek);
                    for c in 0..3 {
                        chroma[c] += ek[c] - lk;
                    }
                }
                let nz = (h01(x as u32, row ^ frame, seed ^ 0x51ED) - 0.5) * (noise * 0.35 + tl * 0.6);
                let mut e = [0.0f32; 3];
                for c in 0..3 {
                    e[c] = (yl + chroma[c] * 0.125 * sat + nz) * 0.94 + 0.03;
                }
                let o = s.get(x, y);
                let eo = enc(straight(o));
                let a = o[3] + (s0[3] - o[3]) * amount;
                premul(dec(mix3(eo, e, amount)), a)
            }),
            CreatorOp::Kaleidoscope { amount, segments, rot, cx, cy, zoom } => {
                let seg = std::f32::consts::TAU / segments;
                per_pixel(img, |s, x, y| {
                    let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                    let r = (dx * dx + dy * dy).sqrt() / zoom;
                    let th = dy.atan2(dx) - rot;
                    let mut m = th - seg * (th / seg).floor();
                    if m > seg * 0.5 {
                        m = seg - m;
                    }
                    let th2 = m + rot;
                    let k = s.sample_bilinear_clamped(cx + r * th2.cos(), cy + r * th2.sin());
                    let o = s.get(x, y);
                    [o[0] + (k[0] - o[0]) * amount, o[1] + (k[1] - o[1]) * amount, o[2] + (k[2] - o[2]) * amount, o[3] + (k[3] - o[3]) * amount]
                })
            }
            CreatorOp::Glitch { amount, block, split, jitter, step, seed } => per_pixel(img, |s, x, y| {
                let (pcx, pcy) = (x as f32 + 0.5, y as f32 + 0.5);
                let (bw, bh) = (block * 2.0, (block * 0.5).max(1.0));
                let (bx, by) = (ifloor(pcx / bw), ifloor(pcy / bh));
                let st = step ^ seed;
                let r = h01(bx, by, st);
                let mut dx = 0.0;
                if r < amount * 0.4 {
                    dx = (h01(by, bx ^ 0x2C1B, st ^ 0x5BD1) - 0.5) * 4.0 * block * amount;
                }
                let row = y as u32;
                if h01(row, step, seed ^ 0x1234) < jitter * amount * 0.5 {
                    dx += (h01(row ^ 0x3C6E, step, seed ^ 0x77) - 0.5) * jitter * block;
                }
                let spl = split * amount * (0.5 + h01(by, step, seed ^ 0xABC));
                let sx = pcx - dx;
                let (rs, gs, bs) = (s.sample_bilinear_clamped(sx + spl, pcy), s.sample_bilinear_clamped(sx, pcy), s.sample_bilinear_clamped(sx - spl, pcy));
                let a = rs[3].max(gs[3]).max(bs[3]);
                let mut o = [rs[0], gs[1], bs[2], a];
                if r > 1.0 - amount * 0.03 {
                    for c in &mut o[..3] {
                        *c = (a - *c).max(0.0);
                    }
                }
                o
            }),
            CreatorOp::ZoomBlur { amount, cx, cy, spin, steps } => per_pixel(img, |s, x, y| {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                let mut acc = [0.0f32; 4];
                for i in 0..steps {
                    let t = i as f32 / (steps - 1) as f32;
                    let (sx, sy) = if spin {
                        let ang = (t - 0.5) * amount * std::f32::consts::FRAC_PI_6;
                        let (sn, cs) = ang.sin_cos();
                        (cx + dx * cs - dy * sn, cy + dx * sn + dy * cs)
                    } else {
                        let k = 1.0 - amount * 0.5 * t;
                        (cx + dx * k, cy + dy * k)
                    };
                    let p = s.sample_bilinear_clamped(sx, sy);
                    for c in 0..4 {
                        acc[c] += p[c];
                    }
                }
                acc.map(|v| v / steps as f32)
            }),
            CreatorOp::MotionBlur { dx, dy, steps, trail, falloff } => per_pixel(img, |s, x, y| {
                let (pcx, pcy) = (x as f32 + 0.5, y as f32 + 0.5);
                let mut acc = [0.0f32; 4];
                let mut wsum = 0.0f32;
                for i in 0..steps {
                    let t = i as f32 / (steps - 1) as f32;
                    let (off, far) = if trail { (-t, t) } else { (t - 0.5, (t - 0.5).abs() * 2.0) };
                    let wt = 1.0 - falloff * 0.95 * far;
                    let p = s.sample_bilinear_clamped(pcx + dx * off, pcy + dy * off);
                    for c in 0..4 {
                        acc[c] += p[c] * wt;
                    }
                    wsum += wt;
                }
                acc.map(|v| v / wsum)
            }),
            CreatorOp::TechHud { amount, color, scan, spacing, grid, grid_size, brackets, vignette, tint, flicker, line } => {
                let cmax = color[0].max(color[1]).max(color[2]).max(1e-4);
                let mn = wf.min(hf);
                let (m, arm, th) = (0.06 * mn, 0.09 * mn, (0.006 * mn).max(1.0));
                per_pixel(img, |s, x, y| {
                    let o = s.get(x, y);
                    if o[3] <= 1e-6 {
                        return o;
                    }
                    let (pcx, pcy) = (x as f32 + 0.5, y as f32 + 0.5);
                    let e0 = enc(straight(o));
                    let l = luma(e0);
                    let mut e = [0.0f32; 3];
                    let sl = 0.5 - 0.5 * (std::f32::consts::TAU * pcy / spacing).cos();
                    let (vx, vy) = ((pcx / wf - 0.5) * 2.0, (pcy / hf - 0.5) * 2.0);
                    let v = smoothstep(0.45, 1.0, (vx * vx + vy * vy).sqrt() / std::f32::consts::SQRT_2);
                    let shade = (1.0 - scan * 0.45 * sl) * (1.0 + flicker) * (1.0 - vignette * 0.75 * v);
                    for c in 0..3 {
                        let tinted = e0[c] + (l * color[c] / cmax - e0[c]) * tint * 0.6;
                        e[c] = tinted * shade;
                    }
                    let gx = (fract(pcx / grid_size + 0.5) - 0.5).abs() * grid_size;
                    let gy = (fract(pcy / grid_size + 0.5) - 0.5).abs() * grid_size;
                    let gl = (1.0 - gx / line).clamp(0.0, 1.0).max((1.0 - gy / line).clamp(0.0, 1.0)) * grid * 0.35;
                    let fx = pcx.min(wf - pcx) - m;
                    let fy = pcy.min(hf - pcy) - m;
                    let cov_h = (fx + 0.5).min(arm - fx + 0.5).min((fy + 0.5).min(th - fy + 0.5)).clamp(0.0, 1.0);
                    let cov_v = (fx + 0.5).min(th - fx + 0.5).min((fy + 0.5).min(arm - fy + 0.5)).clamp(0.0, 1.0);
                    let cov = (gl + cov_h.max(cov_v) * brackets).clamp(0.0, 1.0);
                    let e = mix3(e, color, cov);
                    premul(dec(mix3(e0, e, amount)), o[3])
                })
            }
            CreatorOp::Chromatic { radial, k, dx, dy, cx, cy } => per_pixel(img, |s, x, y| {
                let (pcx, pcy) = (x as f32 + 0.5, y as f32 + 0.5);
                let (rs, bs) = if radial {
                    let (ox, oy) = (pcx - cx, pcy - cy);
                    (s.sample_bilinear_clamped(cx + ox * (1.0 + k), cy + oy * (1.0 + k)), s.sample_bilinear_clamped(cx + ox * (1.0 - k), cy + oy * (1.0 - k)))
                } else {
                    (s.sample_bilinear_clamped(pcx + dx, pcy + dy), s.sample_bilinear_clamped(pcx - dx, pcy - dy))
                };
                let g = s.get(x, y);
                [rs[0], g[1], bs[2], rs[3].max(g[3]).max(bs[3])]
            }),
            CreatorOp::LightSweep { intensity, color, nx, ny, s0, half, inner } => per_pixel(img, |s, x, y| {
                let o = s.get(x, y);
                if o[3] <= 1e-6 {
                    return o;
                }
                let d = (x as f32 + 0.5 - wf * 0.5) * nx + (y as f32 + 0.5 - hf * 0.5) * ny - s0;
                let band = 1.0 - smoothstep(inner, half, d.abs());
                let k = band * intensity * o[3];
                [o[0] + color[0] * k, o[1] + color[1] * k, o[2] + color[2] * k, o[3]]
            }),
            CreatorOp::Duotone { amount, shadows, highlights, contrast } => per_pixel(img, |s, x, y| {
                let o = s.get(x, y);
                if o[3] <= 1e-6 {
                    return o;
                }
                let e = enc(straight(o));
                let l = ((luma(e).clamp(0.0, 1.0) - 0.5) * (1.0 + contrast) + 0.5).clamp(0.0, 1.0);
                premul(dec(mix3(e, mix3(shadows, highlights, l), amount)), o[3])
            }),
            CreatorOp::Halftone { amount, size, cs, sn, color, ink, paper } => per_pixel(img, |s, x, y| {
                let o = s.get(x, y);
                if o[3] <= 1e-6 {
                    return o;
                }
                let (pcx, pcy) = (x as f32 + 0.5, y as f32 + 0.5);
                let u = cs * pcx + sn * pcy;
                let v = -sn * pcx + cs * pcy;
                let cu = ((u / size).floor() + 0.5) * size;
                let cvv = ((v / size).floor() + 0.5) * size;
                let es = enc(straight(s.sample_bilinear_clamped(cs * cu - sn * cvv, sn * cu + cs * cvv)));
                let d = ((u - cu) * (u - cu) + (v - cvv) * (v - cvv)).sqrt();
                let t = if color {
                    es.map(|c| (c.clamp(0.0, 1.0).sqrt() * size * std::f32::consts::FRAC_1_SQRT_2 - d + 0.5).clamp(0.0, 1.0))
                } else {
                    let r = (1.0 - luma(es).clamp(0.0, 1.0)).sqrt() * size * std::f32::consts::FRAC_1_SQRT_2;
                    mix3(paper, ink, (r - d + 0.5).clamp(0.0, 1.0))
                };
                premul(dec(mix3(enc(straight(o)), t, amount)), o[3])
            }),
        }
    }
}

#[cfg(test)]
#[path = "creator_fx_tests.rs"]
mod tests;
