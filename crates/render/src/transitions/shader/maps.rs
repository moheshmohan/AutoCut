//! Generated grayscale maps for the shader stage's map binding (roadmap C2): the luma-wipe
//! engine's shapes and the maps the GL `displacement` and `luma` ports read.
//!
//! Every map is computed here from a [`Spec`], never loaded: radial, linear at any angle, the
//! iris shapes (oval, box, diamond, cross, star), barn doors, blinds, clock and fan sweeps,
//! spirals, checkerboards, rings, waves, random blocks, value noise and Worley cells. A map is
//! one `f32` per frame pixel (rows from the top, like the frames), feathered with a blur of
//! `feather` pixels and then stretched to exactly 0..1, so a wipe that follows it starts at
//! progress 0 and is complete at progress 1. The CPU reference and the GPU (an `R32Float`
//! texture) read the same values, so the two agree wherever the map decides.
//!
//! Maps depend on the parameters and the frame size, not on the progress: [`cached`] keeps the
//! last few, so playing a transition generates its map once.

use std::f32::consts::TAU;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use filmcraft_project::transition_engines::LumaShape;
use rayon::prelude::*;

use super::super::{fbm, hash};

/// The largest map (pixels): 8K UHD. Larger frames get no map (the transition renders as if its
/// map were flat).
pub const MAX_PIXELS: usize = 7680 * 4320;
/// Maps kept by [`cached`].
const CACHE: usize = 6;

/// What to generate. Pixel quantities are frame pixels (the caller has scaled sequence pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    pub shape: LumaShape,
    /// Degrees (wipe angle, rotation or start angle, by shape).
    pub angle: f32,
    /// Centre in frame pixels (y down); NaN = the frame's centre.
    pub center: (f32, f32),
    /// Slats, points, arms, squares, rings, waves, blocks or cells (at least 1).
    pub count: f32,
    /// Inner radius of a star, 0..1.
    pub inner: f32,
    /// Noise cell size, frame pixels.
    pub scale: f32,
    pub seed: u32,
    /// Counter-clockwise (clock, spiral).
    pub ccw: bool,
    /// Blur radius, frame pixels.
    pub feather: f32,
}

impl Spec {
    /// A spec of `shape` with neutral parameters.
    pub fn new(shape: LumaShape) -> Spec {
        Spec { shape, angle: 0.0, center: (f32::NAN, f32::NAN), count: 1.0, inner: 0.45, scale: 64.0, seed: 1, ccw: false, feather: 0.0 }
    }

    /// The same spec with every number finite and in range.
    fn sane(mut self) -> Spec {
        let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
        self.angle = fin(self.angle, 0.0).clamp(-1e6, 1e6);
        self.center = (
            if self.center.0.is_finite() { self.center.0.clamp(-1e6, 1e6) } else { f32::NAN },
            if self.center.1.is_finite() { self.center.1.clamp(-1e6, 1e6) } else { f32::NAN },
        );
        self.count = fin(self.count, 1.0).round().clamp(1.0, 1000.0);
        self.inner = fin(self.inner, 0.45).clamp(0.01, 0.99);
        self.scale = fin(self.scale, 64.0).clamp(0.5, 1e5);
        self.feather = fin(self.feather, 0.0).clamp(0.0, 4000.0);
        self
    }

    fn key(&self, w: usize, h: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut s = std::collections::hash_map::DefaultHasher::new();
        self.shape.hash(&mut s);
        for v in [self.angle, self.center.0, self.center.1, self.count, self.inner, self.scale, self.feather] {
            v.to_bits().hash(&mut s);
        }
        (self.seed, self.ccw, w, h).hash(&mut s);
        s.finish()
    }
}

/// A generated map: `w × h` values in 0..1, rows from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct Map {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
    /// Identifies the spec and size it was made from (the GPU stage caches its texture by it).
    pub key: u64,
}

impl Map {
    /// The value of pixel (x, y), clamped to the map.
    #[inline]
    pub fn at(&self, x: i64, y: i64) -> f32 {
        if self.w == 0 || self.h == 0 {
            return 0.0;
        }
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        self.px.get(y * self.w + x).copied().unwrap_or(0.0)
    }
}

/// The map for `spec` on a `w × h` frame, from the cache when it was made recently. None for an
/// empty or oversized frame.
pub fn cached(spec: &Spec, w: usize, h: usize) -> Option<Arc<Map>> {
    if w == 0 || h == 0 || w.saturating_mul(h) > MAX_PIXELS {
        return None;
    }
    let spec = spec.sane();
    let key = spec.key(w, h);
    static MAPS: OnceLock<Mutex<Vec<Arc<Map>>>> = OnceLock::new();
    let maps = MAPS.get_or_init(Default::default);
    {
        let mut m = maps.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = m.iter().position(|x| x.key == key && (x.w, x.h) == (w, h)) {
            let hit = m.remove(i);
            m.push(hit.clone());
            return Some(hit);
        }
    }
    let map = Arc::new(generate(&spec, w, h, key));
    let mut m = maps.lock().unwrap_or_else(PoisonError::into_inner);
    m.push(map.clone());
    if m.len() > CACHE {
        m.remove(0);
    }
    Some(map)
}

/// The raw value of `spec` at pixel centre (x, y) on a `w × h` frame (before feathering and
/// stretching to 0..1; larger values are reached later).
fn value(s: &Spec, w: f32, h: f32, x: f32, y: f32) -> f32 {
    use LumaShape::*;
    let cx = if s.center.0.is_finite() { s.center.0 } else { w * 0.5 };
    let cy = if s.center.1.is_finite() { s.center.1 } else { h * 0.5 };
    let (dx, dy) = (x - cx, y - cy);
    let a = s.angle.to_radians();
    let (sa, ca) = a.sin_cos();
    // the frame turned by the angle (about the centre)
    let (u, v) = (dx * ca + dy * sa, -dx * sa + dy * ca);
    // the direction a straight edge travels at the angle (90° = left to right; y down)
    let (ex, ey) = (sa, -ca);
    // the position along that direction, 0 at the first corner reached and 1 at the last
    let along = || {
        let corners = [0.0f32, w * ex, h * ey, w * ex + h * ey];
        let lo = corners.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = corners.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (x * ex + y * ey - lo) / (hi - lo).max(1e-6)
    };
    // the angle round the centre from straight up, clockwise, in turns (−½..½)
    let turn = || dx.atan2(-dy) / TAU;
    // the distance from the centre relative to the farthest corner
    let reach = || {
        let far = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)].iter().map(|(px, py)| (px - cx).hypot(py - cy)).fold(1e-6f32, f32::max);
        dx.hypot(dy) / far
    };
    let spin = if s.ccw { -1.0 } else { 1.0 };
    let cell = (w / s.count).max(1e-3);
    match s.shape {
        Linear => along(),
        Radial => dx.hypot(dy),
        Oval => (dx / w).hypot(dy / h),
        Box => (u / w).abs().max((v / h).abs()),
        Diamond => (u / w).abs() + (v / h).abs(),
        Cross => (u / w).abs().min((v / h).abs()),
        Star => {
            let n = s.count.max(3.0);
            let sector = TAU / n;
            // angle from the point straight up, folded into one half point
            let t = ((u.atan2(-v) + sector * 0.5).rem_euclid(sector) - sector * 0.5).abs();
            let r = u.hypot(v);
            let (qx, qy) = (r * t.cos(), r * t.sin());
            // the edge from the tip (1, 0) to the inner corner
            let (vx, vy) = (s.inner * (sector * 0.5).cos(), s.inner * (sector * 0.5).sin());
            let (nx, ny) = (vy, -(vx - 1.0));
            (nx * qx + ny * qy) / nx.max(1e-6)
        }
        BarnDoor => (dx * ex + dy * ey).abs(),
        Blinds => (along() * s.count).fract(),
        Clock => (turn() * spin - s.angle / 360.0).rem_euclid(1.0),
        Fan => {
            let d = (turn() - s.angle / 360.0 + 0.5).rem_euclid(1.0) - 0.5;
            d.abs()
        }
        Spiral => (turn() * spin * s.count + reach()).rem_euclid(1.0),
        Checker => {
            let (i, j) = ((x / cell).floor(), (y / cell).floor());
            let parity = (i + j).rem_euclid(2.0);
            (parity + (x / cell - i)) * 0.5
        }
        Rings => (reach() * s.count).fract(),
        Wave => {
            let across = (x * ca + y * sa) / w.max(h).max(1.0);
            along() + 0.04 * (TAU * s.count * across).sin()
        }
        Blocks => hash((x / cell).floor() as i32, (y / cell).floor() as i32, s.seed.wrapping_add(31)),
        Noise => fbm(x / s.scale, y / s.scale, s.seed.wrapping_add(97)),
        Worley | Crackle => {
            let (gx, gy) = (x / cell, y / cell);
            let (ci, cj) = (gx.floor() as i32, gy.floor() as i32);
            let (mut f1, mut f2) = (f32::INFINITY, f32::INFINITY);
            for j in cj.saturating_sub(1)..=cj.saturating_add(1) {
                for i in ci.saturating_sub(1)..=ci.saturating_add(1) {
                    let fx = i as f32 + hash(i, j, s.seed.wrapping_add(53));
                    let fy = j as f32 + hash(i, j, s.seed.wrapping_add(54));
                    let d = (gx - fx).hypot(gy - fy);
                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
            if s.shape == Worley { f1 } else { f2 - f1 }
        }
    }
}

/// Generate the map of `spec` (already [`Spec::sane`]) on a `w × h` frame.
fn generate(spec: &Spec, w: usize, h: usize, key: u64) -> Map {
    let (wf, hf) = (w as f32, h as f32);
    let mut px = vec![0.0f32; w * h];
    px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = y as f32 + 0.5;
        for (x, o) in row.iter_mut().enumerate() {
            let v = value(spec, wf, hf, x as f32 + 0.5, fy);
            *o = if v.is_finite() { v } else { 0.0 };
        }
    });
    if spec.feather >= 0.5 {
        feather(&mut px, w, h, spec.feather);
    }
    stretch(&mut px);
    Map { w, h, px, key }
}

/// Blur the map by about `radius` pixels: three box blurs in each direction (a close
/// approximation of a Gaussian), each pass O(pixels) whatever the radius.
fn feather(px: &mut [f32], w: usize, h: usize, radius: f32) {
    let r = ((radius / 3.0).round() as usize).clamp(1, w.max(h));
    for _ in 0..3 {
        px.par_chunks_mut(w).for_each(|row| box_line(row, r));
        transpose_blur(px, w, h, r);
    }
}

/// A box blur of radius `r` along one line, edges clamped (running sum in f64).
fn box_line(line: &mut [f32], r: usize) {
    let n = line.len();
    if n < 2 {
        return;
    }
    let src = line.to_vec();
    let at = |i: isize| src.get(i.clamp(0, n as isize - 1) as usize).copied().unwrap_or(0.0) as f64;
    let ri = r as isize;
    let mut sum: f64 = (-ri..=ri).map(at).sum();
    let k = 1.0 / (2 * r + 1) as f64;
    for (i, o) in line.iter_mut().enumerate() {
        *o = (sum * k) as f32;
        let i = i as isize;
        sum += at(i + ri + 1) - at(i - ri);
    }
}

/// The box blur down the columns (through a transposed copy, so it runs on rows in parallel).
fn transpose_blur(px: &mut [f32], w: usize, h: usize, r: usize) {
    let mut t = vec![0.0f32; w * h];
    for (y, row) in px.chunks(w).enumerate() {
        for (x, v) in row.iter().enumerate() {
            if let Some(o) = t.get_mut(x * h + y) {
                *o = *v;
            }
        }
    }
    t.par_chunks_mut(h).for_each(|col| box_line(col, r));
    for (x, col) in t.chunks(h).enumerate() {
        for (y, v) in col.iter().enumerate() {
            if let Some(o) = px.get_mut(y * w + x) {
                *o = *v;
            }
        }
    }
}

/// Stretch the values to exactly 0..1 (a flat map becomes all 0).
fn stretch(px: &mut [f32]) {
    let lo = px.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = hi - lo;
    if !(span.is_finite() && span > 1e-12) {
        px.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    // a division, so the largest value is exactly 1
    px.par_iter_mut().for_each(|v| *v = ((*v - lo) / span).clamp(0.0, 1.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPES: [LumaShape; 19] = {
        use LumaShape::*;
        [Linear, Radial, Oval, Box, Diamond, Cross, Star, BarnDoor, Blinds, Clock, Fan, Spiral, Checker, Rings, Wave, Blocks, Noise, Worley, Crackle]
    };

    #[test]
    fn every_map_spans_zero_to_one() {
        for shape in SHAPES {
            for (w, h) in [(64usize, 36usize), (17, 5), (3, 2)] {
                let mut s = Spec::new(shape);
                s.count = 4.0;
                s.scale = 6.0;
                let m = generate(&s.sane(), w, h, 0);
                assert_eq!(m.px.len(), w * h);
                assert!(m.px.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "{shape:?} {w}x{h}");
                let lo = m.px.iter().copied().fold(1.0f32, f32::min);
                let hi = m.px.iter().copied().fold(0.0f32, f32::max);
                assert!(lo == 0.0 && hi == 1.0 || lo == hi, "{shape:?} {w}x{h}: {lo}..{hi}");
            }
        }
    }

    #[test]
    fn shapes_are_where_their_names_say() {
        let (w, h) = (64usize, 36usize);
        let at = |m: &Map, x: usize, y: usize| m.at(x as i64, y as i64);
        let map = |s: Spec| generate(&s.sane(), w, h, 0);
        // linear at 90°: left first, right last
        let mut s = Spec::new(LumaShape::Linear);
        s.angle = 90.0;
        let m = map(s);
        assert!(at(&m, 0, 10) < 0.02 && at(&m, w - 1, 10) > 0.98);
        // 180°: top first
        s.angle = 180.0;
        let m = map(s);
        assert!(at(&m, 10, 0) < 0.03 && at(&m, 10, h - 1) > 0.97);
        // radial: the centre first, the corners last
        let m = map(Spec::new(LumaShape::Radial));
        assert!(at(&m, w / 2, h / 2) < 0.05 && at(&m, 0, 0) > 0.95);
        // a centre in the top left corner
        let mut s = Spec::new(LumaShape::Radial);
        s.center = (0.0, 0.0);
        let m = map(s);
        assert!(at(&m, 0, 0) < 0.02 && at(&m, w - 1, h - 1) > 0.98);
        // barn doors (90°): the middle column first, both sides last
        let mut s = Spec::new(LumaShape::BarnDoor);
        s.angle = 90.0;
        let m = map(s);
        assert!(at(&m, w / 2, 3) < 0.05 && at(&m, 0, 3) > 0.95 && at(&m, w - 1, 30) > 0.95);
        // clock: just right of 12 o'clock first, just left of it last
        let m = map(Spec::new(LumaShape::Clock));
        assert!(at(&m, w / 2 + 1, 2) < 0.05 && at(&m, w / 2 - 2, 2) > 0.9);
        let mut s = Spec::new(LumaShape::Clock);
        s.ccw = true;
        let m = map(s);
        assert!(at(&m, w / 2 + 1, 2) > 0.9 && at(&m, w / 2 - 2, 2) < 0.1);
        // blinds: every slat runs 0..1
        let mut s = Spec::new(LumaShape::Blinds);
        s.angle = 90.0;
        s.count = 4.0;
        let m = map(s);
        for k in 0..4 {
            assert!(at(&m, k * 16, 5) < 0.1 && at(&m, k * 16 + 15, 5) > 0.85, "slat {k}");
        }
        // star: the centre first; with 5 points one is straight up
        let mut s = Spec::new(LumaShape::Star);
        s.count = 5.0;
        let m = map(s);
        assert!(at(&m, w / 2, h / 2) < 0.05);
        assert!(at(&m, w / 2, 4) < at(&m, w / 2 + 8, 6), "a point reaches up");
        // feathering keeps the order and the full range, and blurs a hard edge
        let mut s = Spec::new(LumaShape::Checker);
        s.count = 4.0;
        let hard = map(s);
        s.feather = 6.0;
        let soft = map(s);
        let steps = |m: &Map| (1..w).map(|x| (at(m, x, 5) - at(m, x - 1, 5)).abs()).fold(0.0f32, f32::max);
        assert!(steps(&soft) < steps(&hard));
    }

    #[test]
    fn hostile_specs_and_sizes_are_safe() {
        let nasty = [f32::NAN, f32::INFINITY, -f32::INFINITY, f32::MAX, -f32::MAX, 0.0, -1.0, 1e-30];
        for shape in SHAPES {
            for (i, v) in nasty.iter().enumerate() {
                let s = Spec { shape, angle: *v, center: (*v, -*v), count: *v, inner: *v, scale: *v, seed: u32::MAX - i as u32, ccw: i % 2 == 0, feather: *v };
                let m = cached(&s, 9, 4).unwrap();
                assert!(m.px.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)), "{shape:?} {v}");
            }
        }
        assert!(cached(&Spec::new(LumaShape::Radial), 0, 10).is_none());
        assert!(cached(&Spec::new(LumaShape::Radial), 100_000, 100_000).is_none());
        let one = cached(&Spec::new(LumaShape::Noise), 1, 1).unwrap();
        assert_eq!(one.px, vec![0.0]);
        assert_eq!(one.at(-5, 99), 0.0);
        // the cache returns the same map for the same spec and size, another for another size
        let a = cached(&Spec::new(LumaShape::Worley), 32, 18).unwrap();
        let b = cached(&Spec::new(LumaShape::Worley), 32, 18).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let c = cached(&Spec::new(LumaShape::Worley), 33, 18).unwrap();
        assert_ne!(a.key, c.key);
        // a huge feather on a small frame is bounded
        let mut s = Spec::new(LumaShape::Linear);
        s.feather = 4000.0;
        assert!(cached(&s, 5, 3).unwrap().px.iter().all(|v| v.is_finite()));
    }
}
