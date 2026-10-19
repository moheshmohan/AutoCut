//! Rust references (CPU oracles) of the ported GL Transitions: each function is its WGSL
//! shader (`shader/<name>.wgsl`) written again in Rust, operation for operation, on the same host
//! contract (see `prelude.wgsl`). Export renders these; the GPU path is tested against them.
//!
//! The transitions are hand-ported from gl-transitions
//! (<https://github.com/gl-transitions/gl-transitions>, MIT; licence text in
//! `shader/LICENSE-gl-transitions.txt`); each function keeps its upstream author and licence.

use std::f32::consts::PI;

use crate::image::Image;

mod engines;
mod styles;
pub(crate) use engines::{content_wipe, flow_engine};
pub(crate) use styles::{camera, light_film, patterns};

/// A 2-D vector with GLSL-style arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct V2 {
    pub x: f32,
    pub y: f32,
}

impl V2 {
    pub const fn new(x: f32, y: f32) -> V2 {
        V2 { x, y }
    }
    const fn splat(v: f32) -> V2 {
        V2 { x: v, y: v }
    }
    fn add(self, o: V2) -> V2 {
        V2::new(self.x + o.x, self.y + o.y)
    }
    fn sub(self, o: V2) -> V2 {
        V2::new(self.x - o.x, self.y - o.y)
    }
    fn mul(self, o: V2) -> V2 {
        V2::new(self.x * o.x, self.y * o.y)
    }
    fn div(self, o: V2) -> V2 {
        V2::new(self.x / o.x, self.y / o.y)
    }
    fn adds(self, s: f32) -> V2 {
        V2::new(self.x + s, self.y + s)
    }
    fn subs(self, s: f32) -> V2 {
        V2::new(self.x - s, self.y - s)
    }
    fn muls(self, s: f32) -> V2 {
        V2::new(self.x * s, self.y * s)
    }
    fn divs(self, s: f32) -> V2 {
        V2::new(self.x / s, self.y / s)
    }
    fn floor(self) -> V2 {
        V2::new(self.x.floor(), self.y.floor())
    }
    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// One evaluation: both inputs, the eased progress, the aspect ratio and the shader's parameters
/// (the uniforms the GPU receives).
pub(crate) struct Ctx<'a> {
    pub a: &'a Image,
    pub b: &'a Image,
    pub w: f32,
    pub h: f32,
    pub progress: f32,
    pub ratio: f32,
    pub p: [f32; super::PARAMS],
    /// The generated map (the GPU's map binding), when the shader reads one.
    pub map: Option<&'a super::maps::Map>,
    /// The content field (the GPU's map binding for a content-aware shader), when it reads one.
    pub field: Option<&'a crate::transitions::content::Field>,
    /// The field's cells per frame pixel (`Host::inv.zw`).
    pub field_ratio: (f32, f32),
}

impl Ctx<'_> {
    /// The host's `getField` at frame position (`x`, `y`) (zeros without a field).
    fn field(&self, x: f32, y: f32) -> [f32; 4] {
        self.field.map_or([0.0; 4], |f| f.sample(x, y, self.field_ratio))
    }
    /// The host's `getFromPx`: the outgoing frame at frame position (`x`, `y`).
    fn outgoing_px(&self, x: f32, y: f32) -> [f32; 4] {
        self.a.sample_bilinear_clamped(x, y)
    }
    /// The host's `getToPx`.
    fn incoming_px(&self, x: f32, y: f32) -> [f32; 4] {
        self.b.sample_bilinear_clamped(x, y)
    }
    /// The host's `uv_px`: the frame position of `uv`.
    fn uv_px(&self, uv: V2) -> (f32, f32) {
        (uv.x * self.w, (1.0 - uv.y) * self.h)
    }
    /// The host's `getMap`: the map's value at the pixel whose centre is at `uv` (0 without a map).
    fn map(&self, uv: V2) -> f32 {
        let Some(m) = self.map else { return 0.0 };
        let x = (uv.x * self.w).floor();
        let y = ((1.0 - uv.y) * self.h).floor();
        m.at(x as i64, y as i64)
    }
    fn sample(img: &Image, w: f32, h: f32, uv: V2) -> [f32; 4] {
        img.sample_bilinear_clamped(uv.x * w, (1.0 - uv.y) * h)
    }
    /// `getFromColor`: the outgoing frame at uv (origin bottom left), clamped bilinear.
    fn from(&self, uv: V2) -> [f32; 4] {
        Self::sample(self.a, self.w, self.h, uv)
    }
    /// `getToColor`.
    fn to(&self, uv: V2) -> [f32; 4] {
        Self::sample(self.b, self.w, self.h, uv)
    }
    /// The host's `rand_pixel`: a hash of the output pixel whose centre is at `uv`.
    fn rand_pixel(&self, uv: V2) -> f32 {
        self.rand_pixel_seeded(uv, 1)
    }
    /// The host's `rand_pixel_seeded`.
    fn rand_pixel_seeded(&self, uv: V2, s: u32) -> f32 {
        let x = (uv.x * self.w).floor();
        let y = ((1.0 - uv.y) * self.h).floor();
        hash(x as i32, y as i32, s)
    }
    /// The host's `rand_column`: a hash of the output pixel column at `uv`.
    fn rand_column(&self, uv: V2, s: u32) -> f32 {
        hash((uv.x * self.w).floor() as i32, 0, s)
    }
    /// The host's `rand_row`: a hash of the output pixel row at `uv`.
    fn rand_row(&self, uv: V2, s: u32) -> f32 {
        hash(0, ((1.0 - uv.y) * self.h).floor() as i32, s)
    }
}

use super::super::hash;

/// WGSL `fract` (`x − floor(x)`; Rust's `f32::fract` truncates instead).
#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}
/// GLSL `mod` for floats (the host's `glsl_mod`).
#[inline]
fn glsl_mod(x: f32, y: f32) -> f32 {
    x - y * (x / y).floor()
}
/// WGSL `step(edge, x)`: 1 when edge ≤ x.
#[inline]
fn step(edge: f32, x: f32) -> f32 {
    if edge <= x { 1.0 } else { 0.0 }
}
/// WGSL `sign` (0 at 0; Rust's `signum` gives 1).
#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
#[inline]
fn dot(a: V2, b: V2) -> f32 {
    a.x * b.x + a.y * b.y
}
#[inline]
fn distance(a: V2, b: V2) -> f32 {
    a.sub(b).length()
}
/// The host's `in_bounds`: strictly inside the unit square.
#[inline]
fn in_bounds(p: V2) -> bool {
    0.0 < p.x && 0.0 < p.y && p.x < 1.0 && p.y < 1.0
}
/// The host's `rand_cell_seeded`.
#[inline]
fn rand_cell_seeded(cell: V2, s: u32) -> f32 {
    hash(cell.x as i32, cell.y as i32, s)
}
#[inline]
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
/// A colour with its alpha clamped to 1 (what an 8-bit target does to upstream's).
#[inline]
fn alpha_at_most_1(c: [f32; 4]) -> [f32; 4] {
    [c[0], c[1], c[2], c[3].min(1.0)]
}

/// GLSL `mix` (`a·(1−t) + b·t`, as WGSL defines it).
#[inline]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}
#[inline]
fn mix4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t), mix(a[3], b[3], t)]
}
/// The host's `smooth_step` (a step at e0 when e1 ≤ e0).
#[inline]
fn smooth_step(e0: f32, e1: f32, x: f32) -> f32 {
    super::super::smoothstep(e0, e1, x)
}
/// The host's `atan2_safe`.
#[inline]
fn atan2_safe(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 { 0.0 } else { y.atan2(x) }
}
/// The host's `rand_cell`.
#[inline]
fn rand_cell(cell: V2) -> f32 {
    super::super::hash(cell.x as i32, cell.y as i32, 2)
}

// Author: Eke Péter <peterekepeter@gmail.com>
// License: MIT
pub(crate) fn crosswarp(c: &Ctx, p: V2) -> [f32; 4] {
    let mut x = c.progress;
    x = smooth_step(0.0, 1.0, x * 2.0 + p.x - 1.0);
    mix4(c.from(p.subs(0.5).muls(1.0 - x).adds(0.5)), c.to(p.subs(0.5).muls(x).adds(0.5)), x)
}

// License: MIT
// Author: rectalogic (ported by gre)
pub(crate) fn cross_zoom(c: &Ctx, uv: V2) -> [f32; 4] {
    fn linear_ease(begin: f32, change: f32, duration: f32, time: f32) -> f32 {
        change * time / duration + begin
    }
    fn exponential_ease_in_out(begin: f32, change: f32, duration: f32, time_in: f32) -> f32 {
        if time_in == 0.0 {
            return begin;
        } else if time_in == duration {
            return begin + change;
        }
        let time = time_in / (duration / 2.0);
        if time < 1.0 {
            return change / 2.0 * 2f32.powf(10.0 * (time - 1.0)) + begin;
        }
        change / 2.0 * (-2f32.powf(-10.0 * (time - 1.0)) + 2.0) + begin
    }
    fn sinusoidal_ease_in_out(begin: f32, change: f32, duration: f32, time: f32) -> f32 {
        -change / 2.0 * ((PI * time / duration).cos() - 1.0) + begin
    }
    let progress = c.progress;
    let tex_coord = uv;
    let center = V2::new(linear_ease(0.25, 0.5, 1.0, progress), 0.5);
    let dissolve = exponential_ease_in_out(0.0, 1.0, 1.0, progress);
    let strength = sinusoidal_ease_in_out(0.0, c.p[0], 0.5, progress);
    let mut color = [0.0f32; 4];
    let mut total = 0.0f32;
    let to_center = center.sub(tex_coord);
    let offset = c.rand_pixel(uv);
    for i in 0..=40 {
        let t = i as f32;
        let percent = (t + offset) / 40.0;
        let weight = 4.0 * (percent - percent * percent);
        let at = tex_coord.add(to_center.muls(percent).muls(strength));
        let s = mix4(c.from(at), c.to(at), dissolve);
        for k in 0..4 {
            color[k] += s[k] * weight;
        }
        total += weight;
    }
    color.map(|v| v / total)
}

// Author: Zeh Fernando
// License: MIT
pub(crate) fn dreamy_zoom(c: &Ctx, uv: V2) -> [f32; 4] {
    // upstream's value, kept for the same look (it is π/80, not 1/180*PI)
    const DEG2RAD: f32 = PI / 80.0;
    let (rotation, scale) = (c.p[0], c.p[1]);
    let progress = c.progress;
    let first = progress < 0.5;
    let phase = if first { progress * 2.0 } else { (progress - 0.5) * 2.0 };
    let angle_offset = if first { mix(0.0, rotation * DEG2RAD, phase) } else { mix(-rotation * DEG2RAD, 0.0, phase) };
    let new_scale = if first { mix(1.0, scale, phase) } else { mix(scale, 1.0, phase) };
    let mut p = uv.sub(V2::splat(0.5)).divs(new_scale).mul(V2::new(c.ratio, 1.0));
    let angle = atan2_safe(p.y, p.x) + angle_offset;
    let dist = p.length();
    p.x = angle.cos() * dist / c.ratio + 0.5;
    p.y = angle.sin() * dist + 0.5;
    let col = if first { c.from(p) } else { c.to(p) };
    let light = if first { mix(0.0, 1.0, phase) } else { mix(1.0, 0.0, phase) };
    [col[0] + light, col[1] + light, col[2] + light, (col[3] + light).min(1.0)]
}

// Author: 0gust1
// License: MIT
pub(crate) fn simple_zoom(c: &Ctx, uv: V2) -> [f32; 4] {
    let zoom = |uv: V2, amount: f32| uv.subs(0.5).muls(1.0 - amount).adds(0.5);
    let n_quick = c.p[0].clamp(0.2, 1.0);
    mix4(c.from(zoom(uv, smooth_step(0.0, n_quick, c.progress))), c.to(uv), smooth_step(n_quick - 0.2, 1.0, c.progress))
}

// Author: pschroen
// License: MIT
pub(crate) fn directional_warp(c: &Ctx, uv: V2) -> [f32; 4] {
    let smoothness = c.p[0];
    let direction = V2::new(c.p[1], c.p[2]);
    let center = V2::splat(0.5);
    let mut v = direction.divs(direction.length());
    v = v.divs(v.x.abs() + v.y.abs());
    let d = v.x * center.x + v.y * center.y;
    let m = 1.0 - smooth_step(-smoothness, 0.0, v.x * uv.x + v.y * uv.y - (d - 0.5 + c.progress * (1.0 + smoothness)));
    mix4(c.from(uv.subs(0.5).muls(1.0 - m).adds(0.5)), c.to(uv.subs(0.5).muls(m).adds(0.5)), m)
}

// Author: gre
// License: MIT
pub(crate) fn pixelize(c: &Ctx, uv: V2) -> [f32; 4] {
    let squares_min = V2::new(c.p[0].max(1.0), c.p[1].max(1.0));
    let steps = c.p[2] as i32;
    let d = c.progress.min(1.0 - c.progress);
    let dist = if steps > 0 { (d * steps as f32).ceil() / steps as f32 } else { d };
    let square_size = V2::splat(2.0 * dist).div(squares_min);
    let p = if dist > 0.0 { uv.div(square_size).adds(0.001).floor().adds(0.5).mul(square_size) } else { uv };
    mix4(c.from(p), c.to(p), c.progress)
}

// License: MIT
// Author: Xaychru (ported by gre)
pub(crate) fn mosaic(c: &Ctx, uv: V2) -> [f32; 4] {
    #[allow(non_snake_case)]
    fn Rand(v: V2) -> f32 {
        if v.x == 0.0 && v.y == 0.0 { 0.0 } else { rand_cell(v) }
    }
    #[allow(non_snake_case)]
    fn Rotate(v: V2, a: f32) -> V2 {
        // mat2(cos a, -sin a, sin a, cos a) (column major) * v
        let (s, co) = (a.sin(), a.cos());
        V2::new(co * v.x + s * v.y, -s * v.x + co * v.y)
    }
    let cos_interpolation = |x: f32| -(x * PI).cos() / 2.0 + 0.5;
    let (endx, endy) = (c.p[0] as i32, c.p[1] as i32);
    let progress = c.progress;
    let p = uv.subs(0.5);
    let mut rp = p;
    let rpr = progress * 2.0 - 1.0;
    let z = -(rpr * rpr * 2.0) + 3.0;
    let az = z.abs();
    rp = rp.muls(az);
    let ci = cos_interpolation(progress);
    let k = ci * ci;
    rp = rp.add(V2::new(mix(0.5, endx as f32 + 0.5, k), mix(0.5, endy as f32 + 0.5, k)));
    let crp = rp.adds(0.001).floor();
    let mut mrp = rp.sub(crp);
    let on_end = crp.x as i32 == endx && crp.y as i32 == endy;
    if !on_end {
        let ang = ((Rand(crp) * 4.0) as i32) as f32 * 0.5 * PI;
        mrp = V2::splat(0.5).add(Rotate(mrp.sub(V2::splat(0.5)), ang));
    }
    if on_end || Rand(crp) > 0.5 { c.to(mrp) } else { c.from(mrp) }
}

// Author: gre
// License: MIT
pub(crate) fn ripple(c: &Ctx, uv: V2) -> [f32; 4] {
    let (amplitude, speed) = (c.p[0], c.p[1]);
    let progress = c.progress;
    let dir = uv.sub(V2::splat(0.5));
    let dist = dir.length();
    let offset = dir.muls((progress * dist * amplitude - progress * speed).sin() + 0.5).divs(30.0).muls(progress);
    mix4(c.from(uv.add(offset)), c.to(uv), smooth_step(0.2, 1.0, progress))
}

// Author: Paweł Płóciennik
// License: MIT
pub(crate) fn water_drop(c: &Ctx, p: V2) -> [f32; 4] {
    let (amplitude, speed) = (c.p[0], c.p[1]);
    let progress = c.progress;
    let dir = p.sub(V2::splat(0.5));
    let dist = dir.length();
    if dist > progress {
        return mix4(c.from(p), c.to(p), progress);
    }
    let offset = dir.muls((dist * amplitude - progress * speed).sin());
    mix4(c.from(p.add(offset)), c.to(p), progress)
}

// Author: gre
// License: MIT
pub(crate) fn circle_open(c: &Ctx, uv: V2) -> [f32; 4] {
    const SQRT_2: f32 = std::f32::consts::SQRT_2;
    let smoothness = c.p[0];
    let opening = c.p[1] > 0.5;
    let x = if opening { c.progress } else { 1.0 - c.progress };
    let m = smooth_step(-smoothness, 0.0, SQRT_2 * uv.sub(V2::splat(0.5)).length() - x * (1.0 + smoothness));
    mix4(c.from(uv), c.to(uv), if opening { 1.0 - m } else { m })
}

// Author: Ben Lucas
// License: MIT
pub(crate) fn star_wipe(c: &Ctx, uv: V2) -> [f32; 4] {
    const STAR_ANGLE: f32 = 1.256_637_1;
    let border_thickness = c.p[0];
    let star_rotation = c.p[1];
    let star_center = V2::new(c.p[2], c.p[3]);
    let border_color = [c.p[4], c.p[5], c.p[6], c.p[7]];
    let rotate = |v: V2, theta: f32| {
        let (cos_theta, sin_theta) = (theta.cos(), theta.sin());
        V2::new(cos_theta * v.x - sin_theta * v.y, sin_theta * v.x + cos_theta * v.y)
    };
    let in_star = |uv: V2, center: V2, radius: f32| {
        let mut uv_centered = uv.sub(center);
        uv_centered = rotate(uv_centered, star_rotation * STAR_ANGLE);
        let theta = atan2_safe(uv_centered.y, uv_centered.x) + PI;
        let uv_rotated = rotate(uv_centered, -STAR_ANGLE * ((theta / STAR_ANGLE).floor() + 0.5));
        let slope = 0.3;
        if uv_rotated.y > 0.0 { radius + uv_rotated.x * slope > uv_rotated.y } else { -radius - uv_rotated.x * slope < uv_rotated.y }
    };
    let progress_scaled = (2.0 * border_thickness + 1.0) * c.progress - border_thickness;
    if in_star(uv, star_center, progress_scaled) {
        c.to(uv)
    } else if in_star(uv, star_center, progress_scaled + border_thickness) {
        border_color
    } else {
        c.from(uv)
    }
}

// Author: Fernando Kuteken
// License: MIT
pub(crate) fn angular(c: &Ctx, uv: V2) -> [f32; 4] {
    let starting_angle = c.p[0];
    let offset = starting_angle * PI / 180.0;
    let angle = atan2_safe(uv.y - 0.5, uv.x - 0.5) + offset;
    let mut normalized_angle = (angle + PI) / (2.0 * PI);
    normalized_angle -= normalized_angle.floor();
    // GLSL step(edge, x): 0 when x < edge
    let k = if c.progress < normalized_angle { 0.0 } else { 1.0 };
    mix4(c.from(uv), c.to(uv), k)
}

// ---- batch 2 ----

// Author: hong
// License: MIT
pub(crate) fn book_flip(c: &Ctx, p: V2) -> [f32; 4] {
    let progress = c.progress;
    // Avoids dividing by zero when the page is edge-on (progress = 0.5).
    let non_zero = |x: f32| {
        if x.abs() < 1e-4 { if x < 0.0 { -1e-4 } else { 1e-4 } } else { x }
    };
    let skew_right = |p: V2| V2::new((p.x - progress) / non_zero(0.5 - progress) * 0.5, (p.y - 0.5) / (0.5 + progress * (p.x - 0.5) / 0.5) * 0.5 + 0.5);
    let skew_left = |p: V2| V2::new((p.x - 0.5) / non_zero(progress - 0.5) * 0.5 + 0.5, (p.y - 0.5) / (0.5 + (1.0 - progress) * (0.5 - p.x) / 0.5) * 0.5 + 0.5);
    let shade_val = 0.7f32.max((progress - 0.5).abs() * 2.0);
    let shade = |col: [f32; 4]| [col[0] * shade_val, col[1] * shade_val, col[2] * shade_val, col[3] * 1.0];
    let pr = step(1.0 - progress - 0.00001, p.x);
    if p.x < 0.5 - 0.00001 { mix4(c.from(p), shade(c.to(skew_left(p))), pr) } else { mix4(shade(c.from(skew_right(p))), c.to(p), pr) }
}

// Author: Adrian Purser
// License: MIT
pub(crate) fn bounce(c: &Ctx, uv: V2) -> [f32; 4] {
    let shadow_colour = [c.p[0], c.p[1], c.p[2], 1.0];
    let (shadow_alpha, shadow_height, bounces) = (c.p[3], c.p[4], c.p[5]);
    let time = c.progress;
    let stime = (time * PI / 2.0).sin();
    let phase = time * PI * bounces;
    let y = phase.cos().abs() * (1.0 - stime);
    let d = uv.y - y;
    let shadow = step(d, shadow_height) * (1.0 - mix(((d / shadow_height) * shadow_alpha) + (1.0 - shadow_alpha), 1.0, smooth_step(0.95, 1.0, c.progress)));
    mix4(mix4(c.to(uv), shadow_colour, shadow), c.from(V2::new(uv.x, uv.y + (1.0 - y))), step(d, 0.0))
}

// Author: gre
// License: MIT
pub(crate) fn burn(c: &Ctx, uv: V2) -> [f32; 4] {
    let color = [c.p[0], c.p[1], c.p[2]];
    let (a, b) = (c.from(uv), c.to(uv));
    let pr = c.progress;
    let a = [a[0] + pr * color[0], a[1] + pr * color[1], a[2] + pr * color[2], a[3] + 1.0];
    let b = [b[0] + (1.0 - pr) * color[0], b[1] + (1.0 - pr) * color[1], b[2] + (1.0 - pr) * color[2], b[3] + 1.0];
    alpha_at_most_1(mix4(a, b, pr))
}

// Author: mandubian
// License: MIT
pub(crate) fn butterfly_wave_scrawler(c: &Ctx, uv: V2) -> [f32; 4] {
    let (amplitude, waves, color_separation) = (c.p[0], c.p[1], c.p[2]);
    let compute = |p: V2, progress: f32, center: V2| {
        let o = p.muls((progress * amplitude).sin()).sub(center);
        // horizontal vector
        let h = V2::new(1.0, 0.0);
        // butterfly polar function (don't ask me why this one :))
        let theta = dot(o, h).clamp(-1.0, 1.0).acos() * waves;
        let s = ((2.0 * theta - PI) / 24.0).sin();
        let s2 = s * s;
        (theta.cos().exp() - 2.0 * (4.0 * theta).cos() + s2 * s2 * s) / 10.0
    };
    let p = uv;
    let progress = c.progress;
    let inv = 1.0 - progress;
    let disp = compute(p, progress, V2::splat(0.5));
    let tex_to = c.to(p.adds(inv * disp));
    let g = c.from(p.adds(progress * disp));
    let tex_from = [c.from(p.adds(progress * disp * (1.0 - color_separation)))[0], g[1], c.from(p.adds(progress * disp * (1.0 + color_separation)))[2], g[3]];
    std::array::from_fn(|k| tex_to[k] * progress + tex_from[k] * inv)
}

// Author: lql
// License: MIT
pub(crate) fn chessboard(c: &Ctx, uv: V2) -> [f32; 4] {
    let grid_num = c.p[0];
    let st = uv.muls(grid_num);
    let idx = st.adds(0.001).floor();
    let grid = st.sub(idx);
    let (a, b) = (c.from(uv), c.to(uv));
    let checker = glsl_mod(idx.x + idx.y, 2.0);
    let progress = c.progress;
    let mix_factor = if progress <= 0.5 {
        if checker > 0.5 { step(grid.x, progress * 2.0) } else { 0.0 }
    } else if checker < 0.5 {
        step(grid.x, (progress - 0.5) * 2.0)
    } else {
        1.0
    };
    mix4(a, b, mix_factor)
}

// Author: gre
// License: MIT
pub(crate) fn colorphase(c: &Ctx, uv: V2) -> [f32; 4] {
    let (a, b) = (c.from(uv), c.to(uv));
    std::array::from_fn(|k| mix(a[k], b[k], smooth_step(c.p[k], c.p[k + 4], c.progress)))
}

// License: MIT
// Author: pthrasher
pub(crate) fn crosshatch(c: &Ctx, p: V2) -> [f32; 4] {
    let center = V2::new(c.p[0], c.p[1]);
    let (threshold, fade_edge) = (c.p[2], c.p[3]);
    let progress = c.progress;
    let dist = distance(center, p) / threshold;
    let r = progress - c.rand_row(p, 4).min(c.rand_column(p, 5));
    let k = mix(0.0, mix(step(dist, r), 1.0, smooth_step(1.0 - fade_edge, 1.0, progress)), smooth_step(0.0, fade_edge, progress));
    mix4(c.from(p), c.to(p), k)
}

const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

// Author: gre
// License: MIT
pub(crate) fn cube(c: &Ctx, op: V2) -> [f32; 4] {
    let (persp, unzoom, reflection, floating) = (c.p[0], c.p[1], c.p[2], c.p[3]);
    let progress = c.progress;
    let project = |p: V2| V2::new(p.x * 1.0 + 0.0, p.y * -1.2 + -floating / 100.0);
    // p : the position; persp : the perspective in [ 0, 1 ]; center : the xcenter in [0, 1] \ 0.5 excluded
    let xskew = |p: V2, persp: f32, center: f32| {
        let x = mix(p.x, 1.0 - p.x, center);
        let dc = (center - 0.5).abs();
        let sgn = if center < 0.5 { 1.0 } else { -1.0 };
        V2::new(x, (p.y - 0.5 * (1.0 - persp) * x) / (1.0 + (persp - 1.0) * x))
            .sub(V2::new(0.5 - dc, 0.0))
            .mul(V2::new(0.5 / dc * sgn, 1.0))
            .add(V2::new(if center < 0.5 { 0.0 } else { 1.0 }, 0.0))
    };
    let uz = unzoom * 2.0 * (0.5 - (0.5 - progress).abs());
    let p = V2::new(-uz * 0.5 + (1.0 + uz) * op.x, -uz * 0.5 + (1.0 + uz) * op.y);
    let from_p = xskew(p.sub(V2::new(progress, 0.0)).div(V2::new(1.0 - progress, 1.0)), 1.0 - mix(progress, 0.0, persp), 0.0);
    let to_p = xskew(p.div(V2::new(progress, 1.0)), mix(progress.powf(2.0), 1.0, persp), 1.0);
    if in_bounds(from_p) {
        return c.from(from_p);
    } else if in_bounds(to_p) {
        return c.to(to_p);
    }
    let mut col = BLACK;
    let pfr = project(from_p);
    if in_bounds(pfr) {
        col = add4(col, mix4([0.0; 4], c.from(pfr), reflection * mix(1.0, 0.0, pfr.y)));
    }
    let pto = project(to_p);
    if in_bounds(pto) {
        col = add4(col, mix4([0.0; 4], c.to(pto), reflection * mix(1.0, 0.0, pto.y)));
    }
    alpha_at_most_1(col)
}

// Author: gre
// License: MIT
pub(crate) fn doorway(c: &Ctx, p: V2) -> [f32; 4] {
    let (reflection, perspective, depth) = (c.p[0], c.p[1], c.p[2]);
    let progress = c.progress;
    let project = |p: V2| V2::new(p.x * 1.0 + 0.0, p.y * -1.2 + -0.02);
    let mut pfr = V2::splat(-1.0);
    let middle_slit = 2.0 * (p.x - 0.5).abs() - progress;
    if middle_slit > 0.0 {
        let sgn = if p.x > 0.5 { -1.0 } else { 1.0 };
        pfr = p.add(V2::new(sgn * (0.5 * progress), sgn * 0.0));
        let d = 1.0 / (1.0 + perspective * progress * (1.0 - middle_slit));
        pfr.y -= d / 2.0;
        pfr.y *= d;
        pfr.y += d / 2.0;
    }
    let size = mix(1.0, depth, 1.0 - progress);
    let pto = p.add(V2::splat(-0.5)).mul(V2::splat(size)).add(V2::splat(0.5));
    if in_bounds(pfr) {
        return c.from(pfr);
    } else if in_bounds(pto) {
        return c.to(pto);
    }
    let mut col = BLACK;
    let pt = project(pto);
    if in_bounds(pt) {
        col = add4(col, mix4(BLACK, c.to(pt), reflection * mix(1.0, 0.0, pt.y)));
    }
    alpha_at_most_1(col)
}

/// FilmBurn's 50 blur directions: `cos` / `sin` of `degrees((i / 50.) * 360.)` (computed in
/// double precision, rounded to f32; the WGSL has the same table).
const FILM_BURN_BLUR: [(f32, f32); 50] = [
    (1.0, 0.0),
    (-0.5562213, -0.8310342),
    (-0.38123566, 0.9244779),
    (0.9803362, -0.19733462),
    (-0.70931876, -0.70488787),
    (-0.19142744, 0.98150676),
    (0.9221181, -0.38690853),
    (-0.834335, -0.55125767),
    (0.0062661488, 0.9999804),
    (0.8277723, -0.5610642),
    (-0.9267111, -0.37577465),
    (0.20347367, 0.97908044),
    (0.70060354, -0.71355075),
    (-0.98273224, -0.1850335),
    (0.39222997, 0.9198672),
    (0.5464278, -0.83750623),
    (-0.99992144, 0.012532052),
    (0.5664413, 0.82410204),
    (0.37041396, -0.9288668),
    (-0.9778884, 0.20912728),
    (0.7175868, 0.69646907),
    (0.1793523, -0.9837849),
    (-0.9171969, 0.3984342),
    (0.8409138, 0.5411691),
    (-0.018309265, -0.9998324),
    (-0.8205365, 0.5715942),
    (0.93152523, 0.36367676),
    (-0.21525078, -0.9765588),
    (-0.69231135, 0.7215989),
    (0.9849739, 0.17270331),
    (-0.40283334, -0.91527337),
    (-0.53547686, 0.8445499),
    (0.9996859, -0.025062134),
    (-0.5763256, -0.8172202),
    (-0.35828844, 0.9336109),
    (0.97508264, -0.22184195),
    (-0.725587, -0.68813044),
    (-0.16700937, 0.9859553),
    (0.91253155, -0.4090063),
    (-0.84814745, -0.52976024),
    (0.029861666, 0.99955404),
    (0.8138766, -0.58103776),
    (-0.9356655, -0.35288817),
    (0.22747216, 0.97378457),
    (0.6825004, -0.7308852),
    (-0.98706084, -0.16034602),
    (0.41427198, 0.91015315),
    (0.5256819, -0.85068125),
    (-0.99932957, 0.03661239),
    (0.58493876, 0.8110775),
];

// Author: Anastasia Dunbar
// License: MIT
pub(crate) fn film_burn(c: &Ctx, uv: V2) -> [f32; 4] {
    let seed = c.p[0].to_bits();
    let progress = c.progress;
    let sigmoid = |x: f32, a: f32| {
        let mut b = (x * 2.0).powf(a) / 2.0;
        if x > 0.5 {
            b = 1.0 - (2.0 - (x * 2.0)).powf(a) / 2.0;
        }
        b
    };
    // upstream: rand(float co) = fract(sin((co * 24.9898) + Seed) * 43758.5453)
    let rand1 = |co: f32| hash(co.to_bits() as i32, 7, seed);
    let apow = |a: f32, b: f32| a.abs().powf(b) * sign(b);
    let smooth_mix = |a: f32, b: f32, cc: f32| mix(a, b, sigmoid(cc, 2.0));
    // upstream: fract(sin(dot(co, vec2(12.9898 + k * .5, 78.233 + Seed))) * 43758.5453)
    let rand_lattice = |co: V2, k: f32| hash(co.x as i32, co.y as i32, k.to_bits() ^ seed.wrapping_mul(0x9E37_79B1));
    let random = |co: V2, shft: f32| {
        let co = co.adds(10.0);
        smooth_mix(rand_lattice(co, shft.floor()), rand_lattice(co, (shft + 1.0).floor()), fract(shft))
    };
    let smooth_random = |co: V2, shft: f32| {
        smooth_mix(
            smooth_mix(random(co.floor(), shft), random(co.add(V2::new(1.0, 0.0)).floor(), shft), fract(co.x)),
            smooth_mix(random(co.add(V2::new(0.0, 1.0)).floor(), shft), random(co.add(V2::new(1.0, 1.0)).floor(), shft), fract(co.x)),
            fract(co.y),
        )
    };
    let p = uv;
    // the three channels of upstream's `f` are equal until `pow3`
    let mut f = 0.0f32;
    for k in 0..13 {
        let i = k as f32;
        f += ((p.x * rand1(i) * 6.0) + (progress * 8.0) + rand1(i + 1.43)).sin() * ((p.y * rand1(i + 4.4) * 6.0) + (progress * 6.0) + rand1(i + 2.4)).sin();
        let spot = V2::new(smooth_random(V2::splat(progress * 1.3), i + 1.0), smooth_random(V2::splat(progress * 0.5), i + 6.25));
        f += 1.0 - (p.sub(spot).length() * mix(20.0, 70.0, rand1(i))).clamp(0.0, 1.0);
    }
    f += 4.0;
    f /= 11.0;
    let s = (progress * PI).sin();
    let f = [apow(f * 1.0, 1.0) * s, apow(f * 0.7, 2.0 - s) * s, apow(f * 0.6, 1.3) * s];

    let mut q = p.subs(0.5);
    q = q.muls(1.0 + (smooth_random(V2::splat(progress * 5.0), 6.3) * s * 0.05));
    q = q.adds(0.5);

    let mut blurred = [0.0f32; 4];
    let bluramount = s * 0.03;
    let k = sigmoid(progress, 10.0);
    for (i, d) in FILM_BURN_BLUR.iter().enumerate() {
        let dir = V2::new(d.0, d.1).muls(c.rand_pixel_seeded(uv, i as u32 + 16) + bluramount);
        let uv2 = q.add(dir.muls(bluramount));
        blurred = add4(blurred, mix4(c.from(uv2), c.to(uv2), k));
    }
    let blurred = blurred.map(|v| v / 50.0);
    [blurred[0] + f[0], blurred[1] + f[1], blurred[2] + f[2], blurred[3] + 0.0]
}

// Author: Matt DesLauriers
// License: MIT
pub(crate) fn glitch_displace(c: &Ctx, uv: V2) -> [f32; 4] {
    fn displace(tex: [f32; 4], tex_coord: V2, dot_depth: f32, texture_depth: f32, strength: f32) -> V2 {
        let dt = tex.map(|v| v * 1.0);
        let dis: [f32; 4] = std::array::from_fn(|k| dt[k] * dot_depth + 1.0 - tex[k] * texture_depth);
        let dx = (dis[0] - 1.0 + texture_depth * dot_depth) * strength;
        let dy = (dis[1] - 1.0 + texture_depth * dot_depth) * strength;
        V2::new(tex_coord.x + dx - 0.0, tex_coord.y + dy)
    }
    fn ease1(t: f32) -> f32 {
        if t == 0.0 || t == 1.0 {
            t
        } else if t < 0.5 {
            0.5 * 2f32.powf((20.0 * t) - 10.0)
        } else {
            -0.5 * 2f32.powf(10.0 - (t * 20.0)) + 1.0
        }
    }
    fn ease2(t: f32) -> f32 {
        if t == 1.0 { t } else { 1.0 - 2f32.powf(-10.0 * t) }
    }
    let p = uv;
    let progress = c.progress;
    let color1 = c.from(p);
    let color2 = c.to(p);
    let disp = displace(color1, p, 0.33, 0.7, 1.0 - ease1(progress));
    let disp2 = displace(color2, p, 0.33, 0.5, ease2(progress));
    let d_color1 = c.to(disp);
    let d_color2 = c.from(disp2);
    let val = ease1(progress);
    let m: [f32; 4] = std::array::from_fn(|k| d_color2[k].min(d_color1[k]));
    let gray = m[0] * 0.299 + m[1] * 0.587 + m[2] * 0.114;
    let d_color2 = [gray * 2.0, gray * 2.0, gray * 2.0, 1.0 * 2.0];
    let color1 = mix4(color1, d_color2, smooth_step(0.0, 0.5, progress));
    let color2 = mix4(color2, d_color1, 1.0 - smooth_step(0.5, 1.0, progress));
    alpha_at_most_1(mix4(color1, color2, val))
}

// Author: Gunnar Roth
// License: MIT
pub(crate) fn glitch_memories(c: &Ctx, p: V2) -> [f32; 4] {
    let progress = c.progress;
    let blk = p.divs(16.0).floor();
    let mut uv_noise = blk.divs(64.0);
    uv_noise = uv_noise.add(V2::new((progress * 1200.0).floor(), (progress * 3500.0).floor()).divs(64.0));
    let dist = if progress > 0.0 { V2::new(fract(uv_noise.x), fract(uv_noise.y)).subs(0.5).muls(0.3).muls(1.0 - progress) } else { V2::splat(0.0) };
    let red = p.add(dist.muls(0.2));
    let green = p.add(dist.muls(0.3));
    let blue = p.add(dist.muls(0.5));
    let g = mix4(c.from(green), c.to(green), progress);
    [mix4(c.from(red), c.to(red), progress)[0], g[1], mix4(c.from(blue), c.to(blue), progress)[2], g[3]]
}

// Author: Fernando Kuteken
// License: MIT
pub(crate) fn hexagonalize(c: &Ctx, uv: V2) -> [f32; 4] {
    const SQRT3: f32 = 1.732_050_807_568_877_2_f64 as f32;
    const SQRT3_3: f32 = 0.577_350_269_189_625_8_f64 as f32;
    const SQRT3_2: f32 = 0.866_025_403_784_438_6_f64 as f32;
    const THIRD: f32 = (1.0f64 / 3.0) as f32;
    const TWO_THIRDS: f32 = (2.0f64 / 3.0) as f32;
    let steps = c.p[0] as i32;
    let horizontal_hexagons = c.p[1];
    // (q, r) of the hexagon nearest to cube coordinates (q, r, -q - r)
    let round_hexagon = |q: f32, r: f32| {
        let s = -q - r;
        let (mut rq, mut rr) = ((q + 0.5).floor(), (r + 0.5).floor());
        let rs = (s + 0.5).floor();
        let (dq, dr, ds) = ((rq - q).abs(), (rr - r).abs(), (rs - s).abs());
        if dq > dr && dq > ds {
            rq = -rr - rs;
        } else if dr > ds {
            rr = -rq - rs;
        }
        (rq, rr)
    };
    let hexagon_from_point = |point: V2, size: f32| {
        let point = V2::new(point.x, point.y / c.ratio).subs(0.5).divs(size);
        let q = SQRT3_3 * point.x + (-THIRD) * point.y;
        let r = 0.0 * point.x + TWO_THIRDS * point.y;
        round_hexagon(q, r)
    };
    let point_from_hexagon = |(q, r): (f32, f32), size: f32| {
        let x = (SQRT3 * q + SQRT3_2 * r) * size + 0.5;
        let y = (0.0 * q + 1.5 * r) * size + 0.5;
        V2::new(x, y * c.ratio)
    };
    let mut dist = 2.0 * c.progress.min(1.0 - c.progress);
    if steps > 0 {
        dist = (dist * steps as f32).ceil() / steps as f32;
    }
    let size = SQRT3_3 * dist / horizontal_hexagons;
    let point = if dist > 0.0 { point_from_hexagon(hexagon_from_point(uv, size), size) } else { uv };
    mix4(c.from(point), c.to(point), c.progress)
}

// Author: nwoeanhinnogaehr
// License: MIT
pub(crate) fn kaleidoscope(c: &Ctx, uv: V2) -> [f32; 4] {
    let (speed, angle, power) = (c.p[0], c.p[1], c.p[2]);
    let progress = c.progress;
    let mut p = uv;
    let q = p;
    let mut t = progress.powf(power) * speed;
    p = p.subs(0.5);
    for _ in 0..7 {
        let (s, co) = (t.sin(), t.cos());
        p = V2::new(s * p.x + co * p.y, s * p.y - co * p.x);
        t += angle;
        p = V2::new((glsl_mod(p.x, 2.0) - 1.0).abs(), (glsl_mod(p.y, 2.0) - 1.0).abs());
    }
    mix4(mix4(c.from(q), c.to(q), progress), mix4(c.from(p), c.to(p), progress), 1.0 - 2.0 * (progress - 0.5).abs())
}

/// Ian McEwan's (Ashima Arts, MIT) 2-D simplex noise, as luminance_melt carries it.
fn snoise(v: V2) -> f32 {
    const CX: f32 = 0.211_324_865_405_187_f64 as f32; // (3.0-sqrt(3.0))/6.0
    const CY: f32 = 0.366_025_403_784_439_f64 as f32; // 0.5*(sqrt(3.0)-1.0)
    const CZ: f32 = -0.577_350_269_189_626_f64 as f32; // -1.0 + 2.0 * C.x
    const CW: f32 = 0.024_390_243_902_439_f64 as f32; // 1.0 / 41.0
    const INV289: f32 = (1.0f64 / 289.0) as f32;
    const K1: f32 = 1.792_842_914_001_59_f64 as f32;
    const K2: f32 = 0.853_734_720_953_14_f64 as f32;
    let mod289 = |x: f32| x - (x * INV289).floor() * 289.0;
    let permute = |x: f32| mod289(((x * 34.0) + 1.0) * x);
    // First corner
    let d = dot(v, V2::splat(CY));
    let i = V2::new((v.x + d).floor(), (v.y + d).floor());
    let di = dot(i, V2::splat(CX));
    let x0 = V2::new(v.x - i.x + di, v.y - i.y + di);
    // Other corners
    let i1 = if x0.x > x0.y { V2::new(1.0, 0.0) } else { V2::new(0.0, 1.0) };
    let x12 = [x0.x + CX - i1.x, x0.y + CX - i1.y, x0.x + CZ, x0.y + CZ];
    // Permutations
    let i = V2::new(mod289(i.x), mod289(i.y));
    let py = [permute(i.y + 0.0), permute(i.y + i1.y), permute(i.y + 1.0)];
    let p = [permute(py[0] + i.x + 0.0), permute(py[1] + i.x + i1.x), permute(py[2] + i.x + 1.0)];
    let mut m = [(0.5 - dot(x0, x0)).max(0.0), (0.5 - (x12[0] * x12[0] + x12[1] * x12[1])).max(0.0), (0.5 - (x12[2] * x12[2] + x12[3] * x12[3])).max(0.0)];
    m = m.map(|v| v * v);
    m = m.map(|v| v * v);
    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    let x = p.map(|v| 2.0 * fract(v * CW) - 1.0);
    let h = x.map(|v| v.abs() - 0.5);
    let a0: [f32; 3] = std::array::from_fn(|k| x[k] - (x[k] + 0.5).floor());
    // Normalise gradients implicitly by scaling m
    for (k, mk) in m.iter_mut().enumerate() {
        *mk *= K1 - K2 * (a0[k] * a0[k] + h[k] * h[k]);
    }
    // Compute final noise value at P
    let g = [a0[0] * x0.x + h[0] * x0.y, a0[1] * x12[0] + h[1] * x12[1], a0[2] * x12[2] + h[2] * x12[3]];
    130.0 * (m[0] * g[0] + m[1] * g[1] + m[2] * g[2])
}

// Author: 0gust1
// License: MIT
pub(crate) fn luminance_melt(c: &Ctx, uv: V2) -> [f32; 4] {
    let direction = c.p[0] > 0.5;
    let l_threshold = c.p[1];
    let above = c.p[2] > 0.5;
    let luminance = |color: [f32; 4]| color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114;
    let center = V2::new(1.0, if direction { 1.0 } else { 0.0 });
    let p = uv;
    let progress = c.progress;
    let x = progress;
    let dist = distance(center, p) - progress * snoise(V2::new(p.x, 0.0)).exp();
    let r = x - c.rand_column(p, 3);
    let from = c.from(p);
    let lum = luminance(from);
    let melts = if above { lum > l_threshold } else { lum < l_threshold };
    let m = if dist <= r && melts { 1.0 } else { progress * progress * progress };
    mix4(from, c.to(p), m)
}

// Author: Rich Harris
// License: MIT
pub(crate) fn perlin(c: &Ctx, uv: V2) -> [f32; 4] {
    let (scale, smoothness, seed) = (c.p[0], c.p[1], c.p[2].to_bits());
    let random = |co: V2| rand_cell_seeded(co, seed);
    // 2D Noise based on Morgan McGuire @morgan3d
    let noise = |st: V2| {
        let i = st.floor();
        let f = st.sub(i);
        // Four corners in 2D of a tile
        let a = random(i);
        let b = random(i.add(V2::new(1.0, 0.0)));
        let cc = random(i.add(V2::new(0.0, 1.0)));
        let d = random(i.add(V2::new(1.0, 1.0)));
        // Cubic Hermine Curve.  Same as SmoothStep()
        let u = V2::new(f.x * f.x * (3.0 - 2.0 * f.x), f.y * f.y * (3.0 - 2.0 * f.y));
        mix(a, b, u.x) + (cc - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y
    };
    let n = noise(uv.muls(scale));
    let p = mix(-smoothness, 1.0 + smoothness, c.progress);
    let q = smooth_step(p - smoothness, p + smoothness, n);
    mix4(c.from(uv), c.to(uv), 1.0 - q)
}

// Author: Mr Speaker
// License: MIT
#[allow(clippy::approx_constant, reason = "upstream's 3.1415, kept for the same look")]
pub(crate) fn pinwheel(c: &Ctx, p: V2) -> [f32; 4] {
    let speed = c.p[0];
    let circ_pos = atan2_safe(p.y - 0.5, p.x - 0.5) + c.progress * speed;
    let mod_pos = glsl_mod(circ_pos, 3.1415 / 4.0);
    let sgn = sign(c.progress - mod_pos);
    mix4(c.to(p), c.from(p), step(sgn, 0.5))
}

// Author: Fernando Kuteken
// License: MIT
pub(crate) fn polar_function(c: &Ctx, uv: V2) -> [f32; 4] {
    let segments = c.p[0] as i32;
    let angle = atan2_safe(uv.y - 0.5, uv.x - 0.5) - 0.5 * PI;
    let radius = ((segments as f32 * angle).cos() + 4.0) / 4.0;
    let difference = uv.sub(V2::splat(0.5)).length();
    if difference > radius * c.progress { c.from(uv) } else { c.to(uv) }
}

// Author: bobylito
// License: MIT
pub(crate) fn polka_dots_curtain(c: &Ctx, uv: V2) -> [f32; 4] {
    let dots = c.p[0];
    let center = V2::new(c.p[1], c.p[2]);
    let cell = uv.muls(dots);
    let next_image = distance(V2::new(fract(cell.x), fract(cell.y)), V2::splat(0.5)) < (c.progress / distance(uv, center).max(1e-6));
    if next_image { c.to(uv) } else { c.from(uv) }
}

// Author: gre
// License: MIT
pub(crate) fn random_squares(c: &Ctx, p: V2) -> [f32; 4] {
    let size = V2::new(c.p[0], c.p[1]);
    let smoothness = c.p[2];
    let r = rand_cell(size.mul(p).adds(0.001).floor());
    let m = 1.0 - smooth_step(-smoothness, 0.0, r - (c.progress * (1.0 + smoothness)));
    mix4(c.from(p), c.to(p), m)
}

// Author: gre
// License: MIT
pub(crate) fn squeeze(c: &Ctx, uv: V2) -> [f32; 4] {
    let color_separation = c.p[0];
    let y = 0.5 + (uv.y - 0.5) / (1.0 - c.progress);
    if !(0.0..=1.0).contains(&y) {
        return c.to(uv);
    }
    let fp = V2::new(uv.x, y);
    let off = V2::new(c.progress * 0.0, c.progress * color_separation);
    let col = c.from(fp);
    let cn = c.from(fp.sub(off));
    let cp = c.from(fp.add(off));
    [cn[0], col[1], cp[2], col[3]]
}

// Author: gre
// License: MIT
pub(crate) fn swap(c: &Ctx, p: V2) -> [f32; 4] {
    let (reflection, perspective, depth) = (c.p[0], c.p[1], c.p[2]);
    let progress = c.progress;
    let project = |p: V2| V2::new(p.x * 1.0 + 0.0, p.y * -1.2 + -0.02);
    let mut size = mix(1.0, depth, progress);
    let mut persp = perspective * progress;
    let pfr = V2::new((p.x + -0.0) * (size / (1.0 - perspective * progress)) + 0.0, (p.y + -0.5) * (size / (1.0 - size * persp * p.x)) + 0.5);
    size = mix(1.0, depth, 1.0 - progress);
    persp = perspective * (1.0 - progress);
    let pto = V2::new((p.x + -1.0) * (size / (1.0 - perspective * (1.0 - progress))) + 1.0, (p.y + -0.5) * (size / (1.0 - size * persp * (0.5 - p.x))) + 0.5);
    if progress < 0.5 {
        if in_bounds(pfr) {
            return c.from(pfr);
        }
        if in_bounds(pto) {
            return c.to(pto);
        }
    }
    if in_bounds(pto) {
        return c.to(pto);
    }
    if in_bounds(pfr) {
        return c.from(pfr);
    }
    let mut col = BLACK;
    let pf = project(pfr);
    if in_bounds(pf) {
        col = add4(col, mix4(BLACK, c.from(pf), reflection * mix(1.0, 0.0, pf.y)));
    }
    let pt = project(pto);
    if in_bounds(pt) {
        col = add4(col, mix4(BLACK, c.to(pt), reflection * mix(1.0, 0.0, pt.y)));
    }
    alpha_at_most_1(col)
}

// License: MIT
// Author: Sergey Kosarevsky
#[allow(clippy::approx_constant, reason = "upstream's 3.14159, kept for the same look")]
pub(crate) fn swirl(c: &Ctx, uv: V2) -> [f32; 4] {
    let radius = 1.0f32;
    let t = c.progress;
    let mut uv = uv.sub(V2::splat(0.5));
    let dist = uv.length();
    if dist < radius {
        let percent = (radius - dist) / radius;
        let a = if t <= 0.5 { mix(0.0, 1.0, t / 0.5) } else { mix(1.0, 0.0, (t - 0.5) / 0.5) };
        let theta = percent * percent * a * 8.0 * 3.14159;
        let (s, co) = (theta.sin(), theta.cos());
        uv = V2::new(dot(uv, V2::new(co, -s)), dot(uv, V2::new(s, co)));
    }
    let uv = uv.adds(0.5);
    mix4(c.from(uv), c.to(uv), t)
}

// Author: Brandon Anzaldi
// License: MIT
pub(crate) fn tv_static(c: &Ctx, p: V2) -> [f32; 4] {
    let offset = c.p[0];
    if c.progress < offset {
        c.from(p)
    } else if c.progress > (1.0 - offset) {
        c.to(p)
    } else {
        let n = c.rand_pixel_seeded(p, c.progress.to_bits());
        [n, n, n, 1.0]
    }
}

// Author: gre
// License: MIT
pub(crate) fn wind(c: &Ctx, uv: V2) -> [f32; 4] {
    let (size, reversed) = (c.p[0], c.p[1] > 0.5);
    let x = if reversed { 1.0 - uv.x } else { uv.x };
    let r = c.rand_row(uv, 6);
    let m = 1.0 - smooth_step(-size, 0.0, x * (1.0 - size) + size * r - (c.progress * (1.0 + size)));
    mix4(c.from(uv), c.to(uv), m)
}

// Author: gre
// License: MIT
pub(crate) fn window_slice(c: &Ctx, p: V2) -> [f32; 4] {
    let (count, smoothness) = (c.p[0], c.p[1]);
    let pr = smooth_step(-smoothness, 0.0, p.x - c.progress * (1.0 + smoothness));
    let s = step(pr, fract(count * p.x + 0.001));
    mix4(c.from(p), c.to(p), s)
}

// Author: Travis Fischer
// License: MIT
pub(crate) fn displacement(c: &Ctx, uv: V2) -> [f32; 4] {
    let strength = c.p[0];
    let displacement = c.map(uv) * strength;
    let uv_from = V2::new(uv.x + c.progress * displacement, uv.y);
    let uv_to = V2::new(uv.x - (1.0 - c.progress) * displacement, uv.y);
    mix4(c.from(uv_from), c.to(uv_to), c.progress)
}

// Author: gre
// License: MIT
pub(crate) fn luma(c: &Ctx, uv: V2) -> [f32; 4] {
    mix4(c.to(uv), c.from(uv), step(c.progress, c.map(uv)))
}

// ---------------------------------------------------------------------------------------------
// The transition engines (original work, MIT OR Apache-2.0): `luma_wipe.wgsl`, `blend_dissolve.wgsl`

/// The luma-mask wipe engine.
pub(crate) fn luma_wipe(c: &Ctx, uv: V2) -> [f32; 4] {
    let softness = c.p[0];
    let mut m = c.map(uv);
    if c.p[1] > 0.5 {
        m = 1.0 - m;
    }
    let k = smooth_step(m, m + softness, c.progress * (1.0 + softness));
    mix4(c.from(uv), c.to(uv), k)
}

/// The incoming picture `b` composited onto the outgoing `a` with blend mode `mode` (the
/// compositor's own formula, `crate::blend::composite_at`).
fn blended(a: [f32; 4], b: [f32; 4], mode: u32) -> [f32; 4] {
    use filmcraft_color::{linear_to_srgb, srgb_to_linear};
    let (sa, da) = (b[3], a[3]);
    if sa <= 0.0 {
        return a;
    }
    if mode == 0 || da <= 0.0 {
        let k = 1.0 - sa;
        return [b[0] + a[0] * k, b[1] + a[1] * k, b[2] + a[2] * k, b[3] + a[3] * k];
    }
    let cs = [linear_to_srgb(b[0] / sa), linear_to_srgb(b[1] / sa), linear_to_srgb(b[2] / sa)];
    let cb = [linear_to_srgb(a[0] / da), linear_to_srgb(a[1] / da), linear_to_srgb(a[2] / da)];
    let bl = crate::blend::blend_rgb(crate::blend::Blend::from_index(mode), cb, cs);
    let mut o = [0.0; 4];
    for k in 0..3 {
        let mixed = srgb_to_linear(bl[k].clamp(0.0, 1.0));
        o[k] = b[k] * (1.0 - da) + a[k] * (1.0 - sa) + sa * da * mixed;
    }
    o[3] = sa + da - sa * da;
    o
}

/// The blend-mode dissolve engine.
pub(crate) fn blend_dissolve(c: &Ctx, uv: V2) -> [f32; 4] {
    let (a, b) = (c.from(uv), c.to(uv));
    let mode = c.p[0] as u32;
    let strength = c.p[1];
    let fade = mix4(a, b, c.progress);
    if mode == 1 {
        let dither = if c.rand_pixel(uv) < c.progress { b } else { a };
        return mix4(fade, dither, strength);
    }
    let k = strength * 4.0 * c.progress * (1.0 - c.progress);
    mix4(fade, blended(a, b, mode), k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glsl_helpers_match_their_definitions() {
        assert_eq!(mix(2.0, 4.0, 0.0), 2.0);
        assert_eq!(mix(2.0, 4.0, 1.0), 4.0);
        assert_eq!(smooth_step(0.0, 0.0, -1.0), 0.0);
        assert_eq!(smooth_step(0.0, 0.0, 0.0), 1.0);
        assert_eq!(atan2_safe(0.0, 0.0), 0.0);
        assert_eq!(atan2_safe(1.0, 0.0), PI / 2.0);
        // Rotate by a quarter turn: GLSL's column-major mat2(cos, -sin, sin, cos) * (1, 0)
        let v = V2::new(1.0, 0.0);
        let (s, co) = ((PI / 2.0).sin(), (PI / 2.0).cos());
        let r = V2::new(co * v.x + s * v.y, -s * v.x + co * v.y);
        assert!((r.x - 0.0).abs() < 1e-6 && (r.y + 1.0).abs() < 1e-6, "{r:?}");
    }
}
