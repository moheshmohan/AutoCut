//! Rust references (CPU oracles) of the ported GL Transitions: each function is its WGSL
//! shader (`shader/<name>.wgsl`) written again in Rust, operation for operation, on the same host
//! contract (see `prelude.wgsl`). Export renders these; the GPU path is tested against them.
//!
//! The transitions are hand-ported from gl-transitions
//! (<https://github.com/gl-transitions/gl-transitions>, MIT; licence text in
//! `shader/LICENSE-gl-transitions.txt`); each function keeps its upstream author and licence.

use std::f32::consts::PI;

use crate::image::Image;

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
}

impl Ctx<'_> {
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
        let x = (uv.x * self.w).floor();
        let y = ((1.0 - uv.y) * self.h).floor();
        super::super::hash(x as i32, y as i32, 1)
    }
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
