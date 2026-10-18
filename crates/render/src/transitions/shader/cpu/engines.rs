//! Rust references of FilmCraft's own content-aware engines (original work, MIT OR Apache-2.0):
//! `flow_engine.wgsl` and `content_wipe.wgsl`, operation for operation.

use super::{Ctx, V2, mix4, smooth_step};
use crate::transitions::{bell, ease_in_out, hash};

#[inline]
fn field_xy(c: &Ctx, x: V2) -> V2 {
    let f = c.field(x.x, x.y);
    V2::new(f[0], f[1])
}
#[inline]
fn field_zw(c: &Ctx, x: V2) -> V2 {
    let f = c.field(x.x, x.y);
    V2::new(f[2], f[3])
}
#[inline]
fn from_px(c: &Ctx, x: V2) -> [f32; 4] {
    c.outgoing_px(x.x, x.y)
}
#[inline]
fn to_px(c: &Ctx, x: V2) -> [f32; 4] {
    c.incoming_px(x.x, x.y)
}
#[inline]
fn scale3(m: [f32; 4], k: f32) -> [f32; 4] {
    [m[0] * k, m[1] * k, m[2] * k, m[3]]
}

/// `flow_src_a`: where the outgoing picture's content shown at `x` comes from, `s` of the way.
fn flow_src_a(c: &Ctx, x: V2, s: f32) -> V2 {
    let first = x.sub(field_xy(c, x).muls(s));
    x.sub(field_xy(c, first).muls(s))
}

/// `flow_src_b`: the same for the incoming picture, `s` of the way back.
fn flow_src_b(c: &Ctx, x: V2, s: f32) -> V2 {
    let first = x.sub(field_zw(c, x).muls(s));
    x.sub(field_zw(c, first).muls(s))
}

fn dissolve_at(c: &Ctx, t: f32) -> f32 {
    let half = 0.5 * c.p[2];
    smooth_step(0.5 - half, 0.5 + half, t)
}

fn morph(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let k = c.p[1];
    let a = from_px(c, flow_src_a(c, x, t * k));
    let b = to_px(c, flow_src_b(c, x, (1.0 - t) * k));
    mix4(a, b, dissolve_at(c, t))
}

fn doppler(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let centre = V2::new(c.p[8], c.p[9]);
    let env = bell(t);
    let q = centre.add(x.sub(centre).divs(1.0 + c.p[4] * env));
    let f = c.field(q.x, q.y);
    let speed = V2::new(f[0] - f[2], f[1] - f[3]).length() * 0.5 / c.w.max(c.h);
    let reach = c.p[3] * env * (0.15 + (speed * 20.0).min(1.0) * 0.85) * 0.35;
    let mut acc = [0.0f32; 4];
    for i in 0..12 {
        let s = i as f32 / 11.0;
        let v = morph(c, q.add(centre.sub(q).muls(reach * s)), t);
        for k in 0..4 {
            acc[k] += v[k];
        }
    }
    acc.map(|v| v / 12.0)
}

fn hype(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let r = c.p[3].max(0.01);
    let th = (r * (2.0 * t - 1.0)).tanh();
    let ramped = 0.5 + 0.5 * th / r.tanh();
    let speed = 1.0 - th * th;
    let centre = V2::new(c.w * 0.5, c.h * 0.5);
    let q = centre.add(x.sub(centre).divs(1.0 + c.p[7] * speed));
    let f = c.field(q.x, q.y);
    let along = V2::new(f[0] - f[2], f[1] - f[3]).muls(0.5 * c.p[1]);
    let radial = q.sub(centre).muls(0.12);
    let len = c.p[4] * speed;
    let mut acc = [0.0f32; 4];
    for i in 0..10 {
        let s = i as f32 / 9.0 - 0.5;
        let v = morph(c, q.add(along.add(radial).muls(len * s)), ramped);
        for k in 0..4 {
            acc[k] += v[k];
        }
    }
    let m = acc.map(|v| v / 10.0);
    let d = (t - 0.5) / 0.12;
    let pulse = 1.0 + c.p[5] * (-d * d).exp();
    scale3(m, pulse)
}

fn glitch(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let base = morph(c, x, t);
    let d = (t - 0.5) / 0.14;
    let g = c.p[4] * (-d * d).exp();
    let bs = c.p[3].max(2.0);
    let cell = x.divs(bs).adds(0.001).floor();
    let seed = (c.p[10] as u32).wrapping_mul(977).wrapping_add(((t * 24.0).floor() as u32).wrapping_mul(7919));
    let r = hash(cell.x as i32, cell.y as i32, seed.wrapping_add(11));
    if r >= g {
        return base;
    }
    let centre = cell.adds(0.5).muls(bs);
    let f = c.field(centre.x, centre.y);
    let smear = c.p[6] * c.p[1];
    let jitter = (hash(cell.x as i32, cell.y as i32, seed.wrapping_add(12)) - 0.5) * bs;
    let split = V2::new(c.p[5], 0.0);
    if t < 0.5 {
        let at = x.sub(V2::new(f[0], f[1]).muls(t * smear)).add(V2::new(jitter, 0.0));
        let mid = from_px(c, at);
        return [from_px(c, at.add(split))[0], mid[1], from_px(c, at.sub(split))[2], mid[3]];
    }
    let at = x.sub(V2::new(f[2], f[3]).muls((1.0 - t) * smear)).add(V2::new(jitter, 0.0));
    let mid = to_px(c, at);
    [to_px(c, at.add(split))[0], mid[1], to_px(c, at.sub(split))[2], mid[3]]
}

fn bands(c: &Ctx, x: V2, t: f32, inv: (f32, f32)) -> [f32; 4] {
    let n = c.p[3].max(1.0);
    let horizontal = c.p[4] < 0.5;
    let stagger = c.p[5].clamp(0.0, 0.9);
    let (along, across, len) = if horizontal { (x.x, x.y * inv.1, c.w) } else { (x.y, x.x * inv.0, c.h) };
    let band = (across * n + 0.001).floor().clamp(0.0, n - 1.0);
    let dir = if super::glsl_mod(band, 2.0) < 0.5 { 1.0 } else { -1.0 };
    let delay = stagger * band / (n - 1.0).max(1.0);
    let q = ((t - delay) / (1.0 - stagger)).clamp(0.0, 1.0);
    let a_along = along - dir * ease_in_out(q) * len;
    let k = c.p[1];
    let at = |v: f32| if horizontal { V2::new(v, x.y) } else { V2::new(x.x, v) };
    if a_along >= 0.0 && a_along < len {
        return from_px(c, flow_src_a(c, at(a_along), t * k));
    }
    let b_along = a_along + dir * len;
    to_px(c, flow_src_b(c, at(b_along), (1.0 - t) * k))
}

fn datamosh(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let bs = c.p[3].max(2.0);
    let cell = x.divs(bs).adds(0.001).floor();
    let tau = 0.3 + c.p[5] * 0.55 * hash(cell.x as i32, cell.y as i32, (c.p[10] as u32).wrapping_mul(977).wrapping_add(31));
    let wb = smooth_step(tau - 0.06, tau + 0.06, t);
    let k = c.p[1];
    let drag = c.p[4] * k * t * t;
    let ca = from_px(c, x.sub(field_xy(c, x).muls(drag)));
    let cb = to_px(c, flow_src_b(c, x, (1.0 - t) * k));
    mix4(ca, cb, wb)
}

/// The flow engine (`flow_engine.wgsl`). `inv` is the host's 1 / width, 1 / height.
pub(crate) fn flow_engine_at(c: &Ctx, uv: V2, inv: (f32, f32)) -> [f32; 4] {
    let (px, py) = c.uv_px(uv);
    let x = V2::new(px, py);
    let t = c.progress;
    match c.p[0] as u32 {
        1 => doppler(c, x, t),
        2 => hype(c, x, t),
        3 => glitch(c, x, t),
        4 => bands(c, x, t, inv),
        5 => datamosh(c, x, t),
        _ => morph(c, x, t),
    }
}

/// The flow engine with the host's 1 / width, 1 / height worked out from the frame size the way
/// the host rounds them (see `Host::inv`).
pub(crate) fn flow_engine(c: &Ctx, uv: V2) -> [f32; 4] {
    flow_engine_at(c, uv, (1.0 / c.w, 1.0 / c.h))
}

/// The content-aware wipe (`content_wipe.wgsl`).
pub(crate) fn content_wipe(c: &Ctx, uv: V2) -> [f32; 4] {
    let softness = c.p[0];
    let (px, py) = c.uv_px(uv);
    let m = c.field(px, py)[0];
    let k = smooth_step(m, m + softness, c.progress * (1.0 + softness));
    mix4(c.from(uv), c.to(uv), k)
}
