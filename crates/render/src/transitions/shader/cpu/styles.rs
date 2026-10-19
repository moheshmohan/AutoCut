//! Rust references of FilmCraft's C5 engines (original work, MIT OR Apache-2.0):
//! `light_film.wgsl`, `patterns.wgsl` and `camera.wgsl`, operation for operation.

use super::{Ctx, V2, glsl_mod, mix4, smooth_step};
use crate::transitions::{bell, ease_in_out, hash, vnoise};

use std::f32::consts::{FRAC_PI_2, PI, TAU};

#[inline]
fn side(c: &Ctx, which: u32, p: V2) -> [f32; 4] {
    if which == 0 { c.outgoing_px(p.x, p.y) } else { c.incoming_px(p.x, p.y) }
}
#[inline]
fn add_rgb(m: [f32; 4], rgb: [f32; 3], k: f32, alpha_k: f32) -> [f32; 4] {
    [m[0] + rgb[0] * k, m[1] + rgb[1] * k, m[2] + rgb[2] * k, (m[3] + alpha_k).min(1.0)]
}
#[inline]
fn tint(c: &Ctx) -> [f32; 3] {
    [c.p[4], c.p[5], c.p[6]]
}
#[inline]
fn dot(a: V2, b: V2) -> f32 {
    a.x * b.x + a.y * b.y
}

// ---------------------------------------------------------------------------------------------
// light_film.wgsl

fn lf_base(c: &Ctx, x: V2, t: f32, e0: f32, e1: f32) -> [f32; 4] {
    mix4(side(c, 0, x), side(c, 1, x), smooth_step(e0, e1, t))
}

fn lf_rainbow(h: f32) -> [f32; 3] {
    let f = |o: f32| 0.5 + 0.5 * (TAU * (h + o)).cos();
    [f(0.0), f(0.333_333_34), f(0.666_666_7)]
}

fn lf_leak(c: &Ctx, x: V2, t: f32, prism: bool) -> [f32; 4] {
    let s = c.p[2] as u32;
    let diag = V2::new(c.w, c.h).length();
    let env = bell(t) * c.p[1];
    let mut light = [0.0f32; 3];
    for i in 0..4 {
        let cx = c.w * (hash(i, 1, s) + (t - 0.5) * (0.6 + 0.8 * hash(i, 2, s)));
        let cy = c.h * (0.15 + 0.7 * hash(i, 3, s));
        let r = diag * (0.18 + 0.2 * hash(i, 4, s));
        let d = x.sub(V2::new(cx, cy));
        let k = (-dot(d, d) / (r * r)).exp();
        let mut col = [0.0f32; 3];
        let base = tint(c);
        let warm = [1.0, 0.85, 0.55];
        let m = 0.5 * hash(i, 6, s);
        for j in 0..3 {
            col[j] = base[j] * (1.0 - m) + warm[j] * m;
        }
        if prism {
            col = lf_rainbow(hash(i, 7, s) + x.x * (1.0 / c.w) * 0.5);
        }
        for j in 0..3 {
            light[j] += col[j] * k;
        }
    }
    let mut base = lf_base(c, x, t, 0.3, 0.7);
    if prism {
        let split = V2::new(c.p[3] * env, 0.0);
        let m = smooth_step(0.3, 0.7, t);
        let r = mix4(side(c, 0, x.add(split)), side(c, 1, x.add(split)), m)[0];
        let b = mix4(side(c, 0, x.sub(split)), side(c, 1, x.sub(split)), m)[2];
        base = [r, base[1], b, base[3]];
    }
    let k = env * 0.9;
    add_rgb(base, light, k, k * light[0].max(light[1].max(light[2])))
}

fn lf_burn(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = c.p[2] as u32;
    let diag = V2::new(c.w, c.h).length();
    let start = V2::new(c.w * (0.25 + 0.5 * hash(0, 11, s)), c.h * (0.25 + 0.5 * hash(0, 12, s)));
    let n1 = x.divs(diag * 0.06);
    let n2 = x.divs(diag * 0.018);
    let n = vnoise(n1.x, n1.y, s) * 0.6 + vnoise(n2.x, n2.y, s.wrapping_add(1)) * 0.4;
    let phi = x.sub(start).length() / diag + 0.25 * n;
    let band = c.p[3].max(0.005);
    let front = t * (1.0 + 3.0 * band) - band;
    let d = phi - front;
    let a = side(c, 0, x);
    let b = side(c, 1, x);
    let hot = tint(c);
    if d < -band {
        return b;
    }
    if d < 0.0 {
        let k = 1.0 + d / band;
        let m = 1.0 - 0.85 * k;
        return [b[0] * m, b[1] * m, b[2] * m, b[3]];
    }
    let warm = smooth_step(0.0, 0.1, t);
    if d < band {
        let g = 1.0 - d / band;
        return add_rgb(a, hot, g * g * 2.5 * warm, g * warm);
    }
    let heat = 0.25 * (-(d - band) / (2.0 * band)).exp() * warm;
    add_rgb(a, hot, heat, heat)
}

fn lf_flare(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let diag = V2::new(c.w, c.h).length();
    let centre = V2::new(c.w * 0.5, c.h * 0.5);
    let dir = V2::new(c.p[8], c.p[9]);
    let src = centre.add(dir.muls((t - 0.5) * diag * 1.1));
    let env = bell(t) * c.p[1];
    let d = x.sub(src);
    let core = (-dot(d, d) / (diag * diag * 0.0036)).exp();
    let streak = (-d.y.abs() / (diag * 0.006)).exp() * (-d.x.abs() / (diag * 0.5)).exp();
    const GC: [[f32; 3]; 3] = [[0.3, 0.8, 0.9], [0.7, 0.4, 0.9], [0.9, 0.7, 0.3]];
    const GK: [f32; 3] = [0.5, 1.0, 1.6];
    const GR: [f32; 3] = [0.03, 0.05, 0.08];
    let mut ghosts = [0.0f32; 3];
    for i in 0..3 {
        let g = centre.add(centre.sub(src).muls(GK[i]));
        let r = diag * GR[i];
        let dist = x.sub(g).length();
        let k = (1.0 - smooth_step(r * 0.8, r, dist)) * 0.25;
        for j in 0..3 {
            ghosts[j] += GC[i][j] * k;
        }
    }
    let col = tint(c);
    let mut light = [0.0f32; 3];
    for j in 0..3 {
        light[j] = col[j] * (core * 1.5 + streak * 0.8 + 0.12) + ghosts[j];
    }
    let base = lf_base(c, x, t, 0.4, 0.6);
    add_rgb(base, light, env, env * (core + 0.12))
}

fn lf_flash(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let d = (t - 0.5) / 0.1;
    let f = c.p[1] * (-d * d).exp();
    let base = lf_base(c, x, t, 0.45, 0.55);
    let col = tint(c);
    let g = 1.0 + 3.0 * f;
    [base[0] * g + col[0] * f, base[1] * g + col[1] * f, base[2] * g + col[2] * f, (base[3] + f).min(1.0)]
}

fn lf_bokeh_side(c: &Ctx, which: u32, x: V2, r: f32) -> [f32; 4] {
    let mut acc = [0.0f32; 4];
    let mut wsum = 0.0f32;
    for i in 0..24 {
        let rr = r * ((i as f32 + 0.5) / 24.0).sqrt();
        let a = super::fract(i as f32 * 0.381_966_02) * TAU;
        let s = side(c, which, x.add(V2::new(a.cos(), a.sin()).muls(rr)));
        let l = 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2];
        let w = 1.0 + c.p[7] * l * l * 4.0;
        for k in 0..4 {
            acc[k] += s[k] * w;
        }
        wsum += w;
    }
    acc.map(|v| v / wsum)
}

fn lf_bokeh(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let env = bell(t);
    let r = c.p[3] * env;
    let m = smooth_step(0.3, 0.7, t);
    let v = mix4(lf_bokeh_side(c, 0, x, r), lf_bokeh_side(c, 1, x, r), m);
    let g = 1.0 + 0.15 * env * c.p[7];
    [v[0] * g, v[1] * g, v[2] * g, v[3]]
}

fn lf_projector(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = c.p[2] as u32;
    let env = bell(t);
    let step = (t * 24.0).floor() as i32;
    let gap = c.p[3];
    let weave = (hash(1, step, s) - 0.5) * 4.0 * env;
    let o = ease_in_out(t) * (c.h + gap);
    let ya = x.y + o;
    let p = V2::new(x.x + weave, ya);
    let mut col = [0.0, 0.0, 0.0, 1.0];
    if ya < c.h {
        col = side(c, 0, p);
    } else if ya >= c.h + gap {
        col = side(c, 1, V2::new(p.x, ya - c.h - gap));
    }
    let flicker = 1.0 - c.p[8] * env * hash(0, step, s) * 0.5;
    let scratch = if hash((x.x * 0.5).floor() as i32, step, s.wrapping_add(5)) > 1.0 - 0.06 * c.p[9] { 0.4 * env } else { 0.0 };
    [col[0] * flicker + scratch * col[3], col[1] * flicker + scratch * col[3], col[2] * flicker + scratch * col[3], col[3]]
}

/// The light & film engine (`light_film.wgsl`).
pub(crate) fn light_film(c: &Ctx, uv: V2) -> [f32; 4] {
    let (px, py) = c.uv_px(uv);
    let x = V2::new(px, py);
    let t = c.progress;
    match c.p[0] as u32 {
        1 => lf_leak(c, x, t, true),
        2 => lf_burn(c, x, t),
        3 => lf_flare(c, x, t),
        4 => lf_flash(c, x, t),
        5 => lf_bokeh(c, x, t),
        6 => lf_projector(c, x, t),
        _ => lf_leak(c, x, t, false),
    }
}

// ---------------------------------------------------------------------------------------------
// patterns.wgsl

const PT_DUR: f32 = 0.35;

fn pt_size(c: &Ctx) -> f32 {
    c.w / c.p[1].max(1.0)
}

fn pt_cell(x: V2, s: f32) -> V2 {
    x.divs(s).adds(0.001).floor()
}

fn pt_delay(c: &Ctx, cell: V2, centre: V2, salt: u32) -> f32 {
    let s = (c.p[3] as u32).wrapping_mul(131).wrapping_add(salt);
    let order = c.p[2] as u32;
    if order == 1 {
        return (centre.x * (1.0 / c.w) * 0.92 + 0.08 * hash(cell.x as i32, cell.y as i32, s)).clamp(0.0, 1.0);
    }
    if order == 2 {
        let half = V2::new(c.w * 0.5, c.h * 0.5);
        return (centre.sub(half).length() / half.length() * 0.92 + 0.08 * hash(cell.x as i32, cell.y as i32, s)).clamp(0.0, 1.0);
    }
    hash(cell.x as i32, cell.y as i32, s)
}

fn pt_local(delay: f32, t: f32) -> f32 {
    ((t - delay * (1.0 - PT_DUR)) / PT_DUR).clamp(0.0, 1.0)
}

fn pt_mix(c: &Ctx, x: V2, k: f32) -> [f32; 4] {
    mix4(side(c, 0, x), side(c, 1, x), k)
}

fn pt_halftone(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = pt_size(c);
    let cell = pt_cell(x, s);
    let centre = cell.adds(0.5).muls(s);
    let q = pt_local(pt_delay(c, cell, centre, 0), t);
    let r = q * (0.71 * s + 1.0);
    let k = (1.0 - smooth_step(r - 0.5, r + 0.5, x.sub(centre).length())) * (q * 20.0).min(1.0);
    pt_mix(c, x, k)
}

fn pt_triangles(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = pt_size(c);
    let cell = pt_cell(x, s);
    let u = x.x / s - cell.x;
    let v = x.y / s - cell.y;
    let even = glsl_mod(cell.x + cell.y, 2.0) < 0.5;
    let (tri, l, cen) = if even {
        if u >= v { (1.0, [1.0 - u, u - v, v], V2::new(2.0 / 3.0, 1.0 / 3.0)) } else { (0.0, [1.0 - v, u, v - u], V2::new(1.0 / 3.0, 2.0 / 3.0)) }
    } else if u + v < 1.0 {
        (0.0, [1.0 - u - v, u, v], V2::new(1.0 / 3.0, 1.0 / 3.0))
    } else {
        (1.0, [1.0 - v, u + v - 1.0, 1.0 - u], V2::new(2.0 / 3.0, 2.0 / 3.0))
    };
    let id = V2::new(cell.x * 2.0 + tri, cell.y);
    let q = pt_local(pt_delay(c, id, cell.add(cen).muls(s), 1), t);
    let m = 1.0 - 3.0 * l[0].min(l[1].min(l[2]));
    let aa = 1.5 / s;
    let edge = q * (1.0 + 2.0 * aa);
    let k = (1.0 - smooth_step(edge - aa, edge + aa, m)) * (q * 20.0).min(1.0);
    pt_mix(c, x, k)
}

fn pt_hearts(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = pt_size(c);
    let cell = pt_cell(x, s);
    let centre = cell.adds(0.5).muls(s);
    let q = pt_local(pt_delay(c, cell, centre, 2), t);
    if q >= 1.0 {
        return side(c, 1, x);
    }
    let size = q * 1.6 * s * 0.5;
    if size <= 0.0 {
        return side(c, 0, x);
    }
    let hx = (x.x - centre.x) / size;
    let hy = (centre.y - x.y) / size + 0.15;
    let a = hx * hx + hy * hy - 1.0;
    let inside = a * a * a - hx * hx * hy * hy * hy <= 0.0;
    side(c, if inside { 1 } else { 0 }, x)
}

fn pt_sparkle(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = pt_size(c);
    let cell = pt_cell(x, s);
    let centre = cell.adds(0.5).muls(s);
    let q = pt_local(pt_delay(c, cell, centre, 3), t);
    let base = pt_mix(c, x, smooth_step(0.0, 1.0, q));
    let mut glint = 0.0f32;
    for j in -1..=1 {
        for i in -1..=1 {
            let nc = cell.add(V2::new(i as f32, j as f32));
            let nctr = nc.adds(0.5).muls(s);
            let peak = pt_delay(c, nc, nctr, 3) * (1.0 - PT_DUR) + PT_DUR * 0.5;
            let dt = (t - peak) / 0.05;
            let g = (-dt * dt).exp();
            let d = V2::new((x.x - nctr.x).abs(), (x.y - nctr.y).abs()).divs(s);
            let star = (-d.x / 0.05).exp() * (-d.y / 0.6).exp() + (-d.y / 0.05).exp() * (-d.x / 0.6).exp() + (-dot(d, d) / 0.01).exp();
            glint += g * star;
        }
    }
    let k = glint * c.p[7] * bell(t) * 1.5;
    add_rgb(base, tint(c), k, k)
}

fn pt_rgb_split(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = c.p[3] as u32;
    let d = (t - 0.5) / 0.15;
    let g = c.p[8] * (-d * d).exp();
    let step = (t * 30.0).floor() as i32;
    let band = (x.y * (1.0 / c.h) * c.p[9].max(1.0) + 0.001).floor() as i32;
    let h = hash(band, step, s);
    let off = (hash(band, step, s.wrapping_add(1)) - 0.5) * 2.0 * g * if h > 0.6 { 1.0 } else { 0.2 };
    let p = x.add(V2::new(off, 0.0));
    let split = V2::new(g * 0.5, 0.0);
    let which = if t < 0.5 { 0 } else { 1 };
    let m = side(c, which, p);
    [side(c, which, p.add(split))[0], m[1], side(c, which, p.sub(split))[2], m[3]]
}

fn pt_scanlines(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = c.p[3] as u32;
    let env = bell(t);
    let lh = c.p[9].max(1.0);
    let line = (x.y / lh + 0.001).floor() as i32;
    let step = (t * 30.0).floor() as i32;
    let tear = if hash(line, step, s.wrapping_add(3)) < 0.3 * env { (hash(line, step, s.wrapping_add(2)) - 0.5) * 2.0 * c.p[8] * env } else { 0.0 };
    let p = x.add(V2::new(tear, 0.0));
    let swap = 0.4 + 0.2 * hash(line, 0, s.wrapping_add(4));
    let col = side(c, if t >= swap { 1 } else { 0 }, p);
    let dark = 1.0 - 0.3 * env * if line & 1 == 1 { 1.0 } else { 0.0 };
    [col[0] * dark, col[1] * dark, col[2] * dark, col[3]]
}

/// The pattern engine (`patterns.wgsl`).
pub(crate) fn patterns(c: &Ctx, uv: V2) -> [f32; 4] {
    let (px, py) = c.uv_px(uv);
    let x = V2::new(px, py);
    let t = c.progress;
    match c.p[0] as u32 {
        1 => pt_triangles(c, x, t),
        2 => pt_hearts(c, x, t),
        3 => pt_sparkle(c, x, t),
        4 => pt_rgb_split(c, x, t),
        5 => pt_scanlines(c, x, t),
        _ => pt_halftone(c, x, t),
    }
}

// ---------------------------------------------------------------------------------------------
// camera.wgsl

fn cam_ease_in(t: f32) -> f32 {
    let u = t.clamp(0.0, 1.0);
    u * u * u
}

fn cam_ease_out(t: f32) -> f32 {
    let u = 1.0 - t.clamp(0.0, 1.0);
    1.0 - u * u * u
}

fn cam_mirror(v: f32, size: f32) -> f32 {
    let m = glsl_mod(v, 2.0 * size);
    if m >= size { 2.0 * size - m } else { m }
}

fn cam_sample(c: &Ctx, which: u32, p: V2) -> [f32; 4] {
    side(c, which, V2::new(cam_mirror(p.x, c.w), cam_mirror(p.y, c.h)))
}

fn cam_view(c: &Ctx, which: u32, x: V2, zoom: f32, angle: f32) -> [f32; 4] {
    let centre = V2::new(c.w * 0.5, c.h * 0.5);
    let d = x.sub(centre).divs(zoom);
    let (s, co) = (angle.sin(), angle.cos());
    cam_sample(c, which, centre.add(V2::new(d.x * co + d.y * s, -d.x * s + d.y * co)))
}

fn cam_zooms(c: &Ctx, x: V2, t: f32, mode: u32) -> [f32; 4] {
    let into = cam_ease_in(t / 0.5);
    let outof = 1.0 - cam_ease_out((t - 0.5) / 0.5);
    let z = c.p[1];
    let (mut za, mut zb) = (1.0 + z * into, 1.0 + z * outof);
    let (mut ra, mut rb) = (0.0, 0.0);
    if mode == 1 {
        za = 1.0 / za;
        zb = 1.0 / zb;
    }
    if mode == 2 {
        za = 1.0 + c.p[2] * into;
        zb = 1.0 + c.p[2] * outof;
        ra = c.p[1] * into;
        rb = -c.p[1] * outof;
    }
    mix4(cam_view(c, 0, x, za, ra), cam_view(c, 1, x, zb, rb), smooth_step(0.42, 0.58, t))
}

fn cam_push(c: &Ctx, p: V2, d: V2, e: f32, len: f32, horizontal: bool) -> [f32; 4] {
    let pa = p.sub(d.muls(e * len));
    let along = if horizontal { pa.x } else { pa.y };
    if along >= 0.0 && along < len {
        return side(c, 0, pa);
    }
    side(c, 1, pa.add(d.muls(len)))
}

fn cam_whip(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let d = V2::new(c.p[1], c.p[2]);
    let horizontal = d.x.abs() > 0.5;
    let len = if horizontal { c.w } else { c.h };
    let e = ease_in_out(t);
    let speed = (if t < 0.5 { 12.0 * t * t } else { 12.0 * (1.0 - t) * (1.0 - t) }) / 3.0;
    let blur = c.p[3] * speed * len * 0.25;
    let mut acc = [0.0f32; 4];
    for i in 0..16 {
        let s = i as f32 / 15.0 - 0.5;
        let v = cam_push(c, x.add(d.muls(blur * s)), d, e, len, horizontal);
        for k in 0..4 {
            acc[k] += v[k];
        }
    }
    acc.map(|v| v / 16.0)
}

fn cam_shake(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let s = c.p[3] as u32;
    let env = bell(t);
    let f = t * c.p[2].max(1.0);
    let k = f.floor() as i32;
    let w = smooth_step(0.0, 1.0, f - f.floor());
    let o0 = V2::new(hash(k, 1, s) - 0.5, hash(k, 2, s) - 0.5);
    let o1 = V2::new(hash(k.wrapping_add(1), 1, s) - 0.5, hash(k.wrapping_add(1), 2, s) - 0.5);
    let o = V2::new(o0.x * (1.0 - w) + o1.x * w, o0.y * (1.0 - w) + o1.y * w).muls(2.0 * c.p[1] * env);
    let p = x.add(o);
    mix4(cam_sample(c, 0, p), cam_sample(c, 1, p), smooth_step(0.45, 0.55, t))
}

/// `cam_card`: card coordinates of screen point `q` on a card turned by `phi`, None off its front.
fn cam_card(q: V2, phi: f32, dist: f32) -> Option<V2> {
    let den = dist * phi.cos() - q.x * phi.sin();
    if den <= 1e-4 {
        return None;
    }
    let u = q.x * dist / den;
    let v = q.y * (dist + u * phi.sin()) / dist;
    Some(V2::new(u, v))
}

fn cam_flip(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let hh = c.h * 0.5;
    let centre = V2::new(c.w * 0.5, c.h * 0.5);
    let theta = PI * ease_in_out(t);
    let back = theta >= FRAC_PI_2;
    let phi = if back { theta - PI } else { theta };
    let vertical_axis = c.p[1] < 0.5;
    let rel = x.sub(centre).divs(hh);
    let q = if vertical_axis { rel } else { V2::new(rel.y, rel.x) };
    let half = if vertical_axis { V2::new(c.w / c.h, 1.0) } else { V2::new(1.0, c.w / c.h) };
    let Some(card) = cam_card(q, phi, c.p[2]) else { return [0.0; 4] };
    if card.x.abs() > half.x || card.y.abs() > half.y {
        return [0.0; 4];
    }
    let uv = if vertical_axis { card } else { V2::new(card.y, card.x) };
    let p = centre.add(uv.muls(hh));
    let shade = 0.6 + 0.4 * phi.cos();
    let col = side(c, if back { 1 } else { 0 }, p);
    [col[0] * shade, col[1] * shade, col[2] * shade, col[3]]
}

fn cam_tilt(c: &Ctx, x: V2, t: f32) -> [f32; 4] {
    let d = V2::new(c.p[1], c.p[2]);
    let horizontal = d.x.abs() > 0.5;
    let hh = c.h * 0.5;
    let centre = V2::new(c.w * 0.5, c.h * 0.5);
    let len = if horizontal { c.w } else { c.h };
    let phi = c.p[3] * bell(t) * if horizontal { d.x } else { d.y };
    let rel = x.sub(centre).divs(hh);
    let q = if horizontal { rel } else { V2::new(rel.y, rel.x) };
    let limit = if horizontal { 1.0 } else { c.w / c.h };
    let Some(card) = cam_card(q, phi, 4.0) else { return [0.0; 4] };
    if card.y.abs() > limit {
        return [0.0; 4];
    }
    let p = centre.add(if horizontal { card } else { V2::new(card.y, card.x) }.muls(hh));
    cam_push(c, p, d, ease_in_out(t), len, horizontal)
}

/// The camera engine (`camera.wgsl`).
pub(crate) fn camera(c: &Ctx, uv: V2) -> [f32; 4] {
    let (px, py) = c.uv_px(uv);
    let x = V2::new(px, py);
    let t = c.progress;
    let mode = c.p[0] as u32;
    match mode {
        3 => cam_whip(c, x, t),
        4 => cam_shake(c, x, t),
        5 => cam_flip(c, x, t),
        6 => cam_tilt(c, x, t),
        _ => cam_zooms(c, x, t, mode),
    }
}
