// FilmCraft GPU effect stage: the standard effects of `filmcraft_render::gpufx::FxOp`, one
// compute pass each, reading the working image (`src`, linear premultiplied RGBA f32) and writing
// the next one (`dst`). The math mirrors `FxOp::apply` (the CPU reference) operation by
// operation; parameters arrive evaluated (keyframes, defaults and clamps applied on the CPU).

struct U {
    i0: vec4<u32>,  // op, width, height, unused
    i1: vec4<u32>,  // integer parameters (box radius, repeat, steps, channel, rect…)
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    p4: vec4<f32>,
    p5: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;
// Unsharp Mask: the image before blurring; a masked effect: the image before the effect.
@group(0) @binding(3) var aux: texture_2d<f32>;
// The masks of OP_MASK_MIX / OP_MASK_SCALE (`fx.rs` `mask_words`): `u.i1.x` headers of 12
// words (first point, points, mode, inverted, feather, expansion, opacity, unused, then the box
// x0, y0, x1, y1 outside which the coverage is the mask's outside value), then the points.
@group(0) @binding(4) var<storage, read> mk: array<u32>;

const OP_BRIGHTNESS_CONTRAST: u32 = 1u;
const OP_PROC_AMP: u32 = 2u;
const OP_TINT: u32 = 3u;
const OP_BLACK_WHITE: u32 = 4u;
const OP_COLOR_BALANCE: u32 = 5u;
const OP_LEAVE_COLOR: u32 = 6u;
const OP_CHANGE_TO_COLOR: u32 = 7u;
const OP_COLOR_PASS: u32 = 8u;
const OP_GAMMA: u32 = 9u;
const OP_LEVELS: u32 = 10u;
const OP_EXTRACT: u32 = 11u;
const OP_INVERT: u32 = 12u;
const OP_INVERT_ALPHA: u32 = 13u;
const OP_POSTERIZE: u32 = 14u;
const OP_ASC_CDL: u32 = 15u;
const OP_CHANNEL_MIX: u32 = 16u;
const OP_COLOR_REPLACE: u32 = 17u;
const OP_ALPHA_ADJUST: u32 = 18u;
const OP_VIGNETTE: u32 = 19u;
const OP_BOX: u32 = 20u;
const OP_DIRECTIONAL: u32 = 22u;
const OP_UNSHARP: u32 = 23u;
const OP_CROP: u32 = 24u;
const OP_RESAMPLE: u32 = 25u;
const OP_HFLIP: u32 = 26u;
const OP_VFLIP: u32 = 27u;
const OP_MIRROR: u32 = 28u;
const OP_OFFSET: u32 = 29u;
const OP_VIDEO_LIMITER: u32 = 30u;
const OP_LUMETRI: u32 = 31u;
const OP_MASK_MIX: u32 = 32u;
const OP_MASK_SCALE: u32 = 33u;

// Mask coverage, the math of `filmcraft_render::mask` (and `mask.wgsl`): signed distance to the
// polygon (nonzero winding), the feather falloff, the masks combined top to bottom by mode.

fn mask_falloff(s: f32, feather: f32) -> f32 {
    let w = max(feather, 1.0);
    let t = clamp(s / w + 0.5, 0.0, 1.0);
    let sm = t * t * (3.0 - 2.0 * t);
    return t + (sm - t) * min(feather, 1.0);
}

// MaskMode index: None 0, Add 1, Subtract 2, Intersect 3, Lighten 4, Darken 5, Difference 6.
fn mask_start(mode: u32) -> f32 {
    return select(0.0, 1.0, mode == 2u || mode == 3u || mode == 5u);
}

fn mask_combine(mode: u32, a: f32, m: f32) -> f32 {
    switch mode {
        case 1u: { return a + m - a * m; }
        case 2u: { return a * (1.0 - m); }
        case 3u: { return a * m; }
        case 4u: { return max(a, m); }
        case 5u: { return min(a, m); }
        case 6u: { return a + m - 2.0 * a * m; }
        default: { return a; }
    }
}

fn mask_pt(base: u32, i: u32) -> vec2<f32> {
    return vec2(bitcast<f32>(mk[base + 2u * i]), bitcast<f32>(mk[base + 2u * i + 1u]));
}

fn mask_seg_dist2(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = b - a;
    let q = p - a;
    let l2 = dot(d, d);
    var t = 0.0;
    if l2 > 0.0 {
        t = clamp(dot(q, d) / l2, 0.0, 1.0);
    }
    let e = q - d * t;
    return dot(e, e);
}

// Mask `i`'s coverage at pixel centre `p`.
fn mask_value(i: u32, p: vec2<f32>) -> f32 {
    let h = i * 12u;
    let start = mk[h];
    let len = mk[h + 1u];
    let inverted = mk[h + 3u] != 0u;
    let feather = bitcast<f32>(mk[h + 4u]);
    let expansion = bitcast<f32>(mk[h + 5u]);
    let opacity = bitcast<f32>(mk[h + 6u]);
    let lo = vec2(bitcast<f32>(mk[h + 8u]), bitcast<f32>(mk[h + 9u]));
    let hi = vec2(bitcast<f32>(mk[h + 10u]), bitcast<f32>(mk[h + 11u]));
    let band = abs(expansion) + max(feather, 1.0) * 0.5 + 1.0;
    var sd = -band;
    // beyond the box no edge is within the band and the winding is zero
    if len >= 3u && all(p >= lo) && all(p <= hi) {
        let pts = u.i1.x * 12u + 2u * start;
        var wind = 0;
        var d2 = band * band;
        for (var k = 0u; k < len; k++) {
            let a = mask_pt(pts, k);
            let c = mask_pt(pts, (k + 1u) % len);
            if (a.y <= p.y) != (c.y <= p.y) {
                let t = (p.y - a.y) / (c.y - a.y);
                if a.x + (c.x - a.x) * t < p.x {
                    wind += select(-1, 1, c.y > a.y);
                }
            }
            d2 = min(d2, mask_seg_dist2(p, a, c));
        }
        let d = sqrt(d2);
        sd = select(-d, d, wind != 0);
    }
    let c = mask_falloff(sd + expansion, feather);
    return opacity * select(c, 1.0 - c, inverted);
}

// The masks' combined coverage at pixel `p`.
fn mask_coverage(p: vec2<i32>) -> f32 {
    let pc = vec2<f32>(p) + 0.5;
    var acc = mask_start(mk[2]);
    for (var i = 0u; i < u.i1.x; i++) {
        acc = clamp(mask_combine(mk[i * 12u + 2u], acc, mask_value(i, pc)), 0.0, 1.0);
    }
    return acc;
}

fn size() -> vec2<i32> {
    return vec2<i32>(i32(u.i0.y), i32(u.i0.z));
}

fn ld(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(src, p, 0);
}

// `Image::get_or_clear`
fn ld_or_clear(p: vec2<i32>) -> vec4<f32> {
    let s = size();
    if p.x < 0 || p.y < 0 || p.x >= s.x || p.y >= s.y {
        return vec4(0.0);
    }
    return textureLoad(src, p, 0);
}

// `Image::sample_bilinear`: pixel centres at +0.5, transparent outside.
fn sample_bilinear(x: f32, y: f32) -> vec4<f32> {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = floor(fx);
    let y0 = floor(fy);
    let tx = fx - x0;
    let ty = fy - y0;
    let i = vec2<i32>(i32(x0), i32(y0));
    let a = ld_or_clear(i);
    let b = ld_or_clear(i + vec2(1, 0));
    let c = ld_or_clear(i + vec2(0, 1));
    let d = ld_or_clear(i + vec2(1, 1));
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

// `Image::sample_bilinear_clamped`
fn sample_clamped(x: f32, y: f32) -> vec4<f32> {
    let s = size();
    let fx = clamp(x - 0.5, 0.0, f32(s.x) - 1.0);
    let fy = clamp(y - 0.5, 0.0, f32(s.y) - 1.0);
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = min(x0 + 1, s.x - 1);
    let y1 = min(y0 + 1, s.y - 1);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let a = ld(vec2(x0, y0));
    let b = ld(vec2(x1, y0));
    let c = ld(vec2(x0, y1));
    let d = ld(vec2(x1, y1));
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

// ---- colour helpers (`filmcraft_color`, `effects::enc` / `dec`)

fn linear_to_srgb1(v: f32) -> f32 {
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear1(v: f32) -> f32 {
    if v <= 0.04045 {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn enc(c: vec3<f32>) -> vec3<f32> {
    let m = max(c, vec3(0.0));
    return vec3(linear_to_srgb1(m.x), linear_to_srgb1(m.y), linear_to_srgb1(m.z));
}

fn dec(c: vec3<f32>) -> vec3<f32> {
    let m = clamp(c, vec3(0.0), vec3(1.0));
    return vec3(srgb_to_linear1(m.x), srgb_to_linear1(m.y), srgb_to_linear1(m.z));
}

fn luma709(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// `f32::powf` for x >= 0 (WGSL leaves pow(0, y) undefined).
fn powf(x: f32, y: f32) -> f32 {
    if x == 0.0 {
        if y == 0.0 {
            return 1.0;
        }
        if y > 0.0 {
            return 0.0;
        }
        return 3.0e38;
    }
    return pow(x, y);
}

// `f32::round`: halves away from zero (WGSL `round` rounds them to even).
fn round_away(x: f32) -> f32 {
    let a = abs(x);
    let f = floor(a);
    let r = select(f, f + 1.0, a - f >= 0.5);
    return select(r, -r, x < 0.0);
}

// `rem_euclid(1.0)`
fn fract_euclid(x: f32) -> f32 {
    return x - floor(x);
}

// `vfx::smoothstep` (not WGSL's: a degenerate edge pair divides by 1e-6)
fn smoothstep_fx(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let r = c.x; let g = c.y; let b = c.z;
    let mx = max(max(r, g), b);
    let mn = min(min(r, g), b);
    let l = (mx + mn) / 2.0;
    if abs(mx - mn) < 1e-7 {
        return vec3(0.0, 0.0, l);
    }
    let d = mx - mn;
    var s: f32;
    if l > 0.5 {
        s = d / (2.0 - mx - mn);
    } else {
        s = d / (mx + mn);
    }
    var h: f32;
    if mx == r {
        h = (g - b) / d + select(0.0, 6.0, g < b);
    } else if mx == g {
        h = (b - r) / d + 2.0;
    } else {
        h = (r - g) / d + 4.0;
    }
    return vec3(h / 6.0, s, l);
}

fn hsl_channel(p: f32, q: f32, t0: f32) -> f32 {
    let t = fract_euclid(t0);
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    return p;
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if s <= 0.0 {
        return vec3(l);
    }
    var q: f32;
    if l < 0.5 {
        q = l * (1.0 + s);
    } else {
        q = l + s - l * s;
    }
    let p = 2.0 * l - q;
    return vec3(hsl_channel(p, q, h + 1.0 / 3.0), hsl_channel(p, q, h), hsl_channel(p, q, h - 1.0 / 3.0));
}

fn linear_to_enc_limiter(v: f32) -> f32 {
    if v <= 1.0 {
        return linear_to_srgb1(max(v, 0.0));
    }
    return 1.0 + (v - 1.0) / 2.4;
}

fn enc_to_linear_limiter(v: f32) -> f32 {
    if v <= 1.0 {
        return srgb_to_linear1(max(v, 0.0));
    }
    return 1.0 + (v - 1.0) * 2.4;
}

fn knee_limiter(v: f32, max_val: f32, comp: f32) -> f32 {
    let k = max_val * (1.0 - comp);
    if comp <= 0.0 || v <= k {
        return min(v, max_val);
    }
    let r = max_val - k;
    return k + r * (1.0 - exp(-(v - k) / r));
}

fn to_rgb_709(y: f32, cb: f32, cr: f32) -> vec3<f32> {
    let kr = 0.2126;
    let kb = 0.0722;
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / kg;
    return vec3(r, g, b);
}

fn limit_video(v: vec3<f32>, max_val: f32, comp: f32, axis: u32) -> vec3<f32> {
    let kr = 0.2126;
    let kb = 0.0722;
    let kg = 1.0 - kr - kb;
    var y = kr * v.x + kg * v.y + kb * v.z;
    var cb = (v.z - y) / (2.0 * (1.0 - kb));
    var cr = (v.x - y) / (2.0 * (1.0 - kr));

    if axis != 1u {
        y = knee_limiter(max(y, 0.0), max_val, comp);
    }

    if axis != 0u {
        let rgb = to_rgb_709(y, cb, cr);
        var s = 1.0;
        let d = rgb - vec3(y);

        if rgb.x > max_val && d.x > 1e-6 {
            s = min(s, max((max_val - y) / d.x, 0.0));
        }
        if rgb.x < 0.0 && d.x < -1e-6 {
            s = min(s, max(y / (-d.x), 0.0));
        }

        if rgb.y > max_val && d.y > 1e-6 {
            s = min(s, max((max_val - y) / d.y, 0.0));
        }
        if rgb.y < 0.0 && d.y < -1e-6 {
            s = min(s, max(y / (-d.y), 0.0));
        }

        if rgb.z > max_val && d.z > 1e-6 {
            s = min(s, max((max_val - y) / d.z, 0.0));
        }
        if rgb.z < 0.0 && d.z < -1e-6 {
            s = min(s, max(y / (-d.z), 0.0));
        }

        if axis == 3u && s < 1.0 - 1e-4 {
            s = min(knee_limiter(s, 1.0, max(comp, 0.03)), 1.0);
        }
        cb *= s;
        cr *= s;
    }

    let out_rgb = to_rgb_709(y, cb, cr);
    return clamp(out_rgb, vec3(0.0), vec3(max_val));
}

// ---- per-pixel colour effects on straight colour (`Image::map_rgb`)

fn color_op(op: u32, c: vec3<f32>) -> vec3<f32> {
    switch op {
        case OP_BRIGHTNESS_CONTRAST: {
            let br = u.p0.x; let co = u.p0.y;
            return dec((enc(c) - 0.5) * co + 0.5 + br);
        }
        case OP_PROC_AMP: {
            let br = u.p0.x; let co = u.p0.y; let hue = u.p0.z; let sat = u.p0.w;
            var hsl = rgb_to_hsl(enc(c));
            hsl.x = fract_euclid(hsl.x + hue);
            hsl.y = clamp(hsl.y * sat, 0.0, 1.0);
            return dec((hsl_to_rgb(hsl.x, hsl.y, hsl.z) - 0.5) * co + 0.5 + br);
        }
        case OP_TINT: {
            let bl = u.p0.xyz; let wh = u.p1.xyz; let amt = u.p0.w;
            let l = linear_to_srgb1(max(luma709(c), 0.0));
            let t = bl + (wh - bl) * l;
            let e = enc(c);
            return dec(e + (t - e) * amt);
        }
        case OP_BLACK_WHITE: {
            return vec3(luma709(c));
        }
        case OP_COLOR_BALANCE: {
            let e = enc(c);
            let l = 0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z;
            let ws = (1.0 - l) * (1.0 - l);
            let wh = l * l;
            let wm = 1.0 - ws - wh;
            var o = e + u.p0.xyz * ws + u.p1.xyz * max(wm, 0.0) + u.p2.xyz * wh;
            if u.i1.x != 0u {
                let l2 = 0.2126 * o.x + 0.7152 * o.y + 0.0722 * o.z;
                o = o + (l - l2);
            }
            return dec(o);
        }
        case OP_LEAVE_COLOR: {
            let amt = u.p0.x; let kh = u.p0.y; let tol = u.p0.z; let soft = u.p0.w;
            let h = rgb_to_hsl(enc(c)).x;
            let d = min(abs(h - kh), 1.0 - abs(h - kh)) * 2.0;
            let keep = 1.0 - clamp((d - tol) / soft, 0.0, 1.0);
            let l = luma709(c);
            let k = amt * (1.0 - keep);
            return c + (l - c) * k;
        }
        case OP_CHANGE_TO_COLOR: {
            let fh = u.p0.x; let th = u.p0.y; let tol = u.p0.z; let soft = u.p0.w;
            var hsl = rgb_to_hsl(enc(c));
            let d = min(abs(hsl.x - fh), 1.0 - abs(hsl.x - fh));
            let w = 1.0 - clamp((d - tol) / soft, 0.0, 1.0);
            hsl.x = fract_euclid(hsl.x + (th - fh) * w);
            return dec(hsl_to_rgb(hsl.x, hsl.y, hsl.z));
        }
        case OP_COLOR_PASS: {
            let e = enc(c);
            let k = e - u.p0.xyz;
            let d = sqrt(k.x * k.x + k.y * k.y + k.z * k.z);
            let passes = (d <= u.p0.w * 1.2) != (u.i1.x != 0u);
            if passes {
                return c;
            }
            return vec3(luma709(c));
        }
        case OP_GAMMA: {
            let e = enc(c);
            let g = u.p0.x;
            return dec(vec3(powf(max(e.x, 0.0), g), powf(max(e.y, 0.0), g), powf(max(e.z, 0.0), g)));
        }
        case OP_LEVELS: {
            let ib = u.p0.x; let iw = u.p0.y; let ob = u.p0.z; let ow = u.p0.w; let g = u.p1.x;
            let v = clamp((enc(c) - ib) / (iw - ib), vec3(0.0), vec3(1.0));
            return dec(ob + vec3(powf(v.x, g), powf(v.y, g), powf(v.z, g)) * (ow - ob));
        }
        case OP_EXTRACT: {
            let lo = u.p0.x; let hi = u.p0.y; let soft = u.p0.z;
            let l = linear_to_srgb1(max(luma709(c), 0.0));
            // Hard cutoffs avoid both division by zero and an artificial narrow fade
            // that amplifies transfer-function rounding relative to the CPU reference.
            var inside = select(0.0, 1.0, l > lo && l < hi);
            if soft > 0.0 {
                inside = min(clamp((l - lo) / soft, 0.0, 1.0), clamp((hi - l) / soft, 0.0, 1.0));
            }
            return vec3(select(inside, 1.0 - inside, u.i1.x != 0u));
        }
        case OP_INVERT: {
            let ch = u.i1.x;
            let blend = u.p0.x;
            let e = enc(c);
            var o = e;
            if ch == 0u || ch == 1u { o.x = 1.0 - e.x; }
            if ch == 0u || ch == 2u { o.y = 1.0 - e.y; }
            if ch == 0u || ch == 3u { o.z = 1.0 - e.z; }
            return dec(o + (e - o) * blend);
        }
        case OP_POSTERIZE: {
            let n = u.p0.x;
            let e = enc(c) * n;
            return dec(vec3(round_away(e.x), round_away(e.y), round_away(e.z)) / n);
        }
        case OP_ASC_CDL: {
            let e = enc(c);
            let s = u.p0.xyz; let o = u.p1.xyz; let pw = max(u.p2.xyz, vec3(0.0)); let sat = u.p0.w;
            let q = clamp(e * s + o, vec3(0.0), vec3(1.0));
            let v = vec3(powf(q.x, pw.x), powf(q.y, pw.y), powf(q.z, pw.z));
            let l = luma709(v);
            return dec(clamp(l + sat * (v - l), vec3(0.0), vec3(1.0)));
        }
        case OP_CHANNEL_MIX: {
            let v = enc(c);
            let r = u.p0; let g = u.p1; let b = u.p2;
            return dec(clamp(vec3(r.x * v.x + r.y * v.y + r.z * v.z + r.w, g.x * v.x + g.y * v.y + g.z * v.z + g.w, b.x * v.x + b.y * v.y + b.z * v.z + b.w), vec3(0.0), vec3(1.0)));
        }
        case OP_COLOR_REPLACE: {
            let v = enc(c);
            let t = u.p0.xyz; let sim = u.p0.w;
            let dd = v - t;
            let d = sqrt(dd.x * dd.x + dd.y * dd.y + dd.z * dd.z);
            let k = 1.0 - smoothstep_fx(sim * 0.85, max(sim, 1e-4), d);
            if k <= 0.0 {
                return c;
            }
            var repl = u.p1.xyz;
            if u.i1.x == 0u {
                let rh = u.p2.xyz;
                repl = hsl_to_rgb(rh.x, rh.y, rgb_to_hsl(v).z);
            }
            return dec(v + (repl - v) * k);
        }
        default: {
            return c;
        }
    }
}

fn lumetri_op(c: vec3<f32>, p: vec2<i32>) -> vec3<f32> {
    let s = size();
    let creative_on = u.i1.x != 0u;
    let vignette_on = u.i1.y != 0u;

    // white balance + exposure in linear light
    let ge = u.p0.xyz;
    let lin = c * ge;
    var v = enc(lin);

    // whites / blacks: endpoints
    let b0 = u.p0.w;
    let w0 = u.p1.x;
    let denom = max(w0 - b0, 1e-3);
    v = (v - vec3(b0)) / denom;

    // highlights / shadows: luma-weighted lift/compress, hue preserving
    let sh_k = u.p1.y;
    let hl_k = u.p1.z;
    let l = luma709(v);
    let ws_val = clamp(1.0 - l, 0.0, 1.0);
    let ws = ws_val * ws_val * ws_val;
    let whl_val = clamp(l, 0.0, 1.0);
    let whl = whl_val * whl_val * whl_val;
    let nl = max(l + sh_k * ws + hl_k * whl, 0.0);
    // below a luma of 0.02 (LIFT_LUMA_FLOOR in effects.rs) the hue-preserving scale nl / l is held
    // at that luma's and the rest of the lift is added equally (lift_scale in effects.rs)
    var k = 1.0;
    if l > 1e-5 {
        k = min(nl / l, 1.0 + max(nl - l, 0.0) / 0.02);
    }
    let add = nl - max(l, 0.0) * k;
    v = v * k + vec3(add);

    // contrast: smooth S-curve around mid grey
    let contrast = u.p1.w;
    if abs(contrast) > 1e-4 {
        let k = 1.0 + contrast;
        let q = clamp(v, vec3(0.0), vec3(1.0));
        let sc = q * q * (vec3(3.0) - 2.0 * q);
        if k >= 1.0 {
            v = q + (sc - q) * (k - 1.0);
        } else {
            v = vec3(0.5) + (q - vec3(0.5)) * k;
        }
    }

    // faded film: lift blacks and compress
    let faded = u.p2.x;
    if faded > 0.0 {
        v = v * (1.0 - 0.25 * faded) + vec3(0.12 * faded);
    }

    // split tone
    if creative_on {
        let st_k = u.p4.xyz;
        let ht_k = u.p5.xyz;
        let l2 = clamp(luma709(v), 0.0, 1.0);
        v = v + st_k * (1.0 - l2) + ht_k * l2;
    }

    // saturation & vibrance
    let sat = u.p2.y;
    let vib = u.p2.z;
    let l3 = luma709(v);
    let cur_sat = max(max(v.x, v.y), v.z) - min(min(v.x, v.y), v.z);
    let sat_mult = sat * (1.0 + vib * (1.0 - clamp(cur_sat, 0.0, 1.0)));
    v = vec3(l3) + (v - vec3(l3)) * sat_mult;

    // vignette
    let va = u.p2.w;
    if vignette_on && abs(va) > 1e-4 {
        let vmid = u.p3.x;
        let vfeather = u.p3.y;
        let vround_aspect = u.p3.z;
        let w = f32(s.x);
        let h = f32(s.y);
        let nx = (f32(p.x) / w - 0.5) * 2.0 * vround_aspect;
        let ny = (f32(p.y) / h - 0.5) * 2.0;
        let d = sqrt(nx * nx + ny * ny) / 1.4142135623730951;
        let edge = clamp((d - vmid * 0.9) / (max(vfeather, 0.01) * 0.9), 0.0, 1.0);
        let e2 = edge * edge * (3.0 - 2.0 * edge);
        let k = 1.0 + va * 0.2 * e2;
        if va < 0.0 {
            v = v * max(k, 0.0);
        } else {
            v = v + (vec3(1.0) - v) * (k - 1.0);
        }
    }

    return dec(v);
}

// ---- the Creator effects (`filmcraft_render::creator_fx`, the same arithmetic step by step)

const OP_CR_FILM_GRAIN: u32 = 40u;
const OP_CR_VHS: u32 = 41u;
const OP_CR_KALEIDOSCOPE: u32 = 42u;
const OP_CR_GLITCH: u32 = 43u;
const OP_CR_ZOOM_BLUR: u32 = 44u;
const OP_CR_MOTION_BLUR: u32 = 45u;
const OP_CR_TECH_HUD: u32 = 46u;
const OP_CR_CHROMATIC: u32 = 47u;
const OP_CR_LIGHT_SWEEP: u32 = 48u;
const OP_CR_DUOTONE: u32 = 49u;
const OP_CR_HALFTONE: u32 = 50u;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash3(x: u32, y: u32, z: u32) -> u32 {
    return pcg(x ^ pcg(y ^ pcg(z)));
}

fn h01(x: u32, y: u32, z: u32) -> f32 {
    return f32(hash3(x, y, z) >> 8u) * (1.0 / 16777216.0);
}

fn ifloor(v: f32) -> u32 {
    return bitcast<u32>(i32(floor(v)));
}

fn straight(o: vec4<f32>) -> vec3<f32> {
    return select(o.rgb / o.a, vec3(0.0), o.a <= 1e-6);
}

fn value_noise(qx: f32, qy: f32, seed: u32) -> f32 {
    let fx0 = floor(qx);
    let fy0 = floor(qy);
    let ix = ifloor(qx);
    let iy = ifloor(qy);
    let fx = qx - fx0;
    let fy = qy - fy0;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let a = h01(ix, iy, seed);
    let b = h01(ix + 1u, iy, seed);
    let c = h01(ix, iy + 1u, seed);
    let d = h01(ix + 1u, iy + 1u, seed);
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    return top + (bot - top) * sy;
}

fn creator(op: u32, p: vec2<i32>) -> vec4<f32> {
    let s = size();
    let wf = f32(s.x);
    let hf = f32(s.y);
    let pc = vec2<f32>(p) + 0.5;
    switch op {
        case OP_CR_FILM_GRAIN: {
            let o = ld(p);
            if o.a <= 1e-6 {
                return o;
            }
            let amount = u.p0.x;
            let gsize = u.p0.y;
            let response = u.p0.z;
            let seed = u.i1.y;
            let qx = pc.x / gsize;
            let qy = pc.y / gsize;
            let e = enc(straight(o));
            let l = clamp(luma709(e), 0.0, 1.0);
            let k = amount * 0.5 * (1.0 - response + response * 4.0 * l * (1.0 - l));
            let g0 = value_noise(qx, qy, seed) - 0.5;
            var g = vec3(g0);
            if u.i1.x != 0u {
                g = vec3(g0, value_noise(qx, qy, seed ^ 0x9E3779B9u) - 0.5, value_noise(qx, qy, seed ^ 0x7F4A7C15u) - 0.5);
            }
            return vec4(dec(e + g * k) * o.a, o.a);
        }
        case OP_CR_VHS: {
            let amount = u.p0.x;
            let bleed = u.p0.y;
            let tracking = u.p0.z;
            let noise = u.p0.w;
            let wobble = u.p1.x;
            let sat = u.p1.y;
            let phase = u.p1.z;
            let band_y = u.p1.w;
            let band_h = u.p2.x;
            let frame = u.i1.x;
            let seed = u.i1.y;
            let row = u32(p.y);
            let rowh = h01(row, frame, seed);
            let wob = wobble * (0.65 * sin(pc.y * 0.031 + phase * 6.0) + 0.35 * (rowh * 2.0 - 1.0));
            let tl = tracking * clamp(1.0 - abs(pc.y - band_y) / band_h, 0.0, 1.0);
            let sx = pc.x - (wob + tl * (rowh - 0.5) * 0.08 * wf);
            let s0 = sample_clamped(sx, pc.y);
            let e0 = enc(straight(s0));
            let yl = luma709(e0);
            var chroma = vec3(0.0);
            for (var k = 0; k < 8; k++) {
                let xk = sx - bleed * (f32(k) / 7.0) - bleed * 0.25;
                let ek = enc(straight(sample_clamped(xk, pc.y)));
                chroma += ek - vec3(luma709(ek));
            }
            let nz = (h01(u32(p.x), row ^ frame, seed ^ 0x51EDu) - 0.5) * (noise * 0.35 + tl * 0.6);
            let e = (vec3(yl) + chroma * 0.125 * sat + vec3(nz)) * 0.94 + 0.03;
            let o = ld(p);
            let eo = enc(straight(o));
            let a = o.a + (s0.a - o.a) * amount;
            return vec4(dec(eo + (e - eo) * amount) * a, a);
        }
        case OP_CR_KALEIDOSCOPE: {
            let amount = u.p0.x;
            let segments = u.p0.y;
            let rot = u.p0.z;
            let cx = u.p0.w;
            let cy = u.p1.x;
            let zoom = u.p1.y;
            let seg = 6.283185307179586 / segments;
            let dx = pc.x - cx;
            let dy = pc.y - cy;
            let r = sqrt(dx * dx + dy * dy) / zoom;
            let th = atan2(dy, dx) - rot;
            var m = th - seg * floor(th / seg);
            if m > seg * 0.5 {
                m = seg - m;
            }
            let th2 = m + rot;
            let k = sample_clamped(cx + r * cos(th2), cy + r * sin(th2));
            let o = ld(p);
            return o + (k - o) * amount;
        }
        case OP_CR_GLITCH: {
            let amount = u.p0.x;
            let blk = u.p0.y;
            let split = u.p0.z;
            let jitter = u.p0.w;
            let gstep = u.i1.x;
            let seed = u.i1.y;
            let bw = blk * 2.0;
            let bh = max(blk * 0.5, 1.0);
            let bx = ifloor(pc.x / bw);
            let by = ifloor(pc.y / bh);
            let st = gstep ^ seed;
            let r = h01(bx, by, st);
            var dx = 0.0;
            if r < amount * 0.4 {
                dx = (h01(by, bx ^ 0x2C1Bu, st ^ 0x5BD1u) - 0.5) * 4.0 * blk * amount;
            }
            let row = u32(p.y);
            if h01(row, gstep, seed ^ 0x1234u) < jitter * amount * 0.5 {
                dx += (h01(row ^ 0x3C6Eu, gstep, seed ^ 0x77u) - 0.5) * jitter * blk;
            }
            let spl = split * amount * (0.5 + h01(by, gstep, seed ^ 0xABCu));
            let sx = pc.x - dx;
            let rs = sample_clamped(sx + spl, pc.y);
            let gs = sample_clamped(sx, pc.y);
            let bs = sample_clamped(sx - spl, pc.y);
            let a = max(max(rs.a, gs.a), bs.a);
            var o = vec4(rs.r, gs.g, bs.b, a);
            if r > 1.0 - amount * 0.03 {
                o = vec4(max(vec3(a) - o.rgb, vec3(0.0)), a);
            }
            return o;
        }
        case OP_CR_ZOOM_BLUR: {
            let amount = u.p0.x;
            let cx = u.p0.y;
            let cy = u.p0.z;
            let steps = u.i1.x;
            let spin = u.i1.y != 0u;
            let dx = pc.x - cx;
            let dy = pc.y - cy;
            var acc = vec4(0.0);
            for (var i = 0u; i < steps; i++) {
                let t = f32(i) / f32(steps - 1u);
                var q: vec2<f32>;
                if spin {
                    let ang = (t - 0.5) * amount * 0.5235987755982988;
                    let sn = sin(ang);
                    let cs = cos(ang);
                    q = vec2(cx + dx * cs - dy * sn, cy + dx * sn + dy * cs);
                } else {
                    let k = 1.0 - amount * 0.5 * t;
                    q = vec2(cx + dx * k, cy + dy * k);
                }
                acc += sample_clamped(q.x, q.y);
            }
            return acc / f32(steps);
        }
        case OP_CR_MOTION_BLUR: {
            let dx = u.p0.x;
            let dy = u.p0.y;
            let falloff = u.p0.z;
            let steps = u.i1.x;
            let trail = u.i1.y != 0u;
            var acc = vec4(0.0);
            var wsum = 0.0;
            for (var i = 0u; i < steps; i++) {
                let t = f32(i) / f32(steps - 1u);
                var off = t - 0.5;
                var far = abs(t - 0.5) * 2.0;
                if trail {
                    off = -t;
                    far = t;
                }
                let wt = 1.0 - falloff * 0.95 * far;
                acc += sample_clamped(pc.x + dx * off, pc.y + dy * off) * wt;
                wsum += wt;
            }
            return acc / wsum;
        }
        case OP_CR_TECH_HUD: {
            let o = ld(p);
            if o.a <= 1e-6 {
                return o;
            }
            let amount = u.p0.x;
            let scan = u.p0.y;
            let spacing = u.p0.z;
            let grid = u.p0.w;
            let grid_size = u.p1.x;
            let brackets = u.p1.y;
            let vignette = u.p1.z;
            let tint = u.p1.w;
            let flicker = u.p2.x;
            let lw = u.p2.y;
            let col = u.p3.xyz;
            let cmax = max(max(max(col.x, col.y), col.z), 1e-4);
            let mn = min(wf, hf);
            let m = 0.06 * mn;
            let arm = 0.09 * mn;
            let th = max(0.006 * mn, 1.0);
            let e0 = enc(straight(o));
            let l = luma709(e0);
            let sl = 0.5 - 0.5 * cos(6.283185307179586 * pc.y / spacing);
            let vx = (pc.x / wf - 0.5) * 2.0;
            let vy = (pc.y / hf - 0.5) * 2.0;
            let v = smoothstep_fx(0.45, 1.0, sqrt(vx * vx + vy * vy) / 1.4142135623730951);
            let shade = (1.0 - scan * 0.45 * sl) * (1.0 + flicker) * (1.0 - vignette * 0.75 * v);
            let tinted = e0 + (vec3(l) * col / cmax - e0) * tint * 0.6;
            var e = tinted * shade;
            let gx = abs(fract(pc.x / grid_size + 0.5) - 0.5) * grid_size;
            let gy = abs(fract(pc.y / grid_size + 0.5) - 0.5) * grid_size;
            let gl = max(clamp(1.0 - gx / lw, 0.0, 1.0), clamp(1.0 - gy / lw, 0.0, 1.0)) * grid * 0.35;
            let fx = min(pc.x, wf - pc.x) - m;
            let fy = min(pc.y, hf - pc.y) - m;
            let cov_h = clamp(min(min(fx + 0.5, arm - fx + 0.5), min(fy + 0.5, th - fy + 0.5)), 0.0, 1.0);
            let cov_v = clamp(min(min(fx + 0.5, th - fx + 0.5), min(fy + 0.5, arm - fy + 0.5)), 0.0, 1.0);
            let cov = clamp(gl + max(cov_h, cov_v) * brackets, 0.0, 1.0);
            e = e + (col - e) * cov;
            return vec4(dec(e0 + (e - e0) * amount) * o.a, o.a);
        }
        case OP_CR_CHROMATIC: {
            let k = u.p0.x;
            let dx = u.p0.y;
            let dy = u.p0.z;
            let cx = u.p0.w;
            let cy = u.p1.x;
            var rs: vec4<f32>;
            var bs: vec4<f32>;
            if u.i1.x != 0u {
                let ox = pc.x - cx;
                let oy = pc.y - cy;
                rs = sample_clamped(cx + ox * (1.0 + k), cy + oy * (1.0 + k));
                bs = sample_clamped(cx + ox * (1.0 - k), cy + oy * (1.0 - k));
            } else {
                rs = sample_clamped(pc.x + dx, pc.y + dy);
                bs = sample_clamped(pc.x - dx, pc.y - dy);
            }
            let g = ld(p);
            return vec4(rs.r, g.g, bs.b, max(max(rs.a, g.a), bs.a));
        }
        case OP_CR_LIGHT_SWEEP: {
            let o = ld(p);
            if o.a <= 1e-6 {
                return o;
            }
            let intensity = u.p0.x;
            let nx = u.p0.y;
            let ny = u.p0.z;
            let s0 = u.p0.w;
            let halfw = u.p1.x;
            let inner = u.p1.y;
            let col = u.p2.xyz;
            let d = (pc.x - wf * 0.5) * nx + (pc.y - hf * 0.5) * ny - s0;
            let band = 1.0 - smoothstep_fx(inner, halfw, abs(d));
            let k = band * intensity * o.a;
            return vec4(o.rgb + col * k, o.a);
        }
        case OP_CR_DUOTONE: {
            let o = ld(p);
            if o.a <= 1e-6 {
                return o;
            }
            let amount = u.p0.x;
            let contrast = u.p0.y;
            let e = enc(straight(o));
            let l = clamp((clamp(luma709(e), 0.0, 1.0) - 0.5) * (1.0 + contrast) + 0.5, 0.0, 1.0);
            let t = u.p1.xyz + (u.p2.xyz - u.p1.xyz) * l;
            return vec4(dec(e + (t - e) * amount) * o.a, o.a);
        }
        case OP_CR_HALFTONE: {
            let o = ld(p);
            if o.a <= 1e-6 {
                return o;
            }
            let amount = u.p0.x;
            let dsize = u.p0.y;
            let cs = u.p0.z;
            let sn = u.p0.w;
            let ink = u.p1.xyz;
            let paper = u.p2.xyz;
            let uu = cs * pc.x + sn * pc.y;
            let vv = -sn * pc.x + cs * pc.y;
            let cu = (floor(uu / dsize) + 0.5) * dsize;
            let cvv = (floor(vv / dsize) + 0.5) * dsize;
            let es = enc(straight(sample_clamped(cs * cu - sn * cvv, sn * cu + cs * cvv)));
            let d = sqrt((uu - cu) * (uu - cu) + (vv - cvv) * (vv - cvv));
            var t: vec3<f32>;
            if u.i1.x != 0u {
                t = clamp(sqrt(clamp(es, vec3(0.0), vec3(1.0))) * dsize * 0.7071068 - vec3(d) + 0.5, vec3(0.0), vec3(1.0));
            } else {
                let r = sqrt(1.0 - clamp(luma709(es), 0.0, 1.0)) * dsize * 0.7071068;
                t = paper + (ink - paper) * clamp(r - d + 0.5, 0.0, 1.0);
            }
            let e = enc(straight(o));
            return vec4(dec(e + (t - e) * amount) * o.a, o.a);
        }
        default: {
            return ld(p);
        }
    }
}

// One output pixel of every op but the running-sum box blur.
fn pixel(op: u32, p: vec2<i32>) -> vec4<f32> {
    let s = size();
    let pc = vec2<f32>(p) + 0.5;
    switch op {
        case OP_INVERT_ALPHA: {
            let o = ld(p);
            let a = o.a;
            let na = 1.0 - a;
            // (the CPU's threshold; `na / a` is only used when a > 1e-6)
            var k = 0.0;
            if a > 1e-6 {
                k = na / a;
            }
            let blend = u.p0.x;
            return vec4(o.rgb * k, na * (1.0 - blend) + a * blend);
        }
        case OP_ALPHA_ADJUST: {
            let o = ld(p);
            let c = select(o.rgb / o.a, vec3(0.0), o.a <= 1e-6);
            var a = select(o.a, 1.0, u.i1.x != 0u);
            if u.i1.y != 0u {
                a = 1.0 - a;
            }
            a = clamp(a * u.p0.x, 0.0, 1.0);
            if u.i1.z != 0u {
                return vec4(vec3(srgb_to_linear1(a)), 1.0);
            }
            return vec4(c * a, a);
        }
        case OP_BOX: {
            // one box pass along x (i1.z = 0) or y (1): mean over 2r+1 pixels, clamped to the
            // edge (repeat) or skipping pixels outside, as `effects::box_rows`
            let r = i32(u.i1.x);
            let repeat = u.i1.y != 0u;
            let vertical = u.i1.z != 0u;
            let n = select(s.x, s.y, vertical);
            let x = select(p.x, p.y, vertical);
            var acc = vec4(0.0);
            for (var i = x - r; i <= x + r; i++) {
                if repeat || (i >= 0 && i < n) {
                    let j = clamp(i, 0, n - 1);
                    acc += ld(select(vec2(j, p.y), vec2(p.x, j), vertical));
                }
            }
            return acc * (1.0 / f32(2 * r + 1));
        }
        case OP_DIRECTIONAL: {
            let steps = u.i1.x;
            let dx = u.p0.x; let dy = u.p0.y;
            var acc = vec4(0.0);
            for (var i = 0u; i < steps; i++) {
                let t = f32(i) / f32(steps - 1u) - 0.5;
                acc += sample_clamped(pc.x + dx * t, pc.y + dy * t);
            }
            return acc / f32(steps);
        }
        case OP_MASK_MIX: {
            // src: the effected image, aux: the original (`mask::mix`)
            let o = textureLoad(aux, p, 0);
            return o + (ld(p) - o) * mask_coverage(p);
        }
        case OP_MASK_SCALE: {
            // opacity masks (`mask::scale_by`)
            return ld(p) * mask_coverage(p);
        }
        case OP_UNSHARP: {
            // src: the blurred image, aux: the original
            var o = textureLoad(aux, p, 0);
            let bq = ld(p);
            let amount = u.p0.x; let th = u.p0.y;
            for (var k = 0; k < 3; k++) {
                let d = o[k] - bq[k];
                if abs(d) >= th * o.a {
                    o[k] = max(o[k] + d * amount, 0.0);
                }
            }
            return o;
        }
        case OP_CROP: {
            let x0 = u.p0.x; let x1 = u.p0.y; let y0 = u.p0.z; let y1 = u.p0.w; let fe = u.p1.x;
            let d = min(min(pc.x - x0, x1 - pc.x), min(pc.y - y0, y1 - pc.y));
            var a: f32;
            if fe > 0.0 {
                a = clamp(d / fe, 0.0, 1.0);
            } else {
                a = clamp(d + 0.5, 0.0, 1.0);
            }
            let o = ld(p);
            return select(o, o * a, a < 1.0);
        }
        case OP_RESAMPLE: {
            // inverse map p0 = (a b c d), p1 = (e f opacity valid); rect i1 = x0 x1 y0 y1
            if u.p1.w == 0.0 || u32(p.x) < u.i1.x || u32(p.x) >= u.i1.y || u32(p.y) < u.i1.z || u32(p.y) >= u.i1.w {
                return vec4(0.0);
            }
            let m0 = u.p0; let m1 = u.p1;
            let uu = m0.x * pc.x + m0.z * pc.y + m1.x;
            let vv = m0.y * pc.x + m0.w * pc.y + m1.y;
            if uu < -1.0 || vv < -1.0 || uu > f32(s.x) + 1.0 || vv > f32(s.y) + 1.0 {
                return vec4(0.0);
            }
            return sample_bilinear(uu, vv) * m1.z;
        }
        case OP_HFLIP: {
            return ld(vec2(s.x - 1 - p.x, p.y));
        }
        case OP_VFLIP: {
            return ld(vec2(p.x, s.y - 1 - p.y));
        }
        case OP_MIRROR: {
            let c = u.p0.xy; let nrm = u.p0.zw;
            let d = (pc.x - c.x) * nrm.x + (pc.y - c.y) * nrm.y;
            if d > 0.0 {
                return sample_bilinear(pc.x - 2.0 * d * nrm.x, pc.y - 2.0 * d * nrm.y);
            }
            return ld(p);
        }
        case OP_OFFSET: {
            let w = f32(s.x); let h = f32(s.y);
            // dx, dy in 0..w / 0..h, so the shifted coordinate is within (-w, w + 0.5)
            var uu = pc.x - u.p0.x;
            var vv = pc.y - u.p0.y;
            if uu < 0.0 { uu += w; }
            if uu >= w { uu -= w; }
            if vv < 0.0 { vv += h; }
            if vv >= h { vv -= h; }
            let q = sample_clamped(uu, vv);
            let o = ld(p);
            return q + (o - q) * u.p0.z;
        }
        case OP_VIGNETTE: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            let amt = u.p0.x;
            if abs(amt) < 1e-5 {
                return o;
            }
            let mid = u.p0.y;
            let round = u.p0.z;
            let feather = u.p0.w;
            let tgt = u.p1.xyz;
            let w = f32(s.x);
            let h = f32(s.y);
            let aspect = w / h;
            var nx = pc.x / w * 2.0 - 1.0;
            let ny = pc.y / h * 2.0 - 1.0;
            if round > 0.0 {
                nx *= 1.0 + (aspect - 1.0) * round;
            }
            let p_exp = select(2.0, 2.0 + (-round) * 6.0, round < 0.0);
            let d = powf(powf(abs(nx), p_exp) + powf(abs(ny), p_exp), 1.0 / p_exp) / powf(2.0, 1.0 / p_exp);
            let edge = smoothstep_fx(mid - feather * 0.5, mid + feather * 0.5, d);
            let c = o.rgb / a;
            let ec = enc(c);
            let res = dec(ec + (tgt - ec) * (edge * abs(amt)));
            return vec4(res * a, a);
        }
        case OP_VIDEO_LIMITER: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            let c = o.rgb / a;
            let v = vec3(linear_to_enc_limiter(c.x), linear_to_enc_limiter(c.y), linear_to_enc_limiter(c.z));
            let max_val = u.p0.x;
            let comp = u.p0.y;
            let axis = u.i1.x;
            let warn = u.i1.y != 0u;
            let wc = u.p0.zw;
            let wc_b = u.p1.x;
            let warning_col = vec3(wc.x, wc.y, wc_b);
            let out_enc = limit_video(v, max_val, comp, axis);
            let diff = abs(v - out_enc);
            if warn && (diff.x > 1e-4 || diff.y > 1e-4 || diff.z > 1e-4) {
                return vec4(warning_col * a, a);
            }
            let res = vec3(enc_to_linear_limiter(out_enc.x), enc_to_linear_limiter(out_enc.y), enc_to_linear_limiter(out_enc.z));
            return vec4(res * a, a);
        }
        case OP_CR_FILM_GRAIN, OP_CR_VHS, OP_CR_KALEIDOSCOPE, OP_CR_GLITCH, OP_CR_ZOOM_BLUR, OP_CR_MOTION_BLUR, OP_CR_TECH_HUD, OP_CR_CHROMATIC, OP_CR_LIGHT_SWEEP, OP_CR_DUOTONE, OP_CR_HALFTONE: {
            return creator(op, p);
        }
        case OP_LUMETRI: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            let c = o.rgb / a;
            let res = lumetri_op(c, p);
            return vec4(res * a, a);
        }
        default: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            return vec4(color_op(op, o.rgb / a) * a, a);
        }
    }
}

@compute @workgroup_size(16, 16)
fn fx_px(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.i0.y || id.y >= u.i0.z {
        return;
    }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, pixel(u.i0.x, p));
}

// A box pass as a running sum along one row (i1.z = 0) or column (1) per invocation: O(1) per
// pixel for any radius, the arithmetic of `effects::box_rows`.
@compute @workgroup_size(64)
fn fx_run(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = size();
    let vertical = u.i1.z != 0u;
    let n = select(s.x, s.y, vertical);
    let lines = select(s.y, s.x, vertical);
    let line = i32(id.x);
    if line >= lines {
        return;
    }
    let r = i32(u.i1.x);
    let repeat = u.i1.y != 0u;
    let inv = 1.0 / f32(2 * r + 1);
    var acc = vec4(0.0);
    for (var i = -r; i <= r; i++) {
        if repeat || (i >= 0 && i < n) {
            let j = clamp(i, 0, n - 1);
            acc += ld(select(vec2(j, line), vec2(line, j), vertical));
        }
    }
    for (var x = 0; x < n; x++) {
        textureStore(dst, select(vec2(x, line), vec2(line, x), vertical), acc * inv);
        let out_i = x - r;
        let in_i = x + r + 1;
        if repeat || out_i >= 0 {
            let j = clamp(out_i, 0, n - 1);
            acc -= ld(select(vec2(j, line), vec2(line, j), vertical));
        }
        if repeat || in_i < n {
            let j = clamp(in_i, 0, n - 1);
            acc += ld(select(vec2(j, line), vec2(line, j), vertical));
        }
    }
}
