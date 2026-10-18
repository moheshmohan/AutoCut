// FilmCraft flow transition engine: original work, MIT OR Apache-2.0.
//
// One shader behind the flow transitions (`filmcraft_project::flow_transitions`): Flow Morph,
// Doppler Warp, Hype Ramp, Glitch Flow, Split Bands and Datamosh. The host computes the dense
// optical flow between the outgoing clip and the incoming one at the cut, both ways, once per
// transition (Lucas–Kanade, smoothed: `filmcraft_render::transitions::content`), and hands it
// over as the content field: `getField(at).xy` is how far the outgoing picture's content at `at`
// moves to reach the incoming picture, `.zw` the way back, in frame pixels.
//
// The morph: at progress t the outgoing picture is warped t of the way along its flow and the
// incoming one (1 − t) of the way back along its own, so both show the same in-between shapes;
// they dissolve across the middle (Dissolve Width). Each warp is a backward map with one
// fixed-point step (the flow is read where the content comes from, not where it lands). The
// field is smooth, so the warp never tears.
// Parameters: host.p0 = (mode, flow strength, dissolve width, a), host.p1 = (b, c, d, e),
// host.p2 = (centre x, centre y in frame pixels, seed, 0); a..e by mode:
//   1 Doppler Warp: a streak length, b zoom;
//   2 Hype Ramp: a ramp, b blur, c brightness pulse, e punch zoom;
//   3 Glitch Flow: a block size (pixels), b amount, c RGB split (pixels), d smear;
//   4 Split Bands: a bands, b orientation (0 horizontal), c stagger;
//   5 Datamosh: a block size (pixels), b smear, c refresh spread.

struct Pair {
    a: vec4<f32>,
    b: vec4<f32>,
}

// Where the outgoing picture's content shown at `x` comes from, `s` of the way along the flow.
fn flow_src_a(x: vec2<f32>, s: f32) -> vec2<f32> {
    let first = x - getField(x).xy * s;
    return x - getField(first).xy * s;
}

// Where the incoming picture's content shown at `x` comes from, `s` of the way back.
fn flow_src_b(x: vec2<f32>, s: f32) -> vec2<f32> {
    let first = x - getField(x).zw * s;
    return x - getField(first).zw * s;
}

fn morph_pair(x: vec2<f32>, t: f32) -> Pair {
    let k = host.p0.y;
    return Pair(getFromPx(flow_src_a(x, t * k)), getToPx(flow_src_b(x, (1.0 - t) * k)));
}

// The incoming picture's weight at t: a dissolve across the middle.
fn dissolve_at(t: f32) -> f32 {
    let half = 0.5 * host.p0.z;
    return smooth_step(0.5 - half, 0.5 + half, t);
}

fn morph(x: vec2<f32>, t: f32) -> vec4<f32> {
    let m = morph_pair(x, t);
    return mix(m.a, m.b, vec4(dissolve_at(t)));
}

// Doppler Warp: the morph through a zoom on the centre, streaked towards it as far as the
// local motion between the clips is fast.
fn doppler(x: vec2<f32>, t: f32) -> vec4<f32> {
    let c = host.p2.xy;
    let env = fx_bell(t);
    let q = c + (x - c) / (1.0 + host.p1.x * env);
    let f = getField(q);
    let speed = length(f.xy - f.zw) * 0.5 / max(host.frame.x, host.frame.y);
    let reach = host.p0.w * env * (0.15 + min(speed * 20.0, 1.0) * 0.85) * 0.35;
    var acc = vec4(0.0);
    for (var i = 0; i < 12; i++) {
        let s = f32(i) / 11.0;
        acc += morph(q + (c - q) * (reach * s), t);
    }
    return acc / 12.0;
}

// Hype Ramp: the morph on a speed ramp (slow, a rush through the swap, slow), blurred along the
// motion and punched in as fast as it goes, with a brightness pulse at the swap.
fn hype(x: vec2<f32>, t: f32) -> vec4<f32> {
    let r = max(host.p0.w, 0.01);
    let th = tanh(r * (2.0 * t - 1.0));
    let ramped = 0.5 + 0.5 * th / tanh(r);
    let speed = 1.0 - th * th;
    let c = host.frame.xy * 0.5;
    let q = c + (x - c) / (1.0 + host.p1.w * speed);
    let f = getField(q);
    let along = (f.xy - f.zw) * (0.5 * host.p0.y);
    let radial = (q - c) * 0.12;
    let len = host.p1.x * speed;
    var acc = vec4(0.0);
    for (var i = 0; i < 10; i++) {
        let s = f32(i) / 9.0 - 0.5;
        acc += morph(q + (along + radial) * (len * s), ramped);
    }
    let m = acc / 10.0;
    let d = (t - 0.5) / 0.12;
    let pulse = 1.0 + host.p1.y * exp(-d * d);
    return vec4(m.rgb * pulse, m.a);
}

// Glitch Flow: the morph, with datamosh blocks around the cut: pushed along their clip's flow
// (smeared), jittered and split into colour channels; the pattern changes 24 times over.
fn glitch(x: vec2<f32>, t: f32) -> vec4<f32> {
    let base = morph(x, t);
    let d = (t - 0.5) / 0.14;
    let g = host.p1.x * exp(-d * d);
    let bs = max(host.p0.w, 2.0);
    let cell = floor(x / bs + vec2(0.001));
    let seed = u32(host.p2.z) * 977u + u32(floor(t * 24.0)) * 7919u;
    let r = hash(i32(cell.x), i32(cell.y), seed + 11u);
    if r >= g {
        return base;
    }
    let f = getField((cell + vec2(0.5)) * bs);
    let smear = host.p1.z * host.p0.y;
    let jitter = (hash(i32(cell.x), i32(cell.y), seed + 12u) - 0.5) * bs;
    let split = vec2(host.p1.y, 0.0);
    if t < 0.5 {
        let at = x - f.xy * (t * smear) + vec2(jitter, 0.0);
        let mid = getFromPx(at);
        return vec4(getFromPx(at + split).r, mid.g, getFromPx(at - split).b, mid.a);
    }
    let at = x - f.zw * ((1.0 - t) * smear) + vec2(jitter, 0.0);
    let mid = getToPx(at);
    return vec4(getToPx(at + split).r, mid.g, getToPx(at - split).b, mid.a);
}

// Split Bands: bands slide out alternately (staggered), the incoming picture following each one
// in; both keep morphing along the flow while they move.
fn bands(x: vec2<f32>, t: f32) -> vec4<f32> {
    let n = max(host.p0.w, 1.0);
    let horizontal = host.p1.x < 0.5;
    let stagger = clamp(host.p1.y, 0.0, 0.9);
    var along = x.x;
    var across = x.y * host.inv.y;
    var len = host.frame.x;
    if !horizontal {
        along = x.y;
        across = x.x * host.inv.x;
        len = host.frame.y;
    }
    let band = clamp(floor(across * n + 0.001), 0.0, n - 1.0);
    let dir = select(-1.0, 1.0, glsl_mod(band, 2.0) < 0.5);
    let delay = stagger * band / max(n - 1.0, 1.0);
    let q = clamp((t - delay) / (1.0 - stagger), 0.0, 1.0);
    let a_along = along - dir * fx_ease_in_out(q) * len;
    let k = host.p0.y;
    var p = x;
    if a_along >= 0.0 && a_along < len {
        if horizontal { p.x = a_along; } else { p.y = a_along; }
        return getFromPx(flow_src_a(p, t * k));
    }
    let b_along = a_along + dir * len;
    if horizontal { p.x = b_along; } else { p.y = b_along; }
    return getToPx(flow_src_b(p, (1.0 - t) * k));
}

// Datamosh: the outgoing picture dragged further and further along the flow (the incoming
// clip's motion applied to the old pixels) until each block refreshes to the incoming picture.
fn datamosh(x: vec2<f32>, t: f32) -> vec4<f32> {
    let bs = max(host.p0.w, 2.0);
    let cell = floor(x / bs + vec2(0.001));
    let tau = 0.3 + host.p1.y * 0.55 * hash(i32(cell.x), i32(cell.y), u32(host.p2.z) * 977u + 31u);
    let wb = smooth_step(tau - 0.06, tau + 0.06, t);
    let k = host.p0.y;
    let drag = host.p1.x * k * t * t;
    let ca = getFromPx(x - getField(x).xy * drag);
    let cb = getToPx(flow_src_b(x, (1.0 - t) * k));
    return mix(ca, cb, vec4(wb));
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let x = uv_px(uv);
    let t = progress;
    switch u32(host.p0.x) {
        case 1u: { return doppler(x, t); }
        case 2u: { return hype(x, t); }
        case 3u: { return glitch(x, t); }
        case 4u: { return bands(x, t); }
        case 5u: { return datamosh(x, t); }
        default: { return morph(x, t); }
    }
}
