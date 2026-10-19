// FilmCraft light & film transition engine: original work, MIT OR Apache-2.0.
//
// One shader behind the Light & Film presets (`filmcraft_project::transition_catalog`): warm and
// prism light leaks, a burn-through, a lens-flare sweep, an exposure flash, a bokeh blur and a
// projector roll. Every light is generated here from a few seeded shapes (no footage, no
// textures): soft Gaussian blobs drifting across, a noisy burn front, a flare's core, streak and
// ghosts, golden-angle disc taps.
// Parameters: host.p0 = (mode, amount, seed, a), host.p1 = (colour r, g, b, b),
// host.p2 = (c, d, 0, 0); a..d by mode:
//   0 Warm Light Leak, 1 Prism Leak (a: colour split, pixels), 2 Burn Through (a: edge width,
//   0..1), 3 Lens Flare Sweep (c, d: the sweep's direction, a unit vector worked out on the
//   host), 4 Exposure Flash, 5 Bokeh Blur (a: radius, pixels; b: highlight boost), 6 Projector
//   Roll (a: frame line, pixels; c: flicker; d: scratches).

// The cross dissolve under the light: switches around the middle.
fn lf_base(x: vec2<f32>, t: f32, e0: f32, e1: f32) -> vec4<f32> {
    return mix(getFromPx(x), getToPx(x), vec4(smooth_step(e0, e1, t)));
}

// A rainbow colour for hue h (0..1).
fn lf_rainbow(h: f32) -> vec3<f32> {
    return vec3(0.5) + 0.5 * cos(6.2831853 * (vec3(h) + vec3(0.0, 0.33333334, 0.6666667)));
}

// Four soft blobs drifting across the frame (seeded), coloured `tint` (prism: each its own
// rainbow hue along x).
fn lf_leak(x: vec2<f32>, t: f32, prism: bool) -> vec4<f32> {
    let s = u32(host.p0.z);
    let diag = length(host.frame.xy);
    let env = fx_bell(t) * host.p0.y;
    var light = vec3(0.0);
    for (var i = 0; i < 4; i++) {
        let cx = host.frame.x * (hash(i, 1, s) + (t - 0.5) * (0.6 + 0.8 * hash(i, 2, s)));
        let cy = host.frame.y * (0.15 + 0.7 * hash(i, 3, s));
        let r = diag * (0.18 + 0.2 * hash(i, 4, s));
        let d = x - vec2(cx, cy);
        let k = exp(-dot(d, d) / (r * r));
        var c = mix(host.p1.rgb, vec3(1.0, 0.85, 0.55), vec3(0.5 * hash(i, 6, s)));
        if prism {
            c = lf_rainbow(hash(i, 7, s) + x.x * host.inv.x * 0.5);
        }
        light += c * k;
    }
    var base = lf_base(x, t, 0.3, 0.7);
    if prism {
        let split = vec2(host.p0.w * env, 0.0);
        let m = smooth_step(0.3, 0.7, t);
        let r = mix(getFromPx(x + split), getToPx(x + split), vec4(m)).r;
        let b = mix(getFromPx(x - split), getToPx(x - split), vec4(m)).b;
        base = vec4(r, base.g, b, base.a);
    }
    let k = env * 0.9;
    return vec4(base.rgb + light * k, min(base.a + k * max(light.r, max(light.g, light.b)), 1.0));
}

// Burn Through: a noisy front grows from a seeded point; the outgoing picture glows, chars and
// opens onto the incoming one.
fn lf_burn(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = u32(host.p0.z);
    let diag = length(host.frame.xy);
    let start = vec2(host.frame.x * (0.25 + 0.5 * hash(0, 11, s)), host.frame.y * (0.25 + 0.5 * hash(0, 12, s)));
    let n = fx_vnoise(x / (diag * 0.06), s) * 0.6 + fx_vnoise(x / (diag * 0.018), s + 1u) * 0.4;
    let phi = length(x - start) / diag + 0.25 * n;
    let band = max(host.p0.w, 0.005);
    let front = t * (1.0 + 3.0 * band) - band;
    let d = phi - front;
    let a = getFromPx(x);
    let b = getToPx(x);
    let hot = host.p1.rgb;
    if d < -band {
        return b;
    }
    if d < 0.0 {
        // the charred rim: dark, opening onto the incoming picture
        let k = 1.0 + d / band;
        return vec4(b.rgb * (1.0 - 0.85 * k), b.a);
    }
    // the glowing edge, and the heat ahead of it
    let warm = smooth_step(0.0, 0.1, t);
    if d < band {
        let g = (1.0 - d / band);
        return vec4(a.rgb + hot * (g * g * 2.5 * warm), min(a.a + g * warm, 1.0));
    }
    let heat = 0.25 * exp(-(d - band) / (2.0 * band)) * warm;
    return vec4(a.rgb + hot * heat, min(a.a + heat, 1.0));
}

// Lens Flare Sweep: a light source crosses the frame along the sweep angle; core, anamorphic
// streak and three ghosts mirrored through the centre; the cut happens as it passes the middle.
fn lf_flare(x: vec2<f32>, t: f32) -> vec4<f32> {
    let diag = length(host.frame.xy);
    let c = host.frame.xy * 0.5;
    let dir = vec2(host.p2.x, host.p2.y);
    let src = c + dir * ((t - 0.5) * diag * 1.1);
    let env = fx_bell(t) * host.p0.y;
    let d = x - src;
    let core = exp(-dot(d, d) / (diag * diag * 0.0036));
    let streak = exp(-abs(d.y) / (diag * 0.006)) * exp(-abs(d.x) / (diag * 0.5));
    var ghosts = vec3(0.0);
    var gc = array<vec3<f32>, 3>(vec3(0.3, 0.8, 0.9), vec3(0.7, 0.4, 0.9), vec3(0.9, 0.7, 0.3));
    var gk = array<f32, 3>(0.5, 1.0, 1.6);
    var gr = array<f32, 3>(0.03, 0.05, 0.08);
    for (var i = 0; i < 3; i++) {
        let g = c + (c - src) * gk[i];
        let r = diag * gr[i];
        let dist = length(x - g);
        ghosts += gc[i] * (1.0 - smooth_step(r * 0.8, r, dist)) * 0.25;
    }
    let light = host.p1.rgb * (core * 1.5 + streak * 0.8 + 0.12) + ghosts;
    let base = lf_base(x, t, 0.4, 0.6);
    return vec4(base.rgb + light * env, min(base.a + env * (core + 0.12), 1.0));
}

// Exposure Flash: the picture blows out to the flash colour at the cut and comes back.
fn lf_flash(x: vec2<f32>, t: f32) -> vec4<f32> {
    let d = (t - 0.5) / 0.1;
    let f = host.p0.y * exp(-d * d);
    let base = lf_base(x, t, 0.45, 0.55);
    return vec4(base.rgb * (1.0 + 3.0 * f) + host.p1.rgb * f, min(base.a + f, 1.0));
}

// Bokeh Blur: both pictures defocus into discs (bright highlights bloom into bokeh) towards the
// cut, then the incoming one refocuses.
fn lf_bokeh_side(side: u32, x: vec2<f32>, r: f32) -> vec4<f32> {
    var acc = vec4(0.0);
    var wsum = 0.0;
    for (var i = 0; i < 24; i++) {
        let rr = r * sqrt((f32(i) + 0.5) / 24.0);
        let a = fract(f32(i) * 0.38196602) * 6.2831853;
        let c = sample_px(side, x + vec2(cos(a), sin(a)) * rr);
        let l = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
        let w = 1.0 + host.p1.w * l * l * 4.0;
        acc += c * w;
        wsum += w;
    }
    return acc / wsum;
}

fn lf_bokeh(x: vec2<f32>, t: f32) -> vec4<f32> {
    let env = fx_bell(t);
    let r = host.p0.w * env;
    let m = smooth_step(0.3, 0.7, t);
    let c = mix(lf_bokeh_side(0u, x, r), lf_bokeh_side(1u, x, r), vec4(m));
    return vec4(c.rgb * (1.0 + 0.15 * env * host.p1.w), c.a);
}

// Projector Roll: the film loses its loop: the frame rolls up past a black frame line to the
// incoming picture, with flicker, gate weave and scratches.
fn lf_projector(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = u32(host.p0.z);
    let env = fx_bell(t);
    let step = i32(floor(t * 24.0));
    let gap = host.p0.w;
    let weave = (hash(1, step, s) - 0.5) * 4.0 * env;
    let o = fx_ease_in_out(t) * (host.frame.y + gap);
    let ya = x.y + o;
    let p = vec2(x.x + weave, ya);
    var c = vec4(0.0, 0.0, 0.0, 1.0);
    if ya < host.frame.y {
        c = getFromPx(p);
    } else if ya >= host.frame.y + gap {
        c = getToPx(vec2(p.x, ya - host.frame.y - gap));
    }
    let flicker = 1.0 - host.p2.x * env * hash(0, step, s) * 0.5;
    let scratch = select(0.0, 0.4 * env, hash(i32(floor(x.x * 0.5)), step, s + 5u) > 1.0 - 0.06 * host.p2.y);
    return vec4(c.rgb * flicker + vec3(scratch) * c.a, c.a);
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let x = uv_px(uv);
    let t = progress;
    switch u32(host.p0.x) {
        case 1u: { return lf_leak(x, t, true); }
        case 2u: { return lf_burn(x, t); }
        case 3u: { return lf_flare(x, t); }
        case 4u: { return lf_flash(x, t); }
        case 5u: { return lf_bokeh(x, t); }
        case 6u: { return lf_projector(x, t); }
        default: { return lf_leak(x, t, false); }
    }
}
