// FilmCraft pattern transition engine: original work, MIT OR Apache-2.0.
//
// One shader behind the Particles & Patterns presets and two glitches
// (`filmcraft_project::transition_catalog`): halftone dots, triangle tiles, heart bubbles and a
// sparkle dissolve on a grid of cells, each cell turning at its own time (random, left to right
// or from the centre out); an RGB split glitch and a scanline glitch around the cut.
// Parameters: host.p0 = (mode, cells per row, order, seed), host.p1 = (colour r, g, b, amount),
// host.p2 = (shift in pixels, bands or line height, 0, 0).

const PT_DUR: f32 = 0.35;

// The size of a cell (pixels) and the cell `x` is in.
fn pt_size() -> f32 {
    return host.frame.x / max(host.p0.y, 1.0);
}

fn pt_cell(x: vec2<f32>, s: f32) -> vec2<f32> {
    return floor(x / s + vec2(0.001));
}

// When a cell (or a point of the pattern at `centre`) starts turning, 0..1, by the order.
fn pt_delay(cell: vec2<f32>, centre: vec2<f32>, salt: u32) -> f32 {
    let s = u32(host.p0.w) * 131u + salt;
    let order = u32(host.p0.z);
    if order == 1u {
        return clamp(centre.x * host.inv.x * 0.92 + 0.08 * hash(i32(cell.x), i32(cell.y), s), 0.0, 1.0);
    }
    if order == 2u {
        let half = host.frame.xy * 0.5;
        return clamp(length(centre - half) / length(half) * 0.92 + 0.08 * hash(i32(cell.x), i32(cell.y), s), 0.0, 1.0);
    }
    return hash(i32(cell.x), i32(cell.y), s);
}

// How far a cell starting at `delay` has turned at t.
fn pt_local(delay: f32, t: f32) -> f32 {
    return clamp((t - delay * (1.0 - PT_DUR)) / PT_DUR, 0.0, 1.0);
}

fn pt_mix(x: vec2<f32>, k: f32) -> vec4<f32> {
    return mix(getFromPx(x), getToPx(x), vec4(k));
}

// Halftone Dots: a dot of the incoming picture grows in every cell until it fills it.
fn pt_halftone(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = pt_size();
    let cell = pt_cell(x, s);
    let centre = (cell + vec2(0.5)) * s;
    let q = pt_local(pt_delay(cell, centre, 0u), t);
    let r = q * (0.71 * s + 1.0);
    let k = (1.0 - smooth_step(r - 0.5, r + 0.5, length(x - centre))) * min(q * 20.0, 1.0);
    return pt_mix(x, k);
}

// Triangle Tiles: every cell is two triangles (the diagonal alternating like a checkerboard);
// each grows the incoming picture from its centroid at its own time.
fn pt_triangles(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = pt_size();
    let cell = pt_cell(x, s);
    let u = x.x / s - cell.x;
    let v = x.y / s - cell.y;
    let even = glsl_mod(cell.x + cell.y, 2.0) < 0.5;
    var tri = 0.0;
    var l = vec3(0.0);
    var cen = vec2(0.0);
    if even {
        if u >= v {
            tri = 1.0;
            l = vec3(1.0 - u, u - v, v);
            cen = vec2(2.0, 1.0) / 3.0;
        } else {
            l = vec3(1.0 - v, u, v - u);
            cen = vec2(1.0, 2.0) / 3.0;
        }
    } else {
        if u + v < 1.0 {
            l = vec3(1.0 - u - v, u, v);
            cen = vec2(1.0, 1.0) / 3.0;
        } else {
            tri = 1.0;
            l = vec3(1.0 - v, u + v - 1.0, 1.0 - u);
            cen = vec2(2.0, 2.0) / 3.0;
        }
    }
    let id = vec2(cell.x * 2.0 + tri, cell.y);
    let q = pt_local(pt_delay(id, (cell + cen) * s, 1u), t);
    let m = 1.0 - 3.0 * min(l.x, min(l.y, l.z));
    let aa = 1.5 / s;
    let edge = q * (1.0 + 2.0 * aa);
    let k = (1.0 - smooth_step(edge - aa, edge + aa, m)) * min(q * 20.0, 1.0);
    return pt_mix(x, k);
}

// Heart Bubbles: a heart of the incoming picture grows in every cell until it fills it.
fn pt_hearts(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = pt_size();
    let cell = pt_cell(x, s);
    let centre = (cell + vec2(0.5)) * s;
    let q = pt_local(pt_delay(cell, centre, 2u), t);
    if q >= 1.0 {
        return getToPx(x);
    }
    let size = q * 1.6 * s * 0.5;
    if size <= 0.0 {
        return getFromPx(x);
    }
    let hx = (x.x - centre.x) / size;
    let hy = (centre.y - x.y) / size + 0.15;
    let a = hx * hx + hy * hy - 1.0;
    let inside = a * a * a - hx * hx * hy * hy * hy <= 0.0;
    return select(getFromPx(x), getToPx(x), inside);
}

// Sparkle Dissolve: cells flip to the incoming picture in random order, each with a four-point
// glint of the sparkle colour as it turns.
fn pt_sparkle(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = pt_size();
    let cell = pt_cell(x, s);
    let centre = (cell + vec2(0.5)) * s;
    let q = pt_local(pt_delay(cell, centre, 3u), t);
    let base = pt_mix(x, smooth_step(0.0, 1.0, q));
    var glint = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let nc = cell + vec2(f32(i), f32(j));
            let nctr = (nc + vec2(0.5)) * s;
            let peak = pt_delay(nc, nctr, 3u) * (1.0 - PT_DUR) + PT_DUR * 0.5;
            let dt = (t - peak) / 0.05;
            let g = exp(-dt * dt);
            let d = abs(x - nctr) / s;
            let star = exp(-d.x / 0.05) * exp(-d.y / 0.6) + exp(-d.y / 0.05) * exp(-d.x / 0.6) + exp(-dot(d, d) / 0.01);
            glint += g * star;
        }
    }
    let k = glint * host.p1.w * fx_bell(t) * 1.5;
    return vec4(base.rgb + host.p1.rgb * k, min(base.a + k, 1.0));
}

// RGB Split Glitch: bands of the picture jump sideways with their colour channels apart,
// strongest at the cut, where the pictures swap.
fn pt_rgb_split(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = u32(host.p0.w);
    let d = (t - 0.5) / 0.15;
    let g = host.p2.x * exp(-d * d);
    let step = i32(floor(t * 30.0));
    let band = i32(floor(x.y * host.inv.y * max(host.p2.y, 1.0) + 0.001));
    let h = hash(band, step, s);
    let off = (hash(band, step, s + 1u) - 0.5) * 2.0 * g * select(0.2, 1.0, h > 0.6);
    let p = x + vec2(off, 0.0);
    let split = vec2(g * 0.5, 0.0);
    if t < 0.5 {
        let m = getFromPx(p);
        return vec4(getFromPx(p + split).r, m.g, getFromPx(p - split).b, m.a);
    }
    let m = getToPx(p);
    return vec4(getToPx(p + split).r, m.g, getToPx(p - split).b, m.a);
}

// Scanline Glitch: thin lines tear sideways, alternate lines darken, and line by line the picture
// switches around the cut.
fn pt_scanlines(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = u32(host.p0.w);
    let env = fx_bell(t);
    let lh = max(host.p2.y, 1.0);
    let line = i32(floor(x.y / lh + 0.001));
    let step = i32(floor(t * 30.0));
    let tear = select(0.0, (hash(line, step, s + 2u) - 0.5) * 2.0 * host.p2.x * env, hash(line, step, s + 3u) < 0.3 * env);
    let p = x + vec2(tear, 0.0);
    let swap = 0.4 + 0.2 * hash(line, 0, s + 4u);
    let c = select(getFromPx(p), getToPx(p), t >= swap);
    let dark = 1.0 - 0.3 * env * select(0.0, 1.0, (line & 1) == 1);
    return vec4(c.rgb * dark, c.a);
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let x = uv_px(uv);
    let t = progress;
    switch u32(host.p0.x) {
        case 1u: { return pt_triangles(x, t); }
        case 2u: { return pt_hearts(x, t); }
        case 3u: { return pt_sparkle(x, t); }
        case 4u: { return pt_rgb_split(x, t); }
        case 5u: { return pt_scanlines(x, t); }
        default: { return pt_halftone(x, t); }
    }
}
