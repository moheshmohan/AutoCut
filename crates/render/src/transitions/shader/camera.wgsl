// FilmCraft camera transition engine: original work, MIT OR Apache-2.0.
//
// One shader behind the camera moves of Zooms & Camera and 3D & Spatial
// (`filmcraft_project::transition_catalog`): punch zoom in, pull zoom out, spin zoom, whip pan
// and camera shake, a 3D card flip and a tilted slide. Zooms and the shake read the frame
// mirrored beyond its edges (a zoom out shows the picture reflected, never a smeared edge); the
// 3D moves project the picture as a card under a perspective camera, transparent around it.
// Parameters: host.p0 = (mode, a, b, c):
//   0 Punch Zoom In, 1 Pull Zoom Out (a: zoom), 2 Spin Zoom (a: rotation, radians; b: zoom),
//   3 Whip Pan (a, b: direction of travel; c: blur), 4 Camera Shake (a: shake, pixels; b: speed;
//   c: seed), 5 Card Flip (a: axis, 0 = turning about the vertical; b: camera distance in half
//   heights), 6 Tilt Slide (a, b: direction of travel; c: tilt, radians).

fn cam_ease_in(t: f32) -> f32 {
    let u = clamp(t, 0.0, 1.0);
    return u * u * u;
}

fn cam_ease_out(t: f32) -> f32 {
    let u = 1.0 - clamp(t, 0.0, 1.0);
    return 1.0 - u * u * u;
}

// `v` reflected into 0..size (the frame mirrored beyond its edges).
fn cam_mirror(v: f32, size: f32) -> f32 {
    let m = glsl_mod(v, 2.0 * size);
    return select(m, 2.0 * size - m, m >= size);
}

fn cam_sample(side: u32, p: vec2<f32>) -> vec4<f32> {
    return sample_px(side, vec2(cam_mirror(p.x, host.frame.x), cam_mirror(p.y, host.frame.y)));
}

// The zoom-and-turn of the outgoing (first half) and incoming (second half) pictures: each is
// read at the centre plus the pixel's offset turned by -angle and divided by the zoom.
fn cam_view(side: u32, x: vec2<f32>, zoom: f32, angle: f32) -> vec4<f32> {
    let c = host.frame.xy * 0.5;
    let d = (x - c) / zoom;
    let s = sin(angle);
    let co = cos(angle);
    return cam_sample(side, c + vec2(d.x * co + d.y * s, -d.x * s + d.y * co));
}

fn cam_zooms(x: vec2<f32>, t: f32, mode: u32) -> vec4<f32> {
    let into = cam_ease_in(t / 0.5);
    let outof = 1.0 - cam_ease_out((t - 0.5) / 0.5);
    let z = host.p0.y;
    var za = 1.0 + z * into;
    var zb = 1.0 + z * outof;
    var ra = 0.0;
    var rb = 0.0;
    if mode == 1u {
        za = 1.0 / za;
        zb = 1.0 / zb;
    }
    if mode == 2u {
        za = 1.0 + host.p0.z * into;
        zb = 1.0 + host.p0.z * outof;
        ra = host.p0.y * into;
        rb = -host.p0.y * outof;
    }
    return mix(cam_view(0u, x, za, ra), cam_view(1u, x, zb, rb), vec4(smooth_step(0.42, 0.58, t)));
}

// Whip Pan: a fast push, blurred along the travel as fast as it goes.
// The push of the outgoing picture `e` of the way along `d` (a whole frame `len` long), the
// incoming one following it in, at `p`.
fn cam_push(p: vec2<f32>, d: vec2<f32>, e: f32, len: f32, horizontal: bool) -> vec4<f32> {
    let pa = p - d * (e * len);
    let along = select(pa.y, pa.x, horizontal);
    if along >= 0.0 && along < len {
        return getFromPx(pa);
    }
    return getToPx(pa + d * len);
}

fn cam_whip(x: vec2<f32>, t: f32) -> vec4<f32> {
    let d = vec2(host.p0.y, host.p0.z);
    let horizontal = abs(d.x) > 0.5;
    let len = select(host.frame.y, host.frame.x, horizontal);
    let e = fx_ease_in_out(t);
    let speed = select(12.0 * (1.0 - t) * (1.0 - t), 12.0 * t * t, t < 0.5) / 3.0;
    let blur = host.p0.w * speed * len * 0.25;
    var acc = vec4(0.0);
    for (var i = 0; i < 16; i++) {
        let s = f32(i) / 15.0 - 0.5;
        acc += cam_push(x + d * (blur * s), d, e, len, horizontal);
    }
    return acc / 16.0;
}

// Camera Shake: the frame jolts about (offsets from a seeded walk, smoothed between steps), the
// pictures swapping at the cut.
fn cam_shake(x: vec2<f32>, t: f32) -> vec4<f32> {
    let s = u32(host.p0.w);
    let env = fx_bell(t);
    let f = t * max(host.p0.z, 1.0);
    let k = i32(floor(f));
    let w = smooth_step(0.0, 1.0, f - floor(f));
    let o0 = vec2(hash(k, 1, s), hash(k, 2, s)) - vec2(0.5);
    let o1 = vec2(hash(k + 1, 1, s), hash(k + 1, 2, s)) - vec2(0.5);
    let o = mix(o0, o1, vec2(w)) * (2.0 * host.p0.y * env);
    let p = x + o;
    return mix(cam_sample(0u, p), cam_sample(1u, p), vec4(smooth_step(0.45, 0.55, t)));
}

// A point of the screen (`q`: half heights from the centre, along the turning axis's normal and
// along the axis) on a card turned by `phi` under a camera `dist` half heights away: the card
// coordinates, and whether the ray meets the card's front.
fn cam_card(q: vec2<f32>, phi: f32, dist: f32) -> vec3<f32> {
    let den = dist * cos(phi) - q.x * sin(phi);
    if den <= 1e-4 {
        return vec3(0.0, 0.0, -1.0);
    }
    let u = q.x * dist / den;
    let v = q.y * (dist + u * sin(phi)) / dist;
    return vec3(u, v, 1.0);
}

// Card Flip: the picture turns over like a card, the incoming picture on its back.
fn cam_flip(x: vec2<f32>, t: f32) -> vec4<f32> {
    let hh = host.frame.y * 0.5;
    let c = host.frame.xy * 0.5;
    let theta = 3.1415927 * fx_ease_in_out(t);
    let back = theta >= 1.5707964;
    let phi = select(theta, theta - 3.1415927, back);
    let vertical_axis = host.p0.y < 0.5;
    let rel = (x - c) / hh;
    let q = select(rel.yx, rel, vertical_axis);
    let half = select(vec2(1.0, host.frame.x / host.frame.y), vec2(host.frame.x / host.frame.y, 1.0), vertical_axis);
    let card = cam_card(q, phi, host.p0.z);
    if card.z < 0.0 || abs(card.x) > half.x || abs(card.y) > half.y {
        return vec4(0.0);
    }
    let uv = select(card.yx, card.xy, vertical_axis);
    let p = c + uv * hh;
    let shade = 0.6 + 0.4 * cos(phi);
    let col = select(getFromPx(p), getToPx(p), back);
    return vec4(col.rgb * shade, col.a);
}

// Tilt Slide: a push whose pictures lie on a wall the camera turns along as it travels.
fn cam_tilt(x: vec2<f32>, t: f32) -> vec4<f32> {
    let d = vec2(host.p0.y, host.p0.z);
    let horizontal = abs(d.x) > 0.5;
    let hh = host.frame.y * 0.5;
    let c = host.frame.xy * 0.5;
    let len = select(host.frame.y, host.frame.x, horizontal);
    let phi = host.p0.w * fx_bell(t) * select(d.y, d.x, horizontal);
    let rel = (x - c) / hh;
    let q = select(rel.yx, rel, horizontal);
    let card = cam_card(q, phi, 4.0);
    let limit = select(host.frame.x / host.frame.y, 1.0, horizontal);
    if card.z < 0.0 || abs(card.y) > limit {
        return vec4(0.0);
    }
    let p = c + select(card.yx, card.xy, horizontal) * hh;
    return cam_push(p, d, fx_ease_in_out(t), len, horizontal);
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let x = uv_px(uv);
    let t = progress;
    let mode = u32(host.p0.x);
    switch mode {
        case 3u: { return cam_whip(x, t); }
        case 4u: { return cam_shake(x, t); }
        case 5u: { return cam_flip(x, t); }
        case 6u: { return cam_tilt(x, t); }
        default: { return cam_zooms(x, t, mode); }
    }
}
