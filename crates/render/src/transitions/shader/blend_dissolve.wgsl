// FilmCraft blend-mode dissolve engine: original work, MIT OR Apache-2.0.
//
// One shader behind every Blend Dissolve (`filmcraft_project::transition_engines`): a cross
// dissolve whose middle is the incoming picture blended onto the outgoing one with one of the
// compositor's blend modes, weighted 4·p·(1 − p) (none at either end, all of it at the
// midpoint) times Strength. The blend is the compositor's own (W3C Compositing and Blending on
// straight, sRGB-encoded colour, `filmcraft_render::blend`); Dissolve becomes a dither from one
// picture to the other.
// Parameters: host.p0.x = blend mode (index into BLEND_MODES), host.p0.y = strength (0..1).

fn bd_screen(b: f32, s: f32) -> f32 { return b + s - b * s; }

fn bd_color_burn(b: f32, s: f32) -> f32 {
    if b >= 1.0 { return 1.0; }
    if s <= 0.0 { return 0.0; }
    return 1.0 - min((1.0 - b) / s, 1.0);
}

fn bd_color_dodge(b: f32, s: f32) -> f32 {
    if b <= 0.0 { return 0.0; }
    if s >= 1.0 { return 1.0; }
    return min(b / (1.0 - s), 1.0);
}

fn bd_hard_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 { return b * 2.0 * s; }
    return bd_screen(b, 2.0 * s - 1.0);
}

fn bd_vivid_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 { return bd_color_burn(b, 2.0 * s); }
    return bd_color_dodge(b, 2.0 * s - 1.0);
}

fn bd_sep(mode: u32, b: f32, s: f32) -> f32 {
    switch mode {
        case 2u: { return min(b, s); }
        case 3u: { return b * s; }
        case 4u: { return bd_color_burn(b, s); }
        case 5u: { return max(b + s - 1.0, 0.0); }
        case 7u: { return max(b, s); }
        case 8u: { return bd_screen(b, s); }
        case 9u: { return bd_color_dodge(b, s); }
        case 10u: { return min(b + s, 1.0); }
        case 12u: { return bd_hard_light(s, b); }
        case 13u: {
            if s <= 0.5 { return b - (1.0 - 2.0 * s) * b * (1.0 - b); }
            var d = sqrt(b);
            if b <= 0.25 { d = ((16.0 * b - 12.0) * b + 4.0) * b; }
            return b + (2.0 * s - 1.0) * (d - b);
        }
        case 14u: { return bd_hard_light(b, s); }
        case 15u: { return bd_vivid_light(b, s); }
        case 16u: { return clamp(b + 2.0 * s - 1.0, 0.0, 1.0); }
        case 17u: {
            if s <= 0.5 { return min(b, 2.0 * s); }
            return max(b, 2.0 * s - 1.0);
        }
        case 18u: { return select(0.0, 1.0, bd_vivid_light(b, s) >= 0.5); }
        case 19u: { return abs(b - s); }
        case 20u: { return b + s - 2.0 * b * s; }
        case 21u: { return max(b - s, 0.0); }
        case 22u: {
            if s <= 0.0 { return 1.0; }
            return min(b / s, 1.0);
        }
        default: { return s; }
    }
}

fn bd_lum(c: vec3<f32>) -> f32 { return 0.3 * c.r + 0.59 * c.g + 0.11 * c.b; }

fn bd_clip_color(c: vec3<f32>) -> vec3<f32> {
    let l = bd_lum(c);
    let n = min(min(c.r, c.g), c.b);
    let x = max(max(c.r, c.g), c.b);
    var o = c;
    if n < 0.0 { o = l + (o - l) * l / max(l - n, 1e-6); }
    if x > 1.0 { o = l + (o - l) * (1.0 - l) / max(x - l, 1e-6); }
    return o;
}

fn bd_set_lum(c: vec3<f32>, l: f32) -> vec3<f32> { return bd_clip_color(c + (l - bd_lum(c))); }

fn bd_sat(c: vec3<f32>) -> f32 { return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b); }

fn bd_set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    if mx - mn <= 1e-6 { return vec3(0.0); }
    return (c - mn) * s / (mx - mn);
}

fn bd_blend_rgb(mode: u32, b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 23u: { return bd_set_lum(bd_set_sat(s, bd_sat(b)), bd_lum(b)); }
        case 24u: { return bd_set_lum(bd_set_sat(b, bd_sat(s)), bd_lum(b)); }
        case 25u: { return bd_set_lum(s, bd_lum(b)); }
        case 26u: { return bd_set_lum(b, bd_lum(s)); }
        case 6u: { return select(b, s, bd_lum(s) < bd_lum(b)); }
        case 11u: { return select(b, s, bd_lum(s) > bd_lum(b)); }
        default: { return vec3(bd_sep(mode, b.r, s.r), bd_sep(mode, b.g, s.g), bd_sep(mode, b.b, s.b)); }
    }
}

fn bd_to_srgb(v: vec3<f32>) -> vec3<f32> {
    let hi = 1.055 * pow(max(v, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return select(hi, v * 12.92, v <= vec3(0.0031308));
}

fn bd_to_linear(v: vec3<f32>) -> vec3<f32> {
    let lo = v / 12.92;
    let hi = pow((max(v, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return select(hi, lo, v <= vec3(0.04045));
}

// The incoming picture `b` composited onto the outgoing `a` with blend mode `mode`.
fn bd_blended(a: vec4<f32>, b: vec4<f32>, mode: u32) -> vec4<f32> {
    let sa = b.a;
    let da = a.a;
    if sa <= 0.0 {
        return a;
    }
    if mode == 0u || da <= 0.0 {
        return b + a * (1.0 - sa);
    }
    let cs = bd_to_srgb(b.rgb / sa);
    let cb = bd_to_srgb(a.rgb / da);
    let mixed = bd_to_linear(clamp(bd_blend_rgb(mode, cb, cs), vec3(0.0), vec3(1.0)));
    return vec4(b.rgb * (1.0 - da) + a.rgb * (1.0 - sa) + sa * da * mixed, sa + da - sa * da);
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let a = getFromColor(uv);
    let b = getToColor(uv);
    let mode = u32(host.p0.x);
    let strength = host.p0.y;
    let fade = mix(a, b, vec4(progress));
    if mode == 1u {
        // Dissolve: each pixel switches at its own random moment
        let dither = select(a, b, rand_pixel(uv) < progress);
        return mix(fade, dither, vec4(strength));
    }
    let k = strength * 4.0 * progress * (1.0 - progress);
    return mix(fade, bd_blended(a, b, mode), vec4(k));
}
