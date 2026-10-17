// Author: gre
// License: MIT
// Description: The two images swap places in 3D, with a floor reflection
// Tags: 3d, swap, reflection
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/swap.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = reflection (strength of the reflection, default 0.4), host.p0.y =
// perspective (amount of perspective, default 0.2), host.p0.z = depth (default 3).
// FilmCraft: the floor is opaque black plus the reflections, whose alpha (above 1 upstream, which
// an 8-bit target clamps) is clamped to 1.

const black = vec4(0.0, 0.0, 0.0, 1.0);

fn project(p: vec2<f32>) -> vec2<f32> {
    return p * vec2(1.0, -1.2) + vec2(0.0, -0.02);
}

fn bgColor(p: vec2<f32>, pfr_in: vec2<f32>, pto_in: vec2<f32>) -> vec4<f32> {
    let reflection = host.p0.x;
    var c = black;
    let pfr = project(pfr_in);
    if in_bounds(pfr) {
        c += mix(black, getFromColor(pfr), vec4(reflection * mix(1.0, 0.0, pfr.y)));
    }
    let pto = project(pto_in);
    if in_bounds(pto) {
        c += mix(black, getToColor(pto), vec4(reflection * mix(1.0, 0.0, pto.y)));
    }
    return c;
}

fn transition(p: vec2<f32>) -> vec4<f32> {
    let perspective = host.p0.y;
    let depth = host.p0.z;

    var size = mix(1.0, depth, progress);
    var persp = perspective * progress;
    let pfr = (p + vec2(-0.0, -0.5)) * vec2(size / (1.0 - perspective * progress), size / (1.0 - size * persp * p.x)) + vec2(0.0, 0.5);

    size = mix(1.0, depth, 1.0 - progress);
    persp = perspective * (1.0 - progress);
    let pto = (p + vec2(-1.0, -0.5)) * vec2(size / (1.0 - perspective * (1.0 - progress)), size / (1.0 - size * persp * (0.5 - p.x))) + vec2(1.0, 0.5);

    if progress < 0.5 {
        if in_bounds(pfr) {
            return getFromColor(pfr);
        }
        if in_bounds(pto) {
            return getToColor(pto);
        }
    }
    if in_bounds(pto) {
        return getToColor(pto);
    }
    if in_bounds(pfr) {
        return getFromColor(pfr);
    }
    let c = bgColor(p, pfr, pto);
    return vec4(c.rgb, min(c.a, 1.0));
}
