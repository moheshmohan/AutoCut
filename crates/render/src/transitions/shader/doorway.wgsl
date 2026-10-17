// Author: gre
// License: MIT
// Description: The outgoing image opens like double doors as the incoming one comes forward, with a floor reflection
// Tags: doors, 3d, reflection
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/doorway.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = reflection (strength of the reflection, default 0.4), host.p0.y =
// perspective (amount of perspective, default 0.4), host.p0.z = depth (default 3).
// FilmCraft: the floor is opaque black plus the reflection, whose alpha (above 1 upstream, which
// an 8-bit target clamps) is clamped to 1.

const black = vec4(0.0, 0.0, 0.0, 1.0);

fn project(p: vec2<f32>) -> vec2<f32> {
    return p * vec2(1.0, -1.2) + vec2(0.0, -0.02);
}

fn bgColor(p: vec2<f32>, pto_in: vec2<f32>) -> vec4<f32> {
    let reflection = host.p0.x;
    var c = black;
    let pto = project(pto_in);
    if in_bounds(pto) {
        c += mix(black, getToColor(pto), vec4(reflection * mix(1.0, 0.0, pto.y)));
    }
    return c;
}

fn transition(p: vec2<f32>) -> vec4<f32> {
    let perspective = host.p0.y;
    let depth = host.p0.z;
    var pfr = vec2(-1.0);
    var pto = vec2(-1.0);
    let middleSlit = 2.0 * abs(p.x - 0.5) - progress;
    if middleSlit > 0.0 {
        pfr = p + select(1.0, -1.0, p.x > 0.5) * vec2(0.5 * progress, 0.0);
        let d = 1.0 / (1.0 + perspective * progress * (1.0 - middleSlit));
        pfr.y -= d / 2.0;
        pfr.y *= d;
        pfr.y += d / 2.0;
    }
    let size = mix(1.0, depth, 1.0 - progress);
    pto = (p + vec2(-0.5, -0.5)) * vec2(size, size) + vec2(0.5, 0.5);
    if in_bounds(pfr) {
        return getFromColor(pfr);
    } else if in_bounds(pto) {
        return getToColor(pto);
    }
    let c = bgColor(p, pto);
    return vec4(c.rgb, min(c.a, 1.0));
}
