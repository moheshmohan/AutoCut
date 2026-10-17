// Author: gre
// License: MIT
// Description: The two images are faces of a rotating 3D cube, with a floor reflection
// Tags: 3d, cube, rotate, reflection
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/cube.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = persp (amount of perspective, default 0.7), host.p0.y = unzoom
// (default 0.3), host.p0.z = reflection (strength of the reflection, default 0.4), host.p0.w =
// floating (height of the cube above its reflection, default 3).
// FilmCraft: the floor is opaque black plus the reflections, whose alpha (above 1 upstream, which
// an 8-bit target clamps) is clamped to 1. A single pass like the others: the "3D" is upstream's
// skew of the two images.

fn project(p: vec2<f32>) -> vec2<f32> {
    let floating = host.p0.w;
    return p * vec2(1.0, -1.2) + vec2(0.0, -floating / 100.0);
}

fn bgColor(p: vec2<f32>, pfr_in: vec2<f32>, pto_in: vec2<f32>) -> vec4<f32> {
    let reflection = host.p0.z;
    var c = vec4(0.0, 0.0, 0.0, 1.0);
    let pfr = project(pfr_in);
    if in_bounds(pfr) {
        c += mix(vec4(0.0), getFromColor(pfr), vec4(reflection * mix(1.0, 0.0, pfr.y)));
    }
    let pto = project(pto_in);
    if in_bounds(pto) {
        c += mix(vec4(0.0), getToColor(pto), vec4(reflection * mix(1.0, 0.0, pto.y)));
    }
    return c;
}

// p : the position
// persp : the perspective in [ 0, 1 ]
// center : the xcenter in [0, 1] \ 0.5 excluded
fn xskew(p: vec2<f32>, persp: f32, center: f32) -> vec2<f32> {
    let x = mix(p.x, 1.0 - p.x, center);
    return (
        (
            vec2(x, (p.y - 0.5 * (1.0 - persp) * x) / (1.0 + (persp - 1.0) * x))
            - vec2(0.5 - distance(center, 0.5), 0.0)
        )
        * vec2(0.5 / distance(center, 0.5) * select(-1.0, 1.0, center < 0.5), 1.0)
        + vec2(select(1.0, 0.0, center < 0.5), 0.0)
    );
}

fn transition(op: vec2<f32>) -> vec4<f32> {
    let persp = host.p0.x;
    let unzoom = host.p0.y;
    let uz = unzoom * 2.0 * (0.5 - distance(0.5, progress));
    let p = -uz * 0.5 + (1.0 + uz) * op;
    let fromP = xskew(
        (p - vec2(progress, 0.0)) / vec2(1.0 - progress, 1.0),
        1.0 - mix(progress, 0.0, persp),
        0.0
    );
    let toP = xskew(
        p / vec2(progress, 1.0),
        mix(pow(progress, 2.0), 1.0, persp),
        1.0
    );
    if in_bounds(fromP) {
        return getFromColor(fromP);
    } else if in_bounds(toP) {
        return getToColor(toP);
    }
    let c = bgColor(op, fromP, toP);
    return vec4(c.rgb, min(c.a, 1.0));
}
