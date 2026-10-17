// License: MIT
// Author: pthrasher
// Description: The incoming image appears through a crosshatch of lines spreading from the center
// Tags: pattern, lines, noise
// adapted by gre from https://gist.github.com/pthrasher/04fd9a7de4012cbb03f6
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/crosshatch.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.xy = center (uv, default (0.5, 0.5); the host takes it from the Center
// point), host.p0.z = threshold (default 3, at least 0.01), host.p0.w = fadeEdge (default 0.1).
// FilmCraft: `rand(vec2(p.y, 0.0))` and `rand(vec2(0.0, p.x))` are the host's integer hashes of
// the output pixel row and column (identical on CPU and GPU).

fn transition(p: vec2<f32>) -> vec4<f32> {
    let center = host.p0.xy;
    let threshold = host.p0.z;
    let fadeEdge = host.p0.w;
    let dist = distance(center, p) / threshold;
    let r = progress - min(rand_row(p, 4u), rand_column(p, 5u));
    return mix(getFromColor(p), getToColor(p), vec4(mix(0.0, mix(step(dist, r), 1.0, smooth_step(1.0 - fadeEdge, 1.0, progress)), smooth_step(0.0, fadeEdge, progress))));
}
