// Author: gre
// License: MIT
// Description: The incoming image appears in random squares
// Tags: squares, grid, dissolve
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/randomsquares.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.xy = size (number of squares: columns, rows; default (10, 10), whole numbers
// of at least 1), host.p0.z = smoothness (softness of each square's switch, default 0.5).
// FilmCraft: `rand` is the host's integer cell hash (identical on CPU and GPU), and the square
// index is biased by 1/1000 of a square so edges fall the same way everywhere. Upstream's
// `smoothstep(0.0, -smoothness, x)` (edges in reverse order, which GLSL leaves undefined but
// every GPU evaluates by the formula) is written `1.0 - smoothstep(-smoothness, 0.0, x)`, the
// same curve.

fn transition(p: vec2<f32>) -> vec4<f32> {
    let size = host.p0.xy;
    let smoothness = host.p0.z;
    let r = rand_cell(floor(size * p + 0.001));
    let m = 1.0 - smooth_step(-smoothness, 0.0, r - (progress * (1.0 + smoothness)));
    return mix(getFromColor(p), getToColor(p), vec4(m));
}
