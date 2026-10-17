// Author: gre
// License: MIT
// Description: A ragged left-to-right wipe, as if the image was blown away by the wind
// Tags: wipe, horizontal, noise
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/wind.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = size (width of the ragged edge, default 0.2), host.p0.y = reversed
// (1 = wipe from right to left; default 0).
// FilmCraft: `rand(vec2(0, uv.y))` is the host's integer hash of the output pixel row (identical
// on CPU and GPU). Upstream's `smoothstep(0.0, -size, x)` (edges in reverse order, which GLSL
// leaves undefined but every GPU evaluates by the formula) is written
// `1.0 - smoothstep(-size, 0.0, x)`, the same curve.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let size = host.p0.x;
    let reversed = host.p0.y > 0.5;
    let x = select(uv.x, 1.0 - uv.x, reversed);
    let r = rand_row(uv, 6u);
    let m = 1.0 - smooth_step(-size, 0.0, x * (1.0 - size) + size * r - (progress * (1.0 + size)));
    return mix(
        getFromColor(uv),
        getToColor(uv),
        vec4(m)
    );
}
