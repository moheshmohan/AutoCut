// Author: gre
// License: MIT
// Description: The incoming image appears through vertical slices that widen
// Tags: slices, stripes, wipe
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/windowslice.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = count (number of slices, default 10), host.p0.y = smoothness (softness
// of the slices' edges, default 0.5).
// FilmCraft: the position in the slice is biased by 1/1000 of a slice, so a pixel centre exactly
// on a slice's edge (uv.x = 0.5 on an odd width) falls the same way on every GPU and on the CPU.

fn transition(p: vec2<f32>) -> vec4<f32> {
    let count = host.p0.x;
    let smoothness = host.p0.y;
    let pr = smooth_step(-smoothness, 0.0, p.x - progress * (1.0 + smoothness));
    let s = step(pr, fract(count * p.x + 0.001));
    return mix(getFromColor(p), getToColor(p), vec4(s));
}
