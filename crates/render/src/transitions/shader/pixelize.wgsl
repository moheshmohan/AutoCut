// Author: gre
// License: MIT
// Description: Both images pixelate, crossfade, then sharpen again
// Tags: pixelate, blend
// forked from https://gist.github.com/benraziel/c528607361d90a072e98
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/pixelize.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.xy = squaresMin (minimum number of squares, at the strongest pixelation;
// default (20, 20), whole numbers of at least 1), host.p0.z = steps (number of pixelation steps,
// 0 for a continuous effect; default 50, a whole number).
// FilmCraft: the square index is biased by 1/1000 of a square, so a pixel centre exactly on a
// square's edge (common on odd frame sizes) falls the same way on every GPU and on the CPU.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let squaresMin = max(host.p0.xy, vec2(1.0));
    let steps = i32(host.p0.z);
    let d = min(progress, 1.0 - progress);
    var dist = d;
    if steps > 0 {
        dist = ceil(d * f32(steps)) / f32(steps);
    }
    let squareSize = 2.0 * dist / squaresMin;
    var p = uv;
    if dist > 0.0 {
        p = (floor(uv / squareSize + 0.001) + 0.5) * squareSize;
    }
    return mix(getFromColor(p), getToColor(p), vec4(progress));
}
