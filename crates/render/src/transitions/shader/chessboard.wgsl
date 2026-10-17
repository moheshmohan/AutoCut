// Author: lql
// License: MIT
// Description: A chessboard pattern of squares reveals the incoming image
// Tags: grid, squares, pattern
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/chessboard.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = grid_num (number of squares per row, default 10, at least 1).
// FilmCraft: the square index is biased by 1/1000 of a square (and the position in the square
// follows it), so a pixel centre exactly on a square's edge falls the same way on every GPU and
// on the CPU.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let grid_num = host.p0.x;
    let st = uv * grid_num;
    let idx = floor(st + 0.001);
    let grid = st - idx;

    let a = getFromColor(uv);
    let b = getToColor(uv);

    let checker = glsl_mod(idx.x + idx.y, 2.0);
    var mixFactor: f32;

    if progress <= 0.5 {
        mixFactor = select(0.0, step(grid.x, progress * 2.0), checker > 0.5);
    } else {
        mixFactor = select(1.0, step(grid.x, (progress - 0.5) * 2.0), checker < 0.5);
    }

    return mix(a, b, vec4(mixFactor));
}
