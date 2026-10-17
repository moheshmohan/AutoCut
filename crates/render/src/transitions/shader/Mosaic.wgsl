// License: MIT
// Author: Xaychru
// Description: The view pans across a mosaic of rotated tiles from both images and settles on the incoming one
// Tags: mosaic, tiles, zoom
// ported by gre from https://gist.github.com/Xaychru/130bb7b7affedbda9df5
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/Mosaic.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = endx (default 2), host.p0.y = endy (default -1), whole numbers.
// FilmCraft: `Rand` is the host's integer cell hash instead of the sin-based one, except at the
// starting cell (0, 0), where it stays 0 as upstream's is (the outgoing image, unrotated). The
// tile index is biased by 1/1000 of a tile (and the position in the tile follows it), so a pixel
// centre exactly on a tile edge falls the same way on every GPU and on the CPU.

fn Rand(v: vec2<f32>) -> f32 {
    if v.x == 0.0 && v.y == 0.0 {
        return 0.0;
    }
    return rand_cell(v);
}

fn Rotate(v: vec2<f32>, a: f32) -> vec2<f32> {
    let rm = mat2x2<f32>(cos(a), -sin(a), sin(a), cos(a));
    return rm * v;
}

fn CosInterpolation(x: f32) -> f32 {
    return -cos(x * PI) / 2.0 + 0.5;
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let endx = i32(host.p0.x);
    let endy = i32(host.p0.y);
    let p = uv.xy - 0.5;
    var rp = p;
    let rpr = (progress * 2.0 - 1.0);
    let z = -(rpr * rpr * 2.0) + 3.0;
    let az = abs(z);
    rp *= az;
    let ci = CosInterpolation(progress);
    rp += mix(vec2(0.5, 0.5), vec2(f32(endx) + 0.5, f32(endy) + 0.5), vec2(ci * ci));
    let crp = floor(rp + vec2(0.001));
    var mrp = rp - crp;
    let onEnd = i32(crp.x) == endx && i32(crp.y) == endy;
    if !onEnd {
        let ang = f32(i32(Rand(crp) * 4.0)) * 0.5 * PI;
        mrp = vec2(0.5) + Rotate(mrp - vec2(0.5), ang);
    }
    if onEnd || Rand(crp) > 0.5 {
        return getToColor(mrp);
    }
    return getFromColor(mrp);
}
