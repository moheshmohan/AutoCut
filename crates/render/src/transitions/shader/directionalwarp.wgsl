// Author: pschroen
// License: MIT
// Description: A diagonal wipe with a warped, smeared edge
// Tags: wipe, warp, directional
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/directionalwarp.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = smoothness (softness of the edge, default 0.1), host.p0.yz = direction
// (of the wipe, default (-1, 1); the host replaces a zero vector with the default).

const center = vec2(0.5, 0.5);

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let smoothness = host.p0.x;
    let direction = host.p0.yz;
    var v = normalize(direction);
    v /= abs(v.x) + abs(v.y);
    let d = v.x * center.x + v.y * center.y;
    let m = 1.0 - smooth_step(-smoothness, 0.0, v.x * uv.x + v.y * uv.y - (d - 0.5 + progress * (1.0 + smoothness)));
    return mix(getFromColor((uv - 0.5) * (1.0 - m) + 0.5), getToColor((uv - 0.5) * m + 0.5), vec4(m));
}
