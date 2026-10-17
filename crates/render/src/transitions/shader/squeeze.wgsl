// Author: gre
// License: MIT
// Description: The outgoing image is squeezed into a line with color separation, revealing the incoming one
// Tags: squeeze, chromatic, vertical
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/squeeze.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = colorSeparation (amount of color separation, default 0.04).

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let colorSeparation = host.p0.x;
    let y = 0.5 + (uv.y - 0.5) / (1.0 - progress);
    if y < 0.0 || y > 1.0 {
        return getToColor(uv);
    }
    let fp = vec2(uv.x, y);
    let off = progress * vec2(0.0, colorSeparation);
    let c = getFromColor(fp);
    let cn = getFromColor(fp - off);
    let cp = getFromColor(fp + off);
    return vec4(cn.r, c.g, cp.b, c.a);
}
