// Author: gre
// License: MIT
// Description: A soft-edged circle opens from the center to reveal the incoming image
// Tags: circle, iris, wipe
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/circleopen.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = smoothness (softness of the circle's edge, default 0.3), host.p0.y =
// opening (1 opens the circle, 0 closes it instead; default 1).

const center = vec2(0.5, 0.5);
const SQRT_2: f32 = 1.414213562373;

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let smoothness = host.p0.x;
    let opening = host.p0.y > 0.5;
    let x = select(1.0 - progress, progress, opening);
    let m = smooth_step(-smoothness, 0.0, SQRT_2 * distance(center, uv) - x * (1.0 + smoothness));
    return mix(getFromColor(uv), getToColor(uv), vec4(select(m, 1.0 - m, opening)));
}
