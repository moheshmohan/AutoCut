// Author: Fernando Kuteken
// License: MIT
// Description: A flower shape with several petals grows to reveal the incoming image
// Tags: shape, wipe, polar
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/polar_function.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = segments (number of petals, default 5, a whole number).
// FilmCraft: the angle at the exact centre is pinned to 0; upstream's unused `normalized` is left
// out; PI is the host's.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let segments = i32(host.p0.x);

    let angle = atan2_safe(uv.y - 0.5, uv.x - 0.5) - 0.5 * PI;

    let radius = (cos(f32(segments) * angle) + 4.0) / 4.0;
    let difference = length(uv - vec2(0.5, 0.5));

    if difference > radius * progress {
        return getFromColor(uv);
    }
    return getToColor(uv);
}
