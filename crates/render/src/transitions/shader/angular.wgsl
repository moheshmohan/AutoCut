// Author: Fernando Kuteken
// License: MIT
// Description: A wipe that sweeps around the center from a starting angle
// Tags: wipe, radial, clock
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/angular.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = startingAngle (@range(0, 360), the angle where the wipe starts, in
// degrees, default 90).
// FilmCraft: the angle at the exact centre is pinned to 0. (Upstream's PI, 3.141592653589, is
// the same f32 as the host's.)

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let startingAngle = host.p0.x;
    let offset = startingAngle * PI / 180.0;
    let angle = atan2_safe(uv.y - 0.5, uv.x - 0.5) + offset;
    var normalizedAngle = (angle + PI) / (2.0 * PI);

    normalizedAngle = normalizedAngle - floor(normalizedAngle);

    return mix(
        getFromColor(uv),
        getToColor(uv),
        vec4(step(normalizedAngle, progress))
    );
}
