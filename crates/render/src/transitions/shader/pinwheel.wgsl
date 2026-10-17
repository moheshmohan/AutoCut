// Author: Mr Speaker
// License: MIT
// Description: A spinning pinwheel reveals the incoming image
// Tags: pinwheel, radial, rotate
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/pinwheel.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = speed (rotation speed, default 2).
// FilmCraft: the angle at the exact centre is pinned to 0.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let speed = host.p0.x;

    let p = uv;

    let circPos = atan2_safe(p.y - 0.5, p.x - 0.5) + progress * speed;
    let modPos = glsl_mod(circPos, 3.1415 / 4.0);
    let sgn = sign(progress - modPos);

    return mix(getToColor(p), getFromColor(p), vec4(step(sgn, 0.5)));
}
