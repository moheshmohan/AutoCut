// Author: nwoeanhinnogaehr
// License: MIT
// Description: A crossfade through a spinning kaleidoscope
// Tags: kaleidoscope, distort, blend
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/kaleidoscope.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = speed (default 1), host.p0.y = angle (default 1), host.p0.z = power
// (default 1.5).
// FilmCraft: upstream's statement `abs(mod(p, 1.0));`, whose value is unused, is left out.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let speed = host.p0.x;
    let angle = host.p0.y;
    let power = host.p0.z;
    var p = uv;
    let q = p;
    var t = pow(progress, power) * speed;
    p = p - 0.5;
    for (var i = 0; i < 7; i++) {
        p = vec2(sin(t) * p.x + cos(t) * p.y, sin(t) * p.y - cos(t) * p.x);
        t += angle;
        p = abs(glsl_mod2(p, 2.0) - 1.0);
    }
    return mix(
        mix(getFromColor(q), getToColor(q), vec4(progress)),
        mix(getFromColor(p), getToColor(p), vec4(progress)), vec4(1.0 - 2.0 * abs(progress - 0.5)));
}
