// Author: mandubian
// License: MIT
// Description: Wavy, color-separated ripples scramble the outgoing image into the incoming one
// Tags: wave, distort, chromatic
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/ButterflyWaveScrawler.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = amplitude (default 1), host.p0.y = waves (default 30), host.p0.z =
// colorSeparation (default 0.3).
// FilmCraft: the `acos` argument is clamped to [-1, 1] (only a large amplitude leaves it, where
// `acos` is undefined); the outgoing image's alpha is its green sample's instead of 1 (the same
// for opaque clips); PI is the host's; the progress == 0 / 1 branches are the host's.

fn compute(p: vec2<f32>, prog: f32, center: vec2<f32>) -> f32 {
    let amplitude = host.p0.x;
    let waves = host.p0.y;
    let o = p * sin(prog * amplitude) - center;
    // horizontal vector
    let h = vec2(1.0, 0.0);
    // butterfly polar function (don't ask me why this one :))
    let theta = acos(clamp(dot(o, h), -1.0, 1.0)) * waves;
    let s = sin((2.0 * theta - PI) / 24.0);
    let s2 = s * s;
    return (exp(cos(theta)) - 2.0 * cos(4.0 * theta) + s2 * s2 * s) / 10.0;
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let colorSeparation = host.p0.z;
    let p = uv;
    let inv = 1.0 - progress;
    let disp = compute(p, progress, vec2(0.5, 0.5));
    let texTo = getToColor(p + inv * disp);
    let g = getFromColor(p + progress * disp);
    let texFrom = vec4(
        getFromColor(p + progress * disp * (1.0 - colorSeparation)).r,
        g.g,
        getFromColor(p + progress * disp * (1.0 + colorSeparation)).b,
        g.a);
    return texTo * progress + texFrom * inv;
}
