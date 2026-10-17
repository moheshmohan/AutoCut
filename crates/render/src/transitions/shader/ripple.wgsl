// Author: gre
// License: MIT
// Description: Ripples spread from the center while crossfading
// Tags: ripple, wave, blend
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/ripple.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = amplitude (frequency of the ripples, default 100), host.p0.y = speed
// (of the ripples, default 50).

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let amplitude = host.p0.x;
    let speed = host.p0.y;
    let dir = uv - vec2(0.5);
    let dist = length(dir);
    let offset = dir * (sin(progress * dist * amplitude - progress * speed) + 0.5) / 30.0 * progress;
    return mix(
        getFromColor(uv + offset),
        getToColor(uv),
        vec4(smooth_step(0.2, 1.0, progress))
    );
}
