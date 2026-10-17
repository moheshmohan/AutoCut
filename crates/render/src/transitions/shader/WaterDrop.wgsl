// Author: Paweł Płóciennik
// License: MIT
// Description: Ripples spread from the center like a drop falling into water, crossfading to the incoming image
// Tags: ripple, water, blend
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/WaterDrop.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = amplitude (frequency of the ripples, default 30), host.p0.y = speed
// (of the ripples, default 30).

fn transition(p: vec2<f32>) -> vec4<f32> {
    let amplitude = host.p0.x;
    let speed = host.p0.y;
    let dir = p - vec2(0.5);
    let dist = length(dir);

    if dist > progress {
        return mix(getFromColor(p), getToColor(p), vec4(progress));
    }
    let offset = dir * sin(dist * amplitude - progress * speed);
    return mix(getFromColor(p + offset), getToColor(p), vec4(progress));
}
