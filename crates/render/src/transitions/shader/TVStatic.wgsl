// Author: Brandon Anzaldi
// License: MIT
// Description: Cuts to full-screen TV static, then to the incoming image
// Tags: static, noise, retro
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/TVStatic.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = offset (part of the transition, at each end, that shows the images
// instead of static; default 0.05).
// FilmCraft: the static is the host's integer hash of the output pixel seeded with the
// progress's bits (identical on CPU and GPU, new on every frame) instead of
// `fract(sin(mod(dot(co * progress, vec2(12.9898, 78.233)), 3.14)) * 43758.5453)`.

fn noise(co: vec2<f32>) -> f32 {
    return rand_pixel_seeded(co, bitcast<u32>(progress));
}

fn transition(p: vec2<f32>) -> vec4<f32> {
    let offset = host.p0.x;
    if progress < offset {
        return getFromColor(p);
    } else if progress > (1.0 - offset) {
        return getToColor(p);
    }
    return vec4(vec3(noise(p)), 1.0);
}
