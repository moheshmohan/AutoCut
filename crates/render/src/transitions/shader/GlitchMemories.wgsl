// Author: Gunnar Roth
// based on work from natewave
// License: MIT
// Description: A crossfade with glitchy, color-shifted blocks
// Tags: glitch, chromatic, blend
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/GlitchMemories.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: none.
// FilmCraft: upstream's result is opaque (alpha 1); here its alpha is the crossfade's at the green
// sample, which is the same for opaque clips and keeps transparent clips transparent.

fn transition(p: vec2<f32>) -> vec4<f32> {
    let blk = floor(p.xy / vec2(16.0));
    var uv_noise = blk / vec2(64.0);
    uv_noise += floor(vec2(progress) * vec2(1200.0, 3500.0)) / vec2(64.0);
    var dist = vec2(0.0);
    if progress > 0.0 {
        dist = (fract(uv_noise) - 0.5) * 0.3 * (1.0 - progress);
    }
    let red = p + dist * 0.2;
    let green = p + dist * 0.3;
    let blue = p + dist * 0.5;

    let g = mix(getFromColor(green), getToColor(green), vec4(progress));
    return vec4(mix(getFromColor(red), getToColor(red), vec4(progress)).r, g.g, mix(getFromColor(blue), getToColor(blue), vec4(progress)).b, g.a);
}
