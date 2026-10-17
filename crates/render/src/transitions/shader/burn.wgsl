// Author: gre
// License: MIT
// Description: A crossfade that brightens toward a color in the middle
// Tags: fade, color, flash
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/burn.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.xyz = color (the color the images brighten toward, linear; default
// (0.9, 0.4, 0.2) display-encoded).
// FilmCraft: the result's alpha (2 upstream, which an 8-bit target clamps) is clamped to 1.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let color = host.p0.xyz;
    let c = mix(
        getFromColor(uv) + vec4(progress * color, 1.0),
        getToColor(uv) + vec4((1.0 - progress) * color, 1.0),
        vec4(progress)
    );
    return vec4(c.rgb, min(c.a, 1.0));
}
