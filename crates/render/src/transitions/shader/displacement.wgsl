// Author: Travis Fischer
// License: MIT
// Description: A crossfade in which a grayscale displacement map pushes both pictures sideways
// Tags: displace, distort, blend
//
// Adapted from a Codrops article by Robin Delaporte
// https://tympanus.net/Development/DistortionHoverEffect
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/displacement.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = strength (default 0.5).
// FilmCraft: upstream's `displacementMap` texture is the host's generated map (`getMap`: clouds,
// cells, spiral… chosen by the `map` parameter), not an image.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let strength = host.p0.x;
    let displacement = getMap(uv) * strength;

    let uvFrom = vec2(uv.x + progress * displacement, uv.y);
    let uvTo = vec2(uv.x - (1.0 - progress) * displacement, uv.y);

    return mix(
        getFromColor(uvFrom),
        getToColor(uvTo),
        vec4(progress)
    );
}
