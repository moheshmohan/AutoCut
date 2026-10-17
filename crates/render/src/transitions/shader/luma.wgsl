// Author: gre
// License: MIT
// Description: A hard wipe that follows the brightness of a grayscale map
// Tags: wipe, luma, mask
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/luma.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: none of its own.
// FilmCraft: upstream's `luma` texture is the host's generated map (`getMap`: clouds, cells,
// spiral… chosen by the `map` parameter), not an image. (The Luma Wipes engine is the soft,
// feathered generalisation of this transition.)

fn transition(uv: vec2<f32>) -> vec4<f32> {
    return mix(
        getToColor(uv),
        getFromColor(uv),
        vec4(step(progress, getMap(uv)))
    );
}
