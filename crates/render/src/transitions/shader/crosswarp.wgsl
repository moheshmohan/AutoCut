// Author: Eke Péter <peterekepeter@gmail.com>
// License: MIT
// Description: The outgoing image zooms in while the incoming one zooms out into place, sweeping from right to left
// Tags: warp, zoom, horizontal
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/crosswarp.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// The FilmCraft host (prelude.wgsl) supplies progress, getFromColor, getToColor and smooth_step.

fn transition(p: vec2<f32>) -> vec4<f32> {
    var x = progress;
    x = smooth_step(0.0, 1.0, (x * 2.0 + p.x - 1.0));
    return mix(getFromColor((p - 0.5) * (1.0 - x) + 0.5), getToColor((p - 0.5) * x + 0.5), vec4(x));
}
