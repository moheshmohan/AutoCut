// Author: gre
// License: MIT
// Description: A crossfade in which each color channel changes at a different time
// Tags: fade, color
//
// Usage: fromStep and toStep must be in [0.0, 1.0] range
// and all(fromStep) must be < all(toStep)
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/colorphase.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0 = fromStep (when each channel (r, g, b, a) starts changing; default
// (0, 0.2, 0.4, 0)), host.p1 = toStep (when each channel finishes; default (0.6, 0.8, 1, 1)).
// The host keeps the alpha steps at upstream's defaults (they only show on transparent clips).
// FilmCraft: a channel whose fromStep is not below its toStep switches at fromStep (the host's
// smoothstep), where GLSL leaves the result undefined.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let fromStep = host.p0;
    let toStep = host.p1;
    let a = getFromColor(uv);
    let b = getToColor(uv);
    return mix(a, b, vec4(
        smooth_step(fromStep.x, toStep.x, progress),
        smooth_step(fromStep.y, toStep.y, progress),
        smooth_step(fromStep.z, toStep.z, progress),
        smooth_step(fromStep.w, toStep.w, progress)
    ));
}
