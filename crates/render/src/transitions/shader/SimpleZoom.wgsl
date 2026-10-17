// Author: 0gust1
// License: MIT
// Description: Zooms into the outgoing image, then fades to the incoming one
// Tags: zoom, fade
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/SimpleZoom.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = zoom_quickness (@range(0.2, 1), default 0.8).

fn zoom(uv: vec2<f32>, amount: f32) -> vec2<f32> {
    return 0.5 + ((uv - 0.5) * (1.0 - amount));
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let nQuick = clamp(host.p0.x, 0.2, 1.0);
    return mix(
        getFromColor(zoom(uv, smooth_step(0.0, nQuick, progress))),
        getToColor(uv),
        vec4(smooth_step(nQuick - 0.2, 1.0, progress))
    );
}
