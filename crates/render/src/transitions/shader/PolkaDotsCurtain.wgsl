// Author: bobylito
// License: MIT
// Description: The incoming image appears through growing polka dots
// Tags: dots, pattern
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/PolkaDotsCurtain.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = dots (number of dots across the image, default 20), host.p0.yz =
// center (uv; default (0, 0), the bottom left corner; the host takes it from the Center point).
// FilmCraft: the distance to the centre is kept above 1e-6, so a centre exactly on a pixel centre
// gives the incoming image there instead of dividing by zero.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let dots = host.p0.x;
    let center = host.p0.yz;
    let nextImage = distance(fract(uv * dots), vec2(0.5, 0.5)) < (progress / max(distance(uv, center), 1e-6));
    if nextImage {
        return getToColor(uv);
    }
    return getFromColor(uv);
}
