// Author: Adrian Purser
// License: MIT
// Description: The outgoing image drops out of the frame with a few bounces, revealing the incoming one
// Tags: bounce, slide, vertical, shadow
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/Bounce.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.xyz = shadow_colour's colour (linear; default black), host.p0.w =
// shadow_colour's alpha (the shadow's strength, default 0.6), host.p1.x = shadow_height
// (relative to the image height, default 0.075, at least 0.01), host.p1.y = bounces (number of
// bounces before settling, default 3).
// FilmCraft: the shadow is drawn in its opaque colour; its alpha only scales the shadow's
// strength, as in upstream's formula (upstream also lowers the output's alpha in the shadow,
// which its opaque canvas ignores). PI is the host's.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let shadow_colour = vec4(host.p0.xyz, 1.0);
    let shadow_alpha = host.p0.w;
    let shadow_height = host.p1.x;
    let bounces = host.p1.y;
    let time = progress;
    let stime = sin(time * PI / 2.0);
    let phase = time * PI * bounces;
    let y = (abs(cos(phase))) * (1.0 - stime);
    let d = uv.y - y;
    return mix(
        mix(
            getToColor(uv),
            shadow_colour,
            vec4(step(d, shadow_height) * (1.0 - mix(
                ((d / shadow_height) * shadow_alpha) + (1.0 - shadow_alpha),
                1.0,
                smooth_step(0.95, 1.0, progress) // fade-out the shadow at the end
            )))
        ),
        getFromColor(vec2(uv.x, uv.y + (1.0 - y))),
        vec4(step(d, 0.0))
    );
}
