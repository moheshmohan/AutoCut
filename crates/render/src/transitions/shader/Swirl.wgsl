// License: MIT
// Author: Sergey Kosarevsky
// Description: Both images swirl around the center while crossfading
// Tags: swirl, rotate, blend
// ( http://www.linderdaum.com )
// ported by gre from https://gist.github.com/corporateshark/cacfedb8cca0f5ce3f7c
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/Swirl.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: none.

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let Radius = 1.0;

    let T = progress;

    var UV = uv - vec2(0.5, 0.5);

    let Dist = length(UV);

    if Dist < Radius {
        let Percent = (Radius - Dist) / Radius;
        let A = select(mix(1.0, 0.0, (T - 0.5) / 0.5), mix(0.0, 1.0, T / 0.5), T <= 0.5);
        let Theta = Percent * Percent * A * 8.0 * 3.14159;
        let S = sin(Theta);
        let C = cos(Theta);
        UV = vec2(dot(UV, vec2(C, -S)), dot(UV, vec2(S, C)));
    }
    UV += vec2(0.5, 0.5);

    let C0 = getFromColor(UV);
    let C1 = getToColor(UV);

    return mix(C0, C1, vec4(T));
}
