// Author: Matt DesLauriers
// License: MIT
// Description: Blocks of the image are displaced in a glitch that flashes into the incoming image
// Tags: glitch, displace
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/GlitchDisplace.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: none.
// FilmCraft: the three `voronoi()` values upstream's `displace()` computes are never used, so
// they (and the `random()` only they call) are left out: the picture is the same. Upstream's
// `smoothstep(1.0, 0.5, progress)` (edges in reverse order, which GLSL leaves undefined but every
// GPU evaluates by the formula) is written `1.0 - smoothstep(0.5, 1.0, progress)`, the same
// curve. The grey flash is `vec4(gray, 1.0) * 2.0`, whose alpha of 2 an 8-bit target clamps:
// the result's alpha is clamped to 1 here too.

fn displace(tex: vec4<f32>, texCoord: vec2<f32>, dotDepth: f32, textureDepth: f32, strength: f32) -> vec2<f32> {
    let dt = tex * 1.0;
    var dis = dt * dotDepth + 1.0 - tex * textureDepth;

    dis.x = dis.x - 1.0 + textureDepth * dotDepth;
    dis.y = dis.y - 1.0 + textureDepth * dotDepth;
    dis.x *= strength;
    dis.y *= strength;
    var res_uv = texCoord;
    res_uv.x = res_uv.x + dis.x - 0.0;
    res_uv.y = res_uv.y + dis.y;
    return res_uv;
}

fn ease1(t: f32) -> f32 {
    if t == 0.0 || t == 1.0 {
        return t;
    }
    if t < 0.5 {
        return 0.5 * pow(2.0, (20.0 * t) - 10.0);
    }
    return -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}

fn ease2(t: f32) -> f32 {
    if t == 1.0 {
        return t;
    }
    return 1.0 - pow(2.0, -10.0 * t);
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let p = uv;
    var color1 = getFromColor(p);
    var color2 = getToColor(p);
    let disp = displace(color1, p, 0.33, 0.7, 1.0 - ease1(progress));
    let disp2 = displace(color2, p, 0.33, 0.5, ease2(progress));
    let dColor1 = getToColor(disp);
    var dColor2 = getFromColor(disp2);
    let val = ease1(progress);
    let gray = vec3(dot(min(dColor2, dColor1).rgb, vec3(0.299, 0.587, 0.114)));
    dColor2 = vec4(gray, 1.0);
    dColor2 *= 2.0;
    color1 = mix(color1, dColor2, vec4(smooth_step(0.0, 0.5, progress)));
    color2 = mix(color2, dColor1, vec4(1.0 - smooth_step(0.5, 1.0, progress)));
    let c = mix(color1, color2, vec4(val));
    return vec4(c.rgb, min(c.a, 1.0));
}
