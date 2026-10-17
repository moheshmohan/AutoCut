// Author: hong
// License: MIT
// Description: The outgoing image turns like a book page from right to left, revealing the incoming one
// Tags: page, flip, 3d
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/BookFlip.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: none. A single pass like the others: the "3D" is upstream's skew of the page.
// FilmCraft: the page's edges are moved by 1/100000 of the width (`p.x < 0.5`, `step(1.0 -
// progress, p.x)`), so a pixel centre exactly on one (uv.x = 0.5 on an odd width) falls the same
// way on every GPU and on the CPU.

// Avoids dividing by zero when the page is edge-on (progress = 0.5).
fn nonZero(x: f32) -> f32 {
    if abs(x) < 1e-4 {
        return select(1e-4, -1e-4, x < 0.0);
    }
    return x;
}

fn skewRight(p: vec2<f32>) -> vec2<f32> {
    let skewX = (p.x - progress) / nonZero(0.5 - progress) * 0.5;
    let skewY = (p.y - 0.5) / (0.5 + progress * (p.x - 0.5) / 0.5) * 0.5 + 0.5;
    return vec2(skewX, skewY);
}

fn skewLeft(p: vec2<f32>) -> vec2<f32> {
    let skewX = (p.x - 0.5) / nonZero(progress - 0.5) * 0.5 + 0.5;
    let skewY = (p.y - 0.5) / (0.5 + (1.0 - progress) * (0.5 - p.x) / 0.5) * 0.5 + 0.5;
    return vec2(skewX, skewY);
}

fn addShade() -> vec4<f32> {
    let shadeVal = max(0.7, abs(progress - 0.5) * 2.0);
    return vec4(vec3(shadeVal), 1.0);
}

fn transition(p: vec2<f32>) -> vec4<f32> {
    let pr = step(1.0 - progress - 0.00001, p.x);

    if p.x < 0.5 - 0.00001 {
        return mix(getFromColor(p), getToColor(skewLeft(p)) * addShade(), vec4(pr));
    }
    return mix(getFromColor(skewRight(p)) * addShade(), getToColor(p), vec4(pr));
}
