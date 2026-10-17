// License: MIT
// Author: rectalogic
// Description: A crossfade through a strong zoom blur
// Tags: zoom, blur, blend
// ported by gre from https://gist.github.com/rectalogic/b86b90161503a0023231
//
// Converted from https://github.com/rectalogic/rendermix-basic-effects/blob/master/assets/com/rendermix/CrossZoom/CrossZoom.frag
// Which is based on https://github.com/evanw/glfx.js/blob/master/src/filters/blur/zoomblur.js
// With additional easing functions from https://github.com/rectalogic/rendermix-basic-effects/blob/master/assets/com/rendermix/Easing/Easing.glsllib
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/CrossZoom.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = strength (@range(0, 1), default 0.4).
// FilmCraft: `rand(uv)` is the host's per-pixel integer hash instead of the sin-based one.

fn Linear_ease(begin: f32, change: f32, duration: f32, time: f32) -> f32 {
    return change * time / duration + begin;
}

fn Exponential_easeInOut(begin: f32, change: f32, duration: f32, time_in: f32) -> f32 {
    if time_in == 0.0 {
        return begin;
    } else if time_in == duration {
        return begin + change;
    }
    let time = time_in / (duration / 2.0);
    if time < 1.0 {
        return change / 2.0 * pow(2.0, 10.0 * (time - 1.0)) + begin;
    }
    return change / 2.0 * (-pow(2.0, -10.0 * (time - 1.0)) + 2.0) + begin;
}

fn Sinusoidal_easeInOut(begin: f32, change: f32, duration: f32, time: f32) -> f32 {
    return -change / 2.0 * (cos(PI * time / duration) - 1.0) + begin;
}

fn rand(co: vec2<f32>) -> f32 {
    return rand_pixel(co);
}

fn crossFade(uv: vec2<f32>, dissolve: f32) -> vec4<f32> {
    return mix(getFromColor(uv), getToColor(uv), vec4(dissolve));
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let texCoord = uv;

    // Linear interpolate center across center half of the image
    let center = vec2(Linear_ease(0.25, 0.5, 1.0, progress), 0.5);
    let dissolve = Exponential_easeInOut(0.0, 1.0, 1.0, progress);

    // Mirrored sinusoidal loop. 0->strength then strength->0
    let strength = Sinusoidal_easeInOut(0.0, host.p0.x, 0.5, progress);

    var color = vec4(0.0);
    var total = 0.0;
    let toCenter = center - texCoord;

    // randomize the lookup values to hide the fixed number of samples
    let offset = rand(uv);

    for (var i = 0; i <= 40; i++) {
        let t = f32(i);
        let percent = (t + offset) / 40.0;
        let weight = 4.0 * (percent - percent * percent);
        color += crossFade(texCoord + toCenter * percent * strength, dissolve) * weight;
        total += weight;
    }
    return color / total;
}
