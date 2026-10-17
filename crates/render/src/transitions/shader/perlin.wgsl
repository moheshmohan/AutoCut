// Author: Rich Harris
// License: MIT
// Description: The incoming image appears through blobs of Perlin noise
// Tags: noise, dissolve
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/perlin.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = scale (of the noise; higher values make smaller blobs; default 4),
// host.p0.y = smoothness (of the blob edges, default 0.01), host.p0.z = seed (default 12.9898).
// FilmCraft: `random` is the host's integer hash of the noise cell, seeded with the seed's bits
// (identical on CPU and GPU), instead of `fract(sin(mod(dot(co, vec2(seed, 78.233)), 3.14)) *
// 43758.5453)`.

fn random(co: vec2<f32>) -> f32 {
    return rand_cell_seeded(co, bitcast<u32>(host.p0.z));
}

// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
fn noise(st: vec2<f32>) -> f32 {
    let i = floor(st);
    let f = fract(st);

    // Four corners in 2D of a tile
    let a = random(i);
    let b = random(i + vec2(1.0, 0.0));
    let c = random(i + vec2(0.0, 1.0));
    let d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    let u = f * f * (3.0 - 2.0 * f);

    // Mix 4 coorners porcentages
    return mix(a, b, u.x) +
            (c - a) * u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    let scale = host.p0.x;
    let smoothness = host.p0.y;
    let fromColor = getFromColor(uv);
    let toColor = getToColor(uv);
    let n = noise(uv * scale);

    let p = mix(-smoothness, 1.0 + smoothness, progress);
    let lower = p - smoothness;
    let higher = p + smoothness;

    let q = smooth_step(lower, higher, n);

    return mix(
        fromColor,
        toColor,
        vec4(1.0 - q)
    );
}
