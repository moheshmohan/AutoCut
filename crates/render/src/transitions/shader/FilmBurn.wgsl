// Author: Anastasia Dunbar
// License: MIT
// Description: Glowing film-burn light leaks wash over the image and reveal the incoming one
// Tags: burn, film, light-leak
//
// Upstream: https://github.com/gl-transitions/gl-transitions/blob/master/transitions/FilmBurn.glsl
// Hand-ported to WGSL for FilmCraft (licence text: LICENSE-gl-transitions.txt next to this file).
// Parameters: host.p0.x = Seed (random seed, default 2.31).
// FilmCraft: the collection's sin-based random numbers become the host's integer hash (identical
// on CPU and GPU): `rand(float)` hashes the float's bits with the seed's, `random(vec2, shft)`
// hashes its lattice cell with floor(shft) and the seed, and the blur's `rand(vec2(i, p.x + p.y))`
// hashes the output pixel with i. The blur's 50 directions, `cos` / `sin` of
// `degrees((i / repeats) * 360.)` (radians far from zero, where GPUs' `cos` loses precision), are
// a table computed once in double precision.

const BLUR_DIRS = array<vec2<f32>, 50>(
    vec2(1.0, 0.0),
    vec2(-0.5562213, -0.8310342),
    vec2(-0.38123566, 0.9244779),
    vec2(0.9803362, -0.19733462),
    vec2(-0.70931876, -0.70488787),
    vec2(-0.19142744, 0.98150676),
    vec2(0.9221181, -0.38690853),
    vec2(-0.834335, -0.55125767),
    vec2(0.0062661488, 0.9999804),
    vec2(0.8277723, -0.5610642),
    vec2(-0.9267111, -0.37577465),
    vec2(0.20347367, 0.97908044),
    vec2(0.70060354, -0.71355075),
    vec2(-0.98273224, -0.1850335),
    vec2(0.39222997, 0.9198672),
    vec2(0.5464278, -0.83750623),
    vec2(-0.99992144, 0.012532052),
    vec2(0.5664413, 0.82410204),
    vec2(0.37041396, -0.9288668),
    vec2(-0.9778884, 0.20912728),
    vec2(0.7175868, 0.69646907),
    vec2(0.1793523, -0.9837849),
    vec2(-0.9171969, 0.3984342),
    vec2(0.8409138, 0.5411691),
    vec2(-0.018309265, -0.9998324),
    vec2(-0.8205365, 0.5715942),
    vec2(0.93152523, 0.36367676),
    vec2(-0.21525078, -0.9765588),
    vec2(-0.69231135, 0.7215989),
    vec2(0.9849739, 0.17270331),
    vec2(-0.40283334, -0.91527337),
    vec2(-0.53547686, 0.8445499),
    vec2(0.9996859, -0.025062134),
    vec2(-0.5763256, -0.8172202),
    vec2(-0.35828844, 0.9336109),
    vec2(0.97508264, -0.22184195),
    vec2(-0.725587, -0.68813044),
    vec2(-0.16700937, 0.9859553),
    vec2(0.91253155, -0.4090063),
    vec2(-0.84814745, -0.52976024),
    vec2(0.029861666, 0.99955404),
    vec2(0.8138766, -0.58103776),
    vec2(-0.9356655, -0.35288817),
    vec2(0.22747216, 0.97378457),
    vec2(0.6825004, -0.7308852),
    vec2(-0.98706084, -0.16034602),
    vec2(0.41427198, 0.91015315),
    vec2(0.5256819, -0.85068125),
    vec2(-0.99932957, 0.03661239),
    vec2(0.58493876, 0.8110775)
);

fn sigmoid(x: f32, a: f32) -> f32 {
    var b = pow(x * 2.0, a) / 2.0;
    if x > 0.5 {
        b = 1.0 - pow(2.0 - (x * 2.0), a) / 2.0;
    }
    return b;
}

fn seed_bits() -> u32 {
    return bitcast<u32>(host.p0.x);
}

// upstream: rand(float co) = fract(sin((co * 24.9898) + Seed) * 43758.5453)
fn rand1(co: f32) -> f32 {
    return hash(bitcast<i32>(co), 7, seed_bits());
}

fn apow(a: f32, b: f32) -> f32 {
    return pow(abs(a), b) * sign(b);
}

fn pow3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3(apow(a.r, b.r), apow(a.g, b.g), apow(a.b, b.b));
}

fn smooth_mix(a: f32, b: f32, c: f32) -> f32 {
    return mix(a, b, sigmoid(c, 2.0));
}

// upstream: fract(sin(dot(co, vec2(12.9898 + k * .5, 78.233 + Seed))) * 43758.5453)
fn rand_lattice(co: vec2<f32>, k: f32) -> f32 {
    return hash(i32(co.x), i32(co.y), bitcast<u32>(k) ^ (seed_bits() * 0x9E3779B1u));
}

fn random(co_in: vec2<f32>, shft: f32) -> f32 {
    let co = co_in + 10.0;
    return smooth_mix(rand_lattice(co, floor(shft)), rand_lattice(co, floor(shft + 1.0)), fract(shft));
}

fn smooth_random(co: vec2<f32>, shft: f32) -> f32 {
    return smooth_mix(
        smooth_mix(random(floor(co), shft), random(floor(co + vec2(1.0, 0.0)), shft), fract(co.x)),
        smooth_mix(random(floor(co + vec2(0.0, 1.0)), shft), random(floor(co + vec2(1.0, 1.0)), shft), fract(co.x)),
        fract(co.y)
    );
}

fn blendedColor(p: vec2<f32>) -> vec4<f32> {
    return mix(getFromColor(p), getToColor(p), vec4(sigmoid(progress, 10.0)));
}

fn transition(uv: vec2<f32>) -> vec4<f32> {
    var p = uv;
    var f = vec3(0.0);
    for (var k = 0; k < 13; k++) {
        let i = f32(k);
        f += sin(((p.x * rand1(i) * 6.0) + (progress * 8.0)) + rand1(i + 1.43)) * sin(((p.y * rand1(i + 4.4) * 6.0) + (progress * 6.0)) + rand1(i + 2.4));
        f += 1.0 - clamp(length(p - vec2(smooth_random(vec2(progress * 1.3), i + 1.0), smooth_random(vec2(progress * 0.5), i + 6.25))) * mix(20.0, 70.0, rand1(i)), 0.0, 1.0);
    }
    f += 4.0;
    f /= 11.0;
    f = pow3(f * vec3(1.0, 0.7, 0.6), vec3(1.0, 2.0 - sin(progress * PI), 1.3));
    f *= sin(progress * PI);

    p -= 0.5;
    p *= 1.0 + (smooth_random(vec2(progress * 5.0), 6.3) * sin(progress * PI) * 0.05);
    p += 0.5;

    var blurred_image = vec4(0.0);
    let bluramount = sin(progress * PI) * 0.03;
    var dirs = BLUR_DIRS;
    for (var k = 0; k < 50; k++) {
        let q = dirs[k] * (rand_pixel_seeded(uv, u32(k) + 16u) + bluramount);
        let uv2 = p + (q * bluramount);
        blurred_image += blendedColor(uv2);
    }
    blurred_image /= 50.0;

    return blurred_image + vec4(f, 0.0);
}
