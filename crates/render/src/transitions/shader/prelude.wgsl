// FilmCraft host for GL Transitions: original work, MIT OR Apache-2.0.
//
// A ported GL Transition defines `fn transition(uv: vec2<f32>) -> vec4<f32>` and reads
// `progress`, `ratio`, `getFromColor(uv)` and `getToColor(uv)` as in the GLSL collection (uv in
// 0..1, origin at the bottom left), plus its parameters from `host.p0..p2`. Colours are linear,
// premultiplied RGBA, the compositor's working space. The host draws one full-frame triangle;
// each fragment maps its pixel centre to uv and reads the inputs with a clamp-to-edge bilinear
// filter written out with `textureLoad`: no sampler and no derivatives, so it is valid in
// non-uniform control flow, and it is the Rust reference's `Image::sample_bilinear_clamped`
// operation for operation. Progress arrives already eased by the host (CPU, exact frame time).
// At progress 0 and 1 the inputs are passed through untouched.

struct Host {
    // width, height, eased progress, ratio (width / height)
    frame: vec4<f32>,
    // the shader's parameters
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    // 1 / width, 1 / height (rounded once, on the CPU), 0, 0: a pixel's uv is its centre times
    // these, one correctly rounded multiplication, so the Rust reference and every GPU give each
    // pixel the same uv (a division may be approximate on a GPU, and a pixel centre exactly on
    // an edge, such as uv.x = 0.5 on an odd width, would fall either way)
    inv: vec4<f32>,
}

@group(0) @binding(0) var from_tex: texture_2d<f32>;
@group(0) @binding(1) var to_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> host: Host;

var<private> progress: f32;
var<private> ratio: f32;

const PI: f32 = 3.141592653589793;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[min(i, 2u)], 0.0, 1.0);
}

fn texel(side: u32, at: vec2<i32>) -> vec4<f32> {
    if side == 0u {
        return textureLoad(from_tex, at, 0);
    }
    return textureLoad(to_tex, at, 0);
}

fn sample_side(side: u32, uv: vec2<f32>) -> vec4<f32> {
    let size = host.frame.xy;
    let at = vec2(uv.x * size.x, (1.0 - uv.y) * size.y);
    let f = clamp(at - vec2(0.5), vec2(0.0), size - vec2(1.0));
    let lo = vec2<i32>(floor(f));
    let hi = min(lo + vec2(1), vec2<i32>(size) - vec2(1));
    let k = f - vec2<f32>(lo);
    let a = texel(side, lo);
    let b = texel(side, vec2(hi.x, lo.y));
    let c = texel(side, vec2(lo.x, hi.y));
    let d = texel(side, hi);
    let top = a + (b - a) * k.x;
    let bot = c + (d - c) * k.x;
    return top + (bot - top) * k.y;
}

fn getFromColor(uv: vec2<f32>) -> vec4<f32> {
    return sample_side(0u, uv);
}

fn getToColor(uv: vec2<f32>) -> vec4<f32> {
    return sample_side(1u, uv);
}

// GLSL `mod` for floats (WGSL's `%` truncates instead of flooring).
fn glsl_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn glsl_mod2(x: vec2<f32>, y: f32) -> vec2<f32> {
    return x - y * floor(x / y);
}

// `smoothstep` with a defined result for every edge order (WGSL and GLSL leave e0 >= e1
// undefined): a step at e0.
fn smooth_step(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return select(1.0, 0.0, x < e0);
    }
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// `atan2` with atan2(0, 0) pinned to 0 (left undefined by WGSL).
fn atan2_safe(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    return atan2(y, x);
}

// Integer hash to [0, 1), the Rust reference's bit for bit. It replaces the collection's
// `fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453)`, whose value depends on the
// precision of the GPU's `sin`. Seeds used by the ports: 1 `rand_pixel`, 2 `rand_cell`, 3–9
// the per-column / per-row noise of single transitions, parameter or progress bits otherwise.
fn hash(x: i32, y: i32, s: u32) -> f32 {
    var h = (bitcast<u32>(x) * 0x8DA6B343u) ^ (bitcast<u32>(y) * 0xD8163841u) ^ (s * 0xCB1AB31Fu) ^ 0x9E3779B9u;
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return f32(h >> 8u) / 16777216.0;
}

// A random value per output pixel (the pixel whose centre is at `uv`).
fn rand_pixel(uv: vec2<f32>) -> f32 {
    let at = floor(vec2(uv.x * host.frame.x, (1.0 - uv.y) * host.frame.y));
    return hash(i32(at.x), i32(at.y), 1u);
}

// A random value per integer cell.
fn rand_cell(cell: vec2<f32>) -> f32 {
    return hash(i32(cell.x), i32(cell.y), 2u);
}

// The same, varied by a seed (a parameter's bits, or the progress's for noise that changes
// every frame).
fn rand_pixel_seeded(uv: vec2<f32>, s: u32) -> f32 {
    let at = floor(vec2(uv.x * host.frame.x, (1.0 - uv.y) * host.frame.y));
    return hash(i32(at.x), i32(at.y), s);
}

fn rand_cell_seeded(cell: vec2<f32>, s: u32) -> f32 {
    return hash(i32(cell.x), i32(cell.y), s);
}

// A random value per output pixel column (the same all the way down it) and per output pixel
// row: the collection's `rand(vec2(uv.x, k))` and `rand(vec2(k, uv.y))`.
fn rand_column(uv: vec2<f32>, s: u32) -> f32 {
    return hash(i32(floor(uv.x * host.frame.x)), 0, s);
}

fn rand_row(uv: vec2<f32>, s: u32) -> f32 {
    return hash(0, i32(floor((1.0 - uv.y) * host.frame.y)), s);
}

// Whether `p` is strictly inside the unit square (the collection's `inBounds`).
fn in_bounds(p: vec2<f32>) -> bool {
    return all(vec2(0.0) < p) && all(p < vec2(1.0));
}

@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    progress = host.frame.z;
    ratio = host.frame.w;
    let px = vec2<i32>(at.xy);
    if progress <= 0.0 {
        return texel(0u, px);
    }
    if progress >= 1.0 {
        return texel(1u, px);
    }
    let uv = vec2(at.x * host.inv.x, (host.frame.y - at.y) * host.inv.y);
    return transition(uv);
}
