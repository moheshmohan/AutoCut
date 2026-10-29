// The merge of filmcraft_render::denoise::merge_plane (same math; see its docs): each pixel of a
// plane averaged with its neighbours' pixels where their motion takes it, each weighted by
// exp(-(d/σ)²) for its difference d from the pixel.

struct Params {
    width: u32,
    height: u32,
    count: u32,
    _pad: u32,
    // the plane's size over the luma's
    scale: vec2<f32>,
    // 1 / σ²
    inv: f32,
    // the samples' largest value (255, 1023…)
    max_value: f32,
};

// One neighbour's motion: `bw` × `bh` blocks of `block` luma pixels, from `vecs[start]`.
struct MotionInfo {
    bw: u32,
    bh: u32,
    block: f32,
    start: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
// layer 0 the plane, 1.. its neighbours'
@group(0) @binding(1) var planes: texture_2d_array<u32>;
@group(0) @binding(2) var<storage, read> infos: array<MotionInfo>;
@group(0) @binding(3) var<storage, read> vecs: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn at(layer: u32, x: u32, y: u32) -> f32 {
    let c = vec2<i32>(i32(min(x, p.width - 1u)), i32(min(y, p.height - 1u)));
    return f32(textureLoad(planes, c, i32(layer), 0).r) / p.max_value;
}

fn bilinear(layer: u32, x: f32, y: f32) -> f32 {
    let fx = clamp(x, 0.0, f32(p.width - 1u));
    let fy = clamp(y, 0.0, f32(p.height - 1u));
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let top = at(layer, x0, y0) * (1.0 - tx) + at(layer, x0 + 1u, y0) * tx;
    let bottom = at(layer, x0, y0 + 1u) * (1.0 - tx) + at(layer, x0 + 1u, y0 + 1u) * tx;
    return top * (1.0 - ty) + bottom * ty;
}

// filmcraft_render::denoise::Motion::at: bilinear between block centres
fn motion_at(m: MotionInfo, x: f32, y: f32) -> vec2<f32> {
    if (m.bw == 0u || m.bh == 0u) {
        return vec2<f32>(0.0, 0.0);
    }
    let fx = clamp(x / m.block - 0.5, 0.0, f32(m.bw - 1u));
    let fy = clamp(y / m.block - 0.5, 0.0, f32(m.bh - 1u));
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, m.bw - 1u);
    let y1 = min(y0 + 1u, m.bh - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let a = vecs[m.start + y0 * m.bw + x0];
    let b = vecs[m.start + y0 * m.bw + x1];
    let c = vecs[m.start + y1 * m.bw + x0];
    let d = vecs[m.start + y1 * m.bw + x1];
    let top = a * (1.0 - tx) + b * tx;
    let bottom = c * (1.0 - tx) + d * tx;
    return top * (1.0 - ty) + bottom * ty;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height) {
        return;
    }
    let c = at(0u, id.x, id.y);
    var sum = c;
    var wsum = 1.0;
    for (var k = 0u; k < p.count; k = k + 1u) {
        // the pixel's luma position, the motion there, back in this plane's pixels
        let v = motion_at(infos[k], (f32(id.x) + 0.5) / p.scale.x, (f32(id.y) + 0.5) / p.scale.y);
        let s = bilinear(k + 1u, f32(id.x) + v.x * p.scale.x, f32(id.y) + v.y * p.scale.y);
        let d = s - c;
        let w = exp(-(d * d) * p.inv);
        sum = sum + w * s;
        wsum = wsum + w;
    }
    out[id.y * p.width + id.x] = sum / wsum;
}
