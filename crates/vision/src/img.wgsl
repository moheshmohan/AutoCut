// The profile pass's picture preparation on the GPU (`gpunn.rs`), the arithmetic of
// `dfine::prepare_exact` and `rtmpose::crop`: the frame is RGBA8, one u32 a pixel (byte 0 red).
//
// - `resize_area`: the frame (w × h) area-averaged to ow × oh (each output pixel the mean of the
//   source pixels from ⌊x·w/ow⌋ to max(⌊(x+1)·w/ow⌋, that + 1), as on the CPU), RGB in 0..1,
//   planes `[3][oh][ow]`.
// - `crop`: a person's crop for the pose network: per crop its centre and source pixels per crop
//   pixel (b: `[cx, cy, k]`), the frame sampled bilinearly (black outside) at 192 × 256,
//   normalised with mmpose's mean and deviation, planes `[n][3][256][192]`.

struct T {
    rows: u32,
    cols: u32,
    cin: u32,
    cout: u32,
    n: u32,
    heads: u32,
    hd: u32,
    k: u32,
    flags: u32,
    act: u32,
    a_off: u32,
    b_off: u32,
    total: u32,
    levels: u32,
    u0: u32,
    u1: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
};

@group(0) @binding(0) var<uniform> p: T;
@group(0) @binding(1) var<storage, read> px: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;

// w = p.cin, h = p.rows (the frame); ow = p.cols, oh = p.cout.
@compute @workgroup_size(64)
fn resize_area(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let w = p.cin;
    let h = p.rows;
    let ow = p.cols;
    let oh = p.cout;
    if x >= ow || y >= oh {
        return;
    }
    let y0 = y * h / oh;
    let y1 = min(max((y + 1u) * h / oh, y0 + 1u), h);
    let x0 = x * w / ow;
    let x1 = min(max((x + 1u) * w / ow, x0 + 1u), w);
    var r = 0u;
    var g = 0u;
    var bl = 0u;
    var n = 0u;
    for (var yy = y0; yy < y1; yy++) {
        for (var xx = x0; xx < x1; xx++) {
            let v = px[yy * w + xx];
            r += v & 255u;
            g += (v >> 8u) & 255u;
            bl += (v >> 16u) & 255u;
            n += 1u;
        }
    }
    let k = f32(max(n, 1u)) * 255.0;
    let plane = ow * oh;
    let i = y * ow + x;
    o[i] = f32(r) / k;
    o[plane + i] = f32(g) / k;
    o[2u * plane + i] = f32(bl) / k;
}

fn pix(x: i32, y: i32, ch: u32) -> f32 {
    if x < 0 || y < 0 || x >= i32(p.cin) || y >= i32(p.rows) {
        return 0.0;
    }
    let v = px[u32(y) * p.cin + u32(x)];
    return f32((v >> (8u * ch)) & 255u);
}

// w = p.cin, h = p.rows (the frame); n = p.n crops.
@compute @workgroup_size(64)
fn crop(@builtin(global_invocation_id) id: vec3<u32>) {
    let cw = 192u;
    let chh = 256u;
    let i = id.x;
    let ci = id.y;
    if i >= cw * chh || ci >= p.n {
        return;
    }
    let u = i % cw;
    let v = i / cw;
    let cx = b[ci * 3u];
    let cy = b[ci * 3u + 1u];
    let k = b[ci * 3u + 2u];
    let sx = cx + (f32(u) - f32(cw) / 2.0) * k;
    let sy = cy + (f32(v) - f32(chh) / 2.0) * k;
    let x0 = floor(sx);
    let y0 = floor(sy);
    let fx = sx - x0;
    let fy = sy - y0;
    let xi = i32(x0);
    let yi = i32(y0);
    let mean = array<f32, 3>(123.675, 116.28, 103.53);
    let dev = array<f32, 3>(58.395, 57.12, 57.375);
    let plane = cw * chh;
    for (var c = 0u; c < 3u; c++) {
        let val = pix(xi, yi, c) * (1.0 - fx) * (1.0 - fy) + pix(xi + 1, yi, c) * fx * (1.0 - fy) + pix(xi, yi + 1, c) * (1.0 - fx) * fy
            + pix(xi + 1, yi + 1, c) * fx * fy;
        o[(ci * 3u + c) * plane + v * cw + u] = (val - mean[c]) / dev[c];
    }
}
