// The detector networks' layers on the GPU (`gpu.rs`), the arithmetic of the CPU passes in
// `net.rs`: feature maps are planes `[c][h][w]` of f32 in storage buffers.
//
// - `stem`: the 3×3/2 convolution of the picture (zero padding 1), its bias and the ReLU.
// - `pointwise`: the 1×1 convolution and its bias, four output channels per invocation.
// - `depthwise`: the 3×3 depthwise convolution over the zero-padded plane, its bias, an optional
//   ReLU and an optional 2×2 max pool (taps summed in the CPU's order: rows top to bottom, each
//   left, centre, right).
// - `maxpool`: a 2×2 max pool. `upadd`: `a + nearest-neighbour 2× upsampling of b`.

struct P {
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    // bit 0: ReLU; bit 1: 2×2 max pool after the layer
    flags: u32,
    oh: u32,
    ow: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
// `upadd`: the half-size map
@group(0) @binding(5) var<storage, read> src2: array<f32>;

@compute @workgroup_size(64)
fn stem(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p.oh * p.ow;
    let o = id.x;
    let co = id.y;
    if o >= n || co >= p.cout {
        return;
    }
    let oy = i32(o / p.ow);
    let ox = i32(o % p.ow);
    var acc = 0.0;
    for (var ci = 0u; ci < p.cin; ci++) {
        for (var ky = 0; ky < 3; ky++) {
            let y = oy * 2 + ky - 1;
            if y < 0 || y >= i32(p.h) {
                continue;
            }
            for (var kx = 0; kx < 3; kx++) {
                let x = ox * 2 + kx - 1;
                if x < 0 || x >= i32(p.w) {
                    continue;
                }
                acc += wt[((co * p.cin + ci) * 3u + u32(ky)) * 3u + u32(kx)] * src[(ci * p.h + u32(y)) * p.w + u32(x)];
            }
        }
    }
    dst[co * n + o] = max(acc + bias[co], 0.0);
}

@compute @workgroup_size(64)
fn pointwise(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p.h * p.w;
    let px = id.x;
    let co0 = id.y * 4u;
    if px >= n || co0 >= p.cout {
        return;
    }
    var acc = vec4(0.0);
    let k = vec4<u32>(co0, co0 + 1u, co0 + 2u, co0 + 3u);
    let ok = k < vec4(p.cout);
    // rows past the last output channel read row 0 and are not stored
    let row = select(vec4(0u), k, ok) * p.cin;
    for (var ci = 0u; ci < p.cin; ci++) {
        let v = src[ci * n + px];
        acc += vec4(wt[row.x + ci], wt[row.y + ci], wt[row.z + ci], wt[row.w + ci]) * v;
    }
    dst[k.x * n + px] = acc.x + bias[k.x];
    if ok.y {
        dst[k.y * n + px] = acc.y + bias[k.y];
    }
    if ok.z {
        dst[k.z * n + px] = acc.z + bias[k.z];
    }
    if ok.w {
        dst[k.w * n + px] = acc.w + bias[k.w];
    }
}

// One depthwise output (before the pool) of channel `c` at (x, y).
fn dw_at(c: u32, y: i32, x: i32) -> f32 {
    var acc = bias[c];
    let plane = c * p.h * p.w;
    for (var ky = 0; ky < 3; ky++) {
        let yy = y + ky - 1;
        if yy < 0 || yy >= i32(p.h) {
            continue;
        }
        for (var kx = 0; kx < 3; kx++) {
            let xx = x + kx - 1;
            if xx < 0 || xx >= i32(p.w) {
                continue;
            }
            acc += wt[c * 9u + u32(ky * 3 + kx)] * src[plane + u32(yy) * p.w + u32(xx)];
        }
    }
    if (p.flags & 1u) != 0u {
        acc = max(acc, 0.0);
    }
    return acc;
}

@compute @workgroup_size(64)
fn depthwise(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p.oh * p.ow;
    let o = id.x;
    let c = id.y;
    if o >= n || c >= p.cout {
        return;
    }
    let oy = i32(o / p.ow);
    let ox = i32(o % p.ow);
    var v: f32;
    if (p.flags & 2u) != 0u {
        let y = oy * 2;
        let x = ox * 2;
        v = max(max(dw_at(c, y, x), dw_at(c, y, x + 1)), max(dw_at(c, y + 1, x), dw_at(c, y + 1, x + 1)));
    } else {
        v = dw_at(c, oy, ox);
    }
    dst[c * n + o] = v;
}

@compute @workgroup_size(64)
fn maxpool(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p.oh * p.ow;
    let o = id.x;
    let c = id.y;
    if o >= n || c >= p.cout {
        return;
    }
    let y = (o / p.ow) * 2u;
    let x = (o % p.ow) * 2u;
    let b = c * p.h * p.w;
    let r0 = b + y * p.w + x;
    let r1 = r0 + p.w;
    dst[c * n + o] = max(max(src[r0], src[r0 + 1u]), max(src[r1], src[r1 + 1u]));
}

@compute @workgroup_size(64)
fn upadd(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p.h * p.w;
    let o = id.x;
    let c = id.y;
    if o >= n || c >= p.cout {
        return;
    }
    let y = o / p.w;
    let x = o % p.w;
    let hw2 = p.w / 2u;
    dst[c * n + o] = src[c * n + o] + src2[(c * (p.h / 2u) + y / 2u) * hw2 + x / 2u];
}
