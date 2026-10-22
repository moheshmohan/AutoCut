// The profile pass's convolutional layers on the GPU (`gpunn.rs`), the arithmetic of the CPU
// passes in `layers.rs`: tensors are `[n][c][h][w]` f32 in storage buffers.
//
// - `conv`: a convolution (dense, or depthwise with `flags` bit 0) with explicit top/left
//   padding, its bias, an activation (`act`: 0 none, 1 ReLU, 2 ReLU then a·x + b, 3 SiLU) and
//   optionally a residual added after it (`flags` bit 1); four output channels an invocation.
// - `maxpool`: k×k max pooling with stride and −∞ padding.
// - `upsample`: nearest-neighbour 2× upsampling.
// - `pointwise`: a 1×1 convolution as a tiled matrix product (bias, activation, residual as
//   `conv`).
// - `copy`: channels of one tensor into a channel range of another (concatenation, splits).
// - `pad`: zeros added on the right and at the bottom (`pad_t` of them).
// - `mean`: each plane's mean. `attn`: hard sigmoid of a 1×1 convolution of the means.
// - `scale`: every plane times its channel's factor.

struct P {
    n: u32,
    cin: u32,
    h: u32,
    w: u32,
    cout: u32,
    oh: u32,
    ow: u32,
    k: u32,
    stride: u32,
    pad_t: u32,
    pad_l: u32,
    flags: u32,
    act: u32,
    act_a: f32,
    act_b: f32,
    // channel offsets and totals of the views read and written
    cin_off: u32,
    cin_total: u32,
    cout_off: u32,
    cout_total: u32,
    _p0: u32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;

fn activate(v: f32) -> f32 {
    switch p.act {
        case 1u: {
            return max(v, 0.0);
        }
        case 2u: {
            return p.act_a * max(v, 0.0) + p.act_b;
        }
        case 3u: {
            return v / (1.0 + exp(-v));
        }
        default: {
            return v;
        }
    }
}

@compute @workgroup_size(64)
fn conv(@builtin(global_invocation_id) id: vec3<u32>) {
    let npx = p.oh * p.ow;
    let px = id.x;
    let co0 = id.y * 4u;
    let b = id.z;
    if px >= npx || co0 >= p.cout || b >= p.n {
        return;
    }
    let oy = i32(px / p.ow);
    let ox = i32(px % p.ow);
    let plane = p.h * p.w;
    let kk = p.k * p.k;
    let ks = i32(p.k);
    let y0 = oy * i32(p.stride) - i32(p.pad_t);
    let x0 = ox * i32(p.stride) - i32(p.pad_l);
    var acc = vec4(0.0);
    let co = vec4<u32>(co0, co0 + 1u, co0 + 2u, co0 + 3u);
    let ok = co < vec4(p.cout);
    let cc = select(vec4(co0), co, ok);
    if (p.flags & 1u) != 0u {
        // depthwise: channel c reads its own input channel
        for (var j = 0u; j < 4u; j++) {
            let c = cc[j];
            let sb = (b * p.cin_total + p.cin_off + c) * plane;
            var a = 0.0;
            for (var ky = 0; ky < ks; ky++) {
                let y = y0 + ky;
                if y < 0 || y >= i32(p.h) {
                    continue;
                }
                for (var kx = 0; kx < ks; kx++) {
                    let x = x0 + kx;
                    if x < 0 || x >= i32(p.w) {
                        continue;
                    }
                    a += wt[c * kk + u32(ky * ks + kx)] * src[sb + u32(y) * p.w + u32(x)];
                }
            }
            acc[j] = a;
        }
    } else {
        let row = cc * p.cin * kk;
        for (var ci = 0u; ci < p.cin; ci++) {
            let sb = (b * p.cin_total + p.cin_off + ci) * plane;
            let wb = ci * kk;
            for (var ky = 0; ky < ks; ky++) {
                let y = y0 + ky;
                if y < 0 || y >= i32(p.h) {
                    continue;
                }
                for (var kx = 0; kx < ks; kx++) {
                    let x = x0 + kx;
                    if x < 0 || x >= i32(p.w) {
                        continue;
                    }
                    let v = src[sb + u32(y) * p.w + u32(x)];
                    let t = wb + u32(ky * ks + kx);
                    acc += vec4(wt[row.x + t], wt[row.y + t], wt[row.z + t], wt[row.w + t]) * v;
                }
            }
        }
    }
    for (var j = 0u; j < 4u; j++) {
        if !ok[j] {
            continue;
        }
        let c = co[j];
        var v = activate(acc[j] + bias[c]);
        if (p.flags & 2u) != 0u {
            v += res[(b * p.cout + c) * npx + px];
        }
        dst[(b * p.cout_total + p.cout_off + c) * npx + px] = v;
    }
}

// A 1×1 convolution as a tiled matrix product: 16 × 16 outputs (channels × pixels) a workgroup,
// the input and weights staged 16 at a time in shared memory.
var<workgroup> tile_w: array<array<f32, 16>, 16>;
var<workgroup> tile_x: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn pointwise(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let npx = p.oh * p.ow;
    let b = wg.z;
    let px = wg.x * 16u + lid.x;
    let co = wg.y * 16u + lid.y;
    var acc = 0.0;
    let steps = (p.cin + 15u) / 16u;
    for (var s = 0u; s < steps; s++) {
        // weights: row co (this thread's lid.y), column s·16 + lid.x
        let wc = s * 16u + lid.x;
        if co < p.cout && wc < p.cin {
            tile_w[lid.y][lid.x] = wt[co * p.cin + wc];
        } else {
            tile_w[lid.y][lid.x] = 0.0;
        }
        // input: channel s·16 + lid.y, pixel px
        let xc = s * 16u + lid.y;
        if px < npx && xc < p.cin && b < p.n {
            tile_x[lid.y][lid.x] = src[(b * p.cin_total + p.cin_off + xc) * npx + px];
        } else {
            tile_x[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < 16u; i++) {
            acc += tile_w[lid.y][i] * tile_x[i][lid.x];
        }
        workgroupBarrier();
    }
    if px >= npx || co >= p.cout || b >= p.n {
        return;
    }
    var v = activate(acc + bias[co]);
    if (p.flags & 2u) != 0u {
        v += res[(b * p.cout + co) * npx + px];
    }
    dst[(b * p.cout_total + p.cout_off + co) * npx + px] = v;
}

@compute @workgroup_size(64)
fn maxpool(@builtin(global_invocation_id) id: vec3<u32>) {
    let npx = p.oh * p.ow;
    let px = id.x;
    let c = id.y;
    let b = id.z;
    if px >= npx || c >= p.cout || b >= p.n {
        return;
    }
    let oy = i32(px / p.ow) * i32(p.stride) - i32(p.pad_t);
    let ox = i32(px % p.ow) * i32(p.stride) - i32(p.pad_l);
    let sb = (b * p.cin_total + p.cin_off + c) * p.h * p.w;
    var m = -3.0e38;
    var seen = false;
    for (var ky = 0; ky < i32(p.k); ky++) {
        let y = oy + ky;
        if y < 0 || y >= i32(p.h) {
            continue;
        }
        for (var kx = 0; kx < i32(p.k); kx++) {
            let x = ox + kx;
            if x < 0 || x >= i32(p.w) {
                continue;
            }
            m = max(m, src[sb + u32(y) * p.w + u32(x)]);
            seen = true;
        }
    }
    if !seen {
        m = -3.0e38;
    }
    dst[(b * p.cout_total + p.cout_off + c) * npx + px] = m;
}

@compute @workgroup_size(64)
fn upsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let npx = p.oh * p.ow;
    let px = id.x;
    let c = id.y;
    let b = id.z;
    if px >= npx || c >= p.cout || b >= p.n {
        return;
    }
    let y = (px / p.ow) / 2u;
    let x = (px % p.ow) / 2u;
    dst[(b * p.cout_total + p.cout_off + c) * npx + px] = src[(b * p.cin_total + p.cin_off + c) * p.h * p.w + y * p.w + x];
}

@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
    let npx = p.oh * p.ow;
    let px = id.x;
    let c = id.y;
    let b = id.z;
    if px >= npx || c >= p.cout || b >= p.n {
        return;
    }
    dst[(b * p.cout_total + p.cout_off + c) * npx + px] = src[(b * p.cin_total + p.cin_off + c) * npx + px];
}

@compute @workgroup_size(64)
fn pad(@builtin(global_invocation_id) id: vec3<u32>) {
    let npx = p.oh * p.ow;
    let px = id.x;
    let c = id.y;
    let b = id.z;
    if px >= npx || c >= p.cout || b >= p.n {
        return;
    }
    let y = px / p.ow;
    let x = px % p.ow;
    var v = 0.0;
    if y < p.h && x < p.w {
        v = src[(b * p.cin_total + p.cin_off + c) * p.h * p.w + y * p.w + x];
    }
    dst[(b * p.cout_total + p.cout_off + c) * npx + px] = v;
}

@compute @workgroup_size(64)
fn mean(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    let b = id.z;
    if c >= p.cin || b >= p.n {
        return;
    }
    let plane = p.h * p.w;
    let sb = (b * p.cin_total + p.cin_off + c) * plane;
    var s = 0.0;
    for (var i = 0u; i < plane; i++) {
        s += src[sb + i];
    }
    dst[b * p.cin + c] = s / f32(max(plane, 1u));
}

@compute @workgroup_size(64)
fn attn(@builtin(global_invocation_id) id: vec3<u32>) {
    let co = id.x;
    let b = id.z;
    if co >= p.cout || b >= p.n {
        return;
    }
    var z = bias[co];
    for (var ci = 0u; ci < p.cin; ci++) {
        z += wt[co * p.cin + ci] * src[b * p.cin + ci];
    }
    dst[b * p.cout + co] = clamp(z / 6.0 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
    let npx = p.h * p.w;
    let px = id.x;
    let c = id.y;
    let b = id.z;
    if px >= npx || c >= p.cout || b >= p.n {
        return;
    }
    let i = (b * p.cout + c) * npx + px;
    dst[i] = src[i] * res[b * p.cout + c];
}
