// MI-GAN's layers (crate::inpaint) as two compute passes: a depthwise convolution (`dw`: plain,
// strided, or reading its input up by 2 with zeros between the pixels) and a pointwise one as a
// tiled matrix product (`pw`), each finished by the bias, the fixed noise, the activation (leaky
// ReLU 0.2 times √2, clamped to ±256) and a map added after, as `flags` say. Maps are planar,
// `[channel][y][x]`.

struct P {
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    k: u32,
    stride: u32,
    pad: u32,
    mode: u32,
    flags: u32,
    pad_: u32,
    stuff: f32,
    f0: f32,
    f1: f32,
    f2: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> noise: array<f32>;
@group(0) @binding(5) var<storage, read> addm: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;

const F_BIAS: u32 = 1u;
const F_NOISE: u32 = 2u;
const F_ACT: u32 = 4u;
const F_ADD: u32 = 8u;

fn act(v: f32) -> f32 {
    let l = select(0.2 * v, v, v >= 0.0);
    return clamp(l * 1.4142135, -256.0, 256.0);
}

fn finish(v0: f32, ch: u32, pix: u32, hw: u32) -> f32 {
    var v = v0;
    if ((p.flags & F_BIAS) != 0u) {
        v = v + bias[ch];
    }
    if ((p.flags & F_NOISE) != 0u) {
        v = v + noise[pix];
    }
    if ((p.flags & F_ACT) != 0u) {
        v = act(v);
    }
    if ((p.flags & F_ADD) != 0u) {
        v = v + addm[ch * hw + pix];
    }
    return v;
}

// one output pixel of one channel
@compute @workgroup_size(16, 16, 1)
fn dw(@builtin(global_invocation_id) g: vec3<u32>) {
    let ox = g.x;
    let oy = g.y;
    let ch = g.z;
    if (ox >= p.ow || oy >= p.oh || ch >= p.cout) {
        return;
    }
    let k = p.k;
    let base = ch * p.h * p.w;
    var s = 0.0;
    for (var ky = 0u; ky < k; ky = ky + 1u) {
        for (var kx = 0u; kx < k; kx = kx + 1u) {
            var v = 0.0;
            if (p.mode == 0u) {
                let iy = i32(oy * p.stride + ky) - i32(p.pad);
                let ix = i32(ox * p.stride + kx) - i32(p.pad);
                if (iy >= 0 && ix >= 0 && iy < i32(p.h) && ix < i32(p.w)) {
                    v = src[base + u32(iy) * p.w + u32(ix)];
                }
            } else {
                // the input up by 2, zeros between its pixels, padded 2 before
                let iy = i32(oy + ky) - 2;
                let ix = i32(ox + kx) - 2;
                if (iy >= 0 && ix >= 0 && (iy & 1) == 0 && (ix & 1) == 0 && iy / 2 < i32(p.h) && ix / 2 < i32(p.w)) {
                    v = p.stuff * src[base + u32(iy / 2) * p.w + u32(ix / 2)];
                }
            }
            s = s + v * wt[(ch * k + ky) * k + kx];
        }
    }
    let pix = oy * p.ow + ox;
    dst[ch * p.oh * p.ow + pix] = finish(s, ch, pix, p.oh * p.ow);
}

// out[co][px] = sum over ci of wt[co][ci] * src[ci][px]: a 64 × 64 tile of the output a
// workgroup, 4 × 4 a thread
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

var<workgroup> a_tile: array<f32, 1024>;
var<workgroup> b_tile: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn pw(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let n0 = wg.x * BN;
    let m0 = wg.y * BM;
    let hw = p.oh * p.ow;
    let tid = l.y * 16u + l.x;
    // four rows (output channels) of four pixels, in registers
    var acc0 = vec4(0.0);
    var acc1 = vec4(0.0);
    var acc2 = vec4(0.0);
    var acc3 = vec4(0.0);
    for (var k0 = 0u; k0 < p.cin; k0 = k0 + BK) {
        // the tiles as [k][m] and [k][n]: a thread's four rows and pixels side by side
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * 256u;
            let am = idx / BK;
            let ak = idx % BK;
            var a = 0.0;
            if (m0 + am < p.cout && k0 + ak < p.cin) {
                a = wt[(m0 + am) * p.cin + k0 + ak];
            }
            a_tile[ak * BM + am] = a;
            let bk = idx / BN;
            let bn = idx % BN;
            var b = 0.0;
            if (k0 + bk < p.cin && n0 + bn < hw) {
                b = src[(k0 + bk) * hw + n0 + bn];
            }
            b_tile[bk * BN + bn] = b;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let ab = kk * BM + l.y * 4u;
            let a = vec4(a_tile[ab], a_tile[ab + 1u], a_tile[ab + 2u], a_tile[ab + 3u]);
            let bb = kk * BN + l.x * 4u;
            let b = vec4(b_tile[bb], b_tile[bb + 1u], b_tile[bb + 2u], b_tile[bb + 3u]);
            acc0 = acc0 + a.x * b;
            acc1 = acc1 + a.y * b;
            acc2 = acc2 + a.z * b;
            acc3 = acc3 + a.w * b;
        }
        workgroupBarrier();
    }
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var mi = 0u; mi < 4u; mi = mi + 1u) {
        let gm = m0 + l.y * 4u + mi;
        if (gm >= p.cout) {
            continue;
        }
        for (var ni = 0u; ni < 4u; ni = ni + 1u) {
            let gn = n0 + l.x * 4u + ni;
            if (gn < hw) {
                dst[gm * hw + gn] = finish(accs[mi][ni], gm, gn, hw);
            }
        }
    }
}
