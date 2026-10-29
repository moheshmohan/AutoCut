// RIFE's IFNet on the GPU (`interp_gpu.rs`), the arithmetic of candle's passes in `interp.rs`.
// Feature maps are `[c][h][w]` f32 in storage buffers; a pass reads and writes channel ranges of
// them (`*_off` the first channel, `*_total` the buffer's channels), so the blocks' inputs are
// written in place instead of concatenated.
//
// - `conv`: a 3 × 3 convolution, stride 1 or 2, zero padding 1, its bias, then `act` 1 a leaky
//   ReLU (0.2), or 2 a residual unit's `leaky(conv · beta + input)`: a tiled matrix product of
//   the output's pixels (64 a workgroup) by its channels (64) over the input's channels × 9 taps
//   (16 at a time through workgroup memory), the input read in place.
// - `convt`: a 4 × 4 transposed convolution, stride 2, padding 1, its bias, written pixel-shuffled
//   by 2 when `shuffle` is 1 (channel c of the transposed output goes to channel c / 4 at twice the
//   size, at row offset (c % 4) / 2 and column offset c % 2).
// - `resize`: a bilinear resize without aligned corners (half-pixel centres, edges held), the
//   channels from `split` on times `factor`.
// - `warp`: channels sampled bilinearly at each pixel plus its motion (two channels of `flow`),
//   the edges held.
// - `axpy`: `dst = keep · dst + factor · src` over a channel range.
// - `fill`: a channel set to a constant.
// - `mix`: the two warped frames mixed by the mask's sigmoid, clamped to 0..1.

struct P {
    // sizes: source (channels, height, width), destination (height, width)
    sc: u32,
    sh: u32,
    sw: u32,
    dh: u32,
    dw: u32,
    // channels: count, source offset and total, destination offset and total
    n: u32,
    s_off: u32,
    s_total: u32,
    d_off: u32,
    d_total: u32,
    // conv / convt: output channels, stride, activation, shuffle
    cout: u32,
    stride: u32,
    act: u32,
    shuffle: u32,
    // a third buffer's channel offset (the flow of `warp`, the residual of `conv`)
    x_off: u32,
    split: u32,
    factor: f32,
    keep: f32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> extra: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

// The convolution's tile (`interp_gpu.rs` makes a module for each of three shapes by replacing
// these lines: 64 pixels × 64 channels, 128 × 32 and 256 × 16, so a layer of few channels keeps
// its lanes busy). WX × WY invocations, each 4 pixels × 4 channels.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const WX: u32 = 16u;
const WY: u32 = 16u;

var<workgroup> a_tile: array<f32, 1024>;
var<workgroup> b_tile: array<f32, 1024>;

fn leaky(v: f32) -> f32 {
    return select(v * 0.2, v, v >= 0.0);
}

// `conv`: src is the input (`sc` channels from `s_off` of `s_total`, `sh` × `sw`), the output
// `dh` × `dw` (`cout` channels at `d_off` of `d_total`); `wt` is `[sc · 9][cout]`; `extra` holds
// the residual units' beta (`cout`) then the residual is the input itself (same size, stride 1).
@compute @workgroup_size(WX, WY)
fn conv(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tid = lid.y * WX + lid.x;
    let opx = p.dh * p.dw;
    let ipx = p.sh * p.sw;
    let m0 = wg.x * BM;
    let n0 = wg.y * BN;
    let kn = p.sc * 9u;
    var acc0 = vec4(0.0);
    var acc1 = vec4(0.0);
    var acc2 = vec4(0.0);
    var acc3 = vec4(0.0);
    for (var k0 = 0u; k0 < kn; k0 += BK) {
        for (var q = 0u; q < (BM * BK + 255u) / 256u; q++) {
            let e = q * 256u + tid;
            if e >= BM * BK {
                break;
            }
            let kk = e / BM;
            let mm = e % BM;
            let k = k0 + kk;
            let m = m0 + mm;
            var v = 0.0;
            if k < kn && m < opx {
                let ci = k / 9u;
                let tap = k % 9u;
                let y = i32((m / p.dw) * p.stride) + i32(tap / 3u) - 1;
                let x = i32((m % p.dw) * p.stride) + i32(tap % 3u) - 1;
                if y >= 0 && y < i32(p.sh) && x >= 0 && x < i32(p.sw) {
                    v = src[(p.s_off + ci) * ipx + u32(y) * p.sw + u32(x)];
                }
            }
            a_tile[kk * BM + mm] = v;
        }
        for (var q = 0u; q < (BN * BK + 255u) / 256u; q++) {
            let e = q * 256u + tid;
            if e >= BN * BK {
                break;
            }
            let kb = e / BN;
            let nb = e % BN;
            let k2 = k0 + kb;
            let n = n0 + nb;
            var wv = 0.0;
            if k2 < kn && n < p.cout {
                wv = wt[k2 * p.cout + n];
            }
            b_tile[kb * BN + nb] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let ab = kk * BM + lid.x * 4u;
            let a = vec4(a_tile[ab], a_tile[ab + 1u], a_tile[ab + 2u], a_tile[ab + 3u]);
            let bb = kk * BN + lid.y * 4u;
            let b = vec4(b_tile[bb], b_tile[bb + 1u], b_tile[bb + 2u], b_tile[bb + 3u]);
            acc0 += a.x * b;
            acc1 += a.y * b;
            acc2 += a.z * b;
            acc3 += a.w * b;
        }
        workgroupBarrier();
    }
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let m = m0 + lid.x * 4u + i;
        if m >= opx {
            continue;
        }
        for (var j = 0u; j < 4u; j++) {
            let n = n0 + lid.y * 4u + j;
            if n >= p.cout {
                continue;
            }
            var v = accs[i][j] + bias[n];
            if p.act == 1u {
                v = leaky(v);
            } else if p.act == 2u {
                v = leaky(v * extra[n] + src[(p.s_off + n) * ipx + m]);
            }
            dst[(p.d_off + n) * opx + m] = v;
        }
    }
}

// `convt`: src `sc` channels (`s_off` of `s_total`) `sh` × `sw`; the transposed output is
// `2sh` × `2sw` with `cout` channels; `wt` is `[sc][cout][4][4]`. Four output channels an
// invocation (`z` the group of four).
@compute @workgroup_size(8, 8, 1)
fn convt(@builtin(global_invocation_id) id: vec3<u32>) {
    let oh = p.sh * 2u;
    let ow = p.sw * 2u;
    let co0 = id.z * 4u;
    if id.x >= ow || id.y >= oh || co0 >= p.cout {
        return;
    }
    let oy = i32(id.y);
    let ox = i32(id.x);
    let ipx = p.sh * p.sw;
    // the two rows and two columns of taps that reach this pixel (the others fall between input
    // pixels): ky ≡ oy + 1 (mod 2), iy = (oy + 1 − ky) / 2
    let py = (oy + 1) % 2;
    let px = (ox + 1) % 2;
    var taps: array<u32, 4>;
    var at: array<u32, 4>;
    var ok: array<bool, 4>;
    for (var a = 0; a < 2; a++) {
        for (var b = 0; b < 2; b++) {
            let ky = py + 2 * a;
            let kx = px + 2 * b;
            let iy = (oy + 1 - ky) / 2;
            let ix = (ox + 1 - kx) / 2;
            let k = u32(a * 2 + b);
            ok[k] = iy >= 0 && iy < i32(p.sh) && ix >= 0 && ix < i32(p.sw);
            taps[k] = u32(ky * 4 + kx);
            at[k] = select(0u, u32(iy) * p.sw + u32(ix), ok[k]);
        }
    }
    var acc = vec4(0.0);
    for (var ci = 0u; ci < p.sc; ci++) {
        let sb = (p.s_off + ci) * ipx;
        for (var k = 0u; k < 4u; k++) {
            if !ok[k] {
                continue;
            }
            let v = src[sb + at[k]];
            for (var j = 0u; j < 4u; j++) {
                let co = co0 + j;
                if co < p.cout {
                    acc[j] += v * wt[(ci * p.cout + co) * 16u + taps[k]];
                }
            }
        }
    }
    for (var j = 0u; j < 4u; j++) {
        let co = co0 + j;
        if co >= p.cout {
            continue;
        }
        let v = acc[j] + bias[co];
        if p.shuffle == 1u {
            let fw = ow * 2u;
            let fh = oh * 2u;
            let y = u32(oy) * 2u + (co % 4u) / 2u;
            let x = u32(ox) * 2u + co % 2u;
            dst[(p.d_off + co / 4u) * fh * fw + y * fw + x] = v;
        } else {
            dst[(p.d_off + co) * oh * ow + u32(oy) * ow + u32(ox)] = v;
        }
    }
}

// `resize`: `n` channels from `s_off` of the source (`sh` × `sw`) to `d_off` of the destination
// (`dh` × `dw`); channels from `split` on (counted within the `n`) times `factor`.
@compute @workgroup_size(8, 8, 1)
fn resize(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.dw || id.y >= p.dh || id.z >= p.n {
        return;
    }
    let sy = max((f32(id.y) + 0.5) * f32(p.sh) / f32(p.dh) - 0.5, 0.0);
    let sx = max((f32(id.x) + 0.5) * f32(p.sw) / f32(p.dw) - 0.5, 0.0);
    let y0 = min(u32(floor(sy)), p.sh - 1u);
    let x0 = min(u32(floor(sx)), p.sw - 1u);
    let y1 = min(y0 + 1u, p.sh - 1u);
    let x1 = min(x0 + 1u, p.sw - 1u);
    let fy = sy - f32(y0);
    let fx = sx - f32(x0);
    let b = (p.s_off + id.z) * p.sh * p.sw;
    let top = src[b + y0 * p.sw + x0] * (1.0 - fx) + src[b + y0 * p.sw + x1] * fx;
    let bottom = src[b + y1 * p.sw + x0] * (1.0 - fx) + src[b + y1 * p.sw + x1] * fx;
    var v = top * (1.0 - fy) + bottom * fy;
    if id.z >= p.split {
        v = v * p.factor;
    }
    dst[(p.d_off + id.z) * p.dh * p.dw + id.y * p.dw + id.x] = v;
}

// `warp`: `n` channels from `s_off` of the source, sampled at each pixel plus the motion in
// channels `x_off`, `x_off + 1` of `extra` (the flow, same size), into `d_off` of the destination.
@compute @workgroup_size(8, 8, 1)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.sw || id.y >= p.sh || id.z >= p.n {
        return;
    }
    let npx = p.sh * p.sw;
    let i = id.y * p.sw + id.x;
    let sx = clamp(f32(id.x) + extra[p.x_off * npx + i], 0.0, f32(p.sw - 1u));
    let sy = clamp(f32(id.y) + extra[(p.x_off + 1u) * npx + i], 0.0, f32(p.sh - 1u));
    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, p.sw - 1u);
    let y1 = min(y0 + 1u, p.sh - 1u);
    let ax = sx - f32(x0);
    let ay = sy - f32(y0);
    let b = (p.s_off + id.z) * npx;
    let top = src[b + y0 * p.sw + x0] * (1.0 - ax) + src[b + y0 * p.sw + x1] * ax;
    let bottom = src[b + y1 * p.sw + x0] * (1.0 - ax) + src[b + y1 * p.sw + x1] * ax;
    dst[(p.d_off + id.z) * npx + i] = top * (1.0 - ay) + bottom * ay;
}

// `axpy`: `dst[d_off + c] = keep · dst[d_off + c] + factor · src[s_off + c]` for `n` channels
// of `sh` × `sw`.
@compute @workgroup_size(8, 8, 1)
fn axpy(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.sw || id.y >= p.sh || id.z >= p.n {
        return;
    }
    let npx = p.sh * p.sw;
    let i = id.y * p.sw + id.x;
    let d = (p.d_off + id.z) * npx + i;
    let k = select(p.keep * dst[d], 0.0, p.keep == 0.0);
    dst[d] = k + p.factor * src[(p.s_off + id.z) * npx + i];
}

// `fill`: channel `d_off` (`sh` × `sw`) set to `factor`.
@compute @workgroup_size(8, 8, 1)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.sw || id.y >= p.sh {
        return;
    }
    dst[p.d_off * p.sh * p.sw + id.y * p.sw + id.x] = p.factor;
}

// `mix`: `src` holds the two warped frames (channels 0..3 and 3..6), `extra` the mask's logit in
// channel `x_off`; the result (3 channels) clamped to 0..1.
@compute @workgroup_size(8, 8, 1)
fn mix(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.sw || id.y >= p.sh || id.z >= 3u {
        return;
    }
    let npx = p.sh * p.sw;
    let i = id.y * p.sw + id.x;
    let m = 1.0 / (1.0 + exp(-extra[p.x_off * npx + i]));
    let v = src[id.z * npx + i] * m + src[(3u + id.z) * npx + i] * (1.0 - m);
    dst[id.z * npx + i] = clamp(v, 0.0, 1.0);
}
