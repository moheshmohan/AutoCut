// MobileSAM's image encoder (TinyViT-5M and its neck, crate::sam_gpu) as five compute passes.
// Maps are planar, [channel][y][x] (a transformer's tokens are the columns: a 1 × 1 convolution
// and a linear layer are the same matrix product). Each pass finishes with what `flags` say: the
// bias, a map added, GELU (candle's tanh form, as the CPU's encoder).

struct P {
    m: u32,      // output channels (rows)
    k: u32,      // input channels (× 9 for a 3 × 3 convolution)
    h: u32,      // input height and width
    w: u32,
    oh: u32,     // output height and width
    ow: u32,
    stride: u32,
    flags: u32,
    heads: u32,  // attention: heads, window side, key width, windows across
    ws: u32,
    kd: u32,
    nw: u32,
    eps: f32,
    scale: f32,
    f0: f32,
    f1: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> addm: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

const F_BIAS: u32 = 1u;
const F_ADD: u32 = 2u;
const F_GELU: u32 = 4u;
// the matrix product's input is a 3 × 3 convolution's (padded 1, `stride`)
const F_CONV3: u32 = 8u;

fn gelu(x: f32) -> f32 {
    let c = 0.7978845608028654;
    // (Metal's tanh is NaN for large arguments; it is ±1 to f32 beyond ±10)
    let t = tanh(clamp(c * (x + 0.044715 * x * x * x), -10.0, 10.0));
    return 0.5 * x * (1.0 + t);
}

fn finish(v0: f32, ch: u32, pix: u32, hw: u32) -> f32 {
    var v = v0;
    if ((p.flags & F_BIAS) != 0u) {
        v = v + bias[ch];
    }
    if ((p.flags & F_ADD) != 0u) {
        v = v + addm[ch * hw + pix];
    }
    if ((p.flags & F_GELU) != 0u) {
        v = gelu(v);
    }
    return v;
}

// the matrix product's input at row `r`, column `c`
fn input(r: u32, c: u32) -> f32 {
    if ((p.flags & F_CONV3) == 0u) {
        return src[r * p.oh * p.ow + c];
    }
    let ci = r / 9u;
    let t = r % 9u;
    let oy = c / p.ow;
    let ox = c % p.ow;
    let iy = i32(oy * p.stride + t / 3u) - 1;
    let ix = i32(ox * p.stride + t % 3u) - 1;
    if (iy < 0 || ix < 0 || iy >= i32(p.h) || ix >= i32(p.w)) {
        return 0.0;
    }
    return src[(ci * p.h + u32(iy)) * p.w + u32(ix)];
}

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

var<workgroup> a_tile: array<f32, 1024>;
var<workgroup> b_tile: array<f32, 1024>;

// out[m][n] = sum over k of wt[m][k] * input(k, n): a 64 × 64 tile a workgroup, 4 × 4 a thread
@compute @workgroup_size(16, 16, 1)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let n0 = wg.x * BN;
    let m0 = wg.y * BM;
    let hw = p.oh * p.ow;
    let tid = l.y * 16u + l.x;
    var acc0 = vec4(0.0);
    var acc1 = vec4(0.0);
    var acc2 = vec4(0.0);
    var acc3 = vec4(0.0);
    for (var k0 = 0u; k0 < p.k; k0 = k0 + BK) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * 256u;
            let am = idx / BK;
            let ak = idx % BK;
            var a = 0.0;
            if (m0 + am < p.m && k0 + ak < p.k) {
                a = wt[(m0 + am) * p.k + k0 + ak];
            }
            a_tile[ak * BM + am] = a;
            let bk = idx / BN;
            let bn = idx % BN;
            var b = 0.0;
            if (k0 + bk < p.k && n0 + bn < hw) {
                b = input(k0 + bk, n0 + bn);
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
        if (gm >= p.m) {
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

// a depthwise 3 × 3 convolution (padded 1, `stride`): one output pixel of one channel
@compute @workgroup_size(16, 16, 1)
fn dw(@builtin(global_invocation_id) g: vec3<u32>) {
    let ox = g.x;
    let oy = g.y;
    let ch = g.z;
    if (ox >= p.ow || oy >= p.oh || ch >= p.m) {
        return;
    }
    let base = ch * p.h * p.w;
    var s = 0.0;
    for (var ky = 0u; ky < 3u; ky = ky + 1u) {
        let iy = i32(oy * p.stride + ky) - 1;
        if (iy < 0 || iy >= i32(p.h)) {
            continue;
        }
        for (var kx = 0u; kx < 3u; kx = kx + 1u) {
            let ix = i32(ox * p.stride + kx) - 1;
            if (ix >= 0 && ix < i32(p.w)) {
                s = s + src[base + u32(iy) * p.w + u32(ix)] * wt[ch * 9u + ky * 3u + kx];
            }
        }
    }
    let pix = oy * p.ow + ox;
    dst[ch * p.oh * p.ow + pix] = finish(s, ch, pix, p.oh * p.ow);
}

// the layer norm of each column over its `m` rows (weights in `wt`, biases in `bias`)
@compute @workgroup_size(64, 1, 1)
fn ln(@builtin(global_invocation_id) g: vec3<u32>) {
    let col = g.x;
    let n = p.oh * p.ow;
    if (col >= n) {
        return;
    }
    var mean = 0.0;
    for (var r = 0u; r < p.m; r = r + 1u) {
        mean = mean + src[r * n + col];
    }
    mean = mean / f32(p.m);
    var var_ = 0.0;
    for (var r = 0u; r < p.m; r = r + 1u) {
        let d = src[r * n + col] - mean;
        var_ = var_ + d * d;
    }
    let inv = 1.0 / sqrt(var_ / f32(p.m) + p.eps);
    for (var r = 0u; r < p.m; r = r + 1u) {
        dst[r * n + col] = (src[r * n + col] - mean) * inv * wt[r] + bias[r];
    }
}

// a map `h` × `w` copied into the top-left of one `oh` × `ow` (zeros round it), `m` channels; or,
// with F_ADD, the top-left `oh` × `ow` of one `h` × `w` taken out and `addm` added
@compute @workgroup_size(16, 16, 1)
fn place(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = g.x;
    let y = g.y;
    let ch = g.z;
    if (x >= p.ow || y >= p.oh || ch >= p.m) {
        return;
    }
    let o = (ch * p.oh + y) * p.ow + x;
    var v = 0.0;
    if (x < p.w && y < p.h) {
        v = src[(ch * p.h + y) * p.w + x];
    }
    if ((p.flags & F_ADD) != 0u) {
        v = v + addm[o];
    }
    dst[o] = v;
}

// window attention: one query token of one head of one window. `src` is the qkv map
// (`heads` × [q, k, v] × `kd` rows, the padded grid's tokens as columns, `oh` × `ow`), `wt` the
// position bias [head][query][key]; the answer goes to rows head × `kd` of `dst`
@compute @workgroup_size(64, 1, 1)
fn attn(@builtin(global_invocation_id) g: vec3<u32>) {
    let nn = p.ws * p.ws;
    let qi = g.x;
    let head = g.y;
    let win = g.z;
    if (qi >= nn || head >= p.heads) {
        return;
    }
    let n = p.oh * p.ow;
    let wy = win / p.nw;
    let wx = win % p.nw;
    let tok = (wy * p.ws + qi / p.ws) * p.ow + wx * p.ws + qi % p.ws;
    let qrow = head * 3u * p.kd;
    let krow = qrow + p.kd;
    let vrow = qrow + 2u * p.kd;
    let brow = (head * nn + qi) * nn;
    // the scores' largest, then the weighted values
    var best = -3.0e38;
    for (var j = 0u; j < nn; j = j + 1u) {
        let tj = (wy * p.ws + j / p.ws) * p.ow + wx * p.ws + j % p.ws;
        var s = 0.0;
        for (var d = 0u; d < p.kd; d = d + 1u) {
            s = s + src[(qrow + d) * n + tok] * src[(krow + d) * n + tj];
        }
        best = max(best, s * p.scale + wt[brow + j]);
    }
    var acc: array<vec4<f32>, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        acc[i] = vec4(0.0);
    }
    var total = 0.0;
    for (var j = 0u; j < nn; j = j + 1u) {
        let tj = (wy * p.ws + j / p.ws) * p.ow + wx * p.ws + j % p.ws;
        var s = 0.0;
        for (var d = 0u; d < p.kd; d = d + 1u) {
            s = s + src[(qrow + d) * n + tok] * src[(krow + d) * n + tj];
        }
        let e = exp(s * p.scale + wt[brow + j] - best);
        total = total + e;
        for (var d4 = 0u; d4 < p.kd / 4u; d4 = d4 + 1u) {
            let r = vrow + d4 * 4u;
            acc[d4] = acc[d4] + e * vec4(src[r * n + tj], src[(r + 1u) * n + tj], src[(r + 2u) * n + tj], src[(r + 3u) * n + tj]);
        }
    }
    let inv = 1.0 / max(total, 1e-30);
    for (var d4 = 0u; d4 < p.kd / 4u; d4 = d4 + 1u) {
        let r = head * p.kd + d4 * 4u;
        dst[r * n + tok] = acc[d4].x * inv;
        dst[(r + 1u) * n + tok] = acc[d4].y * inv;
        dst[(r + 2u) * n + tok] = acc[d4].z * inv;
        dst[(r + 3u) * n + tok] = acc[d4].w * inv;
    }
}
