// The profile pass's transformer parts and heads on the GPU (`gpunn.rs`), the arithmetic of the
// CPU code in `layers.rs`, `dfine.rs` and `rtmpose.rs`: matrices are row-major `[rows, cols]` f32
// in storage buffers. Indices (D-FINE's chosen queries) are kept as exact f32 integers.
//
// - `linear`: `x · W + bias`, an activation (`act`: 0 none, 1 ReLU, 2 GELU, 3 SiLU), a residual
//   added after it (`flags` bit 1), a clamp to ±f0 (bit 2); bias with `flags` bit 0. A tiled
//   matrix product, 16 × 16 outputs a workgroup.
// - `layernorm` (eps f0), `scalenorm` (gain f0), `softmax` over each row.
// - `attention`: multi-head scaled dot-product attention (scale f0), one query and head an
//   invocation, the softmax in two passes as on the CPU.
// - `add`, `concat2`, `gate` (D-FINE's sigmoid gate), `mask_rows`, `planes2rows`, `rows2planes`,
//   `rowmax`, `rank` (the `k` best of a row by value, ties by index: a stable sort's order),
//   `gather`, `refs_init`, `init_box`, `to_box` (FDR's distributions to a box), `lqe_stats`,
//   `add_rowscalar`, `deform` (D-FINE's deformable cross-attention by bilinear samples).
// - RTMPose's head: `gau_attn` (squared-ReLU attention weights between a crop's keypoints),
//   `gau_mix` (u ⊙ (A·v)), `scale_add`, `argmax` of each row (first maximum, as on the CPU).

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
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read> d: array<f32>;
@group(0) @binding(5) var<storage, read> e: array<f32>;
@group(0) @binding(6) var<storage, read> f: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;

// The error function (Abramowitz & Stegun 7.1.26), as `dfine::erf`.
fn erf_(x0: f32) -> f32 {
    let s = sign(x0);
    let x = abs(x0);
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449674) * t + 0.2548296) * t * exp(-x * x);
    return s * y;
}

fn activate(v: f32) -> f32 {
    switch p.act {
        case 1u: {
            return max(v, 0.0);
        }
        case 2u: {
            return 0.5 * v * (1.0 + erf_(v / 1.4142135));
        }
        case 3u: {
            return v / (1.0 + exp(-v));
        }
        default: {
            return v;
        }
    }
}

// Finite (not NaN or ±∞), by the bits: Metal compiles with fast math, where `v == v` and
// comparisons with NaN may be folded away.
fn finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn is_nan(v: f32) -> bool {
    let b = bitcast<u32>(v);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

fn sigm(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

var<workgroup> tile_x: array<array<f32, 16>, 16>;
var<workgroup> tile_w: array<array<f32, 16>, 16>;

// rows = p.rows, in = p.cin, out = p.cout; a: x, b: W [in, out], c: bias, d: residual.
@compute @workgroup_size(16, 16)
fn linear(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let col = wg.x * 16u + lid.x;
    let row = wg.y * 16u + lid.y;
    var acc = 0.0;
    let steps = (p.cin + 15u) / 16u;
    for (var s = 0u; s < steps; s++) {
        let kx = s * 16u + lid.x;
        if row < p.rows && kx < p.cin {
            tile_x[lid.y][lid.x] = a[row * p.cin + kx];
        } else {
            tile_x[lid.y][lid.x] = 0.0;
        }
        let kw = s * 16u + lid.y;
        if kw < p.cin && col < p.cout {
            tile_w[lid.y][lid.x] = b[kw * p.cout + col];
        } else {
            tile_w[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < 16u; i++) {
            acc += tile_x[lid.y][i] * tile_w[i][lid.x];
        }
        workgroupBarrier();
    }
    if row >= p.rows || col >= p.cout {
        return;
    }
    var v = acc;
    if (p.flags & 1u) != 0u {
        v += c[col];
    }
    v = activate(v);
    if (p.flags & 2u) != 0u {
        v += d[row * p.cout + col];
    }
    if (p.flags & 4u) != 0u {
        v = clamp(v, -p.f0, p.f0);
    }
    o[row * p.cout + col] = v;
}

// a: x [rows, cols], b: gain, c: bias, f0: eps.
@compute @workgroup_size(64)
fn layernorm(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if r >= p.rows {
        return;
    }
    let n = p.cols;
    let base = r * n;
    var s = 0.0;
    for (var i = 0u; i < n; i++) {
        s += a[base + i];
    }
    let mean = s / f32(n);
    var v = 0.0;
    for (var i = 0u; i < n; i++) {
        let x = a[base + i] - mean;
        v += x * x;
    }
    let inv = 1.0 / sqrt(v / f32(n) + p.f0);
    for (var i = 0u; i < n; i++) {
        o[base + i] = (a[base + i] - mean) * inv * b[i] + c[i];
    }
}

// x / max(‖x‖ · cols^−½, 1e−5) · f0
@compute @workgroup_size(64)
fn scalenorm(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if r >= p.rows {
        return;
    }
    let n = p.cols;
    let base = r * n;
    var s = 0.0;
    for (var i = 0u; i < n; i++) {
        s += a[base + i] * a[base + i];
    }
    let k = 1.0 / sqrt(f32(n));
    let norm = max(sqrt(s) * k, 1e-5);
    for (var i = 0u; i < n; i++) {
        o[base + i] = a[base + i] / norm * p.f0;
    }
}

@compute @workgroup_size(64)
fn softmax(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if r >= p.rows {
        return;
    }
    let n = p.cols;
    let base = r * n;
    var m = -3.0e38;
    for (var i = 0u; i < n; i++) {
        m = max(m, a[base + i]);
    }
    var s = 0.0;
    for (var i = 0u; i < n; i++) {
        s += exp(a[base + i] - m);
    }
    for (var i = 0u; i < n; i++) {
        let v = exp(a[base + i] - m);
        if s > 0.0 {
            o[base + i] = v / s;
        } else {
            o[base + i] = v;
        }
    }
}

// a: q, b: k, c: v, all [rows, cols]; heads of hd = cols / heads (at most 64); f0: scale.
@compute @workgroup_size(64)
fn attention(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let h = id.y;
    let n = p.rows;
    let dm = p.cols;
    let hd = p.hd;
    if i >= n || h >= p.heads || hd > 64u {
        return;
    }
    let qo = i * dm + h * hd;
    var m = -3.0e38;
    for (var j = 0u; j < n; j++) {
        var s = 0.0;
        let ko = j * dm + h * hd;
        for (var t = 0u; t < hd; t++) {
            s += a[qo + t] * b[ko + t];
        }
        m = max(m, s * p.f0);
    }
    var acc: array<f32, 64>;
    for (var t = 0u; t < 64u; t++) {
        acc[t] = 0.0;
    }
    var l = 0.0;
    for (var j = 0u; j < n; j++) {
        var s = 0.0;
        let ko = j * dm + h * hd;
        for (var t = 0u; t < hd; t++) {
            s += a[qo + t] * b[ko + t];
        }
        let w = exp(s * p.f0 - m);
        l += w;
        for (var t = 0u; t < hd; t++) {
            acc[t] += w * c[ko + t];
        }
    }
    for (var t = 0u; t < hd; t++) {
        if l > 0.0 {
            o[qo + t] = acc[t] / l;
        } else {
            o[qo + t] = acc[t];
        }
    }
}

// o = a + b (rows · cols values).
@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 65535u * 64u;
    if i >= p.rows * p.cols {
        return;
    }
    o[i] = a[i] + b[i];
}

// [a | b]: a [rows, cin], b [rows, cout] side by side.
@compute @workgroup_size(64)
fn concat2(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let r = id.y;
    let w = p.cin + p.cout;
    if j >= w || r >= p.rows {
        return;
    }
    if j < p.cin {
        o[r * w + j] = a[r * p.cin + j];
    } else {
        o[r * w + j] = b[r * p.cout + j - p.cin];
    }
}

// D-FINE's gate in place of the residual: a: the gate's logits [rows, 2·cols], b: h1, c: the
// cross-attention, both [rows, cols].
@compute @workgroup_size(64)
fn gate(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let r = id.y;
    let n = p.cols;
    if j >= n || r >= p.rows {
        return;
    }
    o[r * n + j] = sigm(a[r * 2u * n + j]) * b[r * n + j] + sigm(a[r * 2u * n + n + j]) * c[r * n + j];
}

// Rows whose b is 0 set to zeros.
@compute @workgroup_size(64)
fn mask_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let r = id.y;
    if j >= p.cols || r >= p.rows {
        return;
    }
    if b[r] > 0.5 {
        o[r * p.cols + j] = a[r * p.cols + j];
    } else {
        o[r * p.cols + j] = 0.0;
    }
}

// A map's planes (a: `[cols][rows]`, picture 0) as token rows, written at row `a_off` of o.
@compute @workgroup_size(64)
fn planes2rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let ch = id.y;
    if j >= p.rows || ch >= p.cols {
        return;
    }
    o[(p.a_off + j) * p.cols + ch] = a[ch * p.rows + j];
}

// Token rows (from row `a_off` of a) as a map's planes.
@compute @workgroup_size(64)
fn rows2planes(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let ch = id.y;
    if j >= p.rows || ch >= p.cols {
        return;
    }
    o[ch * p.rows + j] = a[(p.a_off + j) * p.cols + ch];
}

@compute @workgroup_size(64)
fn rowmax(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if r >= p.rows {
        return;
    }
    var m = -3.4028235e38;
    for (var j = 0u; j < p.cols; j++) {
        m = max(m, a[r * p.cols + j]);
    }
    o[r] = m;
}

// The `k` largest of `rows` values (a), in order (ties: the first first), as indices.
@compute @workgroup_size(64)
fn rank(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 65535u * 64u;
    if i >= p.rows {
        return;
    }
    let v = a[i];
    var r = 0u;
    for (var j = 0u; j < p.rows; j++) {
        let w = a[j];
        if w > v || (w == v && j < i) {
            r += 1u;
        }
    }
    if r < p.k {
        o[r] = f32(i);
    }
}

// Rows b[q] of a (cols wide).
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let q = id.y;
    if j >= p.cols || q >= p.rows {
        return;
    }
    let src = u32(b[q]);
    o[q * p.cols + j] = a[src * p.cols + j];
}

// The first boxes: sigmoid(a (the regression) + the chosen anchors (b, by the indices c)).
@compute @workgroup_size(64)
fn refs_init(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.rows * 4u {
        return;
    }
    let q = i / 4u;
    let j = i % 4u;
    let src = u32(c[q]);
    o[i] = sigm(a[i] + b[src * 4u + j]);
}

// sigmoid(a + inverse_sigmoid(b))
@compute @workgroup_size(64)
fn init_box(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.rows * 4u {
        return;
    }
    let x = clamp(b[i], 0.0, 1.0);
    let inv = log(max(x, 1e-5) / max(1.0 - x, 1e-5));
    o[i] = sigm(a[i] + inv);
}

// A query's box from its first layer's box (a) and its four distance distributions (b, `k`
// bins each), the bin positions c, f0 = reg_scale.
@compute @workgroup_size(64)
fn to_box(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = id.x;
    if q >= p.rows {
        return;
    }
    let bins = p.k;
    let side = 4u * bins;
    var dist = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var s = 0u; s < 4u; s++) {
        let base = q * side + s * bins;
        var m = -3.0e38;
        for (var i = 0u; i < bins; i++) {
            m = max(m, b[base + i]);
        }
        var sum = 0.0;
        for (var i = 0u; i < bins; i++) {
            sum += exp(b[base + i] - m);
        }
        var dv = 0.0;
        for (var i = 0u; i < bins; i++) {
            var pr = exp(b[base + i] - m);
            if sum > 0.0 {
                pr = pr / sum;
            }
            dv += pr * c[i];
        }
        dist[s] = dv;
    }
    let r = max(p.f0, 1e-6);
    let px = a[q * 4u];
    let py = a[q * 4u + 1u];
    let pw = a[q * 4u + 2u];
    let ph = a[q * 4u + 3u];
    let x1 = px - (0.5 * r + dist[0]) * (pw / r);
    let y1 = py - (0.5 * r + dist[1]) * (ph / r);
    let x2 = px + (0.5 * r + dist[2]) * (pw / r);
    let y2 = py + (0.5 * r + dist[3]) * (ph / r);
    o[q * 4u] = (x1 + x2) / 2.0;
    o[q * 4u + 1u] = (y1 + y2) / 2.0;
    o[q * 4u + 2u] = x2 - x1;
    o[q * 4u + 3u] = y2 - y1;
}

// The localisation-quality statistics of a query's side: its distribution's `n` (p.n) largest
// probabilities, largest first, and their mean. a: the distributions (`k` bins each).
@compute @workgroup_size(64)
fn lqe_stats(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if t >= p.rows * 4u {
        return;
    }
    let bins = p.k;
    let base = t * bins;
    var m = -3.0e38;
    for (var i = 0u; i < bins; i++) {
        m = max(m, a[base + i]);
    }
    var sum = 0.0;
    for (var i = 0u; i < bins; i++) {
        sum += exp(a[base + i] - m);
    }
    let top = p.n;
    let out = t * (top + 1u);
    // the next largest after (pv, pi) in the order (value down, index up)
    var pv = 3.0e38;
    var pi = 0u;
    var total = 0.0;
    for (var r = 0u; r < top; r++) {
        var bv = -1.0;
        var bi = 0u;
        var found = false;
        for (var i = 0u; i < bins; i++) {
            var v = exp(a[base + i] - m);
            if sum > 0.0 {
                v = v / sum;
            }
            let after = r == 0u || v < pv || (v == pv && i > pi);
            if after && (!found || v > bv) {
                bv = v;
                bi = i;
                found = true;
            }
        }
        o[out + r] = bv;
        total += bv;
        pv = bv;
        pi = bi;
    }
    o[out + top] = total / f32(max(top, 1u));
}

// a [rows, cols] plus each row's b.
@compute @workgroup_size(64)
fn add_rowscalar(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let r = id.y;
    if j >= p.cols || r >= p.rows {
        return;
    }
    o[r * p.cols + j] = a[r * p.cols + j] + b[r];
}

// D-FINE's deformable cross-attention: a: the memory's tokens [*, cols]; b: offsets
// [rows, heads, total, 2]; c: weights [rows, heads, total] (softmaxed); d: boxes [rows, 4];
// e: per point its scale, then its level; f: per level its first token, rows and columns;
// f0: the offset scale. One output channel an invocation.
@compute @workgroup_size(64)
fn deform(@builtin(global_invocation_id) id: vec3<u32>) {
    let ch = id.x;
    let q = id.y;
    let dm = p.cols;
    if ch >= dm || q >= p.rows || p.hd == 0u {
        return;
    }
    let h = ch / p.hd;
    let rx = d[q * 4u];
    let ry = d[q * 4u + 1u];
    let rw = d[q * 4u + 2u];
    let rh = d[q * 4u + 3u];
    var acc = 0.0;
    for (var pt = 0u; pt < p.total; pt++) {
        let at = (q * p.heads + h) * p.total + pt;
        let ox = b[at * 2u];
        let oy = b[at * 2u + 1u];
        let wgt = c[at];
        let ps = e[pt];
        let lvl = u32(e[p.total + pt]);
        if lvl >= p.levels {
            continue;
        }
        let start = u32(f[lvl * 3u]);
        let lh = u32(f[lvl * 3u + 1u]);
        let lw = u32(f[lvl * 3u + 2u]);
        let lx = rx + ox * ps * rw * p.f0;
        let ly = ry + oy * ps * rh * p.f0;
        let px = lx * f32(lw) - 0.5;
        let py = ly * f32(lh) - 0.5;
        // non-finite positions and empty weights add nothing
        if !finite(px) || !finite(py) || wgt == 0.0 {
            continue;
        }
        let x0 = floor(px);
        let y0 = floor(py);
        let fx = px - x0;
        let fy = py - y0;
        for (var k = 0u; k < 4u; k++) {
            let cx = x0 + f32(k & 1u);
            let cy = y0 + f32(k >> 1u);
            var cw = 0.0;
            switch k {
                case 0u: {
                    cw = (1.0 - fx) * (1.0 - fy);
                }
                case 1u: {
                    cw = fx * (1.0 - fy);
                }
                case 2u: {
                    cw = (1.0 - fx) * fy;
                }
                default: {
                    cw = fx * fy;
                }
            }
            if cx < 0.0 || cy < 0.0 || cx >= f32(lw) || cy >= f32(lh) || cw == 0.0 {
                continue;
            }
            let tok = start + u32(cy) * lw + u32(cx);
            acc += cw * wgt * a[tok * dm + ch];
        }
    }
    o[q * dm + ch] = acc;
}

// RTMPose's gated attention unit, its weights: a: the uv rows [crops·k, 2e + s] (u, v, base);
// b: gamma [2, s]; c: beta [2, s]; e = p.cin, s = p.cout, k keypoints a crop.
// o [crops·k, k] = relu(q_a · k_b / √s)².
@compute @workgroup_size(64)
fn gau_attn(@builtin(global_invocation_id) id: vec3<u32>) {
    let kb = id.x;
    let row = id.y;
    let kk = p.k;
    if kb >= kk || row >= p.rows {
        return;
    }
    let crop = row / kk;
    let e2 = p.cin;
    let s = p.cout;
    let w = 2u * e2 + s;
    let qa = row * w + 2u * e2;
    let kbase = (crop * kk + kb) * w + 2u * e2;
    var dot = 0.0;
    for (var j = 0u; j < s; j++) {
        let q = a[qa + j] * b[j] + c[j];
        let k = a[kbase + j] * b[s + j] + c[s + j];
        dot += q * k;
    }
    let v = max(dot / sqrt(f32(s)), 0.0);
    o[row * kk + kb] = v * v;
}

// u ⊙ (A · v): a: the uv rows, b: the weights [crops·k, k]; o [crops·k, e].
@compute @workgroup_size(64)
fn gau_mix(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let row = id.y;
    let kk = p.k;
    let e2 = p.cin;
    if j >= e2 || row >= p.rows {
        return;
    }
    let crop = row / kk;
    let w = 2u * e2 + p.cout;
    var acc = 0.0;
    for (var kb = 0u; kb < kk; kb++) {
        let wt = b[row * kk + kb];
        if wt == 0.0 {
            continue;
        }
        acc += wt * a[(crop * kk + kb) * w + e2 + j];
    }
    o[row * e2 + j] = a[row * w + j] * acc;
}

// o = c[col] · a + b
@compute @workgroup_size(64)
fn scale_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let r = id.y;
    if j >= p.cols || r >= p.rows {
        return;
    }
    o[r * p.cols + j] = c[j] * a[r * p.cols + j] + b[r * p.cols + j];
}

// Each row's first maximum: [index, value].
@compute @workgroup_size(64)
fn argmax(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if r >= p.rows {
        return;
    }
    var bi = 0u;
    var bv = -3.4028235e38;
    var any = false;
    for (var j = 0u; j < p.cols; j++) {
        let v = a[r * p.cols + j];
        // (NaN is never taken)
        if !is_nan(v) && (!any || v > bv) {
            bv = v;
            bi = j;
            any = true;
        }
    }
    o[r * 2u] = f32(bi);
    o[r * 2u + 1u] = bv;
}
