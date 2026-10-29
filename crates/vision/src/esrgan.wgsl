// Real-ESRGAN's compact network on the GPU (`upscale_gpu.rs`), the arithmetic of candle's passes
// in `upscale.rs`. Feature maps are `[c][h][w]` f32 in storage buffers.
//
// - `conv3`: a 3 × 3 convolution (stride 1, zero padding 1) with its bias and, when `act` is 1,
//   a PReLU (one slope a channel): a tiled matrix product of the output's pixels (64 a
//   workgroup) by its channels (64 a workgroup) over the input's channels × 9 taps (16 at a time
//   through workgroup memory), the input's 3 × 3 neighbourhoods read in place (no im2col
//   buffer); 4 pixels × 4 channels an invocation.
// - `shuffle`: the 48 channels made the picture four times as large (pixel shuffle by 4) plus the
//   input enlarged by nearest neighbour.

struct P {
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    act: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> slope: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

var<workgroup> a_tile: array<f32, 1024>;
var<workgroup> b_tile: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn conv3(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let npx = p.h * p.w;
    let m0 = wg.x * BM;
    let n0 = wg.y * BN;
    let kn = p.cin * 9u;
    var acc0 = vec4(0.0);
    var acc1 = vec4(0.0);
    var acc2 = vec4(0.0);
    var acc3 = vec4(0.0);
    for (var k0 = 0u; k0 < kn; k0 += BK) {
        for (var q = 0u; q < 4u; q++) {
            let e = q * 256u + tid;
            // the input's value for pixel m at tap k (ci, ky, kx)
            let kk = e / BM;
            let mm = e % BM;
            let k = k0 + kk;
            let m = m0 + mm;
            var v = 0.0;
            if k < kn && m < npx {
                let ci = k / 9u;
                let tap = k % 9u;
                let y = i32(m / p.w) + i32(tap / 3u) - 1;
                let x = i32(m % p.w) + i32(tap % 3u) - 1;
                if y >= 0 && y < i32(p.h) && x >= 0 && x < i32(p.w) {
                    v = src[ci * npx + u32(y) * p.w + u32(x)];
                }
            }
            a_tile[kk * BM + mm] = v;
            // the weight of tap k for output channel n (`wt` is `[k][cout]`)
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
        if m >= npx {
            continue;
        }
        for (var j = 0u; j < 4u; j++) {
            let n = n0 + lid.y * 4u + j;
            if n >= p.cout {
                continue;
            }
            var v = accs[i][j] + bias[n];
            if p.act == 1u && v < 0.0 {
                v = v * slope[n];
            }
            dst[n * npx + m] = v;
        }
    }
}

struct S {
    h: u32,
    w: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> s: S;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn shuffle(@builtin(global_invocation_id) id: vec3<u32>) {
    let ow = s.w * 4u;
    let oh = s.h * 4u;
    if id.x >= ow || id.y >= oh || id.z >= 3u {
        return;
    }
    let c = id.z;
    let x = id.x / 4u;
    let y = id.y / 4u;
    let sub = (id.y % 4u) * 4u + id.x % 4u;
    let npx = s.h * s.w;
    outp[(c * oh + id.y) * ow + id.x] = feat[(c * 16u + sub) * npx + y * s.w + x] + inp[c * npx + y * s.w + x];
}
