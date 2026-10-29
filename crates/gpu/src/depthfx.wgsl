// The depth effects of `filmcraft_render::depth` per pixel, the same arithmetic as its CPU
// reference: `parallax` (the 3D photo: each pixel looks back along its move for the nearest
// surface whose moved place it is) and `blur` (Depth Blur: the discs that reach a pixel, each
// weighted by the area it stands for over its disc's).

struct Par {
    w: u32,
    h: u32,
    dw: u32,
    dh: u32,
    tx: f32,
    ty: f32,
    push: f32,
    pivot: f32,
    zoom: f32,
    min_steps: u32,
    max_steps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> par: Par;
@group(0) @binding(1) var<storage, read> img: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> held: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;

fn held_at(u: f32, v: f32) -> f32 {
    let x = clamp(u, 0.0, 1.0) * f32(max(par.dw, 1u) - 1u);
    let y = clamp(v, 0.0, 1.0) * f32(max(par.dh, 1u) - 1u);
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, max(par.dw, 1u) - 1u);
    let y1 = min(y0 + 1u, max(par.dh, 1u) - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let a = held[y0 * par.dw + x0];
    let b = held[y0 * par.dw + x1];
    let c = held[y1 * par.dw + x0];
    let d = held[y1 * par.dw + x1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

fn px(x: i32, y: i32) -> vec4<f32> {
    let xx = u32(clamp(x, 0, i32(par.w) - 1));
    let yy = u32(clamp(y, 0, i32(par.h) - 1));
    return img[yy * par.w + xx];
}

fn sample(x: f32, y: f32) -> vec4<f32> {
    let sx = clamp(x, 0.0, f32(max(par.w, 1u) - 1u));
    let sy = clamp(y, 0.0, f32(max(par.h, 1u) - 1u));
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let fx = sx - f32(x0);
    let fy = sy - f32(y0);
    let a = px(x0, y0);
    let b = px(x0 + 1, y0);
    let c = px(x0, y0 + 1);
    let d = px(x0 + 1, y0 + 1);
    let top = a + (b - a) * fx;
    return top + (c + (d - c) * fx - top) * fy;
}

@compute @workgroup_size(8, 8)
fn parallax(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= par.w || gid.y >= par.h) {
        return;
    }
    let cx = f32(max(par.w, 1u) - 1u) * 0.5;
    let cy = f32(max(par.h, 1u) - 1u) * 0.5;
    let qx = cx + (f32(gid.x) - cx) / par.zoom;
    let qy = cy + (f32(gid.y) - cy) / par.zoom;
    let vx = par.tx + (qx - cx) * par.push;
    let vy = par.ty + (qy - cy) * par.push;
    let len = sqrt(vx * vx + vy * vy);
    var sx = qx;
    var sy = qy;
    if (len > 1e-3) {
        let n = clamp(u32(ceil(len)), par.min_steps, par.max_steps);
        let su = 1.0 / f32(max(par.w, 2u) - 1u);
        let sv = 1.0 / f32(max(par.h, 2u) - 1u);
        var ps = 1.0;
        var pf = held_at((qx - (1.0 - par.pivot) * vx) * su, (qy - (1.0 - par.pivot) * vy) * sv) - 1.0;
        var hit = 1.0;
        if (pf < 0.0) {
            hit = 0.0;
            for (var i = 1u; i <= n; i = i + 1u) {
                let s = 1.0 - f32(i) / f32(n);
                let f1 = held_at((qx - (s - par.pivot) * vx) * su, (qy - (s - par.pivot) * vy) * sv) - s;
                if (f1 >= 0.0) {
                    hit = ps + (s - ps) * (pf / (pf - f1));
                    break;
                }
                ps = s;
                pf = f1;
            }
        }
        sx = qx - (hit - par.pivot) * vx;
        sy = qy - (hit - par.pivot) * vy;
    }
    out[gid.y * par.w + gid.x] = sample(sx, sy);
}

struct Blur {
    w: u32,
    h: u32,
    taps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> bl: Blur;
@group(0) @binding(1) var<storage, read> bimg: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> nr: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> taps: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> bout: array<vec4<f32>>;

const PI: f32 = 3.14159265358979;
const BEHIND: f32 = 0.02;

fn weight(area: f32, r: f32) -> f32 {
    return area / max(PI * r * r, 1.0);
}

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= bl.w || gid.y >= bl.h) {
        return;
    }
    let i = gid.y * bl.w + gid.x;
    let me = nr[i];
    let own = weight(1.0, me.y);
    var acc = bimg[i] * own;
    var sum = own;
    for (var k = 0u; k < bl.taps; k = k + 1u) {
        let t = taps[k];
        let qx = i32(gid.x) + i32(t.x);
        let qy = i32(gid.y) + i32(t.y);
        if (qx < 0 || qy < 0 || qx >= i32(bl.w) || qy >= i32(bl.h)) {
            continue;
        }
        let q = u32(qy) * bl.w + u32(qx);
        let o = nr[q];
        if (t.z > o.y + 0.5 || (o.x < me.x - BEHIND && t.z > me.y + 0.5)) {
            continue;
        }
        let wt = weight(t.w, o.y);
        acc = acc + bimg[q] * wt;
        sum = sum + wt;
    }
    bout[i] = acc / sum;
}
