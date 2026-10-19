//! The two detector networks, written out by hand on candle (CPU).
//!
//! Both are YuNet-style networks of depthwise-separable units: a 1×1 convolution (no
//! activation) followed by a 3×3 depthwise convolution and a ReLU, with 2×2 max pools between
//! stages; batch norms are already folded into the convolutions in the ONNX files.
//!
//! - **YuNet** (faces): a 3×3/2 stem (16 channels), stages of 16 → 64 channels, an FPN-like neck
//!   (nearest 2× upsampling and addition) giving features at strides 8, 16 and 32, and per level
//!   heads for the class score, objectness, box and five landmarks.
//! - **LPD-YuNet** (plates): a 3×3/2 stem (64 channels), stages up to 256 channels giving features
//!   at strides 8, 16, 32 and 64, and per level a head with 3, 2, 2 and 3 prior boxes of 17 values.
//!
//! The weights are the ONNX file's initializers. The code takes the file's `Conv` nodes in graph
//! order and checks each one's kernel, stride, padding, groups and weight shapes against the layer
//! it implements ([`Convs::next`]); a file of another architecture is refused, never misread.
//!
//! The 1×1 convolutions (almost all of the arithmetic) are candle matrix products; each is fused
//! with the 3×3 depthwise convolution after it, its bias, the ReLU and (where the network has
//! one) the max pool into one pass over the planes in plain Rust, in parallel.

use std::path::Path;

use candle_core::{Device, Tensor};
use rayon::prelude::*;

use crate::onnx::{self, Model, Node};
use crate::post::{self, YunetLevel};
use crate::{Detection, VisionError, catalog};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(e.to_string())
}

/// One convolution's checked weights.
struct Conv {
    /// `[cout, cin / groups, k, k]`, row-major.
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
}

/// The file's convolutions in graph order, handed out one at a time with their shapes checked.
struct Convs<'a> {
    model: &'a Model,
    nodes: Vec<&'a Node>,
    at: usize,
}

impl<'a> Convs<'a> {
    fn new(model: &'a Model) -> Self {
        Self { model, nodes: model.nodes_of("Conv").collect(), at: 0 }
    }

    /// The next convolution, which must take `cin` channels to `cout` with a `k`×`k` kernel,
    /// `stride`, `groups` (padding `k / 2`, dilation 1).
    fn next(&mut self, cin: usize, cout: usize, k: usize, stride: usize, groups: usize) -> Result<Conv, VisionError> {
        let i = self.at;
        let n = self.nodes.get(i).ok_or_else(|| err(format!("the file has {} convolutions; layer {} is missing", self.nodes.len(), i + 1)))?;
        self.at += 1;
        let what = |m: String| err(format!("convolution {} ({}): {m}", i + 1, n.name));
        let ints = |name: &str, d: &[i64]| -> Vec<i64> { n.ints(name).map(<[i64]>::to_vec).unwrap_or_else(|| d.to_vec()) };
        let pad = k / 2;
        let (k64, s64, p64) = (k as i64, stride as i64, pad as i64);
        if ints("kernel_shape", &[k64, k64]) != [k64, k64]
            || ints("strides", &[1, 1]) != [s64, s64]
            || ints("pads", &[0, 0, 0, 0]) != [p64, p64, p64, p64]
            || ints("dilations", &[1, 1]) != [1, 1]
            || n.int("group").unwrap_or(1) != groups as i64
        {
            return Err(what(format!("attributes differ from the {k}×{k}/{stride} convolution with {groups} group(s) this code implements")));
        }
        let (Some(wn), Some(bn)) = (n.inputs.get(1), n.inputs.get(2)) else { return Err(what("no weight or bias".into())) };
        let (Some(w), Some(b)) = (self.model.initializers.get(wn), self.model.initializers.get(bn)) else {
            return Err(what("its weight or bias is not in the file".into()));
        };
        let per_group = cin.checked_div(groups).filter(|_| cin.is_multiple_of(groups)).ok_or_else(|| what("bad grouping".into()))?;
        if w.dims != [cout, per_group, k, k] || b.dims != [cout] {
            return Err(what(format!("weights {:?} / {:?}, expected [{cout}, {per_group}, {k}, {k}] / [{cout}]", w.dims, b.dims)));
        }
        let (Some(wv), Some(bv)) = (w.f32s(), b.f32s()) else { return Err(what("weights are not float32".into())) };
        if wv.iter().chain(bv).any(|v| !v.is_finite()) {
            return Err(what("weights are not finite".into()));
        }
        Ok(Conv { w: wv.to_vec(), b: bv.to_vec(), cin, cout, k, stride })
    }

    /// A 1×1 convolution to `cout` and the 3×3 depthwise convolution after it.
    fn pair(&mut self, cin: usize, cout: usize, relu: bool) -> Result<PwDw, VisionError> {
        let pw = self.next(cin, cout, 1, 1, 1)?;
        let dw = self.next(cout, cout, 3, 1, cout)?;
        PwDw::new(pw, dw, relu)
    }

    /// Every convolution of the file was used.
    fn finish(&self) -> Result<(), VisionError> {
        if self.at == self.nodes.len() { Ok(()) } else { Err(err(format!("the file has {} convolutions, this code uses {}", self.nodes.len(), self.at))) }
    }
}

/// A feature map: `c` planes of `h` × `w`, row-major (batch 1).
#[derive(Clone)]
struct Map {
    v: Vec<f32>,
    c: usize,
    h: usize,
    w: usize,
}

/// The 3×3/2 stem (a candle convolution) with its bias and ReLU.
struct Stem {
    w: Tensor,
    b: Vec<f32>,
    cout: usize,
}

impl Stem {
    fn new(c: Conv) -> Result<Self, VisionError> {
        if c.k != 3 || c.stride != 2 || c.cin != 3 {
            return Err(err("the stem is not a 3×3/2 convolution of the picture"));
        }
        let w = Tensor::from_vec(c.w, (c.cout, 3, 3, 3), &Device::Cpu).map_err(err)?;
        Ok(Self { w, b: c.b, cout: c.cout })
    }

    fn forward(&self, x: &Tensor) -> Result<Map, VisionError> {
        let y = x.conv2d(&self.w, 1, 2, 1, 1).map_err(err)?;
        let (_, c, h, w) = y.dims4().map_err(err)?;
        if c != self.cout {
            return Err(err("stem output"));
        }
        let mut v: Vec<f32> = y.flatten_all().map_err(err)?.to_vec1().map_err(err)?;
        let hw = h * w;
        v.par_chunks_mut(hw.max(1)).zip(self.b.par_iter()).for_each(|(p, b)| {
            for x in p {
                *x = (*x + b).max(0.0);
            }
        });
        Ok(Map { v, c, h, w })
    }
}

/// A 1×1 convolution (a candle matrix product) followed by a 3×3 depthwise convolution with
/// zero padding, the biases, an optional ReLU and an optional 2×2 max pool, in one pass.
struct PwDw {
    pw: Tensor,
    pw_b: Vec<f32>,
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
    cin: usize,
    cout: usize,
    relu: bool,
}

impl PwDw {
    fn new(pw: Conv, dw: Conv, relu: bool) -> Result<Self, VisionError> {
        if pw.k != 1 || dw.k != 3 || dw.cin != pw.cout || dw.cout != pw.cout || dw.w.len() != pw.cout * 9 {
            return Err(err("a depthwise-separable unit's shapes do not chain"));
        }
        let t = Tensor::from_vec(pw.w, (pw.cout, pw.cin), &Device::Cpu).map_err(err)?;
        Ok(Self { pw: t, pw_b: pw.b, dw_w: dw.w, dw_b: dw.b, cin: pw.cin, cout: pw.cout, relu })
    }

    fn forward(&self, x: &Map, pool: bool) -> Result<Map, VisionError> {
        if x.c != self.cin || x.v.len() != x.c * x.h * x.w {
            return Err(err(format!("a unit takes {} channels, got {}", self.cin, x.c)));
        }
        let (h, w) = (x.h, x.w);
        let hw = h * w;
        let xt = Tensor::from_slice(&x.v, (x.c, hw), &Device::Cpu).map_err(err)?;
        let y: Vec<f32> = self.pw.matmul(&xt).map_err(err)?.flatten_all().map_err(err)?.to_vec1().map_err(err)?;
        if y.len() != self.cout * hw {
            return Err(err("pointwise output"));
        }
        let (oh, ow) = if pool { (h / 2, w / 2) } else { (h, w) };
        let mut out = vec![0f32; self.cout * oh * ow];
        if oh == 0 || ow == 0 {
            return Ok(Map { v: out, c: self.cout, h: oh, w: ow });
        }
        out.par_chunks_mut(oh * ow).enumerate().for_each(|(c, o)| {
            let (Some(src), Some(k), Some(&pb), Some(&db)) = (y.get(c * hw..(c + 1) * hw), self.dw_w.get(c * 9..c * 9 + 9), self.pw_b.get(c), self.dw_b.get(c))
            else {
                return;
            };
            let full = depthwise(src, h, w, k, pb, db, self.relu);
            if pool {
                for (oy, orow) in o.chunks_exact_mut(ow).enumerate() {
                    let (r0, r1) = (&full[2 * oy * w..2 * oy * w + w], &full[(2 * oy + 1) * w..(2 * oy + 1) * w + w]);
                    for (ox, v) in orow.iter_mut().enumerate() {
                        *v = r0[2 * ox].max(r0[2 * ox + 1]).max(r1[2 * ox]).max(r1[2 * ox + 1]);
                    }
                }
            } else {
                o.copy_from_slice(&full);
            }
        });
        Ok(Map { v: out, c: self.cout, h: oh, w: ow })
    }
}

/// One plane through a 3×3 convolution with zero padding: the input plus `pre` (the 1×1
/// convolution's bias, inside the picture only), the kernel `k`, plus `post`, then the ReLU.
fn depthwise(src: &[f32], h: usize, w: usize, k: &[f32], pre: f32, post: f32, relu: bool) -> Vec<f32> {
    let mut out = vec![post; h * w];
    if w == 0 {
        return out;
    }
    let biased: Vec<f32> = src.iter().map(|v| v + pre).collect();
    for y in 0..h {
        let orow = &mut out[y * w..(y + 1) * w];
        for ky in 0..3usize {
            let Some(yy) = (y + ky).checked_sub(1).filter(|yy| *yy < h) else { continue };
            let row = &biased[yy * w..(yy + 1) * w];
            let (k0, k1, k2) = (k[ky * 3], k[ky * 3 + 1], k[ky * 3 + 2]);
            // left neighbour, centre, right neighbour
            for (o, r) in orow[1..].iter_mut().zip(&row[..w - 1]) {
                *o += k0 * r;
            }
            for (o, r) in orow.iter_mut().zip(row) {
                *o += k1 * r;
            }
            for (o, r) in orow[..w - 1].iter_mut().zip(&row[1..]) {
                *o += k2 * r;
            }
        }
    }
    if relu {
        for v in &mut out {
            *v = v.max(0.0);
        }
    }
    out
}

fn max_pool(x: &Map) -> Map {
    let (oh, ow) = (x.h / 2, x.w / 2);
    let mut v = vec![0f32; x.c * oh * ow];
    if oh > 0 && ow > 0 {
        v.par_chunks_mut(oh * ow).enumerate().for_each(|(c, o)| {
            let Some(p) = x.v.get(c * x.h * x.w..(c + 1) * x.h * x.w) else { return };
            for (oy, orow) in o.chunks_exact_mut(ow).enumerate() {
                let (r0, r1) = (&p[2 * oy * x.w..2 * oy * x.w + x.w], &p[(2 * oy + 1) * x.w..(2 * oy + 1) * x.w + x.w]);
                for (ox, v) in orow.iter_mut().enumerate() {
                    *v = r0[2 * ox].max(r0[2 * ox + 1]).max(r1[2 * ox]).max(r1[2 * ox + 1]);
                }
            }
        });
    }
    Map { v, c: x.c, h: oh, w: ow }
}

/// `a + nearest-neighbour 2× upsampling of b` (`b` half `a`'s size, rounded down).
fn add_upsampled(a: &Map, b: &Map) -> Result<Map, VisionError> {
    if a.c != b.c || b.h * 2 != a.h || b.w * 2 != a.w {
        return Err(err(format!("feature sizes {}×{} and {}×{} do not pair (pictures are padded to multiples of 32)", a.w, a.h, b.w, b.h)));
    }
    let mut v = a.v.clone();
    let (h, w) = (a.h, a.w);
    v.par_chunks_mut((h * w).max(1)).enumerate().for_each(|(c, p)| {
        let Some(q) = b.v.get(c * b.h * b.w..(c + 1) * b.h * b.w) else { return };
        for (y, row) in p.chunks_exact_mut(w).enumerate() {
            let qrow = &q[(y / 2) * b.w..(y / 2) * b.w + b.w];
            for (x, v) in row.iter_mut().enumerate() {
                *v += qrow[x / 2];
            }
        }
    });
    Ok(Map { v, c: a.c, h, w })
}

fn run(units: &[PwDw], x: Map) -> Result<Map, VisionError> {
    units.iter().try_fold(x, |x, u| u.forward(&x, false))
}

/// Run `units`, max-pooling after the last one.
fn run_pool(units: &[PwDw], x: Map) -> Result<Map, VisionError> {
    let Some((last, rest)) = units.split_last() else { return Ok(max_pool(&x)) };
    last.forward(&run(rest, x)?, true)
}

/// An RGBA picture as the networks take it: fitted inside `max_side` (never enlarged), its sides
/// padded with black up to a multiple of `multiple`, BGR channels in 0..255, `[1, 3, H, W]`.
/// Returns the tensor and the fitted picture's size (detections are in its pixels).
fn input(rgba: &[u8], w: usize, h: usize, max_side: usize, multiple: usize) -> Result<(Tensor, usize, usize), VisionError> {
    if w == 0 || h == 0 || w > 1 << 15 || h > 1 << 15 {
        return Err(VisionError::Invalid(format!("a {w} × {h} picture cannot be analysed")));
    }
    let len = w.checked_mul(h).and_then(|n| n.checked_mul(4)).ok_or_else(|| VisionError::Invalid("picture too large".into()))?;
    let src = rgba.get(..len).ok_or_else(|| VisionError::Invalid(format!("{} bytes for a {w} × {h} RGBA picture", rgba.len())))?;
    let k = (max_side.max(32) as f64 / w.max(h) as f64).min(1.0);
    let (fw, fh) = (((w as f64 * k).round() as usize).clamp(1, w), ((h as f64 * k).round() as usize).clamp(1, h));
    let m = multiple.max(1);
    let (pw, ph) = (fw.div_ceil(m) * m, fh.div_ceil(m) * m);
    let plane = pw * ph;
    let mut data = vec![0f32; plane * 3];
    // area-average resampling into the fitted size, row by row in parallel
    let rows: Vec<Vec<[f32; 3]>> = (0..fh)
        .into_par_iter()
        .map(|y| {
            let (y0, y1) = (y * h / fh, ((y + 1) * h / fh).max(y * h / fh + 1).min(h));
            (0..fw)
                .map(|x| {
                    let (x0, x1) = (x * w / fw, ((x + 1) * w / fw).max(x * w / fw + 1).min(w));
                    let mut sum = [0u32; 3];
                    let mut n = 0u32;
                    for yy in y0..y1 {
                        let row = yy * w;
                        for xx in x0..x1 {
                            if let Some(p) = src.get((row + xx) * 4..(row + xx) * 4 + 3) {
                                sum[0] += u32::from(p[0]);
                                sum[1] += u32::from(p[1]);
                                sum[2] += u32::from(p[2]);
                                n += 1;
                            }
                        }
                    }
                    let n = n.max(1) as f32;
                    // BGR
                    [sum[2] as f32 / n, sum[1] as f32 / n, sum[0] as f32 / n]
                })
                .collect()
        })
        .collect();
    for (y, row) in rows.iter().enumerate() {
        for (x, p) in row.iter().enumerate() {
            let i = y * pw + x;
            data[i] = p[0];
            data[plane + i] = p[1];
            data[2 * plane + i] = p[2];
        }
    }
    let t = Tensor::from_vec(data, (1, 3, ph, pw), &Device::Cpu).map_err(err)?;
    Ok((t, fw, fh))
}

/// A map's `h·w` cells of `c` interleaved values.
fn cells(m: &Map) -> Vec<f32> {
    let hw = m.h * m.w;
    let mut out = vec![0f32; m.v.len()];
    for ch in 0..m.c {
        for i in 0..hw {
            if let (Some(o), Some(s)) = (out.get_mut(i * m.c + ch), m.v.get(ch * hw + i)) {
                *o = *s;
            }
        }
    }
    out
}

fn sigmoid(v: &mut [f32]) {
    for x in v {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

/// YuNet face detector.
pub struct FaceDetector {
    stem: Stem,
    stage0: PwDw,
    stages: [Vec<PwDw>; 5],
    /// The neck's units for strides 8, 16 and 32.
    lateral: [PwDw; 3],
    /// Per level (strides 8, 16, 32): class, box, objectness, landmarks.
    cls: Vec<PwDw>,
    bbox: Vec<PwDw>,
    obj: Vec<PwDw>,
    kps: Vec<PwDw>,
}

/// Longest side of the picture YuNet looks at by default: a face 5 % of the picture tall is
/// 18–32 px there, inside the 10–300 px it was trained on.
pub const FACE_SIDE: usize = 640;

/// One YuNet level's outputs: stride, rows, columns, then class and objectness (after the
/// sigmoid), box and landmarks, cells interleaved.
type YunetOut = (usize, usize, usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

impl FaceDetector {
    pub fn from_model(m: &Model) -> Result<Self, VisionError> {
        let mut c = Convs::new(m);
        let stem = Stem::new(c.next(3, 16, 3, 2, 1)?)?;
        let stage0 = c.pair(16, 16, true)?;
        let stages = [
            vec![c.pair(16, 16, true)?, c.pair(16, 32, true)?],
            vec![c.pair(32, 32, true)?, c.pair(32, 64, true)?],
            vec![c.pair(64, 64, true)?, c.pair(64, 64, true)?],
            vec![c.pair(64, 64, true)?, c.pair(64, 64, true)?],
            vec![c.pair(64, 64, true)?, c.pair(64, 64, true)?],
        ];
        // the neck's lateral units, in the file's order: stride 32, 16, 8
        let l2 = c.pair(64, 64, true)?;
        let l1 = c.pair(64, 64, true)?;
        let l0 = c.pair(64, 64, true)?;
        let mut head = |cout: usize| -> Result<Vec<PwDw>, VisionError> { (0..3).map(|_| c.pair(64, cout, false)).collect() };
        let cls = head(1)?;
        let bbox = head(4)?;
        let obj = head(1)?;
        let kps = head(10)?;
        c.finish()?;
        Ok(Self { stem, stage0, stages, lateral: [l0, l1, l2], cls, bbox, obj, kps })
    }

    pub fn load(path: &Path) -> Result<Self, VisionError> {
        let m = onnx::read(path, &onnx::Limits::default())?;
        Self::from_model(&m)
    }

    /// The head outputs for a `[1, 3, H, W]` input (H and W multiples of 32), per level.
    fn heads(&self, x: &Tensor) -> Result<Vec<YunetOut>, VisionError> {
        let x = self.stem.forward(x)?;
        let x = self.stage0.forward(&x, true)?;
        let x = run(&self.stages[0], x)?;
        let x = run_pool(&self.stages[1], x)?;
        let p3 = run(&self.stages[2], x)?;
        let p4 = run(&self.stages[3], max_pool(&p3))?;
        let p5 = run(&self.stages[4], max_pool(&p4))?;
        let n5 = self.lateral[2].forward(&p5, false)?;
        let n4 = self.lateral[1].forward(&add_upsampled(&p4, &n5)?, false)?;
        let n3 = self.lateral[0].forward(&add_upsampled(&p3, &n4)?, false)?;
        let feats = [n3, n4, n5];
        let mut out = Vec::with_capacity(3);
        for (lvl, f) in feats.iter().enumerate() {
            let get = |v: &[PwDw]| -> Result<Map, VisionError> { v.get(lvl).ok_or_else(|| err("missing head"))?.forward(f, false) };
            let (cls_m, obj_m) = (get(&self.cls)?, get(&self.obj)?);
            let (mut cls, mut obj) = (cells(&cls_m), cells(&obj_m));
            sigmoid(&mut cls);
            sigmoid(&mut obj);
            out.push((post::YUNET_STRIDES[lvl], cls_m.h, cls_m.w, cls, obj, cells(&get(&self.bbox)?), cells(&get(&self.kps)?)));
        }
        Ok(out)
    }

    /// Faces in an RGBA picture, fitted inside `max_side`: scores of at least `threshold`, after
    /// suppression (IoU 0.3), in shares of the picture.
    pub fn detect(&self, rgba: &[u8], w: usize, h: usize, max_side: usize, threshold: f32) -> Result<Vec<Detection>, VisionError> {
        let (x, fw, fh) = input(rgba, w, h, max_side, 32)?;
        let heads = self.heads(&x)?;
        let levels: Vec<YunetLevel<'_>> = heads
            .iter()
            .map(|(stride, rows, cols, cls, obj, bbox, kps)| YunetLevel { stride: *stride, rows: *rows, cols: *cols, cls, obj, bbox, kps })
            .collect();
        let dets = post::nms(post::decode_yunet(&levels, threshold), 0.3, 750);
        Ok(dets.iter().map(|d| d.normalized(fw as f32, fh as f32)).collect())
    }
}

/// LPD-YuNet licence-plate detector.
pub struct PlateDetector {
    stem: Stem,
    stage0: PwDw,
    stages: [Vec<PwDw>; 6],
    /// Per level (strides 8, 16, 32, 64): a unit, then 1×1 and 3×3 depthwise to the priors' values.
    heads: Vec<(PwDw, PwDw)>,
}

/// Longest side of the picture LPD-YuNet looks at by default.
pub const PLATE_SIDE: usize = 960;

impl PlateDetector {
    pub fn from_model(m: &Model) -> Result<Self, VisionError> {
        let mut c = Convs::new(m);
        let stem = Stem::new(c.next(3, 64, 3, 2, 1)?)?;
        let stage0 = c.pair(64, 64, true)?;
        let stages = [
            vec![c.pair(64, 64, true)?, c.pair(64, 256, true)?],
            vec![c.pair(256, 256, true)?, c.pair(256, 256, true)?],
            vec![c.pair(256, 256, true)?, c.pair(256, 256, true)?],
            vec![c.pair(256, 256, true)?, c.pair(256, 256, true)?],
            vec![c.pair(256, 256, true)?, c.pair(256, 256, true)?],
            vec![c.pair(256, 256, true)?, c.pair(256, 256, true)?],
        ];
        let mut heads = Vec::with_capacity(4);
        for anchors in [3usize, 2, 2, 3] {
            let u = c.pair(256, 256, true)?;
            let o = c.pair(256, anchors * post::LPD_STRIDE, false)?;
            heads.push((u, o));
        }
        c.finish()?;
        Ok(Self { stem, stage0, stages, heads })
    }

    pub fn load(path: &Path) -> Result<Self, VisionError> {
        let m = onnx::read(path, &onnx::Limits::default())?;
        Self::from_model(&m)
    }

    /// The raw head values for a `[1, 3, H, W]` input, in prior order ([`post::lpd_priors`]).
    fn raw(&self, x: &Tensor) -> Result<Vec<f32>, VisionError> {
        let x = self.stem.forward(x)?;
        let x = self.stage0.forward(&x, true)?;
        let x = run(&self.stages[0], x)?;
        let x = run_pool(&self.stages[1], x)?;
        let s8 = run(&self.stages[2], x)?;
        let s16 = run(&self.stages[3], max_pool(&s8))?;
        let s32 = run(&self.stages[4], max_pool(&s16))?;
        let s64 = run(&self.stages[5], max_pool(&s32))?;
        let mut out = Vec::new();
        for ((u, o), f) in self.heads.iter().zip([&s8, &s16, &s32, &s64]) {
            out.extend(cells(&o.forward(&u.forward(f, false)?, false)?));
        }
        Ok(out)
    }

    /// Plates in an RGBA picture, fitted inside `max_side`: scores of at least `threshold`, after
    /// suppression (IoU 0.3), in shares of the picture.
    pub fn detect(&self, rgba: &[u8], w: usize, h: usize, max_side: usize, threshold: f32) -> Result<Vec<Detection>, VisionError> {
        let (x, fw, fh) = input(rgba, w, h, max_side, 1)?;
        let raw = self.raw(&x)?;
        let priors = post::lpd_priors(fw, fh);
        if raw.len() != priors.len() * post::LPD_STRIDE {
            return Err(err(format!("{} head values for {} priors", raw.len(), priors.len())));
        }
        let dets = post::nms(post::decode_lpd(&raw, &priors, fw, fh, threshold), 0.3, 750);
        Ok(dets.iter().map(|d| d.normalized(fw as f32, fh as f32)).collect())
    }
}

/// Both detectors, loaded from the models directory.
pub struct Detectors {
    pub face: Option<FaceDetector>,
    pub plate: Option<PlateDetector>,
}

impl Detectors {
    /// Load the detectors that are wanted (`faces`, `plates`) from `models_dir`.
    pub fn load(models_dir: &Path, faces: bool, plates: bool) -> Result<Self, VisionError> {
        let dir = catalog::package_dir(models_dir);
        let need = |f: &catalog::ModelFile| std::fs::metadata(dir.join(f.name)).is_ok_and(|m| m.len() == f.size);
        if (faces && !need(&catalog::FACE)) || (plates && !need(&catalog::PLATE)) {
            return Err(VisionError::NotInstalled);
        }
        let face = if faces { Some(FaceDetector::load(&dir.join(catalog::FACE.name))?) } else { None };
        let plate = if plates { Some(PlateDetector::load(&dir.join(catalog::PLATE.name))?) } else { None };
        Ok(Self { face, plate })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The models directory (`FILMCRAFT_VISION_MODELS`, holding `privacy-detectors/`) and the
    /// oracle directory (`FILMCRAFT_VISION_ORACLE`: `picture.png` and ONNX Runtime's outputs from
    /// `oracle.py`), or `None` to skip.
    fn live() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let m = std::env::var_os("FILMCRAFT_VISION_MODELS")?;
        let o = std::env::var_os("FILMCRAFT_VISION_ORACLE")?;
        Some((m.into(), o.into()))
    }

    fn f32s(p: &std::path::Path) -> Vec<f32> {
        std::fs::read(p).unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }

    fn picture(dir: &std::path::Path) -> (Vec<u8>, usize, usize) {
        let img = image::open(dir.join("picture.png")).unwrap().to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        (img.into_raw(), w, h)
    }

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// The fused 3×3 depthwise pass against a direct convolution with zero padding.
    #[test]
    fn depthwise_matches_a_direct_convolution() {
        let (h, w) = (5, 7);
        let src: Vec<f32> = (0..h * w).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let k: Vec<f32> = (0..9).map(|i| i as f32 * 0.25 - 1.0).collect();
        let got = depthwise(&src, h, w, &k, 0.5, -0.25, false);
        for y in 0..h {
            for x in 0..w {
                let mut acc = -0.25;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (yy, xx) = (y as i64 + ky - 1, x as i64 + kx - 1);
                        if yy >= 0 && yy < h as i64 && xx >= 0 && xx < w as i64 {
                            acc += k[(ky * 3 + kx) as usize] * (src[yy as usize * w + xx as usize] + 0.5);
                        }
                    }
                }
                assert!((got[y * w + x] - acc).abs() < 1e-5, "({x}, {y})");
            }
        }
        // one-pixel-wide planes
        assert_eq!(depthwise(&[1.0], 1, 1, &k, 0.0, 0.0, false), vec![k[4]]);
        assert!(depthwise(&[], 0, 0, &k, 0.0, 0.0, true).is_empty());
    }

    #[test]
    fn bad_pictures_are_errors() {
        assert!(input(&[0; 16], 2, 2, 640, 32).is_ok());
        assert!(input(&[0; 15], 2, 2, 640, 32).is_err());
        assert!(input(&[], 0, 2, 640, 32).is_err());
        assert!(input(&[], 1 << 20, 1 << 20, 640, 32).is_err());
    }

    /// YuNet's raw outputs match ONNX Runtime's on a real frame, and the faces match OpenCV's
    /// `FaceDetectorYN`.
    #[test]
    #[ignore = "needs the downloaded models and an oracle directory (FILMCRAFT_VISION_MODELS, FILMCRAFT_VISION_ORACLE)"]
    fn yunet_matches_onnx_runtime() {
        let Some((models, oracle)) = live() else { return };
        let det = FaceDetector::load(&catalog::package_dir(&models).join(catalog::FACE.name)).unwrap();
        let (rgba, w, h) = picture(&oracle);
        let (x, ..) = input(&rgba, w, h, w.max(h), 32).unwrap();
        let heads = det.heads(&x).unwrap();
        for (stride, _, _, cls, obj, bbox, kps) in &heads {
            for (name, ours) in [("cls", cls), ("obj", obj), ("bbox", bbox), ("kps", kps)] {
                let theirs = f32s(&oracle.join(format!("yunet_{name}_{stride}.f32")));
                let e = max_err(ours, &theirs);
                let scale = theirs.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
                eprintln!("yunet {name}_{stride}: max abs error {e:.2e} (largest value {scale:.1})");
                assert!(e <= 1e-3 * scale, "{name}_{stride}: {e}");
            }
        }
        let faces = det.detect(&rgba, w, h, w.max(h), 0.6).unwrap();
        let o: serde_json::Value = serde_json::from_slice(&std::fs::read(oracle.join("oracle.json")).unwrap()).unwrap();
        let theirs = o["faces"].as_array().unwrap();
        assert_eq!(faces.len(), theirs.len(), "{faces:?}");
        for t in theirs {
            let t: Vec<f32> = t.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let b = [t[0] / w as f32, t[1] / h as f32, t[2] / w as f32, t[3] / h as f32];
            let m = faces.iter().map(|f| crate::post::iou(&f.bbox, &b)).fold(0.0, f32::max);
            assert!(m > 0.99, "OpenCV's face {t:?} has no match (best IoU {m})");
        }
    }

    /// LPD-YuNet's raw outputs match ONNX Runtime's on a real frame.
    #[test]
    #[ignore = "needs the downloaded models and an oracle directory (FILMCRAFT_VISION_MODELS, FILMCRAFT_VISION_ORACLE)"]
    fn lpd_matches_onnx_runtime() {
        let Some((models, oracle)) = live() else { return };
        let det = PlateDetector::load(&catalog::package_dir(&models).join(catalog::PLATE.name)).unwrap();
        let (rgba, w, h) = picture(&oracle);
        let (x, ..) = input(&rgba, w, h, w.max(h), 1).unwrap();
        let raw = det.raw(&x).unwrap();
        let (loc, conf, iou) = (f32s(&oracle.join("lpd_loc.f32")), f32s(&oracle.join("lpd_conf.f32")), f32s(&oracle.join("lpd_iou.f32")));
        let n = iou.len();
        assert_eq!(raw.len(), n * post::LPD_STRIDE);
        assert_eq!(n, post::lpd_priors(w, h).len());
        let (mut el, mut ec, mut ei) = (0f32, 0f32, 0f32);
        for (i, v) in raw.as_chunks::<{ post::LPD_STRIDE }>().0.iter().enumerate() {
            for k in 0..14 {
                el = el.max((v[k] - loc[i * 14 + k]).abs());
            }
            let m = v[14].max(v[15]);
            let p = (v[15] - m).exp() / ((v[14] - m).exp() + (v[15] - m).exp());
            ec = ec.max((p - conf[i * 2 + 1]).abs());
            ei = ei.max((v[16] - iou[i]).abs());
        }
        eprintln!("lpd: max abs error loc {el:.2e}, plate probability {ec:.2e}, iou {ei:.2e}");
        assert!(el < 1e-2 && ec < 1e-3 && ei < 1e-3);
    }

    /// How long one detection takes on this machine (release builds give the real figure).
    #[test]
    #[ignore = "a timing, with the downloaded models (FILMCRAFT_VISION_MODELS, FILMCRAFT_VISION_ORACLE)"]
    fn detection_speed() {
        let Some((models, oracle)) = live() else { return };
        let d = Detectors::load(&models, true, true).unwrap();
        let (rgba, w, h) = picture(&oracle);
        for (side, label) in [(640usize, "face 640"), (320, "face 320"), (640, "plate 640"), (480, "plate 480")] {
            let t0 = std::time::Instant::now();
            let n = 10;
            for _ in 0..n {
                if label.starts_with("face") {
                    d.face.as_ref().unwrap().detect(&rgba, w, h, side, 0.6).unwrap();
                } else {
                    d.plate.as_ref().unwrap().detect(&rgba, w, h, side, 0.8).unwrap();
                }
            }
            eprintln!("{label}: {:.1} ms a picture", t0.elapsed().as_secs_f64() * 1000.0 / n as f64);
        }
    }
}
