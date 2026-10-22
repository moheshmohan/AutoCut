//! RTMPose (Jiang et al., "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose",
//! arXiv 2303.07399; Apache-2.0, OpenMMLab), the pose network of the profile pass, written out by
//! hand on the CPU ([`crate::layers`], the reference) and on the GPU ([`crate::gpunn`]: the
//! crops, the backbone, the head and the SimCC maxima, one read-back for every person of a frame).
//! Top-down: each person's box, grown by a quarter and widened to 3:4, is warped to 192 × 256 and
//! looked at on its own.
//!
//! The weights are the `end2end.onnx` file of OpenMMLab's deployment zip (batch norms already
//! folded into the convolutions). The code takes the file's `Conv` nodes in graph order and
//! checks each one's kernel, stride, padding, groups and weight shapes against the layer it
//! implements, and finds the head's matrices and scales by shape; a file of another architecture
//! is refused, never misread.
//!
//! - **Backbone** CSPNeXt: a stem of three 3×3 convolutions; four stages of a 3×3/2 convolution
//!   and a CSP layer (two 1×1 branches, CSPNeXt blocks of 3×3, depthwise 5×5 and 1×1
//!   convolutions on one with a residual except in the last stage, the branches joined, a
//!   channel attention (global average, 1×1, hard sigmoid) and a 1×1 convolution); an SPP
//!   bottleneck (max pools 5, 9, 13) before the last CSP layer. SiLU everywhere.
//! - **Head** RTMCC: a 7×7 convolution to one 8 × 6 map per keypoint, flattened, a scale norm
//!   and a linear layer to 256 values, a gated attention unit (scale norm, a SiLU linear layer
//!   split into u, v and a shared base, per-keypoint queries and keys from the base, squared ReLU
//!   attention, u ⊙ (A·v), a linear layer, a scaled residual), then two linear layers to the
//!   SimCC logits along x (384 bins, half a pixel each) and y (512).

use std::path::Path;

use rayon::prelude::*;

use crate::VisionError;
use crate::gpunn::{GImg, GM, GT, GpuUse, Nn, Run, TAct};
use crate::head::{KEYPOINTS, Keypoint};
use crate::layers::{Act, Backend, Conv, Cpu, Feat, Linear, err};
use crate::onnx::{self, Data, Model, Node};

/// The network's input, pixels.
pub const INPUT_W: usize = 192;
pub const INPUT_H: usize = 256;
/// SimCC bins per input pixel.
const SPLIT: f32 = 2.0;
/// A person's box is grown by this before it is cropped.
const PADDING: f32 = 1.25;
/// mmpose's normalisation (RGB).
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];
/// Most people looked at in one batch.
pub const MAX_BATCH: usize = 16;

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

    fn peek(&self) -> Option<&'a Node> {
        self.nodes.get(self.at).copied()
    }

    /// The next convolution, which must take `cin` channels to `cout` with a `k`×`k` kernel,
    /// `stride`, `groups` (padding `k / 2`).
    fn next(&mut self, cin: usize, cout: usize, k: usize, stride: usize, groups: usize, act: Act) -> Result<Conv, VisionError> {
        let i = self.at;
        let n = self.nodes.get(i).ok_or_else(|| err(format!("the pose file has {} convolutions; layer {} is missing", self.nodes.len(), i + 1)))?;
        self.at += 1;
        let what = |m: String| err(format!("pose convolution {} ({}): {m}", i + 1, n.name));
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
        let per = cin.checked_div(groups).filter(|_| cin.is_multiple_of(groups)).ok_or_else(|| what("bad grouping".into()))?;
        if w.dims != [cout, per, k, k] || b.dims != [cout] {
            return Err(what(format!("weights {:?} / {:?}, expected [{cout}, {per}, {k}, {k}] / [{cout}]", w.dims, b.dims)));
        }
        let (Some(wv), Some(bv)) = (w.f32s(), b.f32s()) else { return Err(what("weights are not float32".into())) };
        Conv::new(wv.to_vec(), bv.to_vec(), cin, cout, k, stride, pad, groups, act)
    }

    /// The next convolution's output channels, if it is a `k`×`k` one taking `cin`.
    fn next_is(&self, cin: usize, k: usize) -> Option<usize> {
        let n = self.peek()?;
        let w = self.model.initializers.get(n.inputs.get(1)?)?;
        match w.dims[..] {
            [cout, c, kh, kw] if c == cin && kh == k && kw == k => Some(cout),
            _ => None,
        }
    }
}

/// A CSPNeXt block: 3×3, depthwise 5×5, 1×1, and the input added back.
struct Block {
    conv: Conv,
    dw: Conv,
    pw: Conv,
    residual: bool,
}

/// A CSP layer with channel attention.
struct Csp {
    short: Conv,
    main: Conv,
    blocks: Vec<Block>,
    /// The channel attention's 1×1 convolution (with a bias, no activation).
    attn: Conv,
    fin: Conv,
}

impl Csp {
    fn forward<B: Backend>(&self, be: &B, x: &B::T) -> Result<B::T, VisionError> {
        let short = be.conv(&self.short, x, None)?;
        let mut m = be.conv(&self.main, x, None)?;
        for b in &self.blocks {
            let y = be.conv(&b.dw, &be.conv(&b.conv, &m, None)?, None)?;
            m = be.conv(&b.pw, &y, if b.residual { Some(&m) } else { None })?;
        }
        let cat = be.concat(&[&m, &short])?;
        be.conv(&self.fin, &be.chan_attn(&cat, &self.attn)?, None)
    }

    fn upload(&mut self, nn: &Nn) {
        for c in [&mut self.short, &mut self.main, &mut self.attn, &mut self.fin] {
            c.upload(nn);
        }
        for b in &mut self.blocks {
            b.conv.upload(nn);
            b.dw.upload(nn);
            b.pw.upload(nn);
        }
    }
}

struct Stage {
    down: Conv,
    spp: Option<(Conv, Conv)>,
    csp: Csp,
}

/// RTMPose.
pub struct RtmPose {
    stem: [Conv; 3],
    stages: Vec<Stage>,
    final_layer: Conv,
    /// The head's first scale norm's gain, then the linear layer (`[48, 256]`).
    mlp_g: f32,
    mlp: Linear,
    gau_g: f32,
    uv: Linear,
    /// `[2, s]` each: the per-keypoint query and key from the shared base.
    gamma: Vec<f32>,
    beta: Vec<f32>,
    o: Linear,
    res_scale: Vec<f32>,
    cls_x: Linear,
    cls_y: Linear,
    e: usize,
    s: usize,
    gpu: Option<GpuUse>,
    /// The head's vectors on the GPU: gamma, beta, the residual's scale.
    consts: Option<[wgpu::Buffer; 3]>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Scale norm over each row of `x` (rows of `d`): `x / max(‖x‖·d^−½, 1e−5) · g`.
pub(crate) fn scale_norm(x: &mut [f32], d: usize, g: f32) {
    let k = (d as f32).sqrt().recip();
    for r in x.chunks_exact_mut(d.max(1)) {
        let n = (r.iter().map(|v| v * v).sum::<f32>().sqrt() * k).max(1e-5);
        for v in r {
            *v = *v / n * g;
        }
    }
}

impl RtmPose {
    /// Load from OpenMMLab's deployment zip (its `end2end.onnx`).
    pub fn load(zip_path: &Path) -> Result<Self, VisionError> {
        let len = std::fs::metadata(zip_path).map_err(|e| err(format!("{}: {e}", zip_path.display())))?.len();
        if len > crate::zip::MAX_ENTRY as u64 {
            return Err(err(format!("{} is too large", zip_path.display())));
        }
        let bytes = std::fs::read(zip_path).map_err(|e| err(format!("{}: {e}", zip_path.display())))?;
        let model_bytes = crate::zip::read_entry(&bytes, "end2end.onnx", 128 << 20)?;
        let m = onnx::parse(&model_bytes, &onnx::Limits { max_bytes: 128 << 20, ..onnx::Limits::default() })?;
        Self::from_model(&m)
    }

    pub fn from_model(m: &Model) -> Result<Self, VisionError> {
        let mut c = Convs::new(m);
        let c0 = c.next_is(3, 3).ok_or_else(|| err("the pose file does not start with a 3×3 convolution of the picture"))?;
        let stem = [c.next(3, c0, 3, 2, 1, Act::Silu)?, c.next(c0, c0, 3, 1, 1, Act::Silu)?, c.next(c0, 2 * c0, 3, 1, 1, Act::Silu)?];
        let mut cin = 2 * c0;
        let mut stages = Vec::with_capacity(4);
        for si in 0..4 {
            let last = si == 3;
            let cout = c.next_is(cin, 3).ok_or_else(|| err(format!("pose stage {}: no downsampling convolution", si + 1)))?;
            let down = c.next(cin, cout, 3, 2, 1, Act::Silu)?;
            let spp = if last { Some((c.next(cout, cout / 2, 1, 1, 1, Act::Silu)?, c.next(cout * 2, cout, 1, 1, 1, Act::Silu)?)) } else { None };
            let half = cout / 2;
            let short = c.next(cout, half, 1, 1, 1, Act::Silu)?;
            let main = c.next(cout, half, 1, 1, 1, Act::Silu)?;
            let mut blocks = Vec::new();
            while c.next_is(half, 3) == Some(half) && blocks.len() < 16 {
                blocks.push(Block {
                    conv: c.next(half, half, 3, 1, 1, Act::Silu)?,
                    dw: c.next(half, half, 5, 1, half, Act::Silu)?,
                    pw: c.next(half, half, 1, 1, 1, Act::Silu)?,
                    residual: !last,
                });
            }
            if blocks.is_empty() {
                return Err(err(format!("pose stage {}: no blocks", si + 1)));
            }
            // the channel attention's 1×1 convolution (with a bias, no activation)
            let attn = c.next(cout, cout, 1, 1, 1, Act::None)?;
            let fin = c.next(cout, cout, 1, 1, 1, Act::Silu)?;
            stages.push(Stage { down, spp, csp: Csp { short, main, blocks, attn, fin } });
            cin = cout;
        }
        let final_layer = c.next(cin, KEYPOINTS, 7, 1, 1, Act::None)?;
        if c.at != c.nodes.len() {
            return Err(err(format!("the pose file has {} convolutions, this code uses {}", c.nodes.len(), c.at)));
        }
        // the head's matrices, in graph order, and its scales by name and shape
        let mats: Vec<(usize, usize, &[f32])> = m
            .nodes_of("MatMul")
            .filter_map(|n| {
                let t = m.initializers.get(n.inputs.get(1)?)?;
                match (&t.dims[..], &t.data) {
                    ([a, b], Data::F32(v)) => Some((*a, *b, v.as_slice())),
                    _ => None,
                }
            })
            .collect();
        let [(fl, hid, mlp_w), (h2, uvn, uv_w), (e2, h3, o_w), (h4, xn, x_w), (h5, yn, y_w)] = mats[..] else {
            return Err(err(format!("the pose head has {} weight matrices, expected 5", mats.len())));
        };
        // u and v are e2 wide each, the shared base s_dim
        let s_dim = uvn.saturating_sub(2 * e2);
        if fl != 48 || h2 != hid || h3 != hid || h4 != hid || h5 != hid || xn != INPUT_W * 2 || yn != INPUT_H * 2 || s_dim == 0 || e2 == 0 {
            return Err(err("the pose head's matrices do not chain"));
        }
        let scalar = |name: &str| -> Result<f32, VisionError> {
            match m.initializers.get(name).and_then(|t| t.f32s()) {
                Some([v]) if v.is_finite() => Ok(*v),
                _ => Err(err(format!("the pose head's `{name}` is missing"))),
            }
        };
        let named = |name: &str, len: usize| -> Result<Vec<f32>, VisionError> {
            match m.initializers.get(name).and_then(|t| t.f32s()) {
                Some(v) if v.len() == len && v.iter().all(|x| x.is_finite()) => Ok(v.to_vec()),
                _ => Err(err(format!("the pose head's `{name}` is missing"))),
            }
        };
        // gamma has no name in the file: the [1, 1, 2, s] tensor multiplied by the base
        let gamma = m
            .initializers
            .values()
            .find(|t| t.dims == [1, 1, 2, s_dim] && t.f32s().is_some())
            .and_then(|t| t.f32s())
            .map(<[f32]>::to_vec)
            .ok_or_else(|| err("the pose head's gamma is missing"))?;
        Ok(Self {
            stem,
            stages,
            final_layer,
            mlp_g: scalar("head.mlp.0.g")?,
            mlp: Linear::from_in_out(mlp_w, fl, hid)?,
            gau_g: scalar("head.gau.ln.g")?,
            uv: Linear::from_in_out(uv_w, hid, uvn)?,
            gamma,
            beta: named("head.gau.beta", 2 * s_dim)?,
            o: Linear::from_in_out(o_w, e2, hid)?,
            res_scale: named("head.gau.res_scale.scale", hid)?,
            cls_x: Linear::from_in_out(x_w, hid, xn)?,
            cls_y: Linear::from_in_out(y_w, hid, yn)?,
            e: e2,
            s: s_dim,
            gpu: None,
            consts: None,
        })
    }

    /// Put the network on this machine's GPU (the CPU stays the fallback); `false` without one
    /// (or when a weight does not go up: the CPU then).
    pub fn use_gpu(&mut self) -> bool {
        let Some(nn) = Nn::shared() else { return false };
        for c in &mut self.stem {
            c.upload(&nn);
        }
        for st in &mut self.stages {
            st.down.upload(&nn);
            if let Some((a, b)) = st.spp.as_mut() {
                a.upload(&nn);
                b.upload(&nn);
            }
            st.csp.upload(&nn);
        }
        self.final_layer.upload(&nn);
        for l in [&mut self.mlp, &mut self.uv, &mut self.o, &mut self.cls_x, &mut self.cls_y] {
            if l.upload(&nn).is_err() {
                return false;
            }
        }
        self.consts = Some([nn.upload(&self.gamma), nn.upload(&self.beta), nn.upload(&self.res_scale)]);
        self.gpu = Some(GpuUse::new(nn));
        true
    }

    pub fn on_gpu(&self) -> bool {
        GpuUse::active(&self.gpu)
    }

    /// The backbone and the head's 7×7 convolution: one 8 × 6 map per keypoint and crop.
    fn maps<B: Backend>(&self, be: &B, xi: &B::T) -> Result<B::T, VisionError> {
        let [a, b, cc] = &self.stem;
        let mut y = be.conv(cc, &be.conv(b, &be.conv(a, xi, None)?, None)?, None)?;
        for st in &self.stages {
            y = be.conv(&st.down, &y, None)?;
            if let Some((c1, c2)) = &st.spp {
                let z = be.conv(c1, &y, None)?;
                let (p5, p9, p13) = (be.max_pool(&z, 5, 1, 2)?, be.max_pool(&z, 9, 1, 4)?, be.max_pool(&z, 13, 1, 6)?);
                y = be.conv(c2, &be.concat(&[&z, &p5, &p9, &p13])?, None)?;
            }
            y = st.csp.forward(be, &y)?;
        }
        be.conv(&self.final_layer, &y, None)
    }

    /// The SimCC logits for `n` normalised crops `[n, 3, 256, 192]`: `[n, 17, 384]` and
    /// `[n, 17, 512]`. On the GPU when it has one, else (or when that fails) on the CPU.
    pub(crate) fn raw(&self, x: &Feat) -> Result<(Vec<f32>, Vec<f32>), VisionError> {
        if x.c != 3 || x.h != INPUT_H || x.w != INPUT_W || x.n == 0 {
            return Err(err("the pose network takes 192 × 256 RGB crops"));
        }
        if let Some(v) = GpuUse::run(&self.gpu, |r| {
            let (sx, sy) = self.head_gpu(r, &self.maps(r, &r.input(x)?)?)?;
            let mut v = r.read_mats(&[&sx, &sy])?.into_iter();
            v.next().zip(v.next()).ok_or_else(|| err("the pose head's outputs did not come back"))
        }) {
            return Ok(v);
        }
        self.raw_cpu(x)
    }

    /// [`RtmPose::raw`] on the CPU (the reference).
    pub(crate) fn raw_cpu(&self, x: &Feat) -> Result<(Vec<f32>, Vec<f32>), VisionError> {
        if x.c != 3 || x.h != INPUT_H || x.w != INPUT_W || x.n == 0 {
            return Err(err("the pose network takes 192 × 256 RGB crops"));
        }
        let f = self.maps(&Cpu, x)?;
        self.head_cpu(&f)
    }

    /// The head (on the CPU) over the keypoint maps `[n, 17, 8, 6]`: the SimCC logits.
    pub(crate) fn head_cpu(&self, f: &Feat) -> Result<(Vec<f32>, Vec<f32>), VisionError> {
        let n = f.n;
        let flat = f.h * f.w;
        if flat != self.mlp.cin || f.c != KEYPOINTS {
            return Err(err(format!("the pose head takes {} values per keypoint, got {flat}", self.mlp.cin)));
        }
        // [n·17, 48]
        let mut t = f.v.clone();
        scale_norm(&mut t, flat, self.mlp_g);
        let h = self.mlp.forward(&t)?;
        let hid = self.mlp.cout;
        let mut z = h.clone();
        scale_norm(&mut z, hid, self.gau_g);
        let mut uv = self.uv.forward(&z)?;
        for v in &mut uv {
            *v = silu(*v);
        }
        let gated = gau_cpu(&uv, &self.gamma, &self.beta, self.e, self.s, n);
        let o = self.o.forward(&gated)?;
        let k = KEYPOINTS;
        let mut feat = vec![0f32; n * k * hid];
        for ((f, hh), oo) in feat.chunks_exact_mut(hid).zip(h.chunks_exact(hid)).zip(o.chunks_exact(hid)) {
            for j in 0..hid {
                f[j] = self.res_scale[j] * hh[j] + oo[j];
            }
        }
        Ok((self.cls_x.forward(&feat)?, self.cls_y.forward(&feat)?))
    }

    /// The head on the GPU over the keypoint maps there: the SimCC logits `[n·17, 384]` and
    /// `[n·17, 512]`, not read back.
    pub(crate) fn head_gpu(&self, r: &Run<'_>, f: &GT) -> Result<(GM, GM), VisionError> {
        let [gamma, beta, res_scale] = self.consts.as_ref().ok_or_else(|| err("the pose head is not on the GPU"))?;
        let (_, c, h, w) = Run::shape(f);
        if c != KEYPOINTS || h * w != self.mlp.cin {
            return Err(err(format!("the pose head takes {} values per keypoint, got {}", self.mlp.cin, h * w)));
        }
        let rows = r.as_rows(f, h * w)?;
        let hh = r.linear(&r.scalenorm(&rows, self.mlp_g)?, &self.mlp, TAct::None, None, None)?;
        let uv = r.linear(&r.scalenorm(&hh, self.gau_g)?, &self.uv, TAct::Silu, None, None)?;
        let gated = r.gau(&uv, gamma, beta, self.e, self.s, KEYPOINTS)?;
        let o = r.linear(&gated, &self.o, TAct::None, None, None)?;
        let feat = r.scale_add(&hh, &o, res_scale)?;
        Ok((r.linear(&feat, &self.cls_x, TAct::None, None, None)?, r.linear(&feat, &self.cls_y, TAct::None, None, None)?))
    }

    /// The poses of the people in boxes `boxes` (`[x, y, w, h]`, pixels) of a frame already on
    /// the GPU, in its pixels: crops, network and SimCC maxima there, one read-back for all.
    pub(crate) fn poses_gpu(&self, r: &Run<'_>, img: &GImg, boxes: &[[f32; 4]]) -> Result<Vec<Vec<Keypoint>>, VisionError> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        let affines: Vec<Affine> = boxes.iter().map(affine_of).collect::<Result<_, _>>()?;
        let mut maxima: Vec<(GM, GM)> = Vec::new();
        for chunk in affines.chunks(MAX_BATCH) {
            let at: Vec<[f32; 3]> = chunk.iter().map(|a| [a.cx, a.cy, a.k]).collect();
            let crops = r.crops(img, &at)?;
            let (sx, sy) = self.head_gpu(r, &self.maps(r, &crops)?)?;
            maxima.push((r.argmax(&sx)?, r.argmax(&sy)?));
        }
        let mats: Vec<&GM> = maxima.iter().flat_map(|(x, y)| [x, y]).collect();
        let got = r.read_mats(&mats)?;
        let mut out = Vec::with_capacity(boxes.len());
        for (chunk, [mx, my]) in affines.chunks(MAX_BATCH).zip(got.as_chunks::<2>().0) {
            for (i, a) in chunk.iter().enumerate() {
                let kps = (0..KEYPOINTS)
                    .map(|k| {
                        let row = i * KEYPOINTS + k;
                        let at = |m: &[f32]| (m.get(row * 2).copied().unwrap_or(0.0), m.get(row * 2 + 1).copied().unwrap_or(f32::NEG_INFINITY));
                        let ((xi, xv), (yi, yv)) = (at(mx), at(my));
                        // indices come back as exact f32 integers
                        keypoint(a, xi.max(0.0) as usize, xv, yi.max(0.0) as usize, yv)
                    })
                    .collect();
                out.push(kps);
            }
        }
        if out.len() != boxes.len() {
            return Err(err("the keypoints did not come back"));
        }
        Ok(out)
    }

    /// The poses of the people in boxes `boxes` (`[x, y, w, h]`, pixels) of an RGBA picture, in
    /// the picture's pixels.
    pub fn poses(&self, rgba: &[u8], w: usize, h: usize, boxes: &[[f32; 4]]) -> Result<Vec<Vec<Keypoint>>, VisionError> {
        let mut out = Vec::with_capacity(boxes.len());
        for chunk in boxes.chunks(MAX_BATCH) {
            let crops: Vec<(Vec<f32>, Affine)> = chunk.par_iter().map(|b| crop(rgba, w, h, b)).collect::<Result<_, _>>()?;
            let n = crops.len();
            let mut v = Vec::with_capacity(n * 3 * INPUT_H * INPUT_W);
            for (c, _) in &crops {
                v.extend_from_slice(c);
            }
            let (sx, sy) = self.raw(&Feat::new(v, n, 3, INPUT_H, INPUT_W)?)?;
            let (bx, by) = (INPUT_W * 2, INPUT_H * 2);
            for (i, (_, a)) in crops.iter().enumerate() {
                let kps = (0..KEYPOINTS)
                    .map(|k| {
                        let row_x = sx.get((i * KEYPOINTS + k) * bx..(i * KEYPOINTS + k + 1) * bx).unwrap_or(&[]);
                        let row_y = sy.get((i * KEYPOINTS + k) * by..(i * KEYPOINTS + k + 1) * by).unwrap_or(&[]);
                        let (xi, xv) = argmax(row_x);
                        let (yi, yv) = argmax(row_y);
                        keypoint(a, xi, xv, yi, yv)
                    })
                    .collect();
                out.push(kps);
            }
        }
        Ok(out)
    }
}

/// A keypoint in the picture from its SimCC maxima (bins along x and y and their logits) in the
/// crop `a`: scored by the smaller logit.
fn keypoint(a: &Affine, xi: usize, xv: f32, yi: usize, yv: f32) -> Keypoint {
    let score = xv.min(yv);
    let (px, py) = (xi as f32 / SPLIT, yi as f32 / SPLIT);
    Keypoint { x: a.cx + (px - INPUT_W as f32 / 2.0) * a.k, y: a.cy + (py - INPUT_H as f32 / 2.0) * a.k, score: if score.is_finite() { score } else { 0.0 } }
}

/// RTMPose's gated attention unit on the CPU over the uv rows of `n` crops (`[n·17, 2e + s]`:
/// u, v, the shared base): per keypoint a query and key from the base (`gamma`, `beta`),
/// squared-ReLU attention, `u ⊙ (A·v)`. `[n·17, e]`.
pub(crate) fn gau_cpu(uv: &[f32], gamma: &[f32], beta: &[f32], e: usize, s: usize, n: usize) -> Vec<f32> {
    let row = 2 * e + s;
    let k = KEYPOINTS;
    let mut gated = vec![0f32; n * k * e];
    gated.par_chunks_mut((k * e).max(1)).enumerate().for_each(|(i, out)| {
        let Some(rows) = uv.get(i * k * row..(i + 1) * k * row) else { return };
        // q, k per keypoint from the shared base
        let qk: Vec<(Vec<f32>, Vec<f32>)> = rows
            .chunks_exact(row.max(1))
            .map(|r| {
                let base = r.get(2 * e..).unwrap_or(&[]);
                let q = base.iter().enumerate().map(|(j, b)| b * gamma.get(j).copied().unwrap_or(0.0) + beta.get(j).copied().unwrap_or(0.0)).collect();
                let kk = base.iter().enumerate().map(|(j, b)| b * gamma.get(s + j).copied().unwrap_or(0.0) + beta.get(s + j).copied().unwrap_or(0.0)).collect();
                (q, kk)
            })
            .collect();
        let sq = (s as f32).sqrt();
        for (a, (qa, _)) in qk.iter().enumerate() {
            let mut acc = vec![0f32; e];
            for (bi, (_, kb)) in qk.iter().enumerate() {
                let dot: f32 = qa.iter().zip(kb).map(|(x, y)| x * y).sum::<f32>() / sq;
                let w = dot.max(0.0).powi(2);
                if w == 0.0 {
                    continue;
                }
                let v = rows.get(bi * row + e..bi * row + 2 * e).unwrap_or(&[]);
                for (o, x) in acc.iter_mut().zip(v) {
                    *o += w * x;
                }
            }
            let u = rows.get(a * row..a * row + e).unwrap_or(&[]);
            if let Some(o) = out.get_mut(a * e..(a + 1) * e) {
                for ((o, u), x) in o.iter_mut().zip(u).zip(&acc) {
                    *o = u * x;
                }
            }
        }
    });
    gated
}

pub(crate) fn argmax(v: &[f32]) -> (usize, f32) {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, x)| if *x > bv { (i, *x) } else { (bi, bv) })
}

/// Where a crop came from: its centre and pixels of the picture per crop pixel.
#[derive(Clone, Copy, Debug)]
struct Affine {
    cx: f32,
    cy: f32,
    k: f32,
}

/// Where a person's box `[x, y, w, h]` is cropped from: grown by [`PADDING`], widened or
/// heightened to 3:4.
fn affine_of(b: &[f32; 4]) -> Result<Affine, VisionError> {
    if !b.iter().all(|v| v.is_finite()) || b[2] <= 0.0 || b[3] <= 0.0 {
        return Err(VisionError::Invalid("a person's box cannot be cropped".into()));
    }
    let (cx, cy) = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0);
    // the grown box widened or heightened to 3:4: its width over the crop's
    let aspect = INPUT_W as f32 / INPUT_H as f32;
    let sw = (b[2] * PADDING).max(b[3] * PADDING * aspect);
    let k = sw / INPUT_W as f32;
    if !(k.is_finite() && k > 0.0) {
        return Err(VisionError::Invalid("a person's box cannot be cropped".into()));
    }
    Ok(Affine { cx, cy, k })
}

/// A person's box as the pose network takes it: grown by [`PADDING`], widened or heightened to
/// 3:4, warped to 192 × 256 (bilinear, black outside), normalised, RGB planes.
fn crop(rgba: &[u8], w: usize, h: usize, b: &[f32; 4]) -> Result<(Vec<f32>, Affine), VisionError> {
    if w.checked_mul(h).and_then(|n| n.checked_mul(4)).is_none_or(|n| rgba.len() < n) {
        return Err(VisionError::Invalid("a person's box cannot be cropped".into()));
    }
    let Affine { cx, cy, k } = affine_of(b)?;
    let plane = INPUT_W * INPUT_H;
    let mut out = vec![0f32; 3 * plane];
    let px = |x: i64, y: i64, c: usize| -> f32 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            return 0.0;
        }
        rgba.get((y as usize * w + x as usize) * 4 + c).map_or(0.0, |v| f32::from(*v))
    };
    for v in 0..INPUT_H {
        for u in 0..INPUT_W {
            let sx = cx + (u as f32 - INPUT_W as f32 / 2.0) * k;
            let sy = cy + (v as f32 - INPUT_H as f32 / 2.0) * k;
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (xi, yi) = (x0 as i64, y0 as i64);
            for c in 0..3 {
                let val = px(xi, yi, c) * (1.0 - fx) * (1.0 - fy)
                    + px(xi + 1, yi, c) * fx * (1.0 - fy)
                    + px(xi, yi + 1, c) * (1.0 - fx) * fy
                    + px(xi + 1, yi + 1, c) * fx * fy;
                out[c * plane + v * INPUT_W + u] = (val - MEAN[c]) / STD[c];
            }
        }
    }
    Ok((out, Affine { cx, cy, k }))
}

/// The CPU's crop of a person's box and where it came from (`[cx, cy, k]`), for the GPU's tests.
#[cfg(test)]
pub(crate) fn crop_for_test(rgba: &[u8], w: usize, h: usize, b: &[f32; 4]) -> Result<(Vec<f32>, [f32; 3]), VisionError> {
    let (c, a) = crop(rgba, w, h, b)?;
    Ok((c, [a.cx, a.cy, a.k]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crops_are_three_by_four_round_the_box() {
        let (w, h) = (64usize, 48usize);
        // a white square at (20..30, 10..30) on black
        let mut rgba = vec![0u8; w * h * 4];
        for y in 10..30 {
            for x in 20..30 {
                rgba[(y * w + x) * 4..(y * w + x) * 4 + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        let (c, a) = crop(&rgba, w, h, &[20.0, 10.0, 10.0, 20.0]).unwrap();
        // 20 px tall · 1.25 = 25 px over 256: a tall box keeps its height
        assert!((a.k - 25.0 / 256.0).abs() < 1e-6);
        assert_eq!((a.cx, a.cy), (25.0, 20.0));
        // the centre is white, the corners black (normalised)
        let white = (255.0 - MEAN[0]) / STD[0];
        let black = -MEAN[0] / STD[0];
        assert!((c[128 * INPUT_W + 96] - white).abs() < 1e-3);
        assert!((c[0] - black).abs() < 1e-3);
        assert!(crop(&rgba, w, h, &[0.0, 0.0, f32::NAN, 1.0]).is_err());
        assert!(crop(&rgba, w, h, &[0.0, 0.0, 0.0, 1.0]).is_err());
        assert!(crop(&rgba[..10], w, h, &[0.0, 0.0, 5.0, 5.0]).is_err());
        // far outside the picture: black, no panic
        assert!(crop(&rgba, w, h, &[1e6, -1e6, 5.0, 5.0]).is_ok());
    }

    #[test]
    fn a_file_of_another_network_is_refused() {
        assert!(RtmPose::from_model(&Model::default()).is_err());
        assert_eq!(argmax(&[0.1, 0.7, 0.3]), (1, 0.7));
        assert_eq!(argmax(&[]).0, 0);
    }
}
