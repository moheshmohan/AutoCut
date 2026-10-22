//! The building blocks of the person detector (D-FINE, `dfine.rs`) and the pose network
//! (RTMPose, `rtmpose.rs`), on the CPU: candle matrix products and convolutions for the
//! arithmetic, plain Rust (in parallel) for the depthwise convolutions, biases, activations,
//! normalisations and the rest. Feature maps are [`Feat`]s (`n` pictures of `c` planes,
//! row-major); token sequences are row-major `[tokens, channels]` slices.

use candle_core::{Device, Tensor};
use rayon::prelude::*;

use crate::VisionError;

pub(crate) fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(e.to_string())
}

/// `n` pictures of `c` planes of `h` × `w`, row-major.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Feat {
    pub(crate) v: Vec<f32>,
    pub(crate) n: usize,
    pub(crate) c: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
}

impl Feat {
    pub(crate) fn new(v: Vec<f32>, n: usize, c: usize, h: usize, w: usize) -> Result<Self, VisionError> {
        let len = n.checked_mul(c).and_then(|x| x.checked_mul(h)).and_then(|x| x.checked_mul(w)).ok_or_else(|| err("feature map too large"))?;
        if v.len() != len {
            return Err(err(format!("{} values for {n}×{c}×{h}×{w}", v.len())));
        }
        Ok(Self { v, n, c, h, w })
    }

    pub(crate) fn zeros(n: usize, c: usize, h: usize, w: usize) -> Result<Self, VisionError> {
        let len = n.checked_mul(c).and_then(|x| x.checked_mul(h)).and_then(|x| x.checked_mul(w)).ok_or_else(|| err("feature map too large"))?;
        Self::new(vec![0.0; len], n, c, h, w)
    }

    fn plane(&self) -> usize {
        self.h * self.w
    }

    /// Picture `i`'s planes.
    pub(crate) fn image(&self, i: usize) -> &[f32] {
        let s = self.c * self.plane();
        self.v.get(i * s..(i + 1) * s).unwrap_or(&[])
    }

    /// The tokens of picture `i`: `[h·w, c]`.
    pub(crate) fn tokens(&self, i: usize) -> Vec<f32> {
        let (hw, c) = (self.plane(), self.c);
        let src = self.image(i);
        let mut out = vec![0f32; hw * c];
        for (ch, p) in src.chunks_exact(hw.max(1)).enumerate() {
            for (j, v) in p.iter().enumerate() {
                if let Some(o) = out.get_mut(j * c + ch) {
                    *o = *v;
                }
            }
        }
        out
    }

    /// One picture from tokens `[h·w, c]`.
    pub(crate) fn from_tokens(t: &[f32], c: usize, h: usize, w: usize) -> Result<Self, VisionError> {
        let hw = h * w;
        if t.len() != hw * c {
            return Err(err("token count"));
        }
        let mut v = vec![0f32; hw * c];
        for (j, row) in t.chunks_exact(c.max(1)).enumerate() {
            for (ch, x) in row.iter().enumerate() {
                if let Some(o) = v.get_mut(ch * hw + j) {
                    *o = *x;
                }
            }
        }
        Self::new(v, 1, c, h, w)
    }

    /// The pictures' planes side by side: `[a's channels, b's channels]`.
    pub(crate) fn concat(parts: &[&Feat]) -> Result<Feat, VisionError> {
        let Some(first) = parts.first() else { return Err(err("nothing to join")) };
        let (n, h, w) = (first.n, first.h, first.w);
        if parts.iter().any(|p| p.n != n || p.h != h || p.w != w) {
            return Err(err(format!("feature maps of different sizes do not join ({:?})", parts.iter().map(|p| (p.n, p.h, p.w)).collect::<Vec<_>>())));
        }
        let c: usize = parts.iter().map(|p| p.c).sum();
        let mut v = Vec::with_capacity(n * c * h * w);
        for i in 0..n {
            for p in parts {
                v.extend_from_slice(p.image(i));
            }
        }
        Feat::new(v, n, c, h, w)
    }

    /// Channels `from..to` of every picture.
    pub(crate) fn channels(&self, from: usize, to: usize) -> Result<Feat, VisionError> {
        if from > to || to > self.c {
            return Err(err("channel range"));
        }
        let hw = self.plane();
        let mut v = Vec::with_capacity(self.n * (to - from) * hw);
        for i in 0..self.n {
            let img = self.image(i);
            v.extend_from_slice(img.get(from * hw..to * hw).ok_or_else(|| err("channel range"))?);
        }
        Feat::new(v, self.n, to - from, self.h, self.w)
    }

    /// Element-wise sum with `o` (same shape).
    pub(crate) fn add(&self, o: &Feat) -> Result<Feat, VisionError> {
        if (self.n, self.c, self.h, self.w) != (o.n, o.c, o.h, o.w) {
            return Err(err("sum of differently shaped maps"));
        }
        let mut v = self.v.clone();
        v.par_iter_mut().zip(o.v.par_iter()).for_each(|(a, b)| *a += b);
        Feat::new(v, self.n, self.c, self.h, self.w)
    }

    /// Zeros added on the right and at the bottom.
    pub(crate) fn pad_right_bottom(&self, by: usize) -> Result<Feat, VisionError> {
        let (h2, w2) = (self.h + by, self.w + by);
        let mut out = Feat::zeros(self.n, self.c, h2, w2)?;
        let (hw, hw2) = (self.plane(), h2 * w2);
        out.v.par_chunks_mut(hw2.max(1)).zip(self.v.par_chunks(hw.max(1))).for_each(|(o, s)| {
            for (y, row) in s.chunks_exact(self.w.max(1)).enumerate() {
                if let Some(d) = o.get_mut(y * w2..y * w2 + self.w) {
                    d.copy_from_slice(row);
                }
            }
        });
        Ok(out)
    }

    /// Nearest-neighbour 2× upsampling.
    pub(crate) fn upsample2(&self) -> Result<Feat, VisionError> {
        let (h2, w2) = (self.h * 2, self.w * 2);
        let mut out = Feat::zeros(self.n, self.c, h2, w2)?;
        let w = self.w;
        out.v.par_chunks_mut((h2 * w2).max(1)).zip(self.v.par_chunks(self.plane().max(1))).for_each(|(o, s)| {
            for (y, row) in o.chunks_exact_mut(w2.max(1)).enumerate() {
                let Some(src) = s.get((y / 2) * w..(y / 2) * w + w) else { continue };
                for (x, v) in row.iter_mut().enumerate() {
                    *v = src.get(x / 2).copied().unwrap_or(0.0);
                }
            }
        });
        Ok(out)
    }

    /// `k`×`k` max pooling with stride `s`, `pad` (−∞) on every side, rounding down.
    pub(crate) fn max_pool(&self, k: usize, s: usize, pad: usize) -> Result<Feat, VisionError> {
        let (h, w) = (self.h, self.w);
        let s = s.max(1);
        let (Some(oh), Some(ow)) = ((h + 2 * pad).checked_sub(k), (w + 2 * pad).checked_sub(k)) else { return Err(err("pooling window larger than the map")) };
        let (oh, ow) = (oh / s + 1, ow / s + 1);
        let mut out = Feat::zeros(self.n, self.c, oh, ow)?;
        out.v.par_chunks_mut((oh * ow).max(1)).zip(self.v.par_chunks((h * w).max(1))).for_each(|(o, src)| {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut m = f32::NEG_INFINITY;
                    for ky in 0..k {
                        let Some(iy) = (oy * s + ky).checked_sub(pad).filter(|y| *y < h) else { continue };
                        for kx in 0..k {
                            let Some(ix) = (ox * s + kx).checked_sub(pad).filter(|x| *x < w) else { continue };
                            if let Some(v) = src.get(iy * w + ix) {
                                m = m.max(*v);
                            }
                        }
                    }
                    if let Some(d) = o.get_mut(oy * ow + ox) {
                        *d = m;
                    }
                }
            }
        });
        Ok(out)
    }

    fn tensor(&self) -> Result<Tensor, VisionError> {
        Tensor::from_slice(&self.v, (self.n, self.c, self.h, self.w), &Device::Cpu).map_err(err)
    }
}

/// An activation (after the bias).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Act {
    None,
    Relu,
    /// ReLU, then a learnable `scale · x + bias` (HGNetv2's learnable affine block).
    ReluAffine(f32, f32),
    Silu,
}

impl Act {
    #[inline]
    pub(crate) fn apply(self, x: f32) -> f32 {
        match self {
            Act::None => x,
            Act::Relu => x.max(0.0),
            Act::ReluAffine(s, b) => s * x.max(0.0) + b,
            Act::Silu => x / (1.0 + (-x).exp()),
        }
    }
}

/// How a convolution is computed.
enum ConvKind {
    /// A 1×1 convolution with stride 1: a matrix product.
    Pointwise(Tensor),
    /// One filter per channel, in plain Rust.
    Depthwise,
    /// Anything else: candle's convolution.
    Dense(Tensor),
}

/// A 2-D convolution with its bias (batch norm folded in) and activation.
pub(crate) struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    pub(crate) cin: usize,
    pub(crate) cout: usize,
    pub(crate) k: usize,
    pub(crate) stride: usize,
    pub(crate) pad: usize,
    kind: ConvKind,
    pub(crate) act: Act,
    /// The weights on the GPU (`gpunn`).
    pub(crate) dev: Option<crate::gpunn::ConvW>,
}

impl Conv {
    /// `w`: `[cout, cin / groups, k, k]`; groups 1, or `cin` (= `cout`, depthwise).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        w: Vec<f32>,
        b: Vec<f32>,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        act: Act,
    ) -> Result<Self, VisionError> {
        if cin == 0 || cout == 0 || k == 0 || stride == 0 || k > 15 || pad > k {
            return Err(err(format!("a {k}×{k}/{stride} convolution of {cin} → {cout} channels cannot be built")));
        }
        let depthwise = groups == cin && groups == cout && groups > 1;
        if !(groups == 1 || depthwise) {
            return Err(err(format!("convolutions in {groups} groups are not supported")));
        }
        let per = if depthwise { 1 } else { cin };
        if w.len() != cout * per * k * k || b.len() != cout {
            return Err(err(format!("a convolution's weights hold {} / {} values, expected {} / {cout}", w.len(), b.len(), cout * per * k * k)));
        }
        if w.iter().chain(&b).any(|v| !v.is_finite()) {
            return Err(err("a convolution's weights are not finite"));
        }
        let kind = if depthwise {
            ConvKind::Depthwise
        } else if k == 1 && stride == 1 && pad == 0 {
            ConvKind::Pointwise(Tensor::from_slice(&w, (cout, cin), &Device::Cpu).map_err(err)?)
        } else {
            ConvKind::Dense(Tensor::from_slice(&w, (cout, cin, k, k), &Device::Cpu).map_err(err)?)
        };
        Ok(Self { w, b, cin, cout, k, stride, pad, kind, act, dev: None })
    }

    pub(crate) fn is_depthwise(&self) -> bool {
        matches!(self.kind, ConvKind::Depthwise)
    }

    /// The weights `[cout, cin / groups, k, k]` and the bias.
    pub(crate) fn weights(&self) -> (Vec<f32>, Vec<f32>) {
        (self.w.clone(), self.b.clone())
    }

    /// The output size for an `h` × `w` input.
    pub(crate) fn out_size(&self, h: usize, w: usize) -> Result<(usize, usize), VisionError> {
        let (Some(a), Some(b)) = ((h + 2 * self.pad).checked_sub(self.k), (w + 2 * self.pad).checked_sub(self.k)) else {
            return Err(err(format!("a {}×{} map is smaller than a {}×{} kernel", w, h, self.k, self.k)));
        };
        Ok((a / self.stride + 1, b / self.stride + 1))
    }

    pub(crate) fn forward(&self, x: &Feat) -> Result<Feat, VisionError> {
        if x.c != self.cin {
            return Err(err(format!("a convolution takes {} channels, got {}", self.cin, x.c)));
        }
        let (oh, ow) = self.out_size(x.h, x.w)?;
        let mut y = match &self.kind {
            ConvKind::Depthwise => self.depthwise(x, oh, ow)?,
            ConvKind::Pointwise(wt) => {
                let hw = x.h * x.w;
                let mut v = Vec::with_capacity(x.n * self.cout * hw);
                for i in 0..x.n {
                    let xt = Tensor::from_slice(x.image(i), (self.cin, hw), &Device::Cpu).map_err(err)?;
                    v.extend(wt.matmul(&xt).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?);
                }
                Feat::new(v, x.n, self.cout, oh, ow)?
            }
            ConvKind::Dense(wt) => {
                let y = x.tensor()?.conv2d(wt, self.pad, self.stride, 1, 1).map_err(err)?;
                let v: Vec<f32> = y.flatten_all().map_err(err)?.to_vec1().map_err(err)?;
                Feat::new(v, x.n, self.cout, oh, ow)?
            }
        };
        // the bias and the activation (the depthwise pass added its own bias)
        let depthwise = matches!(self.kind, ConvKind::Depthwise);
        let plane = (oh * ow).max(1);
        let (cout, act) = (self.cout, self.act);
        y.v.par_chunks_mut(plane).enumerate().for_each(|(j, p)| {
            let b = if depthwise { 0.0 } else { self.b.get(j % cout).copied().unwrap_or(0.0) };
            for v in p {
                *v = act.apply(*v + b);
            }
        });
        Ok(y)
    }

    fn depthwise(&self, x: &Feat, oh: usize, ow: usize) -> Result<Feat, VisionError> {
        let (h, w, k, s, p) = (x.h, x.w, self.k, self.stride, self.pad);
        let mut out = Feat::zeros(x.n, self.cout, oh, ow)?;
        let c = self.cout;
        out.v.par_chunks_mut((oh * ow).max(1)).zip(x.v.par_chunks((h * w).max(1))).enumerate().for_each(|(j, (o, src))| {
            let ch = j % c;
            let (Some(kk), Some(&b)) = (self.w.get(ch * k * k..(ch + 1) * k * k), self.b.get(ch)) else { return };
            for oy in 0..oh {
                let Some(orow) = o.get_mut(oy * ow..(oy + 1) * ow) else { continue };
                orow.fill(b);
                for ky in 0..k {
                    let Some(iy) = (oy * s + ky).checked_sub(p).filter(|y| *y < h) else { continue };
                    let Some(row) = src.get(iy * w..(iy + 1) * w) else { continue };
                    for kx in 0..k {
                        let wv = kk[ky * k + kx];
                        // the outputs whose input column ox·s + kx − p is inside the row
                        let lo = if kx >= p { 0 } else { (p - kx).div_ceil(s) };
                        let hi = if w + p > kx { ((w + p - kx - 1) / s + 1).min(ow) } else { 0 };
                        for ox in lo..hi {
                            if let (Some(d), Some(v)) = (orow.get_mut(ox), row.get(ox * s + kx - p)) {
                                *d += wv * v;
                            }
                        }
                    }
                }
            }
        });
        Ok(out)
    }
}

/// The operations the convolutional parts are made of: on the CPU ([`Cpu`], the reference) or
/// as WGSL passes on the GPU (`gpunn::Run`). The networks are written once over it.
pub(crate) trait Backend {
    /// A tensor (`n` pictures of `c` planes).
    type T;
    fn input(&self, x: &Feat) -> Result<Self::T, VisionError>;
    /// The convolution with its activation, then `res` added.
    fn conv(&self, c: &Conv, x: &Self::T, res: Option<&Self::T>) -> Result<Self::T, VisionError>;
    fn concat(&self, parts: &[&Self::T]) -> Result<Self::T, VisionError>;
    fn channels(&self, x: &Self::T, from: usize, to: usize) -> Result<Self::T, VisionError>;
    /// One row and column of zeros added at the bottom and on the right.
    fn pad_rb(&self, x: &Self::T) -> Result<Self::T, VisionError>;
    /// `k`×`k` max pooling, stride `s`, `pad` of −∞ on every side.
    fn max_pool(&self, x: &Self::T, k: usize, s: usize, pad: usize) -> Result<Self::T, VisionError>;
    fn upsample2(&self, x: &Self::T) -> Result<Self::T, VisionError>;
    /// Channel attention: every plane times the hard sigmoid of the 1×1 convolution `fc` of the
    /// planes' means.
    fn chan_attn(&self, x: &Self::T, fc: &Conv) -> Result<Self::T, VisionError>;
    /// `(n, c, h, w)`.
    fn shape(t: &Self::T) -> (usize, usize, usize, usize);
    /// The tensors on the CPU (the GPU's recorded work is submitted and waited for here).
    fn read(&self, xs: Vec<Self::T>) -> Result<Vec<Feat>, VisionError>;
}

/// The plain-Rust passes (the reference).
pub(crate) struct Cpu;

impl Backend for Cpu {
    type T = Feat;
    fn input(&self, x: &Feat) -> Result<Feat, VisionError> {
        Ok(x.clone())
    }
    fn conv(&self, c: &Conv, x: &Feat, res: Option<&Feat>) -> Result<Feat, VisionError> {
        let y = c.forward(x)?;
        match res {
            Some(r) => y.add(r),
            None => Ok(y),
        }
    }
    fn concat(&self, parts: &[&Feat]) -> Result<Feat, VisionError> {
        Feat::concat(parts)
    }
    fn channels(&self, x: &Feat, from: usize, to: usize) -> Result<Feat, VisionError> {
        x.channels(from, to)
    }
    fn pad_rb(&self, x: &Feat) -> Result<Feat, VisionError> {
        x.pad_right_bottom(1)
    }
    fn max_pool(&self, x: &Feat, k: usize, s: usize, pad: usize) -> Result<Feat, VisionError> {
        x.max_pool(k, s, pad)
    }
    fn upsample2(&self, x: &Feat) -> Result<Feat, VisionError> {
        x.upsample2()
    }
    fn chan_attn(&self, x: &Feat, fc: &Conv) -> Result<Feat, VisionError> {
        if fc.cin != x.c || fc.cout != x.c || fc.k != 1 {
            return Err(err("channel attention shapes"));
        }
        let (c, hw) = (x.c, (x.h * x.w).max(1));
        let means: Vec<f32> = x.v.chunks_exact(hw).map(|p| p.iter().sum::<f32>() / hw as f32).collect();
        let mut scales = vec![0f32; means.len()];
        for (i, s) in scales.iter_mut().enumerate() {
            let (n, o) = (i / c, i % c);
            let row = fc.w.get(o * c..(o + 1) * c).unwrap_or(&[]);
            let img = means.get(n * c..(n + 1) * c).unwrap_or(&[]);
            let z: f32 = row.iter().zip(img).map(|(a, b)| a * b).sum::<f32>() + fc.b.get(o).copied().unwrap_or(0.0);
            *s = (z / 6.0 + 0.5).clamp(0.0, 1.0);
        }
        let mut y = x.clone();
        y.v.par_chunks_mut(hw).zip(scales.par_iter()).for_each(|(p, s)| {
            for v in p {
                *v *= s;
            }
        });
        Ok(y)
    }
    fn shape(t: &Feat) -> (usize, usize, usize, usize) {
        (t.n, t.c, t.h, t.w)
    }
    fn read(&self, xs: Vec<Feat>) -> Result<Vec<Feat>, VisionError> {
        Ok(xs)
    }
}

/// A batch norm (`gamma`, `beta`, running mean and variance) folded into a convolution's weights
/// `w` (`[cout, …]`) and bias `b` (none: zeros). Returns the new bias.
pub(crate) fn fold_bn(w: &mut [f32], b: Option<&[f32]>, gamma: &[f32], beta: &[f32], mean: &[f32], var: &[f32], eps: f32) -> Result<Vec<f32>, VisionError> {
    let cout = gamma.len();
    if cout == 0 || beta.len() != cout || mean.len() != cout || var.len() != cout || b.is_some_and(|b| b.len() != cout) || !w.len().is_multiple_of(cout) {
        return Err(err("a batch norm does not fit its convolution"));
    }
    let per = w.len() / cout;
    let mut bias = Vec::with_capacity(cout);
    for (o, chunk) in w.chunks_exact_mut(per.max(1)).enumerate() {
        let s = gamma[o] / (var[o] + eps).max(1e-12).sqrt();
        for v in chunk.iter_mut() {
            *v *= s;
        }
        let b0 = b.map_or(0.0, |b| b[o]);
        bias.push((b0 - mean[o]) * s + beta[o]);
    }
    if bias.iter().chain(w.iter()).any(|v| !v.is_finite()) {
        return Err(err("a folded batch norm is not finite"));
    }
    Ok(bias)
}

/// A linear layer: `y = x Wᵀ + b`.
pub(crate) struct Linear {
    /// `[in, out]`.
    wt: Tensor,
    b: Option<Vec<f32>>,
    pub(crate) cin: usize,
    pub(crate) cout: usize,
}

impl Linear {
    /// `w`: `[out, in]` (PyTorch's layout).
    pub(crate) fn new(w: &[f32], b: Option<Vec<f32>>, cin: usize, cout: usize) -> Result<Self, VisionError> {
        if w.len() != cin * cout || b.as_ref().is_some_and(|b| b.len() != cout) || cin == 0 || cout == 0 {
            return Err(err(format!("a linear layer's weights do not fit {cin} → {cout}")));
        }
        if w.iter().chain(b.iter().flatten()).any(|v| !v.is_finite()) {
            return Err(err("a linear layer's weights are not finite"));
        }
        let wt = Tensor::from_slice(w, (cout, cin), &Device::Cpu).map_err(err)?.t().map_err(err)?.contiguous().map_err(err)?;
        Ok(Self { wt, b, cin, cout })
    }

    /// `w` already `[in, out]` (ONNX MatMul weights).
    pub(crate) fn from_in_out(w: &[f32], cin: usize, cout: usize) -> Result<Self, VisionError> {
        if w.len() != cin * cout || cin == 0 || cout == 0 || w.iter().any(|v| !v.is_finite()) {
            return Err(err(format!("a linear layer's weights do not fit {cin} → {cout}")));
        }
        Ok(Self { wt: Tensor::from_slice(w, (cin, cout), &Device::Cpu).map_err(err)?, b: None, cin, cout })
    }

    /// `x`: `[rows, in]` → `[rows, out]`.
    pub(crate) fn forward(&self, x: &[f32]) -> Result<Vec<f32>, VisionError> {
        if !x.len().is_multiple_of(self.cin) {
            return Err(err(format!("a linear layer takes rows of {}, got {} values", self.cin, x.len())));
        }
        let rows = x.len() / self.cin;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let xt = Tensor::from_slice(x, (rows, self.cin), &Device::Cpu).map_err(err)?;
        let mut y: Vec<f32> = xt.matmul(&self.wt).map_err(err)?.flatten_all().map_err(err)?.to_vec1().map_err(err)?;
        if let Some(b) = &self.b {
            y.par_chunks_mut(self.cout).for_each(|r| {
                for (v, b) in r.iter_mut().zip(b) {
                    *v += b;
                }
            });
        }
        Ok(y)
    }
}

/// Layer normalisation over the last dimension (`c`), in place.
pub(crate) struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl Norm {
    pub(crate) fn new(g: Vec<f32>, b: Vec<f32>, eps: f32) -> Result<Self, VisionError> {
        if g.is_empty() || g.len() != b.len() || g.iter().chain(&b).any(|v| !v.is_finite()) {
            return Err(err("a layer norm's weights do not fit"));
        }
        Ok(Self { g, b, eps })
    }

    pub(crate) fn apply(&self, x: &mut [f32]) {
        let c = self.g.len();
        x.par_chunks_mut(c).for_each(|r| {
            let n = r.len() as f32;
            let mean = r.iter().sum::<f32>() / n;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = 1.0 / (var + self.eps).sqrt();
            for ((v, g), b) in r.iter_mut().zip(&self.g).zip(&self.b) {
                *v = (*v - mean) * inv * g + b;
            }
        });
    }
}

/// Softmax over each row of `cols` values, in place.
pub(crate) fn softmax_rows(x: &mut [f32], cols: usize) {
    if cols == 0 {
        return;
    }
    x.par_chunks_mut(cols).for_each(|r| {
        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in r.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        if s > 0.0 {
            for v in r.iter_mut() {
                *v /= s;
            }
        }
    });
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Multi-head attention (PyTorch's): queries and keys from `qk`, values from `v` (`[tokens, d]`
/// both), `heads` heads, scaled dot products.
pub(crate) struct Attention {
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) o: Linear,
    pub(crate) heads: usize,
}

impl Attention {
    pub(crate) fn forward(&self, qk: &[f32], v: &[f32]) -> Result<Vec<f32>, VisionError> {
        let d = self.q.cout;
        let heads = self.heads.max(1);
        if !d.is_multiple_of(heads) || qk.len() != v.len() {
            return Err(err("attention shapes"));
        }
        let n = qk.len() / self.q.cin.max(1);
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut q = self.q.forward(qk)?;
        for x in &mut q {
            *x *= scale;
        }
        let k = self.k.forward(qk)?;
        let vv = self.v.forward(v)?;
        let dev = Device::Cpu;
        let split = |x: Vec<f32>| -> Result<Tensor, VisionError> {
            Tensor::from_vec(x, (n, heads, hd), &dev).map_err(err)?.transpose(0, 1).map_err(err)?.contiguous().map_err(err)
        };
        let (qt, kt, vt) = (split(q)?, split(k)?, split(vv)?);
        let kt = kt.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?;
        let mut s: Vec<f32> = qt.matmul(&kt).map_err(err)?.flatten_all().map_err(err)?.to_vec1().map_err(err)?;
        softmax_rows(&mut s, n);
        let p = Tensor::from_vec(s, (heads, n, n), &dev).map_err(err)?;
        let o = p.matmul(&vt).map_err(err)?.transpose(0, 1).map_err(err)?.contiguous().map_err(err)?;
        let o: Vec<f32> = o.flatten_all().map_err(err)?.to_vec1().map_err(err)?;
        self.o.forward(&o)
    }
}

/// A small MLP: linear layers with ReLU between them.
pub(crate) struct Mlp {
    pub(crate) layers: Vec<Linear>,
}

impl Mlp {
    pub(crate) fn forward(&self, x: &[f32]) -> Result<Vec<f32>, VisionError> {
        let mut y = x.to_vec();
        let last = self.layers.len().saturating_sub(1);
        for (i, l) in self.layers.iter().enumerate() {
            y = l.forward(&y)?;
            if i < last {
                for v in &mut y {
                    *v = v.max(0.0);
                }
            }
        }
        Ok(y)
    }
}

/// Bilinear sampling of a `[h, w]` grid of `c`-vectors (`values[(y·w + x)·stride + off ..][..c]`)
/// at pixel position `(x, y)` (pixel centres at integers), zero outside; the result is added to
/// `out` times `weight`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn sample_add(values: &[f32], stride: usize, off: usize, c: usize, h: usize, w: usize, x: f32, y: f32, weight: f32, out: &mut [f32]) {
    if !(x.is_finite() && y.is_finite()) || weight == 0.0 {
        return;
    }
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let corners = [(x0, y0, (1.0 - fx) * (1.0 - fy)), (x0 + 1.0, y0, fx * (1.0 - fy)), (x0, y0 + 1.0, (1.0 - fx) * fy), (x0 + 1.0, y0 + 1.0, fx * fy)];
    for (cx, cy, cw) in corners {
        if cx < 0.0 || cy < 0.0 || cx >= w as f32 || cy >= h as f32 || cw == 0.0 {
            continue;
        }
        let at = (cy as usize * w + cx as usize) * stride + off;
        let Some(src) = values.get(at..at + c) else { continue };
        let k = cw * weight;
        for (o, v) in out.iter_mut().zip(src) {
            *o += k * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A direct convolution, for checking the fast paths.
    fn direct(x: &Feat, w: &[f32], b: &[f32], cout: usize, k: usize, s: usize, p: usize, groups: usize) -> Vec<f32> {
        let (oh, ow) = ((x.h + 2 * p - k) / s + 1, (x.w + 2 * p - k) / s + 1);
        let per = x.c / groups;
        let mut out = vec![0f32; x.n * cout * oh * ow];
        for n in 0..x.n {
            for o in 0..cout {
                let g = o / (cout / groups);
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = b[o];
                        for ci in 0..per {
                            let c = g * per + ci;
                            for ky in 0..k {
                                for kx in 0..k {
                                    let (iy, ix) = ((oy * s + ky) as i64 - p as i64, (ox * s + kx) as i64 - p as i64);
                                    if iy >= 0 && ix >= 0 && (iy as usize) < x.h && (ix as usize) < x.w {
                                        acc += w[((o * per + ci) * k + ky) * k + kx] * x.v[((n * x.c + c) * x.h + iy as usize) * x.w + ix as usize];
                                    }
                                }
                            }
                        }
                        out[((n * cout + o) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
        out
    }

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        (0..n).map(|i| (((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 7) % 1000) as f32 / 500.0 - 1.0).collect()
    }

    #[test]
    fn convolutions_match_a_direct_one() {
        let x = Feat::new(pseudo(2 * 6 * 9 * 11, 1), 2, 6, 9, 11).unwrap();
        for (cout, k, s, p, groups) in [(4, 1, 1, 0, 1), (5, 3, 1, 1, 1), (3, 3, 2, 1, 1), (6, 3, 2, 1, 6), (6, 5, 1, 2, 6), (6, 3, 1, 1, 6), (2, 2, 1, 0, 1)] {
            let per = 6 / groups;
            let w = pseudo(cout * per * k * k, 7);
            let b = pseudo(cout, 3);
            let conv = Conv::new(w.clone(), b.clone(), 6, cout, k, s, p, groups, Act::None).unwrap();
            let got = conv.forward(&x).unwrap();
            let want = direct(&x, &w, &b, cout, k, s, p, groups);
            assert_eq!(got.v.len(), want.len(), "{cout} {k} {s} {p} {groups}");
            let e = got.v.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(e < 1e-4, "{cout} {k} {s} {p} {groups}: {e}");
        }
        // hostile shapes are errors
        assert!(Conv::new(vec![0.0; 4], vec![0.0], 1, 1, 2, 0, 0, 1, Act::None).is_err());
        assert!(Conv::new(vec![0.0; 3], vec![0.0], 1, 1, 2, 1, 0, 1, Act::None).is_err());
        assert!(Conv::new(vec![f32::NAN; 4], vec![0.0], 1, 1, 2, 1, 0, 1, Act::None).is_err());
        let c = Conv::new(vec![0.0; 9], vec![0.0], 1, 1, 3, 1, 0, 1, Act::None).unwrap();
        assert!(c.forward(&Feat::zeros(1, 1, 2, 2).unwrap()).is_err());
        assert!(c.forward(&Feat::zeros(1, 2, 5, 5).unwrap()).is_err());
    }

    #[test]
    fn batch_norm_folding_is_the_same_as_normalising() {
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        let b = fold_bn(&mut w, Some(&[0.5, -0.5]), &[2.0, 1.0], &[0.1, 0.2], &[1.0, 0.0], &[3.0, 0.0], 1.0).unwrap();
        // channel 0: (x·w + 0.5 − 1) / 2 · 2 + 0.1
        assert!((w[0] - 1.0).abs() < 1e-6 && (b[0] - (-0.5 + 0.1)).abs() < 1e-6);
        assert!((w[2] - 3.0).abs() < 1e-6 && (b[1] - (-0.5 + 0.2)).abs() < 1e-6);
        assert!(fold_bn(&mut w, None, &[1.0], &[0.0, 0.0], &[0.0], &[1.0], 1e-5).is_err());
    }

    #[test]
    fn maps_join_split_pool_and_upsample() {
        let a = Feat::new((0..8).map(|v| v as f32).collect(), 1, 2, 2, 2).unwrap();
        let b = Feat::new(vec![9.0; 4], 1, 1, 2, 2).unwrap();
        let j = Feat::concat(&[&a, &b]).unwrap();
        assert_eq!(j.c, 3);
        assert_eq!(j.channels(2, 3).unwrap().v, vec![9.0; 4]);
        assert!(j.channels(2, 4).is_err());
        let u = a.upsample2().unwrap();
        assert_eq!((u.h, u.w), (4, 4));
        assert_eq!(&u.v[..4], &[0.0, 0.0, 1.0, 1.0]);
        let p = a.max_pool(3, 1, 1).unwrap();
        assert_eq!(p.v[..4], [3.0, 3.0, 3.0, 3.0]);
        let pr = a.pad_right_bottom(1).unwrap();
        assert_eq!((pr.h, pr.w), (3, 3));
        assert_eq!(&pr.v[..3], &[0.0, 1.0, 0.0]);
        let t = a.tokens(0);
        assert_eq!(&t[..4], &[0.0, 4.0, 1.0, 5.0]);
        assert_eq!(Feat::from_tokens(&t, 2, 2, 2).unwrap(), a);
        assert!(Feat::concat(&[&a, &Feat::zeros(1, 1, 3, 3).unwrap()]).is_err());
    }

    #[test]
    fn attention_and_sampling() {
        let ident = |n: usize| -> Vec<f32> { (0..n * n).map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.0 }).collect() };
        let lin = || Linear::new(&ident(4), Some(vec![0.0; 4]), 4, 4).unwrap();
        let att = Attention { q: lin(), k: lin(), v: lin(), o: lin(), heads: 2 };
        // two identical tokens attend equally: the output is their mean
        let x = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let y = att.forward(&x, &[1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((y[0] - 2.0).abs() < 1e-5 && (y[4] - 2.0).abs() < 1e-5);
        // bilinear sampling between four values, zero outside
        let vals = vec![0.0, 10.0, 20.0, 30.0];
        let mut out = [0.0f32];
        sample_add(&vals, 1, 0, 1, 2, 2, 0.5, 0.5, 1.0, &mut out);
        assert!((out[0] - 15.0).abs() < 1e-5);
        let mut out = [0.0f32];
        sample_add(&vals, 1, 0, 1, 2, 2, -0.5, 0.0, 1.0, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-5);
        sample_add(&vals, 1, 0, 1, 2, 2, f32::NAN, 0.0, 1.0, &mut out);
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut x, 3);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
