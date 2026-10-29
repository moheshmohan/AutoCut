//! MI-GAN (Sargsyan et al., Picsart AI Research, ICCV 2023; MIT, [`crate::catalog::INPAINTER`]):
//! a hole in a picture filled with what could be there, for object removal.
//!
//! The network as its authors export it to ONNX (512 × 512, trained on Places2): an encoder from
//! 512 to 4 pixels and a synthesis back, StyleGAN2-like, every layer a depthwise 3 × 3 and a
//! pointwise convolution — written out here, its weights and fixed noise read from the file
//! (`crate::onnx`, which runs nothing). candle runs a grouped convolution as one convolution a
//! group (512 for a depthwise layer), so the layers are plain loops on the CPU (the pointwise
//! ones as candle matrix products) and WGSL passes on the GPU (`inpaint_gpu`, `migan.wgsl`).
//!
//! - **In.** The picture as `2x/255 − 1` times the mask (1 kept, 0 the hole), beside the mask
//!   minus ½: four channels.
//! - **Activation.** Leaky ReLU (0.2) times √2, clamped to ±256.
//! - **Encoder block** (`b512` … `b8`): `fromrgb` (the first only); depthwise 3 × 3 + bias, act,
//!   pointwise, act — kept as the block's feature —; depthwise 3 × 3 + bias, act, a 4 × 4
//!   depthwise filter down by 2, pointwise to the next width, act. `b4` has no filter down.
//! - **Synthesis block** (`b4`: the encoder's two convolution pairs, its feature added after the
//!   first): depthwise 3 × 3 + bias, act, pointwise, up by 2 (zeros between the pixels, padded 2
//!   and 1, a 4 × 4 depthwise filter), + noise, act, + the encoder's feature; depthwise 3 × 3 +
//!   bias, act, pointwise, + noise, act. Its picture is the last one's filtered up by 2 plus
//!   `torgb` of its features.
//! - **Out.** `(x + 1) / 2` of the last picture, in the hole; the picture's own pixels elsewhere.
//!
//! [`Inpainter::inpaint`] fills a hole in a picture of any size: a square round the hole (three
//! times its size, at least 512 wide where the picture allows) scaled to 512, filled, scaled back
//! and blended in by the hole's coverage.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::onnx::{Model, Node};
use crate::{VisionError, catalog};

/// The network's side.
pub const SIDE: usize = 512;
/// The resolutions of the blocks, largest first.
const RES: [usize; 8] = [512, 256, 128, 64, 32, 16, 8, 4];
const GAIN: f32 = std::f32::consts::SQRT_2;
const CLAMP: f32 = 256.0;
/// The square cut round a hole: this many times the hole's size, at least [`SIDE`].
const CONTEXT: f32 = 3.0;

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("inpainter: {e}"))
}

/// A depthwise convolution: `[c][k][k]` and its bias.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dw {
    pub c: usize,
    pub k: usize,
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}

/// A pointwise convolution: `[cout][cin]` and its bias.
#[derive(Clone, Debug, Default)]
pub(crate) struct Pw {
    pub cin: usize,
    pub cout: usize,
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}

/// One encoder block.
#[derive(Clone, Debug, Default)]
pub(crate) struct Enc {
    pub fromrgb: Option<Pw>,
    pub dw1: Dw,
    pub pw1: Pw,
    pub dw2: Dw,
    pub down: Option<Dw>,
    pub pw2: Pw,
}

/// One synthesis block (`up`, `up_rgb` and the noise absent at 4 pixels).
#[derive(Clone, Debug, Default)]
pub(crate) struct Syn {
    pub dw1: Dw,
    pub pw1: Pw,
    pub up: Option<Dw>,
    pub noise1: Option<Vec<f32>>,
    pub dw2: Dw,
    pub pw2: Pw,
    pub noise2: Option<Vec<f32>>,
    pub up_rgb: Option<Dw>,
    pub torgb: Pw,
}

/// The network's weights, by resolution (largest first for the encoder, smallest for the
/// synthesis), and the value the zero-stuffing keeps (the filters' gain is in them).
#[derive(Clone, Debug, Default)]
pub(crate) struct Weights {
    pub enc: Vec<Enc>,
    pub syn: Vec<Syn>,
    pub stuff: f32,
}

fn tensor(m: &Model, name: &str, dims: &[usize]) -> Result<Vec<f32>, VisionError> {
    let t = m.initializers.get(name).ok_or_else(|| err(format!("no `{name}`")))?;
    if t.dims != dims {
        return Err(err(format!("`{name}` is {:?}, not {dims:?}", t.dims)));
    }
    t.f32s().map(<[f32]>::to_vec).ok_or_else(|| err(format!("`{name}` is not float")))
}

fn dims(m: &Model, name: &str) -> Option<Vec<usize>> {
    m.initializers.get(name).map(|t| t.dims.clone())
}

fn dw(m: &Model, name: &str, bias: bool) -> Result<Dw, VisionError> {
    let d = dims(m, &format!("{name}.weight")).ok_or_else(|| err(format!("no `{name}`")))?;
    let [c, 1, k, k2] = d[..] else { return Err(err(format!("`{name}` is {d:?}"))) };
    if k != k2 || !(k == 3 || k == 4) {
        return Err(err(format!("`{name}` is {d:?}")));
    }
    let w = tensor(m, &format!("{name}.weight"), &[c, 1, k, k])?;
    let b = if bias { Some(tensor(m, &format!("{name}.bias"), &[c])?) } else { None };
    Ok(Dw { c, k, w, b })
}

fn pw(m: &Model, name: &str, bias: bool) -> Result<Pw, VisionError> {
    let d = dims(m, &format!("{name}.weight")).ok_or_else(|| err(format!("no `{name}`")))?;
    let [cout, cin, 1, 1] = d[..] else { return Err(err(format!("`{name}` is {d:?}"))) };
    let w = tensor(m, &format!("{name}.weight"), &[cout, cin, 1, 1])?;
    let b = if bias { Some(tensor(m, &format!("{name}.bias"), &[cout])?) } else { None };
    Ok(Pw { cin, cout, w, b })
}

/// The fixed noise added after the convolution whose weight is `weight`: the other input of the
/// `Add` taking that convolution's output, `[side][side]`.
fn noise(m: &Model, weight: &str, side: usize) -> Result<Vec<f32>, VisionError> {
    let conv: &Node =
        m.nodes.iter().find(|n| n.op == "Conv" && n.inputs.iter().any(|i| i == weight)).ok_or_else(|| err(format!("no convolution of `{weight}`")))?;
    let out = conv.outputs.first().ok_or_else(|| err("a convolution without an output"))?;
    let add = m.nodes.iter().find(|n| n.op == "Add" && n.inputs.iter().any(|i| i == out)).ok_or_else(|| err(format!("no noise after `{weight}`")))?;
    let name = add.inputs.iter().find(|i| *i != out).ok_or_else(|| err("an Add of one input"))?;
    tensor(m, name, &[side, side])
}

impl Weights {
    pub(crate) fn read(m: &Model) -> Result<Self, VisionError> {
        let mut enc = Vec::with_capacity(RES.len());
        for (i, r) in RES.iter().enumerate() {
            let p = format!("model.encoder.b{r}");
            let last = i + 1 == RES.len();
            enc.push(Enc {
                fromrgb: if i == 0 { Some(pw(m, &format!("{p}.fromrgb"), true)?) } else { None },
                dw1: dw(m, &format!("{p}.conv1.conv1"), true)?,
                pw1: pw(m, &format!("{p}.conv1.conv2"), false)?,
                dw2: dw(m, &format!("{p}.conv2.conv1"), true)?,
                down: if last { None } else { Some(dw(m, &format!("{p}.conv2.downsample.filter"), false)?) },
                pw2: pw(m, &format!("{p}.conv2.conv2"), false)?,
            });
        }
        let mut syn = Vec::with_capacity(RES.len());
        let mut stuff = None;
        for r in RES.iter().rev() {
            let p = format!("model.synthesis.b{r}");
            let first = *r == 4;
            let s = Syn {
                dw1: dw(m, &format!("{p}.conv1.conv1"), true)?,
                pw1: pw(m, &format!("{p}.conv1.conv2"), false)?,
                up: if first { None } else { Some(dw(m, &format!("{p}.conv1.upsample.filter"), false)?) },
                noise1: if first { None } else { Some(noise(m, &format!("{p}.conv1.upsample.filter.weight"), *r)?) },
                dw2: dw(m, &format!("{p}.conv2.conv1"), true)?,
                pw2: pw(m, &format!("{p}.conv2.conv2"), false)?,
                noise2: if first { None } else { Some(noise(m, &format!("{p}.conv2.conv2.weight"), *r)?) },
                up_rgb: if first { None } else { Some(dw(m, &format!("{p}.upsample.filter"), false)?) },
                torgb: pw(m, &format!("{p}.torgb"), true)?,
            };
            if !first {
                // the zero-stuffing: a mask keeping one pixel of each 2 × 2
                let k = tensor(m, &format!("{p}.conv1.upsample.filter_const"), &[1, 1, *r, *r])?;
                let keep = k.first().copied().unwrap_or(0.0);
                let ok = k.iter().enumerate().all(|(i, v)| if (i / r) % 2 == 0 && (i % r) % 2 == 0 { *v == keep } else { *v == 0.0 });
                if !ok || stuff.is_some_and(|s| s != keep) {
                    return Err(err(format!("b{r}'s upsampling is not zeros between pixels")));
                }
                stuff = Some(keep);
            }
            syn.push(s);
        }
        let w = Self { enc, syn, stuff: stuff.unwrap_or(1.0) };
        w.check()?;
        Ok(w)
    }

    /// The widths chain from block to block.
    fn check(&self) -> Result<(), VisionError> {
        let mut c = 4usize;
        for e in &self.enc {
            if let Some(f) = &e.fromrgb {
                if f.cin != c {
                    return Err(err("fromrgb's width"));
                }
                c = f.cout;
            }
            let ok = e.dw1.c == c && e.pw1.cin == c && e.dw2.c == e.pw1.cout && e.down.as_ref().is_none_or(|d| d.c == e.pw1.cout) && e.pw2.cin == e.pw1.cout;
            if !ok {
                return Err(err("an encoder block's widths"));
            }
            c = e.pw2.cout;
        }
        for (s, e) in self.syn.iter().zip(self.enc.iter().rev()) {
            let ok = s.dw1.c == c
                && s.pw1.cin == c
                && s.up.as_ref().is_none_or(|u| u.c == s.pw1.cout)
                && s.pw1.cout == e.pw1.cout
                && s.dw2.c == s.pw1.cout
                && s.pw2.cin == s.pw1.cout
                && s.torgb.cin == s.pw2.cout
                && s.torgb.cout == 3
                && s.up_rgb.as_ref().is_none_or(|u| u.c == 3);
            if !ok {
                return Err(err("a synthesis block's widths"));
            }
            c = s.pw2.cout;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// the CPU reference
// ---------------------------------------------------------------------------------------------

/// Leaky ReLU times √2, clamped.
#[inline]
pub(crate) fn act(v: f32) -> f32 {
    (if v >= 0.0 { v } else { 0.2 * v } * GAIN).clamp(-CLAMP, CLAMP)
}

/// What follows a convolution: noise added (`[h][w]`), the activation, a map added after.
#[derive(Clone, Copy, Default)]
struct Then<'a> {
    noise: Option<&'a [f32]>,
    act: bool,
    add: Option<&'a [f32]>,
}

impl Then<'_> {
    #[inline]
    fn apply(&self, v: f32, ch: usize, p: usize, hw: usize) -> f32 {
        let mut v = v;
        if let Some(n) = self.noise {
            v += n.get(p).copied().unwrap_or(0.0);
        }
        if self.act {
            v = act(v);
        }
        if let Some(a) = self.add {
            v += a.get(ch * hw + p).copied().unwrap_or(0.0);
        }
        v
    }
}

/// How a depthwise convolution reads its input: plainly (`stride`, padded `pad` before), or
/// up by 2 (zeros between the pixels, `stuff` kept, padded 2 before).
#[derive(Clone, Copy)]
enum Read {
    Plain { stride: usize, pad: usize },
    Up { stuff: f32 },
}

/// A depthwise convolution of `x` (`[c][h][w]`) to `oh` × `ow`.
fn depthwise(x: &[f32], h: usize, w: usize, d: &Dw, read: Read, oh: usize, ow: usize, then: Then) -> Vec<f32> {
    let k = d.k;
    let mut out = vec![0f32; d.c * oh * ow];
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(ch, o)| {
        let src = x.get(ch * h * w..(ch + 1) * h * w).unwrap_or_default();
        let wt = d.w.get(ch * k * k..(ch + 1) * k * k).unwrap_or_default();
        let bias = d.b.as_ref().and_then(|b| b.get(ch)).copied().unwrap_or(0.0);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut s = bias;
                for ky in 0..k {
                    for kx in 0..k {
                        let v = match read {
                            Read::Plain { stride, pad } => {
                                let (iy, ix) = ((oy * stride + ky) as i64 - pad as i64, (ox * stride + kx) as i64 - pad as i64);
                                if iy < 0 || ix < 0 || iy >= h as i64 || ix >= w as i64 {
                                    0.0
                                } else {
                                    src.get(iy as usize * w + ix as usize).copied().unwrap_or(0.0)
                                }
                            }
                            Read::Up { stuff } => {
                                // the stuffed picture is 2h × 2w, padded 2 before
                                let (iy, ix) = ((oy + ky) as i64 - 2, (ox + kx) as i64 - 2);
                                if iy < 0 || ix < 0 || iy % 2 != 0 || ix % 2 != 0 || iy / 2 >= h as i64 || ix / 2 >= w as i64 {
                                    0.0
                                } else {
                                    stuff * src.get((iy / 2) as usize * w + (ix / 2) as usize).copied().unwrap_or(0.0)
                                }
                            }
                        };
                        s += v * wt.get(ky * k + kx).copied().unwrap_or(0.0);
                    }
                }
                let p = oy * ow + ox;
                if let Some(slot) = o.get_mut(p) {
                    *slot = then.apply(s, ch, p, oh * ow);
                }
            }
        }
    });
    out
}

/// A pointwise convolution of `x` (`[cin][hw]`): a matrix product on candle.
fn pointwise(x: &[f32], hw: usize, p: &Pw, then: Then) -> Result<Vec<f32>, VisionError> {
    use candle_core::{Device, Tensor};
    let dev = Device::Cpu;
    let a = Tensor::from_slice(&p.w, (p.cout, p.cin), &dev).map_err(err)?;
    let b = Tensor::from_slice(x.get(..p.cin * hw).ok_or_else(|| err("a short map"))?, (p.cin, hw), &dev).map_err(err)?;
    let mut out: Vec<f32> = a.matmul(&b).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1()).map_err(err)?;
    out.par_chunks_mut(hw).enumerate().for_each(|(co, o)| {
        let bias = p.b.as_ref().and_then(|b| b.get(co)).copied().unwrap_or(0.0);
        for (i, v) in o.iter_mut().enumerate() {
            *v = then.apply(*v + bias, co, i, hw);
        }
    });
    Ok(out)
}

/// The network on the CPU: four channels in (`[4][512][512]`), the picture out (`[3][512][512]`,
/// about −1 … 1).
pub(crate) fn forward(w: &Weights, input: &[f32]) -> Result<Vec<f32>, VisionError> {
    if input.len() != 4 * SIDE * SIDE {
        return Err(err("the input is not 4 × 512 × 512"));
    }
    let act_ = Then { act: true, ..Default::default() };
    let mut x = input.to_vec();
    let mut feats: Vec<Vec<f32>> = Vec::with_capacity(RES.len());
    for (e, r) in w.enc.iter().zip(RES) {
        let hw = r * r;
        if let Some(f) = &e.fromrgb {
            x = pointwise(&x, hw, f, act_)?;
        }
        x = depthwise(&x, r, r, &e.dw1, Read::Plain { stride: 1, pad: 1 }, r, r, act_);
        x = pointwise(&x, hw, &e.pw1, act_)?;
        feats.push(x.clone());
        x = depthwise(&x, r, r, &e.dw2, Read::Plain { stride: 1, pad: 1 }, r, r, act_);
        let r2 = match &e.down {
            Some(d) => {
                x = depthwise(&x, r, r, d, Read::Plain { stride: 2, pad: 1 }, r / 2, r / 2, Then::default());
                r / 2
            }
            None => r,
        };
        x = pointwise(&x, r2 * r2, &e.pw2, act_)?;
    }
    let mut img: Vec<f32> = Vec::new();
    let mut r = 4usize;
    for (s, feat) in w.syn.iter().zip(feats.iter().rev()) {
        let hw0 = r * r;
        x = depthwise(&x, r, r, &s.dw1, Read::Plain { stride: 1, pad: 1 }, r, r, act_);
        match (&s.up, &s.noise1) {
            (Some(up), Some(n)) => {
                x = pointwise(&x, hw0, &s.pw1, Then::default())?;
                let r2 = r * 2;
                x = depthwise(&x, r, r, up, Read::Up { stuff: w.stuff }, r2, r2, Then { noise: Some(n), act: true, add: Some(feat) });
                r = r2;
            }
            _ => x = pointwise(&x, hw0, &s.pw1, Then { act: true, add: Some(feat), ..Default::default() })?,
        }
        let hw = r * r;
        x = depthwise(&x, r, r, &s.dw2, Read::Plain { stride: 1, pad: 1 }, r, r, act_);
        x = pointwise(&x, hw, &s.pw2, Then { noise: s.noise2.as_deref(), act: true, add: None })?;
        let up_img = match &s.up_rgb {
            Some(u) if !img.is_empty() => Some(depthwise(&img, r / 2, r / 2, u, Read::Up { stuff: w.stuff }, r, r, Then::default())),
            _ => None,
        };
        img = pointwise(&x, hw, &s.torgb, Then { add: up_img.as_deref(), ..Default::default() })?;
    }
    Ok(img)
}

/// The network's four input channels for a 512 × 512 RGBA picture and what to keep of it
/// (`[512 × 512]`, 1 kept, 0 the hole).
pub(crate) fn input(rgba: &[u8], keep: &[f32]) -> Vec<f32> {
    let hw = SIDE * SIDE;
    let mut x = vec![0f32; 4 * hw];
    for p in 0..hw {
        let m = keep.get(p).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        x[p] = m - 0.5;
        for c in 0..3 {
            let v = rgba.get(p * 4 + c).copied().map_or(0.0, f32::from);
            x[(c + 1) * hw + p] = (v * 2.0 / 255.0 - 1.0) * m;
        }
    }
    x
}

// ---------------------------------------------------------------------------------------------
// the inpainter
// ---------------------------------------------------------------------------------------------

/// MI-GAN, loaded: on the GPU when there is one, on the CPU otherwise and after a GPU failure.
pub struct Inpainter {
    w: Weights,
    gpu: Option<crate::inpaint_gpu::GpuNet>,
    gpu_failed: AtomicBool,
}

impl Inpainter {
    /// Load MI-GAN from the models directory (on the GPU when `gpu` and there is one).
    pub fn load(models_dir: &Path, gpu: bool) -> Result<Self, VisionError> {
        if !catalog::inpainter_installed(models_dir) {
            return Err(VisionError::NotInstalled);
        }
        let path = catalog::inpaint_dir(models_dir).join(catalog::INPAINTER.name);
        let m = crate::onnx::read(&path, &crate::onnx::Limits::default())?;
        let w = Weights::read(&m)?;
        let gpu = if gpu { crate::inpaint_gpu::GpuNet::new(&w) } else { None };
        Ok(Self { w, gpu, gpu_failed: AtomicBool::new(false) })
    }

    /// Where it runs: `"gpu"` or `"cpu"`.
    pub fn device_name(&self) -> &'static str {
        if self.gpu.is_some() && !self.gpu_failed.load(Ordering::Relaxed) { "gpu" } else { "cpu" }
    }

    /// The network's picture (`[3][512][512]`, about −1 … 1) for a 512 × 512 RGBA picture and
    /// what to keep of it (1 kept, 0 the hole).
    fn net(&self, rgba: &[u8], keep: &[f32]) -> Result<Vec<f32>, VisionError> {
        let x = input(rgba, keep);
        if let Some(g) = &self.gpu
            && !self.gpu_failed.load(Ordering::Relaxed)
        {
            match g.run(&x) {
                Ok(v) => return Ok(v),
                Err(_) => self.gpu_failed.store(true, Ordering::Relaxed),
            }
        }
        forward(&self.w, &x)
    }

    /// The hole of an RGBA picture (`w` × `h`) filled: `hole` its coverage a pixel (0 kept … 255
    /// filled). Pixels outside the hole are the picture's own.
    pub fn inpaint(&self, rgba: &[u8], w: usize, h: usize, hole: &[u8]) -> Result<Vec<u8>, VisionError> {
        let n = w.checked_mul(h).ok_or_else(|| err("a picture too large"))?;
        if w == 0 || h == 0 || rgba.len() < n * 4 || hole.len() < n {
            return Err(VisionError::Invalid(format!("{} bytes and a {}-pixel hole for a {w} × {h} picture", rgba.len(), hole.len())));
        }
        let Some((x0, y0, x1, y1)) = bounds(hole, w, h) else { return Ok(rgba.get(..n * 4).unwrap_or_default().to_vec()) };
        let (cx, cy, cw, ch) = square(x0, y0, x1, y1, w, h);
        // the square at 512, the hole a little larger (it must cover what it scales from)
        let crop = |src: &[u8], bpp: usize| -> Vec<u8> {
            let mut out = vec![0u8; cw * ch * bpp];
            for y in 0..ch {
                let from = ((cy + y) * w + cx) * bpp;
                if let (Some(s), Some(d)) = (src.get(from..from + cw * bpp), out.get_mut(y * cw * bpp..(y + 1) * cw * bpp)) {
                    d.copy_from_slice(s);
                }
            }
            out
        };
        let pic = area(&crop(rgba, 4), cw, ch, 4, SIDE, SIDE);
        let mut keep = vec![1f32; SIDE * SIDE];
        let hc = crop(hole, 1);
        for y in 0..ch {
            for x in 0..cw {
                if hc.get(y * cw + x).copied().unwrap_or(0) > 8 {
                    // every 512 pixel this one reaches, grown by one
                    let (sx0, sy0) = ((x * SIDE / cw).saturating_sub(1), (y * SIDE / ch).saturating_sub(1));
                    let (sx1, sy1) = (((x + 1) * SIDE).div_ceil(cw).min(SIDE - 1) + 1, ((y + 1) * SIDE).div_ceil(ch).min(SIDE - 1) + 1);
                    for sy in sy0..sy1.min(SIDE) {
                        for sx in sx0..sx1.min(SIDE) {
                            keep[sy * SIDE + sx] = 0.0;
                        }
                    }
                }
            }
        }
        let made = self.net(&pic, &keep)?;
        // back to the square's size (bilinear), blended in by the hole's coverage
        let hw = SIDE * SIDE;
        let mut out = rgba.get(..n * 4).unwrap_or_default().to_vec();
        let (kx, ky) = (SIDE as f32 / cw as f32, SIDE as f32 / ch as f32);
        for y in 0..ch {
            for x in 0..cw {
                let a = f32::from(hc.get(y * cw + x).copied().unwrap_or(0)) / 255.0;
                if a <= 0.0 {
                    continue;
                }
                let (sx, sy) = (((x as f32 + 0.5) * kx - 0.5).clamp(0.0, (SIDE - 1) as f32), ((y as f32 + 0.5) * ky - 0.5).clamp(0.0, (SIDE - 1) as f32));
                let (ix, iy) = (sx as usize, sy as usize);
                let (fx, fy) = (sx - ix as f32, sy - iy as f32);
                let (ix1, iy1) = ((ix + 1).min(SIDE - 1), (iy + 1).min(SIDE - 1));
                let o = ((cy + y) * w + cx + x) * 4;
                for c in 0..3 {
                    let at = |yy: usize, xx: usize| made.get(c * hw + yy * SIDE + xx).copied().unwrap_or(0.0);
                    let v = at(iy, ix) * (1.0 - fx) * (1.0 - fy) + at(iy, ix1) * fx * (1.0 - fy) + at(iy1, ix) * (1.0 - fx) * fy + at(iy1, ix1) * fx * fy;
                    let v = ((v + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0);
                    if let Some(p) = out.get_mut(o + c) {
                        *p = (f32::from(*p) * (1.0 - a) + v * a).round() as u8;
                    }
                }
            }
        }
        Ok(out)
    }
}

/// The box of a hole's pixels (coverage above 8): `(x0, y0, x1, y1)`, exclusive ends.
fn bounds(hole: &[u8], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if hole.get(y * w + x).copied().unwrap_or(0) > 8 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// The square cut round a hole's box: [`CONTEXT`] times its larger side, at least [`SIDE`],
/// centred on it and kept inside the picture (a rectangle where the picture is narrower).
fn square(x0: usize, y0: usize, x1: usize, y1: usize, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let side = (((x1 - x0).max(y1 - y0) as f32 * CONTEXT) as usize).max(SIDE);
    let (sw, sh) = (side.min(w).max(x1 - x0), side.min(h).max(y1 - y0));
    let place = |lo: usize, hi: usize, len: usize, total: usize| -> usize {
        let mid = (lo + hi) / 2;
        mid.saturating_sub(len / 2).min(total.saturating_sub(len))
    };
    (place(x0, x1, sw, w), place(y0, y1, sh, h), sw.min(w), sh.min(h))
}

/// A picture (`bpp` bytes a pixel, `w` × `h`) averaged over the areas of a `tw` × `th` one (or
/// sampled bilinearly where it grows).
fn area(src: &[u8], w: usize, h: usize, bpp: usize, tw: usize, th: usize) -> Vec<u8> {
    let mut out = vec![0u8; tw * th * bpp];
    out.par_chunks_mut(tw * bpp).enumerate().for_each(|(ty, row)| {
        let (fy0, fy1) = (ty as f32 * h as f32 / th as f32, (ty + 1) as f32 * h as f32 / th as f32);
        for tx in 0..tw {
            let (fx0, fx1) = (tx as f32 * w as f32 / tw as f32, (tx + 1) as f32 * w as f32 / tw as f32);
            let (y0, y1) = (fy0 as usize, (fy1.ceil() as usize).clamp(fy0 as usize + 1, h));
            let (x0, x1) = (fx0 as usize, (fx1.ceil() as usize).clamp(fx0 as usize + 1, w));
            for c in 0..bpp {
                let (mut s, mut n) = (0f32, 0f32);
                for y in y0..y1 {
                    for x in x0..x1 {
                        s += src.get((y * w + x) * bpp + c).copied().map_or(0.0, f32::from);
                        n += 1.0;
                    }
                }
                if let Some(p) = row.get_mut(tx * bpp + c) {
                    *p = (s / n.max(1.0)).round() as u8;
                }
            }
        }
    });
    out
}

#[cfg(test)]
#[path = "inpaint_tests.rs"]
mod tests;
