//! Depth Anything V2 Small (Apache-2.0, [`crate::catalog::DEPTH`]): how near each part of a
//! picture is, from the picture alone — a DINOv2 ViT-S/14 encoder and a DPT head (Yang et al.
//! 2024, arXiv 2406.09414) written out on candle from the published architecture, on the GPU
//! (Metal) on macOS, the CPU elsewhere and as the fallback.
//!
//! A picture is fitted with its short side at [`SIDE`] (518, the training size) and its long
//! side the nearest multiple of 14 (at most [`MAX_ASPECT`] times the short: a wider picture is
//! squeezed), area-averaged, normalised as ImageNet's. The encoder's learned positions (a 37 × 37
//! grid) are resized to the picture's grid bicubically as PyTorch does (A = −0.75, edges clamped,
//! the grid's size plus 0.1 as the scale); the head's resizes are bilinear with the corners
//! aligned. The answer — inverse depth up to an unknown scale and shift, larger nearer — is made
//! a [`DepthMap`]: 0 the farthest, 1 the nearest, by its 1st and 99th percentiles.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm, Linear, VarBuilder};

use crate::{VisionError, catalog};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("depth: {e}"))
}

/// The short side of the picture the network sees.
pub const SIDE: usize = 518;
/// The long side at most this many times the short one.
pub const MAX_ASPECT: usize = 2;
const PATCH: usize = 14;
/// The learned positions' grid (518 / 14).
const GRID: usize = 37;
const DIM: usize = 384;
const HEADS: usize = 6;
const LAYERS: usize = 12;
/// The encoder layers the head reads.
const TAKE: [usize; 4] = [2, 5, 8, 11];
const FEATURES: usize = 64;
const OUT: [usize; 4] = [48, 96, 192, 384];
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// The share of a map left out at each end when it is made 0..1.
const TAIL: f32 = 0.01;

/// How near each part of a picture is: `near` row by row, 0 the farthest and 1 the nearest. It
/// covers the whole picture it was made from (at its own size, [`DepthMap::w`] × [`DepthMap::h`]).
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    pub w: usize,
    pub h: usize,
    pub near: Vec<f32>,
}

impl DepthMap {
    /// The nearness at `(u, v)`, shares of the picture (0..1, corners aligned), bilinear; 0 for
    /// an empty map.
    pub fn at(&self, u: f32, v: f32) -> f32 {
        if self.w == 0 || self.h == 0 || self.near.len() < self.w * self.h {
            return 0.0;
        }
        let x = (if u.is_finite() { u.clamp(0.0, 1.0) } else { 0.5 }) * (self.w - 1) as f32;
        let y = (if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 }) * (self.h - 1) as f32;
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |x: usize, y: usize| self.near.get(y * self.w + x).copied().unwrap_or(0.0);
        let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * fx;
        let bottom = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * fx;
        top + (bottom - top) * fy
    }
}

/// The size the network sees a `w` × `h` picture at: the short side [`SIDE`], the long side the
/// nearest multiple of 14 to the picture's shape, at most [`MAX_ASPECT`] times the short.
pub fn fitted_size(w: usize, h: usize) -> (usize, usize) {
    let (short, long) = (w.min(h).max(1) as f64, w.max(h).max(1) as f64);
    let cells = ((long / short * SIDE as f64 / PATCH as f64).round() as usize).clamp(GRID, GRID * MAX_ASPECT);
    let l = cells * PATCH;
    if w >= h { (l, SIDE) } else { (SIDE, l) }
}

/// An RGBA picture fitted to `(fw, fh)`, planar RGB normalised: area-averaged where it shrinks,
/// bilinear where it grows.
fn prepare(rgba: &[u8], w: usize, h: usize, (fw, fh): (usize, usize)) -> Vec<f32> {
    let mut px = vec![0f32; 3 * fw * fh];
    let get = |x: usize, y: usize, c: usize| rgba.get((y.min(h - 1) * w + x.min(w - 1)) * 4 + c).map_or(0.0, |v| f32::from(*v) / 255.0);
    let (sx, sy) = (w as f32 / fw as f32, h as f32 / fh as f32);
    for y in 0..fh {
        for x in 0..fw {
            let mut acc = [0f32; 3];
            if sx > 1.0 || sy > 1.0 {
                // the source pixels under this one
                let (x0, x1) = ((x as f32 * sx) as usize, (((x + 1) as f32 * sx).ceil() as usize).clamp(1, w));
                let (y0, y1) = ((y as f32 * sy) as usize, (((y + 1) as f32 * sy).ceil() as usize).clamp(1, h));
                let (x0, y0) = (x0.min(x1 - 1), y0.min(y1 - 1));
                let n = ((x1 - x0) * (y1 - y0)) as f32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        for (c, a) in acc.iter_mut().enumerate() {
                            *a += get(xx, yy, c);
                        }
                    }
                }
                acc.iter_mut().for_each(|a| *a /= n.max(1.0));
            } else {
                let (u, v) = (((x as f32 + 0.5) * sx - 0.5).max(0.0), ((y as f32 + 0.5) * sy - 0.5).max(0.0));
                let (x0, y0) = (u.floor() as usize, v.floor() as usize);
                let (fx, fy) = (u - x0 as f32, v - y0 as f32);
                for (c, a) in acc.iter_mut().enumerate() {
                    let top = get(x0, y0, c) * (1.0 - fx) + get(x0 + 1, y0, c) * fx;
                    let bottom = get(x0, y0 + 1, c) * (1.0 - fx) + get(x0 + 1, y0 + 1, c) * fx;
                    *a = top * (1.0 - fy) + bottom * fy;
                }
            }
            for c in 0..3 {
                px[c * fw * fh + y * fw + x] = (acc[c] - MEAN[c]) / STD[c];
            }
        }
    }
    px
}

/// PyTorch's bicubic weights (A = −0.75) at fraction `t`.
fn cubic(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let near = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let far = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

/// One axis of PyTorch's bicubic resize with a scale factor (`F.interpolate(…, scale_factor,
/// mode="bicubic", align_corners=False)`): for each output index, the four input indices
/// (clamped) and their weights.
fn cubic_taps(n_in: usize, n_out: usize, scale: f64) -> Vec<([usize; 4], [f32; 4])> {
    (0..n_out)
        .map(|o| {
            let src = ((o as f64 + 0.5) / scale - 0.5) as f32;
            let i = src.floor();
            let w = cubic(src - i);
            let i = i as isize;
            let idx = [i - 1, i, i + 1, i + 2].map(|k| k.clamp(0, n_in as isize - 1) as usize);
            (idx, w)
        })
        .collect()
}

/// The learned positions (`GRID²` × `DIM`, row by row) resized to a `gh` × `gw` grid, as
/// DINOv2's `interpolate_pos_encoding` does (the sizes plus 0.1 over the grid as the scales).
fn resize_positions(pos: &[f32], gh: usize, gw: usize) -> Vec<f32> {
    let ty = cubic_taps(GRID, gh, (gh as f64 + 0.1) / GRID as f64);
    let tx = cubic_taps(GRID, gw, (gw as f64 + 0.1) / GRID as f64);
    let at = |y: usize, x: usize, c: usize| pos.get((y * GRID + x) * DIM + c).copied().unwrap_or(0.0);
    // along x, then along y
    let mut rows = vec![0f32; GRID * gw * DIM];
    for y in 0..GRID {
        for (x, (idx, w)) in tx.iter().enumerate() {
            for c in 0..DIM {
                rows[(y * gw + x) * DIM + c] = (0..4).map(|k| w[k] * at(y, idx[k], c)).sum();
            }
        }
    }
    let mut out = vec![0f32; gh * gw * DIM];
    for (y, (idx, w)) in ty.iter().enumerate() {
        for x in 0..gw {
            for c in 0..DIM {
                out[(y * gw + x) * DIM + c] = (0..4).map(|k| w[k] * rows.get((idx[k] * gw + x) * DIM + c).copied().unwrap_or(0.0)).sum();
            }
        }
    }
    out
}

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ls1: Tensor,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ls2: Tensor,
}

impl Block {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        let ln = |name: &str| candle_nn::layer_norm(DIM, 1e-6, vb.pp(name));
        Ok(Self {
            norm1: ln("norm1")?,
            qkv: candle_nn::linear(DIM, 3 * DIM, vb.pp("attn.qkv"))?,
            proj: candle_nn::linear(DIM, DIM, vb.pp("attn.proj"))?,
            ls1: vb.get(DIM, "ls1.gamma")?,
            norm2: ln("norm2")?,
            fc1: candle_nn::linear(DIM, 4 * DIM, vb.pp("mlp.fc1"))?,
            fc2: candle_nn::linear(4 * DIM, DIM, vb.pp("mlp.fc2"))?,
            ls2: vb.get(DIM, "ls2.gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let hd = c / HEADS;
        let qkv = self.qkv.forward(&self.norm1.forward(x)?)?.reshape((b, n, 3, HEADS, hd))?.permute((2, 0, 3, 1, 4))?;
        let q = (qkv.i(0)?.contiguous()? * (1.0 / (hd as f64).sqrt()))?;
        let (k, v) = (qkv.i(1)?.contiguous()?, qkv.i(2)?.contiguous()?);
        let att = candle_nn::ops::softmax_last_dim(&q.matmul(&k.t()?)?)?;
        let o = att.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        let x = (x + self.proj.forward(&o)?.broadcast_mul(&self.ls1)?)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        x + m.broadcast_mul(&self.ls2)?
    }
}

/// A residual unit of the head: ReLU, 3 × 3, ReLU, 3 × 3, plus its input.
struct Residual {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl Residual {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: candle_nn::conv2d(FEATURES, FEATURES, 3, cfg, vb.pp("conv1"))?,
            conv2: candle_nn::conv2d(FEATURES, FEATURES, 3, cfg, vb.pp("conv2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.conv2.forward(&self.conv1.forward(&x.relu()?)?.relu()?)? + x
    }
}

/// A fusion block: the coarser path plus this level's features, refined, resized, projected.
struct Fusion {
    unit1: Residual,
    unit2: Residual,
    out: Conv2d,
}

impl Fusion {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            unit1: Residual::new(vb.pp("resConfUnit1"))?,
            unit2: Residual::new(vb.pp("resConfUnit2"))?,
            out: candle_nn::conv2d(FEATURES, FEATURES, 1, Default::default(), vb.pp("out_conv"))?,
        })
    }
    fn forward(&self, x: &Tensor, skip: Option<&Tensor>, (h, w): (usize, usize)) -> candle_core::Result<Tensor> {
        let x = match skip {
            Some(s) => (x + self.unit1.forward(s)?)?,
            None => x.clone(),
        };
        self.out.forward(&self.unit2.forward(&x)?.upsample_bilinear2d(h, w, true)?)
    }
}

/// The whole network on one device.
struct Net {
    patch: Conv2d,
    cls: Tensor,
    /// The learned positions on the CPU: the class token's, then the grid's.
    pos: Vec<f32>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    projects: Vec<Conv2d>,
    up4: ConvTranspose2d,
    up2: ConvTranspose2d,
    down2: Conv2d,
    rn: Vec<Conv2d>,
    fusion: Vec<Fusion>,
    out1: Conv2d,
    out2a: Conv2d,
    out2b: Conv2d,
    dev: Device,
    /// The positions resized for the last grid used (most pictures share their shape).
    sized: Mutex<Option<((usize, usize), Tensor)>>,
}

impl Net {
    fn load(path: &Path, dev: &Device) -> Result<Self, VisionError> {
        let vb = VarBuilder::from_pth(path, DType::F32, dev).map_err(err)?;
        let build = || -> candle_core::Result<Self> {
            let e = vb.pp("pretrained");
            let patch = candle_nn::conv2d(3, DIM, PATCH, Conv2dConfig { stride: PATCH, ..Default::default() }, e.pp("patch_embed.proj"))?;
            let cls = e.get((1, 1, DIM), "cls_token")?;
            let pos = e.get((1, GRID * GRID + 1, DIM), "pos_embed")?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let blocks = (0..LAYERS).map(|i| Block::new(e.pp(format!("blocks.{i}")))).collect::<candle_core::Result<Vec<_>>>()?;
            let norm = candle_nn::layer_norm(DIM, 1e-6, e.pp("norm"))?;
            let d = vb.pp("depth_head");
            let projects = OUT
                .iter()
                .enumerate()
                .map(|(i, &o)| candle_nn::conv2d(DIM, o, 1, Default::default(), d.pp(format!("projects.{i}"))))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let tr = |k: usize| ConvTranspose2dConfig { stride: k, ..Default::default() };
            let up4 = candle_nn::conv_transpose2d(OUT[0], OUT[0], 4, tr(4), d.pp("resize_layers.0"))?;
            let up2 = candle_nn::conv_transpose2d(OUT[1], OUT[1], 2, tr(2), d.pp("resize_layers.1"))?;
            let down2 = candle_nn::conv2d(OUT[3], OUT[3], 3, Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, d.pp("resize_layers.3"))?;
            let s = d.pp("scratch");
            let pad = Conv2dConfig { padding: 1, ..Default::default() };
            let rn = OUT
                .iter()
                .enumerate()
                .map(|(i, &o)| candle_nn::conv2d_no_bias(o, FEATURES, 3, pad, s.pp(format!("layer{}_rn", i + 1))))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let fusion = (1..=4).map(|i| Fusion::new(s.pp(format!("refinenet{i}")))).collect::<candle_core::Result<Vec<_>>>()?;
            let out1 = candle_nn::conv2d(FEATURES, FEATURES / 2, 3, pad, s.pp("output_conv1"))?;
            let out2a = candle_nn::conv2d(FEATURES / 2, 32, 3, pad, s.pp("output_conv2.0"))?;
            let out2b = candle_nn::conv2d(32, 1, 1, Default::default(), s.pp("output_conv2.2"))?;
            Ok(Self { patch, cls, pos, blocks, norm, projects, up4, up2, down2, rn, fusion, out1, out2a, out2b, dev: dev.clone(), sized: Mutex::new(None) })
        };
        build().map_err(err)
    }

    /// The positions for a `gh` × `gw` grid, the class token's first: `(1, 1 + gh·gw, DIM)`.
    fn positions(&self, gh: usize, gw: usize) -> candle_core::Result<Tensor> {
        let mut sized = self.sized.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((k, t)) = sized.as_ref()
            && *k == (gh, gw)
        {
            return Ok(t.clone());
        }
        let (cls, grid) = self.pos.split_at(DIM.min(self.pos.len()));
        let mut all = cls.to_vec();
        if gh == GRID && gw == GRID {
            all.extend_from_slice(grid);
        } else {
            all.extend(resize_positions(grid, gh, gw));
        }
        let t = Tensor::from_vec(all, (1, 1 + gh * gw, DIM), &self.dev)?;
        *sized = Some(((gh, gw), t.clone()));
        Ok(t)
    }

    /// The network's answer for a prepared picture (`(1, 3, fh, fw)`): `(fh, fw)` values.
    fn forward(&self, img: &Tensor) -> candle_core::Result<Vec<f32>> {
        let (_, _, fh, fw) = img.dims4()?;
        let (gh, gw) = (fh / PATCH, fw / PATCH);
        let x = self.patch.forward(img)?.flatten_from(2)?.transpose(1, 2)?;
        let mut x = (Tensor::cat(&[&self.cls, &x], 1)? + self.positions(gh, gw)?)?;
        let mut taken = Vec::with_capacity(TAKE.len());
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x)?;
            if TAKE.contains(&i) {
                // normed, the class token dropped, back to a grid
                taken.push(self.norm.forward(&x)?.i((.., 1..))?.transpose(1, 2)?.reshape((1, DIM, gh, gw))?);
            }
        }
        let mut layers = Vec::with_capacity(4);
        for (i, t) in taken.iter().enumerate() {
            let p = self.projects.get(i).ok_or_else(|| candle_core::Error::Msg("projects".into()))?.forward(t)?;
            let p = match i {
                0 => self.up4.forward(&p)?,
                1 => self.up2.forward(&p)?,
                3 => self.down2.forward(&p)?,
                _ => p,
            };
            let rn = self.rn.get(i).ok_or_else(|| candle_core::Error::Msg("rn".into()))?;
            layers.push(rn.forward(&p)?);
        }
        let [l1, l2, l3, l4] = layers.as_slice() else { candle_core::bail!("four levels") };
        let [f1, f2, f3, f4] = self.fusion.as_slice() else { candle_core::bail!("four fusion blocks") };
        let hw = |t: &Tensor| t.dims4().map(|d| (d.2, d.3));
        let p4 = f4.forward(l4, None, hw(l3)?)?;
        let p3 = f3.forward(&p4, Some(l3), hw(l2)?)?;
        let p2 = f2.forward(&p3, Some(l2), hw(l1)?)?;
        let (h1, w1) = hw(l1)?;
        let p1 = f1.forward(&p2, Some(l1), (h1 * 2, w1 * 2))?;
        let out = self.out1.forward(&p1)?.upsample_bilinear2d(gh * PATCH, gw * PATCH, true)?;
        let out = self.out2b.forward(&self.out2a.forward(&out)?.relu()?)?.relu()?;
        out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }
}

/// The device the network runs on: Metal on macOS when there is a GPU, else the CPU.
fn device(gpu: bool) -> Device {
    #[cfg(target_os = "macos")]
    if gpu && let Ok(d) = Device::new_metal(0) {
        return d;
    }
    let _ = gpu;
    Device::Cpu
}

/// The depth network, loaded.
pub struct DepthEstimator {
    net: Net,
    path: PathBuf,
    /// A copy on the CPU, loaded when the GPU fails a picture.
    cpu: Mutex<Option<Net>>,
}

impl DepthEstimator {
    /// Load the network from the models directory, on the GPU when `gpu` and there is one.
    pub fn load(models_dir: &Path, gpu: bool) -> Result<Self, VisionError> {
        if !catalog::depth_installed(models_dir) {
            return Err(VisionError::NotInstalled);
        }
        let path = catalog::depth_dir(models_dir).join(catalog::DEPTH.name);
        let dev = device(gpu);
        let net = match Net::load(&path, &dev) {
            Ok(n) => n,
            // (the GPU refused it: the CPU)
            Err(_) if !matches!(dev, Device::Cpu) => Net::load(&path, &Device::Cpu)?,
            Err(e) => return Err(e),
        };
        Ok(Self { net, path, cpu: Mutex::new(None) })
    }

    /// Where it runs: `"gpu"` or `"cpu"`.
    pub fn device_name(&self) -> &'static str {
        if matches!(self.net.dev, Device::Cpu) { "cpu" } else { "gpu" }
    }

    /// How near each part of an RGBA picture (`w` × `h`, sRGB) is, at the network's size for it
    /// ([`fitted_size`]).
    pub fn estimate(&self, rgba: &[u8], w: usize, h: usize) -> Result<DepthMap, VisionError> {
        if w == 0 || h == 0 || w.checked_mul(h).and_then(|n| n.checked_mul(4)).is_none_or(|n| rgba.len() < n) {
            return Err(VisionError::Invalid(format!("{} bytes for a {w} × {h} RGBA picture", rgba.len())));
        }
        let (fw, fh) = fitted_size(w, h);
        let px = prepare(rgba, w, h, (fw, fh));
        let run = |net: &Net| -> Result<Vec<f32>, VisionError> {
            let img = Tensor::from_vec(px.clone(), (1, 3, fh, fw), &net.dev).map_err(err)?;
            net.forward(&img).map_err(err)
        };
        let raw = match run(&self.net) {
            Ok(v) => v,
            Err(_) if !matches!(self.net.dev, Device::Cpu) => {
                let mut cpu = self.cpu.lock().unwrap_or_else(PoisonError::into_inner);
                if cpu.is_none() {
                    *cpu = Some(Net::load(&self.path, &Device::Cpu)?);
                }
                match cpu.as_ref() {
                    Some(n) => run(n)?,
                    None => return Err(err("no network")),
                }
            }
            Err(e) => return Err(e),
        };
        if raw.len() != fw * fh {
            return Err(err(format!("{} values for {fw} × {fh}", raw.len())));
        }
        Ok(DepthMap { w: fw, h: fh, near: normalised(raw) })
    }
}

/// Inverse depth made 0..1 by its [`TAIL`] percentiles (non-finite values the farthest).
pub fn normalised(mut v: Vec<f32>) -> Vec<f32> {
    let mut finite: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).step_by(7).collect();
    if finite.is_empty() {
        v.iter_mut().for_each(|x| *x = 0.0);
        return v;
    }
    let n = finite.len();
    let k = ((n as f32 * TAIL) as usize).min(n - 1);
    let lo = *finite.select_nth_unstable_by(k, f32::total_cmp).1;
    let hi = *finite.select_nth_unstable_by(n - 1 - k, f32::total_cmp).1;
    let span = (hi - lo).max(1e-6);
    v.iter_mut().for_each(|x| *x = if x.is_finite() { ((*x - lo) / span).clamp(0.0, 1.0) } else { 0.0 });
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pictures_are_fitted_to_the_patch_grid() {
        assert_eq!(fitted_size(518, 518), (518, 518));
        // a 16:9 frame: 518 high, 924 wide (66 patches), as the reference's lower bound gives
        assert_eq!(fitted_size(1920, 1080), (924, 518));
        assert_eq!(fitted_size(3024, 4032), (518, 686));
        // a panorama is squeezed to 2:1; a sliver and an empty picture stay square at least
        assert_eq!(fitted_size(8000, 1000), (1036, 518));
        assert_eq!(fitted_size(1, 4000), (518, 1036));
        assert_eq!(fitted_size(0, 0), (518, 518));
    }

    #[test]
    fn bicubic_weights_match_pytorchs() {
        // at a whole index the source value itself
        let w = cubic(0.0);
        assert!((w[1] - 1.0).abs() < 1e-6 && w[0].abs() < 1e-6 && w[2].abs() < 1e-6 && w[3].abs() < 1e-6);
        // halfway: the A = −0.75 kernel's values (−0.09375, 0.59375, 0.59375, −0.09375)
        let w = cubic(0.5);
        for (a, b) in w.iter().zip([-0.09375f32, 0.59375, 0.59375, -0.09375]) {
            assert!((a - b).abs() < 1e-6, "{w:?}");
        }
        for t in [0.1f32, 0.33, 0.77] {
            assert!((cubic(t).iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        // the positions of a 37 grid resized to 37 by 37.1 / 37: close to themselves, shifted
        // at most 0.1 of a cell (the 0.1 the reference adds: 36.5 × (1 − 37 / 37.1) at the end),
        // and A = −0.75 bends a ramp by up to 0.05 between samples
        let pos: Vec<f32> = (0..GRID * GRID * DIM).map(|i| ((i / DIM) % GRID) as f32).collect();
        let r = resize_positions(&pos, GRID, GRID);
        let worst = r.iter().zip(&pos).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 0.16, "a plain ramp stays a ramp ({worst})");
        // a wider grid: still a ramp along x, from the first column to the last
        let r = resize_positions(&pos, GRID, 66);
        assert!(r[0].abs() < 0.2 && (r[65 * DIM] - 36.0).abs() < 0.2);
        assert!((0..65).all(|x| r[(x + 1) * DIM] >= r[x * DIM] - 1e-4));
    }

    #[test]
    fn maps_are_made_nought_to_one_and_read_between_samples() {
        let v: Vec<f32> = (0..1000).map(|i| i as f32).chain([f32::NAN, f32::INFINITY]).collect();
        let n = normalised(v);
        assert!(n.iter().all(|x| (0.0..=1.0).contains(x)));
        assert_eq!(n[1000], 0.0);
        assert!(n[0] == 0.0 && n[999] == 1.0 && (n[500] - 0.5).abs() < 0.02);
        assert!(normalised(vec![f32::NAN; 4]).iter().all(|x| *x == 0.0));
        let m = DepthMap { w: 2, h: 2, near: vec![0.0, 1.0, 0.0, 1.0] };
        assert!((m.at(0.5, 0.5) - 0.5).abs() < 1e-6 && m.at(1.0, 0.0) == 1.0 && m.at(f32::NAN, 0.0) == 0.5);
        assert_eq!(DepthMap { w: 0, h: 0, near: vec![] }.at(0.5, 0.5), 0.0);
    }

    /// The downloaded model on a drawn scene: a checkered ground running to the horizon under a
    /// plain sky. The ground at the bottom is nearer than at the horizon, and the sky the
    /// farthest. (Skips when the model is not downloaded.)
    #[test]
    fn the_ground_near_the_camera_is_nearer_than_the_horizon() {
        let Some(dir) = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/FilmCraft/models")) else { return };
        let Ok(est) = DepthEstimator::load(&dir, std::env::var_os("DEPTH_CPU").is_none()) else {
            eprintln!("Depth Anything V2 not downloaded; skipping");
            return;
        };
        let (w, h) = (960usize, 540usize);
        let horizon = 200.0f32;
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let v = if (y as f32) < horizon {
                    let t = y as f32 / horizon;
                    [(120.0 + 60.0 * t) as u8, (170.0 + 40.0 * t) as u8, 235]
                } else {
                    // the ground plane in perspective: a checker 1 unit wide at distance 1/(y−horizon)
                    let d = 2000.0 / (y as f32 - horizon + 0.5);
                    let gx = (x as f32 - w as f32 / 2.0) * d / 400.0;
                    let on = ((gx.floor() as i64 + (d / 2.0).floor() as i64) & 1) == 0;
                    if on { [200, 190, 160] } else { [80, 110, 60] }
                };
                rgba[i..i + 4].copy_from_slice(&[v[0], v[1], v[2], 255]);
            }
        }
        let t = std::time::Instant::now();
        let m = est.estimate(&rgba, w, h).unwrap();
        let first = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let _ = est.estimate(&rgba, w, h).unwrap();
        eprintln!("Depth Anything V2 Small on the {}: {first:.0} ms, then {:.0} ms ({} × {})", est.device_name(), t.elapsed().as_secs_f64() * 1000.0, m.w, m.h);
        assert_eq!((m.w, m.h), (924, 518));
        let row = |v: f32| (0..20).map(|i| m.at(0.3 + 0.02 * i as f32, v)).sum::<f32>() / 20.0;
        let (sky, far, mid, near) = (row(0.15), row((horizon + 20.0) / h as f32), row(0.6), row(0.95));
        eprintln!("sky {sky:.2}, ground at the horizon {far:.2}, halfway {mid:.2}, at the bottom {near:.2}");
        assert!(near > mid && mid > far && far >= sky, "nearer down the ground");
        assert!(near > 0.8 && sky < 0.1);
        assert!(est.estimate(&rgba[..100], w, h).is_err());
    }

    /// Each PNG of `FILMCRAFT_DEPTH_PICTURES` with its depth beside it (`<name>.depth.png`, near
    /// white), and the time each took.
    #[test]
    #[ignore = "needs the model and FILMCRAFT_DEPTH_PICTURES"]
    fn depth_maps_of_pictures() {
        let Some(pics) = std::env::var_os("FILMCRAFT_DEPTH_PICTURES").map(PathBuf::from) else { return };
        let dir = PathBuf::from(std::env::var_os("HOME").unwrap()).join("Library/Application Support/FilmCraft/models");
        let est = DepthEstimator::load(&dir, std::env::var_os("DEPTH_CPU").is_none()).unwrap();
        let mut names: Vec<PathBuf> = std::fs::read_dir(&pics)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "png") && !p.to_string_lossy().contains(".depth"))
            .collect();
        names.sort();
        for p in names {
            let img = image::open(&p).unwrap().to_rgba8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let t = std::time::Instant::now();
            let m = est.estimate(img.as_raw(), w, h).unwrap();
            eprintln!("{}: {w} × {h} in {:.0} ms ({} × {})", p.display(), t.elapsed().as_secs_f64() * 1000.0, m.w, m.h);
            let mut out = image::RgbaImage::new(2 * w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    out.put_pixel(x as u32, y as u32, *img.get_pixel(x as u32, y as u32));
                    let v = (m.at(x as f32 / (w - 1).max(1) as f32, y as f32 / (h - 1).max(1) as f32) * 255.0) as u8;
                    out.put_pixel((w + x) as u32, y as u32, image::Rgba([v, v, v, 255]));
                }
            }
            out.save(p.with_extension("depth.png")).unwrap();
        }
    }
}
