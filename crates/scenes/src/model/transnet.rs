//! TransNetV2 (Souček & Lokoč, 2020; MIT): a shot-boundary network over 48×27 RGB frames, 100
//! at a time, written here on candle from the paper and the authors' published architecture
//! (`inference/transnetv2.py`, `inference-pytorch/transnetv2_pytorch.py`, MIT), with the
//! weights read from their TensorFlow checkpoint by layer name ([`super::tfckpt`]).
//!
//! The network: three stacked blocks of two dilated (2+1)D convolution cells — a 1×3×3 spatial
//! convolution then a 3×1×1 temporal one, dilated 1, 2, 4 and 8 in four parallel branches,
//! concatenated and batch-normalised — each block with a residual connection and a 1×2×2 average
//! pool; then the flattened features, a learned frame-similarity matrix of the blocks' mean
//! features and an RGB-histogram similarity matrix (each read in a ±50-frame window) go through a
//! dense layer to one transition probability per frame. The 3-D convolutions factorise: the
//! spatial one is a per-frame `conv2d`, the temporal one a per-pixel dilated `conv1d`.

use std::path::Path;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Conv1dConfig, Conv2dConfig};

use super::tfckpt::Checkpoint;
use super::{R, err};
use crate::boundaries::{WINDOW, WINDOW_KEEP, transnet_windows};
use crate::catalog;

/// Frame size the network sees.
pub const WIDTH: usize = 48;
pub const HEIGHT: usize = 27;
/// Bytes of one frame (RGB).
pub const FRAME_BYTES: usize = WIDTH * HEIGHT * 3;
/// The similarity windows: ±50 frames.
const LOOKUP: usize = 101;
const BN_EPS: f64 = 1e-3;
/// Frames a spatial convolution runs on at once (bounds its im2col buffer).
const CONV_CHUNK: usize = 25;

struct Branch {
    /// `[2F, in, 3, 3]`
    spatial: Tensor,
    /// `[F, 2F, 3]`
    temporal: Tensor,
    dilation: usize,
}

struct Cell {
    branches: Vec<Branch>,
    /// Batch norm folded into a per-channel scale and shift, `[1, 4F, 1, 1]`.
    scale: Tensor,
    shift: Tensor,
    relu: bool,
}

struct Block {
    first: Cell,
    second: Cell,
}

struct Dense {
    /// `[in, out]`
    w: Tensor,
    /// `[out]`
    b: Tensor,
}

impl Dense {
    fn load(c: &Checkpoint, name: &str, inp: usize, out: usize, dev: &Device) -> R<Dense> {
        let w = Tensor::from_vec(c.load(&format!("{name}/kernel"), &[inp, out])?, (inp, out), dev).map_err(err)?;
        let b = Tensor::from_vec(c.load(&format!("{name}/bias"), &[out])?, out, dev).map_err(err)?;
        Ok(Dense { w, b })
    }

    /// `x [.., in]` → `[.., out]`.
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        x.broadcast_matmul(&self.w).and_then(|y| y.broadcast_add(&self.b)).map_err(err)
    }
}

/// The network, loaded.
pub struct TransNet {
    dev: Device,
    blocks: Vec<Block>,
    sim_proj: Dense,
    sim_fc: Dense,
    hist_fc: Dense,
    fc1: Dense,
    cls: Dense,
}

/// Filters of the first block (doubling in each), blocks, cells per block.
const F: usize = 16;
const BLOCKS: usize = 3;

impl TransNet {
    /// Load the checkpoint from the model's directory (`<models>/transnetv2`).
    pub fn load(dir: &Path, dev: &Device) -> R<TransNet> {
        let m = catalog::find(catalog::TRANSNET).ok_or_else(|| err("TransNetV2 is not in the catalogue"))?;
        let (Some(index), Some(data)) = (m.files.first(), m.files.get(1)) else { return Err(err("TransNetV2's catalogue entry is incomplete")) };
        let c = Checkpoint::open(&dir.join(index.name), &dir.join(data.name))?;
        let mut blocks = Vec::new();
        let mut inp = 3;
        for b in 0..BLOCKS {
            let f = F << b;
            let first = cell(&c, b + 1, 1, inp, f, true, dev)?;
            let second = cell(&c, b + 1, 2, 4 * f, f, false, dev)?;
            blocks.push(Block { first, second });
            inp = 4 * f;
        }
        let feats: usize = (0..BLOCKS).map(|b| 4 * (F << b)).sum();
        let flat = 4 * (F << (BLOCKS - 1)) * 3 * 6;
        Ok(TransNet {
            dev: dev.clone(),
            blocks,
            sim_proj: Dense::load(&c, "TransNet/FrameSimilarity/dense_3", feats, 128, dev)?,
            sim_fc: Dense::load(&c, "TransNet/FrameSimilarity/dense_4", LOOKUP, 128, dev)?,
            hist_fc: Dense::load(&c, "TransNet/ColorHistograms/dense_5", LOOKUP, 128, dev)?,
            fc1: Dense::load(&c, "TransNet/dense", flat + 256, 1024, dev)?,
            cls: Dense::load(&c, "TransNet/dense_1", 1024, 1, dev)?,
        })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The probability that each frame is a transition, for `frames` (each [`FRAME_BYTES`] of
    /// 48×27 RGB). `cancelled` is asked between windows.
    pub fn predict(&self, frames: &[Vec<u8>], cancelled: &dyn Fn() -> bool) -> R<Vec<f32>> {
        if frames.iter().any(|f| f.len() != FRAME_BYTES) {
            return Err(crate::SceneError::Invalid(format!("TransNetV2 frames are {WIDTH}×{HEIGHT} RGB ({FRAME_BYTES} bytes)")));
        }
        let n = frames.len();
        let mut out = Vec::with_capacity(n + WINDOW_KEEP);
        for w in transnet_windows(n) {
            if cancelled() {
                return Err(crate::SceneError::Cancelled);
            }
            let window: Vec<&[u8]> = w.iter().filter_map(|i| frames.get(*i).map(Vec::as_slice)).collect();
            if window.len() != WINDOW {
                return Err(err("TransNetV2: a window is short"));
            }
            let p = self.window(&window)?;
            let start = (WINDOW - WINDOW_KEEP) / 2;
            out.extend_from_slice(p.get(start..start + WINDOW_KEEP).ok_or_else(|| err("TransNetV2: short prediction"))?);
        }
        out.truncate(n);
        Ok(out)
    }

    /// One window of [`WINDOW`] frames → a probability per frame.
    fn window(&self, frames: &[&[u8]]) -> R<Vec<f32>> {
        let t = frames.len();
        let mut px = Vec::with_capacity(t * FRAME_BYTES);
        for f in frames {
            // HWC → CHW per frame, 0..1
            for c in 0..3 {
                for i in 0..WIDTH * HEIGHT {
                    px.push(f32::from(f.get(i * 3 + c).copied().unwrap_or(0)) / 255.0);
                }
            }
        }
        let mut x = Tensor::from_vec(px, (t, 3, HEIGHT, WIDTH), &self.dev).map_err(err)?;
        let mut feats = Vec::new();
        for b in &self.blocks {
            let first = b.first.forward(&x, t)?;
            let second = b.second.forward(&first, t)?;
            let y = (second.relu().map_err(err)? + &first).map_err(err)?;
            x = y.avg_pool2d(2).map_err(err)?;
            feats.push(x.mean(D::Minus1).and_then(|m| m.mean(D::Minus1)).map_err(err)?);
        }
        // the learned frame similarity of the blocks' mean features
        let f = Tensor::cat(&feats, 1).map_err(err)?;
        let p = self.sim_proj.forward(&f)?;
        let norm = p.sqr().and_then(|s| s.sum_keepdim(1)).and_then(|s| s.clamp(1e-12, f64::MAX)).and_then(|s| s.sqrt()).map_err(err)?;
        let p = p.broadcast_div(&norm).map_err(err)?;
        let sim: Vec<Vec<f32>> = p.matmul(&p.t().map_err(err)?).and_then(|s| s.to_dtype(DType::F32)).and_then(|s| s.to_vec2()).map_err(err)?;
        let sim_w = Tensor::from_vec(windowed(&sim), (t, LOOKUP), &self.dev).map_err(err)?;
        let sim_f = self.sim_fc.forward(&sim_w)?.relu().map_err(err)?;
        // RGB histogram similarity
        let hist = histogram_similarity(frames);
        let hist_w = Tensor::from_vec(windowed(&hist), (t, LOOKUP), &self.dev).map_err(err)?;
        let hist_f = self.hist_fc.forward(&hist_w)?.relu().map_err(err)?;
        // the last block's features, flattened channels last (as the TensorFlow model does)
        let flat = x.permute((0, 2, 3, 1)).and_then(|v| v.contiguous()).and_then(|v| v.flatten_from(1)).map_err(err)?;
        let all = Tensor::cat(&[&hist_f, &sim_f, &flat], 1).map_err(err)?;
        let h = self.fc1.forward(&all)?.relu().map_err(err)?;
        let logits = self.cls.forward(&h)?;
        let probs = candle_nn::ops::sigmoid(&logits)
            .and_then(|p| p.flatten_all())
            .and_then(|p| p.to_dtype(DType::F32))
            .and_then(|p| p.to_vec1::<f32>())
            .map_err(err)?;
        Ok(probs)
    }
}

fn cell(c: &Checkpoint, block: usize, cell: usize, inp: usize, f: usize, relu: bool, dev: &Device) -> R<Cell> {
    let base = format!("TransNet/SDDCNN_{block}/DDCNN_{cell}");
    let mut branches = Vec::new();
    for (k, dilation) in [(1usize, 1usize), (2, 2), (4, 4), (8, 8)] {
        let s = c.load(&format!("{base}/Conv3D_{k}/conv_spatial/kernel"), &[1, 3, 3, inp, 2 * f])?;
        let t = c.load(&format!("{base}/Conv3D_{k}/conv_temporal/kernel"), &[3, 1, 1, 2 * f, f])?;
        branches.push(Branch {
            spatial: Tensor::from_vec(spatial_kernel(&s, inp, 2 * f), (2 * f, inp, 3, 3), dev).map_err(err)?,
            temporal: Tensor::from_vec(temporal_kernel(&t, 2 * f, f), (f, 2 * f, 3), dev).map_err(err)?,
            dilation,
        });
    }
    let n = 4 * f;
    let g = c.load(&format!("{base}/bn/gamma"), &[n])?;
    let b = c.load(&format!("{base}/bn/beta"), &[n])?;
    let m = c.load(&format!("{base}/bn/moving_mean"), &[n])?;
    let v = c.load(&format!("{base}/bn/moving_variance"), &[n])?;
    let scale: Vec<f32> = g.iter().zip(&v).map(|(g, v)| (f64::from(*g) / (f64::from(*v) + BN_EPS).max(1e-12).sqrt()) as f32).collect();
    let shift: Vec<f32> = b.iter().zip(&m).zip(&scale).map(|((b, m), s)| b - m * s).collect();
    Ok(Cell {
        branches,
        scale: Tensor::from_vec(scale, (1, n, 1, 1), dev).map_err(err)?,
        shift: Tensor::from_vec(shift, (1, n, 1, 1), dev).map_err(err)?,
        relu,
    })
}

/// TensorFlow's `[1, 3, 3, in, out]` → candle's `[out, in, 3, 3]`.
pub(crate) fn spatial_kernel(k: &[f32], inp: usize, out: usize) -> Vec<f32> {
    let mut w = vec![0f32; out * inp * 9];
    for ky in 0..3 {
        for kx in 0..3 {
            for i in 0..inp {
                for o in 0..out {
                    if let (Some(d), Some(s)) = (w.get_mut(((o * inp + i) * 3 + ky) * 3 + kx), k.get(((ky * 3 + kx) * inp + i) * out + o)) {
                        *d = *s;
                    }
                }
            }
        }
    }
    w
}

/// TensorFlow's `[3, 1, 1, in, out]` → candle's `[out, in, 3]`.
pub(crate) fn temporal_kernel(k: &[f32], inp: usize, out: usize) -> Vec<f32> {
    let mut w = vec![0f32; out * inp * 3];
    for t in 0..3 {
        for i in 0..inp {
            for o in 0..out {
                if let (Some(d), Some(s)) = (w.get_mut((o * inp + i) * 3 + t), k.get((t * inp + i) * out + o)) {
                    *d = *s;
                }
            }
        }
    }
    w
}

impl Cell {
    /// `x [t, in, h, w]` → `[t, 4F, h, w]`.
    fn forward(&self, x: &Tensor, t: usize) -> R<Tensor> {
        let mut outs = Vec::with_capacity(self.branches.len());
        for b in &self.branches {
            let s = spatial(x, &b.spatial)?;
            outs.push(temporal(&s, &b.temporal, b.dilation, t)?);
        }
        let y = Tensor::cat(&outs, 1).and_then(|y| y.broadcast_mul(&self.scale)).and_then(|y| y.broadcast_add(&self.shift)).map_err(err)?;
        if self.relu { y.relu().map_err(err) } else { Ok(y) }
    }
}

/// The 1×3×3 convolution ("same" padding) of every frame.
pub(crate) fn spatial(x: &Tensor, w: &Tensor) -> R<Tensor> {
    let cfg = Conv2dConfig { padding: 1, stride: 1, dilation: 1, groups: 1, cudnn_fwd_algo: None };
    let n = x.dim(0).map_err(err)?;
    let mut parts = Vec::new();
    let mut at = 0;
    while at < n {
        let k = CONV_CHUNK.min(n - at);
        parts.push(x.narrow(0, at, k).and_then(|c| c.conv2d(w, cfg.padding, cfg.stride, cfg.dilation, cfg.groups)).map_err(err)?);
        at += k;
    }
    Tensor::cat(&parts, 0).map_err(err)
}

/// The 3×1×1 convolution over time, dilated by `d` ("same" padding): `x [t, c, h, w]` →
/// `[t, out, h, w]`, as a `conv1d` over each pixel's time series.
pub(crate) fn temporal(x: &Tensor, w: &Tensor, d: usize, t: usize) -> R<Tensor> {
    let (n, c, h, wd) = x.dims4().map_err(err)?;
    let out = w.dim(0).map_err(err)?;
    if n != t {
        return Err(err("TransNetV2: frame count mismatch"));
    }
    let cfg = Conv1dConfig { padding: d, stride: 1, dilation: d, groups: 1, cudnn_fwd_algo: None };
    let series = x.reshape((t, c, h * wd)).and_then(|v| v.permute((2, 1, 0))).and_then(|v| v.contiguous()).map_err(err)?;
    let y = series.conv1d(w, cfg.padding, cfg.stride, cfg.dilation, cfg.groups).map_err(err)?;
    y.permute((2, 1, 0)).and_then(|v| v.contiguous()).and_then(|v| v.reshape((t, out, h, wd))).map_err(err)
}

/// Each frame's similarities to the frames ±50 around it (0 past the ends), `[t · 101]`.
pub(crate) fn windowed(sim: &[Vec<f32>]) -> Vec<f32> {
    let t = sim.len();
    let half = (LOOKUP - 1) / 2;
    let mut out = vec![0f32; t * LOOKUP];
    for i in 0..t {
        for l in 0..LOOKUP {
            let j = i + l;
            if j >= half && j - half < t {
                out[i * LOOKUP + l] = sim.get(i).and_then(|r| r.get(j - half)).copied().unwrap_or(0.0);
            }
        }
    }
    out
}

/// The cosine similarity of the frames' 512-bin RGB histograms (3 bits a channel).
pub(crate) fn histogram_similarity(frames: &[&[u8]]) -> Vec<Vec<f32>> {
    let hist: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| {
            let mut h = vec![0f32; 512];
            for p in f.as_chunks::<3>().0 {
                let bin = (usize::from(p[0] >> 5) << 6) | (usize::from(p[1] >> 5) << 3) | usize::from(p[2] >> 5);
                h[bin] += 1.0;
            }
            let n = h.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            h.iter_mut().for_each(|v| *v /= n);
            h
        })
        .collect();
    hist.iter().map(|a| hist.iter().map(|b| a.iter().zip(b).map(|(x, y)| x * y).sum()).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 8) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn kernels_reorder_like_the_reference() {
        let (inp, out) = (2, 3);
        let k = rnd(9 * inp * out, 1);
        let w = spatial_kernel(&k, inp, out);
        // TF [0, ky, kx, i, o] == candle [o, i, ky, kx]
        let (o, i, ky, kx) = (2, 1, 2, 0);
        assert_eq!(w[((o * inp + i) * 3 + ky) * 3 + kx], k[((ky * 3 + kx) * inp + i) * out + o]);
        let k = rnd(3 * inp * out, 2);
        let w = temporal_kernel(&k, inp, out);
        let (o, i, tap) = (1, 0, 2);
        assert_eq!(w[(o * inp + i) * 3 + tap], k[(tap * inp + i) * out + o]);
    }

    #[test]
    fn temporal_convolution_matches_loops() {
        let dev = super::super::device();
        let (t, c, h, w, o, d) = (11, 3, 2, 3, 4, 2);
        let x = rnd(t * c * h * w, 3);
        let k = rnd(o * c * 3, 4);
        let xt = Tensor::from_vec(x.clone(), (t, c, h, w), &dev).unwrap();
        let kt = Tensor::from_vec(k.clone(), (o, c, 3), &dev).unwrap();
        let y: Vec<f32> = temporal(&xt, &kt, d, t).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for ti in 0..t {
            for oi in 0..o {
                for p in 0..h * w {
                    let mut acc = 0f32;
                    for tap in 0..3 {
                        let src = ti as isize + (tap as isize - 1) * d as isize;
                        if src < 0 || src >= t as isize {
                            continue;
                        }
                        for ci in 0..c {
                            acc += k[(oi * c + ci) * 3 + tap] * x[(src as usize * c + ci) * h * w + p];
                        }
                    }
                    let got = y[(ti * o + oi) * h * w + p];
                    assert!((got - acc).abs() < 1e-4, "t{ti} o{oi} p{p}: {got} vs {acc}");
                }
            }
        }
    }

    #[test]
    fn spatial_convolution_matches_loops() {
        let dev = super::super::device();
        let (t, c, h, w, o) = (30, 2, 4, 5, 3);
        let x = rnd(t * c * h * w, 5);
        let k = rnd(o * c * 9, 6);
        let xt = Tensor::from_vec(x.clone(), (t, c, h, w), &dev).unwrap();
        let kt = Tensor::from_vec(k.clone(), (o, c, 3, 3), &dev).unwrap();
        let y: Vec<f32> = spatial(&xt, &kt).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for ti in [0, 17, 29] {
            for oi in 0..o {
                for yy in 0..h {
                    for xx in 0..w {
                        let mut acc = 0f32;
                        for ci in 0..c {
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let (sy, sx) = (yy as isize + ky as isize - 1, xx as isize + kx as isize - 1);
                                    if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                                        continue;
                                    }
                                    acc += k[((oi * c + ci) * 3 + ky) * 3 + kx] * x[((ti * c + ci) * h + sy as usize) * w + sx as usize];
                                }
                            }
                        }
                        let got = y[((ti * o + oi) * h + yy) * w + xx];
                        assert!((got - acc).abs() < 1e-4, "{got} vs {acc}");
                    }
                }
            }
        }
    }

    /// A synthetic shot's frame: a pattern chosen by `shot`, moving with time `t`, 48×27 RGB.
    pub(crate) fn synthetic(shot: u32, t: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(FRAME_BYTES);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let xx = x as f32 + 0.7 * t as f32;
                let (r, g, b) = match shot % 4 {
                    0 => (40.0 + 150.0 * (xx * 0.2).sin().abs(), 90.0, 160.0),
                    1 => (200.0, 180.0 - y as f32 * 3.0, 40.0 + (xx * 0.3).cos().abs() * 60.0),
                    2 => (if ((xx as usize) / 4 + y / 4).is_multiple_of(2) { 230.0 } else { 20.0 }, 120.0, 60.0),
                    _ => (30.0, 30.0 + (y as f32 * 6.0).min(200.0), 30.0 + (xx * 0.1).sin().abs() * 200.0),
                };
                v.extend_from_slice(&[r as u8, g as u8, b as u8]);
            }
        }
        v
    }

    /// With the real weights (`FILMCRAFT_SCENES_MODELS` holding `transnetv2/`): the cuts of a
    /// synthetic clip of four moving shots are found where they are, and nowhere else.
    #[test]
    fn finds_the_cuts_of_a_synthetic_clip() {
        let Some(dir) = std::env::var_os("FILMCRAFT_SCENES_MODELS").map(std::path::PathBuf::from) else { return };
        if !catalog::installed_id(&dir, catalog::TRANSNET) {
            return;
        }
        let net = TransNet::load(&dir.join(catalog::TRANSNET), &super::super::device()).unwrap();
        let lens = [70u32, 55, 90, 60];
        let mut frames = Vec::new();
        for (s, n) in lens.iter().enumerate() {
            for t in 0..*n {
                frames.push(synthetic(s as u32, t));
            }
        }
        let p = net.predict(&frames, &|| false).unwrap();
        assert_eq!(p.len(), frames.len());
        let cuts = crate::boundaries::cuts_from_probabilities(&p, 0.5);
        assert_eq!(cuts, vec![70, 125, 215], "probabilities around the cuts: {:?}", [&p[66..74], &p[121..129], &p[211..219]]);
        assert!(net.predict(&[vec![0u8; 3]], &|| false).is_err(), "a frame of the wrong size");
        assert!(matches!(net.predict(&frames, &|| true), Err(crate::SceneError::Cancelled)));
    }

    #[test]
    fn similarity_windows_and_histograms() {
        let sim = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let w = windowed(&sim);
        assert_eq!(w.len(), 2 * LOOKUP);
        assert_eq!((w[50], w[51], w[49]), (1.0, 0.5, 0.0));
        assert_eq!((w[LOOKUP + 50], w[LOOKUP + 49]), (1.0, 0.5));
        let red = [255u8, 0, 0].repeat(4);
        let blue = [0u8, 0, 255].repeat(4);
        let h = histogram_similarity(&[&red, &red, &blue]);
        assert!((h[0][1] - 1.0).abs() < 1e-6 && h[0][2].abs() < 1e-6);
    }
}
