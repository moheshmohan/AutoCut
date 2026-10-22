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
//! dense layer to one transition probability per frame.
//!
//! Everything is kept channels-last (`[frames, height, width, channels]`, TensorFlow's layout),
//! and both convolutions are one matrix product each over shifted copies of the input: the
//! spatial one over the nine 3×3 neighbours, the temporal one over the frames `−d, 0, +d` away.
//! TensorFlow's kernels are already in that order, so they are used as they are. (candle's own
//! `conv2d` on Metal ran this network eight times slower.)

use std::path::Path;

use candle_core::{D, DType, Device, Tensor};

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

struct Branch {
    /// `[9 · in, 2F]`: the 3×3 neighbours (row by row) of every input channel.
    spatial: Tensor,
    /// `[3 · 2F, F]`: the frames `−d, 0, +d` away.
    temporal: Tensor,
    dilation: usize,
}

struct Cell {
    branches: Vec<Branch>,
    /// Batch norm folded into a per-channel scale and shift, `[4F]`.
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
    dtype: DType,
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
        // the convolutions in half precision on the GPU (they are memory-bound), f32 on the CPU
        let dtype = if dev.is_cpu() { DType::F32 } else { DType::F16 };
        let mut blocks = Vec::new();
        let mut inp = 3;
        for b in 0..BLOCKS {
            let f = F << b;
            let first = cell(&c, b + 1, 1, inp, f, true, dev, dtype)?;
            let second = cell(&c, b + 1, 2, 4 * f, f, false, dev, dtype)?;
            blocks.push(Block { first, second });
            inp = 4 * f;
        }
        let feats: usize = (0..BLOCKS).map(|b| 4 * (F << b)).sum();
        let flat = 4 * (F << (BLOCKS - 1)) * 3 * 6;
        Ok(TransNet {
            dev: dev.clone(),
            dtype,
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
            px.extend(f.iter().map(|v| f32::from(*v) / 255.0));
        }
        // frames, height, width, channels
        let mut x = Tensor::from_vec(px, (t, HEIGHT, WIDTH, 3), &self.dev).and_then(|x| x.to_dtype(self.dtype)).map_err(err)?;
        let mut feats = Vec::new();
        let (mut h, mut w) = (HEIGHT, WIDTH);
        for b in &self.blocks {
            let masks = edge_masks(t, h, w, self.dtype, &self.dev)?;
            let first = b.first.forward(&x, &masks)?;
            let second = b.second.forward(&first, &masks)?;
            (h, w) = (h / 2, w / 2);
            let y = (second.relu().map_err(err)? + &first).map_err(err)?;
            x = avg_pool(&y)?;
            feats.push(spatial_mean(&x)?);
        }
        // the learned frame similarity of the blocks' mean features
        let f = Tensor::cat(&feats, 1).and_then(|f| f.to_dtype(DType::F32)).map_err(err)?;
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
        let flat = x.flatten_from(1).and_then(|f| f.to_dtype(DType::F32)).map_err(err)?;
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

fn cell(c: &Checkpoint, block: usize, cell: usize, inp: usize, f: usize, relu: bool, dev: &Device, dtype: DType) -> R<Cell> {
    let base = format!("TransNet/SDDCNN_{block}/DDCNN_{cell}");
    let mut branches = Vec::new();
    for (k, dilation) in [(1usize, 1usize), (2, 2), (4, 4), (8, 8)] {
        // TensorFlow's [1, 3, 3, in, out] and [3, 1, 1, in, out] are already rows of
        // (neighbour, input channel): the matrices as they are
        let s = c.load(&format!("{base}/Conv3D_{k}/conv_spatial/kernel"), &[1, 3, 3, inp, 2 * f])?;
        let t = c.load(&format!("{base}/Conv3D_{k}/conv_temporal/kernel"), &[3, 1, 1, 2 * f, f])?;
        branches.push(Branch {
            spatial: Tensor::from_vec(s, (9 * inp, 2 * f), dev).and_then(|w| w.to_dtype(dtype)).map_err(err)?,
            temporal: Tensor::from_vec(t, (3 * 2 * f, f), dev).and_then(|w| w.to_dtype(dtype)).map_err(err)?,
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
    let t = |v: Vec<f32>| Tensor::from_vec(v, n, dev).and_then(|t| t.to_dtype(dtype)).map_err(err);
    Ok(Cell { branches, scale: t(scale)?, shift: t(shift)?, relu })
}

impl Cell {
    /// `x [t, h, w, in]` → `[t, h, w, 4F]`.
    fn forward(&self, x: &Tensor, masks: &[Tensor]) -> R<Tensor> {
        let mut outs = Vec::with_capacity(self.branches.len());
        for b in &self.branches {
            let s = spatial(x, &b.spatial, masks)?;
            outs.push(temporal(&s, &b.temporal, b.dilation)?);
        }
        let y = Tensor::cat(&outs, D::Minus1).and_then(|y| y.broadcast_mul(&self.scale)).and_then(|y| y.broadcast_add(&self.shift)).map_err(err)?;
        if self.relu { y.relu().map_err(err) } else { Ok(y) }
    }
}

/// The nine 3×3 neighbour taps of a `t`×`h`×`w` grid of pixels: for each, a column `[t·h·w, 1]`
/// that is 1 where the neighbour is inside the frame and 0 where it falls off an edge.
pub(crate) fn edge_masks(t: usize, h: usize, w: usize, dtype: DType, dev: &Device) -> R<Vec<Tensor>> {
    let mut out = Vec::with_capacity(9);
    for ky in 0..3usize {
        for kx in 0..3usize {
            let mut m = Vec::with_capacity(t * h * w);
            for _ in 0..t {
                for y in 0..h {
                    for x in 0..w {
                        let inside = (y + ky >= 1) && (y + ky - 1 < h) && (x + kx >= 1) && (x + kx - 1 < w);
                        m.push(if inside { 1f32 } else { 0.0 });
                    }
                }
            }
            out.push(Tensor::from_vec(m, (t * h * w, 1), dev).and_then(|m| m.to_dtype(dtype)).map_err(err)?);
        }
    }
    Ok(out)
}

/// The 1×3×3 convolution ("same" padding) of every frame, channels last: `x [t, h, w, in]` and
/// `k [9 · in, out]` → `[t, h, w, out]`. The frames are rows of `in` values; a neighbour is the
/// same rows shifted by `±w ± 1`, so each tap is one matrix product over a shifted view of the
/// (zero-padded) rows, its wrap across a frame's edge masked off ([`edge_masks`]): no copy of
/// the input per tap.
pub(crate) fn spatial(x: &Tensor, k: &Tensor, masks: &[Tensor]) -> R<Tensor> {
    let (t, h, w, c) = x.dims4().map_err(err)?;
    let out = k.dim(1).map_err(err)?;
    let n = t * h * w;
    if masks.len() != 9 {
        return Err(err("TransNetV2: missing edge masks"));
    }
    let pad = w + 1;
    let p = x.contiguous().and_then(|v| v.reshape((n, c))).and_then(|v| v.pad_with_zeros(0, pad, pad)).map_err(err)?;
    let mut acc: Option<Tensor> = None;
    for ky in 0..3 {
        for kx in 0..3 {
            let tap = ky * 3 + kx;
            let y = p.narrow(0, ky * w + kx, n).and_then(|v| v.matmul(&k.narrow(0, tap * c, c)?)).map_err(err)?;
            let y = if tap == 4 {
                y
            } else {
                let mask = masks.get(tap).ok_or_else(|| err("TransNetV2: missing an edge mask"))?;
                y.broadcast_mul(mask).map_err(err)?
            };
            acc = Some(match acc {
                Some(a) => (a + y).map_err(err)?,
                None => y,
            });
        }
    }
    acc.ok_or_else(|| err("TransNetV2: empty convolution"))?.reshape((t, h, w, out)).map_err(err)
}

/// The 3×1×1 convolution over time, dilated by `d` ("same" padding), channels last:
/// `x [t, h, w, in]` and `k [3 · in, out]` → `[t, h, w, out]`; a frame `±d` away is the rows
/// shifted by `±d · h · w` (zero past the ends), one matrix product a tap.
pub(crate) fn temporal(x: &Tensor, k: &Tensor, d: usize) -> R<Tensor> {
    let (t, h, w, c) = x.dims4().map_err(err)?;
    let out = k.dim(1).map_err(err)?;
    let (n, hw) = (t * h * w, h * w);
    let p = x.contiguous().and_then(|v| v.reshape((n, c))).and_then(|v| v.pad_with_zeros(0, d * hw, d * hw)).map_err(err)?;
    let mut acc: Option<Tensor> = None;
    for tap in 0..3 {
        let y = p.narrow(0, tap * d * hw, n).and_then(|v| v.matmul(&k.narrow(0, tap * c, c)?)).map_err(err)?;
        acc = Some(match acc {
            Some(a) => (a + y).map_err(err)?,
            None => y,
        });
    }
    acc.ok_or_else(|| err("TransNetV2: empty convolution"))?.reshape((t, h, w, out)).map_err(err)
}

/// The 1×2×2 average pool (a trailing odd row or column is dropped), channels last. Written with
/// reshapes, slices and sums only: candle 0.9's Metal reductions over a middle axis of a
/// many-axis tensor gave wrong sums.
fn avg_pool(x: &Tensor) -> R<Tensor> {
    let (t, h, w, c) = x.dims4().map_err(err)?;
    let (h2, w2) = (h / 2, w / 2);
    let x = x.narrow(1, 0, 2 * h2).and_then(|v| v.narrow(2, 0, 2 * w2)).and_then(|v| v.contiguous()).map_err(err)?;
    // neighbouring columns are neighbouring runs of `c` values
    let cols = x.reshape((t, 2 * h2, w2, 2 * c)).map_err(err)?;
    let x = (cols.narrow(3, 0, c).map_err(err)? + cols.narrow(3, c, c).map_err(err)?).map_err(err)?;
    // neighbouring rows are neighbouring runs of `w2 · c` values
    let rows = x.contiguous().and_then(|v| v.reshape((t, h2, 2 * w2 * c))).map_err(err)?;
    let x = (rows.narrow(2, 0, w2 * c).map_err(err)? + rows.narrow(2, w2 * c, w2 * c).map_err(err)?).map_err(err)?;
    (x * 0.25).and_then(|v| v.reshape((t, h2, w2, c))).map_err(err)
}

/// The mean over height and width of `x [t, h, w, c]` → `[t, c]`, reduced along the last axis
/// (see [`avg_pool`]).
fn spatial_mean(x: &Tensor) -> R<Tensor> {
    let (t, h, w, c) = x.dims4().map_err(err)?;
    x.reshape((t, h * w, c)).and_then(|v| v.transpose(1, 2)).and_then(|v| v.contiguous()).and_then(|v| v.mean(D::Minus1)).map_err(err)
}

/// Each frame's similarities to the frames ±50 around it (0 past the ends), `[t · 101]`.
pub(crate) fn windowed(sim: &[Vec<f32>]) -> Vec<f32> {
    let t = sim.len();
    let half = (LOOKUP - 1) / 2;
    let mut out = vec![0f32; t * LOOKUP];
    for i in 0..t {
        for l in 0..LOOKUP {
            let j = i + l;
            if j >= half
                && j - half < t
                && let Some(o) = out.get_mut(i * LOOKUP + l)
            {
                *o = sim.get(i).and_then(|r| r.get(j - half)).copied().unwrap_or(0.0);
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
                if let Some(v) = h.get_mut(bin) {
                    *v += 1.0;
                }
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
    fn temporal_convolution_matches_loops() {
        let dev = super::super::device();
        let (t, h, w, c, o, d) = (11, 2, 3, 3, 4, 2);
        let x = rnd(t * h * w * c, 3);
        // TensorFlow's [3, 1, 1, in, out]
        let k = rnd(3 * c * o, 4);
        let xt = Tensor::from_vec(x.clone(), (t, h, w, c), &dev).unwrap();
        let kt = Tensor::from_vec(k.clone(), (3 * c, o), &dev).unwrap();
        let y: Vec<f32> = temporal(&xt, &kt, d).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for ti in 0..t {
            for p in 0..h * w {
                for oi in 0..o {
                    let mut acc = 0f32;
                    for tap in 0..3 {
                        let src = ti as isize + (tap as isize - 1) * d as isize;
                        if src < 0 || src >= t as isize {
                            continue;
                        }
                        for ci in 0..c {
                            acc += k[(tap * c + ci) * o + oi] * x[(src as usize * h * w + p) * c + ci];
                        }
                    }
                    let got = y[(ti * h * w + p) * o + oi];
                    assert!((got - acc).abs() < 1e-4, "t{ti} p{p} o{oi}: {got} vs {acc}");
                }
            }
        }
    }

    #[test]
    fn spatial_convolution_matches_loops() {
        let dev = super::super::device();
        let (t, h, w, c, o) = (3, 4, 5, 2, 3);
        let x = rnd(t * h * w * c, 5);
        // TensorFlow's [1, 3, 3, in, out]
        let k = rnd(9 * c * o, 6);
        let xt = Tensor::from_vec(x.clone(), (t, h, w, c), &dev).unwrap();
        let kt = Tensor::from_vec(k.clone(), (9 * c, o), &dev).unwrap();
        let masks = edge_masks(t, h, w, DType::F32, &dev).unwrap();
        let y: Vec<f32> = spatial(&xt, &kt, &masks).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for ti in 0..t {
            for yy in 0..h {
                for xx in 0..w {
                    for oi in 0..o {
                        let mut acc = 0f32;
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let (sy, sx) = (yy as isize + ky as isize - 1, xx as isize + kx as isize - 1);
                                if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                                    continue;
                                }
                                for ci in 0..c {
                                    acc += k[((ky * 3 + kx) * c + ci) * o + oi] * x[((ti * h + sy as usize) * w + sx as usize) * c + ci];
                                }
                            }
                        }
                        let got = y[((ti * h + yy) * w + xx) * o + oi];
                        assert!((got - acc).abs() < 1e-4, "{got} vs {acc}");
                    }
                }
            }
        }
    }

    #[test]
    fn average_pool_drops_the_odd_edge() {
        let dev = super::super::device();
        let x = Tensor::from_vec((0..5 * 3).map(|v| v as f32).collect::<Vec<_>>(), (1, 3, 5, 1), &dev).unwrap();
        let y: Vec<f32> = avg_pool(&x).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // rows 0-1, columns 0-1 and 2-3 of [[0..5], [5..10], [10..15]]
        assert_eq!(y, vec![3.0, 5.0]);
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
