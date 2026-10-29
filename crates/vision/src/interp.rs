//! RIFE 4.26 (MIT, Practical-RIFE, [`crate::catalog::INTERPOLATOR`]): the picture between two
//! frames at any time between them — IFNet's five blocks estimating the motion both ways, coarse
//! to fine, warping both frames and their features to that time and mixing them by a learned
//! mask (Huang et al., ECCV 2022) — written out on candle from the published architecture (the
//! CPU: the reference and the fallback) and, on macOS, as WGSL passes on the GPU (`interp_gpu`:
//! candle on Metal took 1.1 s a 1080p frame in hundreds of whole-frame operations).
//!
//! The frames are padded with zeros on the right and at the bottom to a multiple of 128 (256 for
//! pictures over 2048 pixels, which are looked at at half the scale), the blocks look at them at
//! 1/16, 1/8, 1/4, 1/2 and the full size (twice those over 2048), each resize bilinear without
//! aligned corners; a frame is warped by its motion with bilinear sampling, the edges held (as
//! PyTorch's `grid_sample` with `padding_mode="border"` and aligned corners does). The weights
//! are the zip's `flownet.pkl`, taken out once (its CRC-32 checked) beside it.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder};

use crate::{VisionError, catalog};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("interpolator: {e}"))
}

/// The largest frame taken, pixels a side.
pub const MAX_SIDE: usize = 4096;
/// Frames whose long side is over this are looked at at half the scale.
pub const HALF_ABOVE: usize = 2048;
const SLOPE: f64 = 0.2;
/// The blocks' channels and inputs.
const BLOCKS: [(usize, usize); 5] = [(15, 192), (28, 128), (28, 96), (28, 64), (28, 32)];

fn conv(cin: usize, cout: usize, k: usize, stride: usize, vb: VarBuilder) -> candle_core::Result<Conv2d> {
    candle_nn::conv2d(cin, cout, k, Conv2dConfig { stride, padding: k / 2, ..Default::default() }, vb)
}

fn up2(cin: usize, cout: usize, vb: VarBuilder) -> candle_core::Result<ConvTranspose2d> {
    candle_nn::conv_transpose2d(cin, cout, 4, ConvTranspose2dConfig { stride: 2, padding: 1, ..Default::default() }, vb)
}

fn lrelu(x: &Tensor) -> candle_core::Result<Tensor> {
    candle_nn::ops::leaky_relu(x, SLOPE)
}

/// A bilinear resize to `(h, w)` without aligned corners (nothing when it is already that size).
fn resize(x: &Tensor, h: usize, w: usize) -> candle_core::Result<Tensor> {
    let (_, _, xh, xw) = x.dims4()?;
    if (xh, xw) == (h, w) { Ok(x.clone()) } else { x.upsample_bilinear2d(h, w, false) }
}

/// The features each frame is described by: three convolutions, the first halving, and a
/// transposed one back to the frame's size (4 channels).
struct Head {
    c0: Conv2d,
    c1: Conv2d,
    c2: Conv2d,
    c3: ConvTranspose2d,
}

impl Head {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            c0: conv(3, 16, 3, 2, vb.pp("cnn0"))?,
            c1: conv(16, 16, 3, 1, vb.pp("cnn1"))?,
            c2: conv(16, 16, 3, 1, vb.pp("cnn2"))?,
            c3: up2(16, 4, vb.pp("cnn3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = lrelu(&self.c0.forward(x)?)?;
        let x = lrelu(&self.c1.forward(&x)?)?;
        let x = lrelu(&self.c2.forward(&x)?)?;
        self.c3.forward(&x)
    }
}

/// A block: its inputs at `1 / scale`, two halving convolutions, eight residual units, a
/// transposed convolution and a pixel shuffle back, resized to the full size: the motion both ways
/// (4 channels, in full-size pixels), the mask's logit and 8 channels of features.
struct Block {
    c0: Conv2d,
    c1: Conv2d,
    res: Vec<(Conv2d, Tensor)>,
    last: ConvTranspose2d,
}

impl Block {
    fn new(cin: usize, c: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let res = (0..8)
            .map(|i| {
                let r = vb.pp(format!("convblock.{i}"));
                Ok((conv(c, c, 3, 1, r.pp("conv"))?, r.get((1, c, 1, 1), "beta")?))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            c0: conv(cin, c / 2, 3, 2, vb.pp("conv0.0.0"))?,
            c1: conv(c / 2, c, 3, 2, vb.pp("conv0.1.0"))?,
            res,
            last: up2(c, 4 * 13, vb.pp("lastconv.0"))?,
        })
    }

    fn forward(&self, x: &Tensor, flow: Option<&Tensor>, scale: usize) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let (_, _, h, w) = x.dims4()?;
        let (sh, sw) = (h / scale, w / scale);
        let mut x = resize(x, sh, sw)?;
        if let Some(f) = flow {
            let f = (resize(f, sh, sw)? * (1.0 / scale as f64))?;
            x = Tensor::cat(&[&x, &f], 1)?;
        }
        let mut feat = lrelu(&self.c1.forward(&lrelu(&self.c0.forward(&x)?)?)?)?;
        for (c, beta) in &self.res {
            feat = lrelu(&(c.forward(&feat)?.broadcast_mul(beta)? + &feat)?)?;
        }
        let tmp = resize(&candle_nn::ops::pixel_shuffle(&self.last.forward(&feat)?, 2)?, h, w)?;
        Ok(((tmp.i((.., 0..4))? * scale as f64)?, tmp.i((.., 4..5))?, tmp.i((.., 5..13))?))
    }
}

/// `x` (`(1, c, h, w)`) warped by `flow` (`(1, 2, h, w)`, pixels): each pixel the bilinear sample
/// at itself plus its motion, the edges held.
pub fn warp(x: &Tensor, flow: &Tensor) -> candle_core::Result<Tensor> {
    let (_, c, h, w) = x.dims4()?;
    let dev = x.device();
    let gx = Tensor::arange(0f32, w as f32, dev)?.reshape((1, w))?.broadcast_as((h, w))?;
    let gy = Tensor::arange(0f32, h as f32, dev)?.reshape((h, 1))?.broadcast_as((h, w))?;
    let sx = (gx + flow.i((0, 0))?)?.clamp(0f32, (w - 1) as f32)?;
    let sy = (gy + flow.i((0, 1))?)?.clamp(0f32, (h - 1) as f32)?;
    let (x0, y0) = (sx.floor()?, sy.floor()?);
    let (ax, ay) = ((&sx - &x0)?, (&sy - &y0)?);
    let x1 = (&x0 + 1.0)?.minimum((w - 1) as f32)?;
    let y1 = (&y0 + 1.0)?.minimum((h - 1) as f32)?;
    let flat = x.reshape((c, h * w))?;
    let at = |xx: &Tensor, yy: &Tensor| -> candle_core::Result<Tensor> {
        let idx = ((yy * w as f64)? + xx)?.flatten_all()?.to_dtype(DType::U32)?;
        flat.index_select(&idx, 1)?.reshape((c, h, w))
    };
    let (one_x, one_y) = ((1.0 - &ax)?, (1.0 - &ay)?);
    let top = (at(&x0, &y0)?.broadcast_mul(&one_x)? + at(&x1, &y0)?.broadcast_mul(&ax)?)?;
    let bottom = (at(&x0, &y1)?.broadcast_mul(&one_x)? + at(&x1, &y1)?.broadcast_mul(&ax)?)?;
    (top.broadcast_mul(&one_y)? + bottom.broadcast_mul(&ay)?)?.unsqueeze(0)
}

struct Net {
    head: Head,
    blocks: Vec<Block>,
    dev: Device,
}

impl Net {
    fn load(path: &Path, dev: &Device) -> Result<Self, VisionError> {
        let vb = VarBuilder::from_pth(path, DType::F32, dev).map_err(err)?.pp("module");
        let build = || -> candle_core::Result<Self> {
            let head = Head::new(vb.pp("encode"))?;
            let blocks =
                BLOCKS.iter().enumerate().map(|(i, &(cin, c))| Block::new(cin, c, vb.pp(format!("block{i}")))).collect::<candle_core::Result<Vec<_>>>()?;
            Ok(Self { head, blocks, dev: dev.clone() })
        };
        build().map_err(err)
    }

    /// The frame at `t` between `a` and `b` (`(1, 3, h, w)`, 0..1, sides multiples of the
    /// coarsest scale × 8), looked at at `scales`.
    fn forward(&self, a: &Tensor, b: &Tensor, t: f32, scales: [usize; 5]) -> candle_core::Result<Tensor> {
        let (_, _, h, w) = a.dims4()?;
        let (f0, f1) = (self.head.forward(a)?, self.head.forward(b)?);
        let ts = Tensor::full(t, (1, 1, h, w), &self.dev)?;
        let mut flow: Option<Tensor> = None;
        let (mut mask, mut feat) = (None, None);
        let (mut wa, mut wb) = (a.clone(), b.clone());
        for (i, blk) in self.blocks.iter().enumerate() {
            let s = scales.get(i).copied().unwrap_or(1).max(1);
            match (&flow, &mask, &feat) {
                (Some(fl), Some(m), Some(ft)) => {
                    let wf0 = warp(&f0, &fl.i((.., 0..2))?)?;
                    let wf1 = warp(&f1, &fl.i((.., 2..4))?)?;
                    let x = Tensor::cat(&[&wa, &wb, &wf0, &wf1, &ts, m, ft], 1)?;
                    let (fd, m0, ft0) = blk.forward(&x, Some(fl), s)?;
                    flow = Some((fl + fd)?);
                    mask = Some(m0);
                    feat = Some(ft0);
                }
                _ => {
                    let x = Tensor::cat(&[a, b, &f0, &f1, &ts], 1)?;
                    let (fl, m, ft) = blk.forward(&x, None, s)?;
                    flow = Some(fl);
                    mask = Some(m);
                    feat = Some(ft);
                }
            }
            if let Some(fl) = &flow {
                wa = warp(a, &fl.i((.., 0..2))?)?;
                wb = warp(b, &fl.i((.., 2..4))?)?;
            }
        }
        let m = candle_nn::ops::sigmoid(mask.as_ref().ok_or_else(|| candle_core::Error::Msg("no mask".into()))?)?;
        let out = (wa.broadcast_mul(&m)? + wb.broadcast_mul(&(1.0 - &m)?)?)?;
        out.clamp(0f32, 1f32)
    }
}

impl Net {
    /// Every layer's weights, for the GPU.
    fn weights(&self) -> candle_core::Result<crate::interp_gpu::NetWeights> {
        use crate::interp_gpu::ConvWeights;
        let v = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        let conv = |c: &Conv2d, beta: Option<&Tensor>| -> candle_core::Result<ConvWeights> {
            let (cout, cin, _, _) = c.weight().dims4()?;
            Ok(ConvWeights {
                cin,
                cout,
                weight: v(c.weight())?,
                bias: c.bias().map(v).transpose()?.unwrap_or_else(|| vec![0.0; cout]),
                beta: beta.map(v).transpose()?,
            })
        };
        let convt = |c: &ConvTranspose2d| -> candle_core::Result<ConvWeights> {
            let (cin, cout, _, _) = c.weight().dims4()?;
            Ok(ConvWeights { cin, cout, weight: v(c.weight())?, bias: c.bias().map(v).transpose()?.unwrap_or_else(|| vec![0.0; cout]), beta: None })
        };
        let h = &self.head;
        let head = [conv(&h.c0, None)?, conv(&h.c1, None)?, conv(&h.c2, None)?, convt(&h.c3)?];
        let mut blocks = Vec::new();
        for (b, &(_, c)) in self.blocks.iter().zip(BLOCKS.iter()) {
            let res = b.res.iter().map(|(cv, beta)| conv(cv, Some(beta))).collect::<candle_core::Result<Vec<_>>>()?;
            blocks.push((conv(&b.c0, None)?, conv(&b.c1, None)?, res, convt(&b.last)?, c));
        }
        Ok(crate::interp_gpu::NetWeights { head, blocks })
    }
}

/// The weights taken out of the zip beside it (once; again when they are missing or the wrong
/// size).
fn weights(models_dir: &Path) -> Result<PathBuf, VisionError> {
    let dir = catalog::interp_dir(models_dir);
    let out = dir.join(catalog::INTERP_WEIGHTS);
    if std::fs::metadata(&out).is_ok_and(|m| m.len() == catalog::INTERP_WEIGHTS_SIZE) {
        return Ok(out);
    }
    let zip = std::fs::read(dir.join(catalog::INTERPOLATOR.name)).map_err(err)?;
    let data = crate::zip::read_entry(&zip, &format!("/{}", catalog::INTERP_WEIGHTS), 64 << 20)?;
    if data.len() as u64 != catalog::INTERP_WEIGHTS_SIZE {
        return Err(err(format!("{} is {} bytes, not {}", catalog::INTERP_WEIGHTS, data.len(), catalog::INTERP_WEIGHTS_SIZE)));
    }
    let tmp = out.with_extension("part");
    std::fs::write(&tmp, &data).map_err(err)?;
    std::fs::rename(&tmp, &out).map_err(err)?;
    Ok(out)
}

/// The interpolator, loaded.
pub struct Interpolator {
    /// candle on the CPU: the reference and the fallback.
    cpu: Net,
    gpu: Option<crate::interp_gpu::GpuNet>,
    /// Set when the GPU failed a frame: the CPU from then on.
    gpu_failed: std::sync::atomic::AtomicBool,
}

/// A frame's RGB (planar, 0..1) padded with zeros to `(pw, ph)`.
fn planar(rgba: &[u8], w: usize, h: usize, pw: usize, ph: usize) -> Vec<f32> {
    let mut px = vec![0f32; 3 * pw * ph];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            for c in 0..3 {
                px[c * pw * ph + y * pw + x] = rgba.get(i + c).map_or(0.0, |v| f32::from(*v) / 255.0);
            }
        }
    }
    px
}

/// How a `w` × `h` frame is looked at: its padded size and the blocks' scales.
pub fn layout(w: usize, h: usize) -> ((usize, usize), [usize; 5]) {
    let half = w.max(h) > HALF_ABOVE;
    let unit = if half { 256 } else { 128 };
    let pad = |v: usize| v.div_ceil(unit).max(1) * unit;
    let scales = if half { [32, 16, 8, 4, 2] } else { [16, 8, 4, 2, 1] };
    ((pad(w), pad(h)), scales)
}

impl Interpolator {
    /// Load the network from the models directory, on the GPU when `gpu` and there is one.
    pub fn load(models_dir: &Path, gpu: bool) -> Result<Self, VisionError> {
        if !catalog::interpolator_installed(models_dir) {
            return Err(VisionError::NotInstalled);
        }
        let path = weights(models_dir)?;
        let cpu = Net::load(&path, &Device::Cpu)?;
        let gpu = if gpu && cfg!(target_os = "macos") { cpu.weights().ok().and_then(|w| crate::interp_gpu::GpuNet::new(&w)) } else { None };
        Ok(Self { cpu, gpu, gpu_failed: std::sync::atomic::AtomicBool::new(false) })
    }

    /// Where it runs: `"gpu"` or `"cpu"`.
    pub fn device_name(&self) -> &'static str {
        if self.gpu.is_some() && !self.gpu_failed.load(std::sync::atomic::Ordering::Relaxed) { "gpu" } else { "cpu" }
    }

    /// The padded frames through candle on the CPU (the reference).
    fn run_cpu(&self, pa: &[f32], pb: &[f32], (pw, ph): (usize, usize), t: f32, scales: [usize; 5]) -> Result<Vec<f32>, VisionError> {
        let ta = Tensor::from_vec(pa.to_vec(), (1, 3, ph, pw), &self.cpu.dev).map_err(err)?;
        let tb = Tensor::from_vec(pb.to_vec(), (1, 3, ph, pw), &self.cpu.dev).map_err(err)?;
        self.cpu.forward(&ta, &tb, t, scales).and_then(|o| o.flatten_all()).and_then(|o| o.to_vec1::<f32>()).map_err(err)
    }

    /// The frame at `t` (0..1) between RGBA frames `a` and `b` (`w` × `h`, sRGB): RGBA, its alpha
    /// the nearer frame's.
    pub fn between(&self, a: &[u8], b: &[u8], w: usize, h: usize, t: f32) -> Result<Vec<u8>, VisionError> {
        let n = w.checked_mul(h).and_then(|n| n.checked_mul(4));
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || n.is_none_or(|n| a.len() < n || b.len() < n) {
            return Err(VisionError::Invalid(format!("frames of {} and {} bytes for {w} × {h} (at most {MAX_SIDE} a side)", a.len(), b.len())));
        }
        let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.5 };
        let ((pw, ph), scales) = layout(w, h);
        let (pa, pb) = (planar(a, w, h, pw, ph), planar(b, w, h, pw, ph));
        let mut out = None;
        if let Some(g) = &self.gpu
            && !self.gpu_failed.load(std::sync::atomic::Ordering::Relaxed)
        {
            match g.between(&pa, &pb, pw, ph, t, scales) {
                Ok(v) => out = Some(v),
                Err(_) => self.gpu_failed.store(true, std::sync::atomic::Ordering::Relaxed),
            }
        }
        let out = match out {
            Some(v) => v,
            None => self.run_cpu(&pa, &pb, (pw, ph), t, scales)?,
        };
        if out.len() != 3 * pw * ph {
            return Err(err(format!("{} values for {pw} × {ph}", out.len())));
        }
        let alpha = if t < 0.5 { a } else { b };
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                for c in 0..3 {
                    let v = out.get(c * pw * ph + y * pw + x).copied().unwrap_or(0.0);
                    rgba[o + c] = if v.is_finite() { (v.clamp(0.0, 1.0) * 255.0).round() as u8 } else { 0 };
                }
                rgba[o + 3] = alpha.get(o + 3).copied().unwrap_or(255);
            }
        }
        Ok(rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_padded_to_the_coarsest_grid() {
        assert_eq!(layout(1920, 1080), ((1920, 1152), [16, 8, 4, 2, 1]));
        assert_eq!(layout(854, 480), ((896, 512), [16, 8, 4, 2, 1]));
        assert_eq!(layout(3840, 2160), ((3840, 2304), [32, 16, 8, 4, 2]));
        assert_eq!(layout(1, 1), ((128, 128), [16, 8, 4, 2, 1]));
    }

    #[test]
    fn a_warp_moves_the_picture_by_its_motion_and_holds_the_edges() {
        let dev = Device::Cpu;
        // a ramp along x: warping by +2.5 px reads 2.5 further on
        let (w, h) = (16usize, 4usize);
        let x = Tensor::from_vec((0..w * h).map(|i| (i % w) as f32).collect::<Vec<_>>(), (1, 1, h, w), &dev).unwrap();
        let flow = Tensor::from_vec([vec![2.5f32; w * h], vec![0.0; w * h]].concat(), (1, 2, h, w), &dev).unwrap();
        let v = warp(&x, &flow).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[3] - 5.5).abs() < 1e-5 && (v[w - 1] - (w - 1) as f32).abs() < 1e-5, "{v:?}");
        // far outside: the edge
        let flow = Tensor::from_vec([vec![-100f32; w * h], vec![50.0; w * h]].concat(), (1, 2, h, w), &dev).unwrap();
        let v = warp(&x, &flow).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| *x == 0.0));
    }

    /// The downloaded model: a square moving 16 px right between two frames is halfway at 0.5,
    /// on the GPU and the CPU alike. (Skips when RIFE is not downloaded.)
    #[test]
    fn the_frame_between_shows_a_moving_square_halfway() {
        let Some(dir) = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/FilmCraft/models")) else { return };
        let Ok(net) = Interpolator::load(&dir, std::env::var_os("INTERP_CPU").is_none()) else {
            eprintln!("RIFE not downloaded; skipping");
            return;
        };
        let (w, h) = (256usize, 160usize);
        let frame = |x0: usize| {
            let mut v = vec![0u8; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let on = (x0..x0 + 40).contains(&x) && (60..100).contains(&y);
                    let bg = (40 + (x * 3 + y * 5) % 60) as u8;
                    v[i..i + 4].copy_from_slice(&if on { [230, 200, 60, 255] } else { [bg, bg, bg + 80, 255] });
                }
            }
            v
        };
        let (a, b) = (frame(80), frame(96));
        let t = std::time::Instant::now();
        let mid = net.between(&a, &b, w, h, 0.5).unwrap();
        eprintln!("RIFE on the {}: {w} × {h} in {:.0} ms", net.device_name(), t.elapsed().as_secs_f64() * 1e3);
        let yellow = |img: &[u8], x: usize| img[(80 * w + x) * 4 + 2] < 100;
        // the square's left edge at 88: yellow from 90 to 126, not at 84 nor 132
        assert!((90..126).all(|x| yellow(&mid, x)), "the square halfway");
        assert!(!yellow(&mid, 84) && !yellow(&mid, 132));
        let truth = frame(88);
        let diff = mid.iter().zip(&truth).map(|(p, q)| f64::from(p.abs_diff(*q))).sum::<f64>() / mid.len() as f64;
        eprintln!("mean difference from the true middle frame: {diff:.2} levels");
        assert!(diff < 6.0, "{diff}");
        assert!(net.between(&a[..10], &b, w, h, 0.5).is_err());
    }

    /// The GPU's passes against candle on the CPU, on two drawn frames with motion.
    #[test]
    fn the_gpu_matches_the_cpu() {
        let Some(dir) = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/FilmCraft/models")) else { return };
        let Ok(net) = Interpolator::load(&dir, true) else { return };
        let Some(g) = &net.gpu else {
            eprintln!("no GPU; skipping");
            return;
        };
        let (w, h) = (256usize, 128usize);
        let frame = |dx: f32| -> Vec<u8> {
            (0..w * h)
                .flat_map(|i| {
                    let (x, y) = ((i % w) as f32 + dx, (i / w) as f32);
                    let v = 128.0 + 90.0 * (x * 0.11).sin() * (y * 0.07).cos();
                    [v as u8, (255.0 - v) as u8, ((x + y) * 0.5) as u8, 255]
                })
                .collect()
        };
        let (a, b) = (frame(0.0), frame(6.0));
        let ((pw, ph), scales) = layout(w, h);
        let (pa, pb) = (planar(&a, w, h, pw, ph), planar(&b, w, h, pw, ph));
        let cpu = net.run_cpu(&pa, &pb, (pw, ph), 0.3, scales).unwrap();
        let t = std::time::Instant::now();
        let gpu = g.between(&pa, &pb, pw, ph, 0.3, scales).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let worst = cpu.iter().zip(&gpu).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let mean = cpu.iter().zip(&gpu).map(|(x, y)| (x - y).abs()).sum::<f32>() / cpu.len() as f32;
        eprintln!("GPU against CPU: at most {worst:.2e}, mean {mean:.2e} ({ms:.0} ms)");
        assert!(worst < 2e-3 && mean < 1e-4, "{worst} {mean}");
        assert!(g.between(&pa[..10], &pb, pw, ph, 0.5, scales).is_err());
    }

    /// Each triplet of `FILMCRAFT_INTERP_TRIPLETS` (`<name>-0.png`, `-1`, `-2`: three frames,
    /// evenly spaced): the middle made from the outer two, its PSNR against the real one beside a
    /// cross-fade's and a held frame's, and the time; the made frame written as `<name>-rife.png`.
    #[test]
    #[ignore = "needs the model and FILMCRAFT_INTERP_TRIPLETS"]
    fn the_middle_of_real_triplets() {
        let Some(dir) = std::env::var_os("FILMCRAFT_INTERP_TRIPLETS").map(PathBuf::from) else { return };
        let models = PathBuf::from(std::env::var_os("HOME").unwrap()).join("Library/Application Support/FilmCraft/models");
        let net = Interpolator::load(&models, std::env::var_os("INTERP_CPU").is_none()).unwrap();
        let psnr = |a: &[u8], b: &[u8]| {
            let (mut se, mut n) = (0f64, 0f64);
            for (p, q) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
                for c in 0..3 {
                    let d = f64::from(p[c]) - f64::from(q[c]);
                    se += d * d;
                    n += 1.0;
                }
            }
            10.0 * (255.0f64 * 255.0 / (se / n.max(1.0)).max(1e-9)).log10()
        };
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|n| n.strip_suffix("-0.png").map(str::to_string))
            .collect();
        names.sort();
        for n in names {
            let load = |k: &str| image::open(dir.join(format!("{n}-{k}.png"))).unwrap().to_rgba8();
            let (a, m, b) = (load("0"), load("1"), load("2"));
            let (w, h) = (a.width() as usize, a.height() as usize);
            let t = std::time::Instant::now();
            let mid = net.between(a.as_raw(), b.as_raw(), w, h, 0.5).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let blend: Vec<u8> = a.as_raw().iter().zip(b.as_raw()).map(|(p, q)| (u16::from(*p) + u16::from(*q)).div_ceil(2) as u8).collect();
            eprintln!(
                "{n} ({w} × {h}): RIFE {:.2} dB, cross-fade {:.2} dB, held {:.2} dB, in {ms:.0} ms on the {}",
                psnr(&mid, m.as_raw()),
                psnr(&blend, m.as_raw()),
                psnr(a.as_raw(), m.as_raw()),
                net.device_name()
            );
            image::RgbaImage::from_raw(w as u32, h as u32, mid).unwrap().save(dir.join(format!("{n}-rife.png"))).unwrap();
        }
    }
}
