//! The Oobleck VAE decoder (diffusers' `AutoencoderOobleck`, the Stable Audio architecture; the
//! weights are ACE-Step's own 48 kHz stereo VAE, MIT): 64 latent channels at 25 frames a second →
//! stereo at 48 kHz.
//!
//! A 7-tap convolution to 2048 channels; five blocks, one per stride (10, 6, 4, 4, 2), each a
//! Snake activation, a transposed convolution (kernel 2·stride) halving the channels and three
//! residual units (Snake, a 7-tap convolution dilated 1, 3 and 9, Snake, a 1×1 convolution);
//! Snake and a 7-tap convolution to two channels. Weight norm is folded on load; Snake's α and β
//! are stored as logarithms.
//!
//! Convolutions are sums of shifted matrix products and the transposed ones a matrix product plus
//! an overlap-add, so everything runs as candle tensors on the GPU or CPU without im2col buffers.
//! The decoder is not causal but its receptive field is finite (about ±9 latent frames), so
//! [`Vae::decode`] works in windows with [`OVERLAP`] frames of context either side: memory stays
//! bounded however long the music, and the result equals one pass over everything.

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::config::VaeConfig;
use super::weights::SafeTensors;
use super::{R, err};

/// Latent frames decoded per window (5.12 s).
pub(crate) const WINDOW: usize = 128;
/// Frames of context either side of a window (above the receptive field).
pub(crate) const OVERLAP: usize = 16;

struct Conv {
    /// Per tap `[cout, cin]`.
    taps: Vec<Tensor>,
    /// `[cout, 1]`.
    bias: Option<Tensor>,
    dil: usize,
}

impl Conv {
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let t = x.dim(1).map_err(err)?;
        let k = self.taps.len();
        let pad = (k - 1) * self.dil / 2;
        let xp = if pad > 0 { x.pad_with_zeros(1, pad, pad).map_err(err)? } else { x.clone() };
        let mut y: Option<Tensor> = None;
        for (j, w) in self.taps.iter().enumerate() {
            let xs = if k == 1 { xp.clone() } else { xp.narrow(1, j * self.dil, t).map_err(err)?.contiguous().map_err(err)? };
            let p = w.matmul(&xs).map_err(err)?;
            y = Some(match y {
                None => p,
                Some(a) => (a + p).map_err(err)?,
            });
        }
        let y = y.ok_or_else(|| err("convolution without taps"))?;
        match &self.bias {
            Some(b) => y.broadcast_add(b).map_err(err),
            None => Ok(y),
        }
    }
}

/// Transposed convolution with kernel `2 · stride`, padding `stride / 2`: exactly `stride` times
/// longer.
struct ConvT {
    /// `[cout · 2s, cin]`, row `o · 2s + j`.
    w: Tensor,
    bias: Tensor,
    cout: usize,
    stride: usize,
}

impl ConvT {
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let l = x.dim(1).map_err(err)?;
        let s = self.stride;
        let y = self.w.matmul(x).map_err(err)?.reshape((self.cout, 2 * s, l)).map_err(err)?;
        // output n' = t·s + j gets tap j of x[t] and tap j + s of x[t − 1]
        let a = y.narrow(1, 0, s).map_err(err)?.pad_with_zeros(2, 0, 1).map_err(err)?;
        let b = y.narrow(1, s, s).map_err(err)?.pad_with_zeros(2, 1, 0).map_err(err)?;
        let full = (a + b).map_err(err)?.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?.reshape((self.cout, (l + 1) * s)).map_err(err)?;
        full.narrow(1, s / 2, l * s).map_err(err)?.broadcast_add(&self.bias).map_err(err)
    }
}

struct Snake {
    /// `[c, 1]`: e^α and 1 / (e^β + 10⁻⁹).
    alpha: Tensor,
    inv_beta: Tensor,
}

impl Snake {
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let s = x.broadcast_mul(&self.alpha).map_err(err)?.sin().map_err(err)?.sqr().map_err(err)?;
        (x + s.broadcast_mul(&self.inv_beta).map_err(err)?).map_err(err)
    }
}

struct Residual {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

struct Block {
    snake: Snake,
    up: ConvT,
    res: Vec<Residual>,
}

pub(crate) struct Vae {
    conv1: Conv,
    blocks: Vec<Block>,
    snake: Snake,
    conv2: Conv,
    dev: Device,
    latent: usize,
}

struct Ld<'a> {
    st: &'a mut SafeTensors,
    dev: &'a Device,
}

impl Ld<'_> {
    /// A weight-normalised weight `g · v / ‖v‖` (norm over every dimension but the first), on the CPU.
    fn wn(&mut self, p: &str, shape: &[usize]) -> R<Tensor> {
        let v = self.st.load(&format!("{p}.weight_v"), shape, &Device::Cpu)?;
        let mut gs = vec![1usize; shape.len()];
        if let (Some(g0), Some(s0)) = (gs.first_mut(), shape.first()) {
            *g0 = *s0;
        }
        let g = self.st.load(&format!("{p}.weight_g"), &gs, &Device::Cpu)?;
        let d0 = *shape.first().ok_or_else(|| err("empty weight shape"))?;
        let norm =
            v.reshape((d0, ())).map_err(err)?.sqr().map_err(err)?.sum_keepdim(1).map_err(err)?.sqrt().map_err(err)?.reshape(gs.as_slice()).map_err(err)?;
        v.broadcast_div(&norm).map_err(err)?.broadcast_mul(&g).map_err(err)
    }

    fn conv(&mut self, p: &str, cin: usize, cout: usize, k: usize, dil: usize, bias: bool) -> R<Conv> {
        let w = self.wn(p, &[cout, cin, k])?;
        let taps = (0..k)
            .map(|j| w.narrow(2, j, 1).map_err(err)?.squeeze(2).map_err(err)?.contiguous().map_err(err)?.to_device(self.dev).map_err(err))
            .collect::<R<Vec<_>>>()?;
        let bias = if bias { Some(self.st.load(&format!("{p}.bias"), &[cout], self.dev)?.reshape((cout, 1)).map_err(err)?) } else { None };
        Ok(Conv { taps, bias, dil })
    }

    fn conv_t(&mut self, p: &str, cin: usize, cout: usize, stride: usize) -> R<ConvT> {
        let k = 2 * stride;
        let w = self.wn(p, &[cin, cout, k])?;
        let w = w.permute((1, 2, 0)).map_err(err)?.contiguous().map_err(err)?.reshape((cout * k, cin)).map_err(err)?.to_device(self.dev).map_err(err)?;
        let bias = self.st.load(&format!("{p}.bias"), &[cout], self.dev)?.reshape((cout, 1)).map_err(err)?;
        Ok(ConvT { w, bias, cout, stride })
    }

    fn snake(&mut self, p: &str, c: usize) -> R<Snake> {
        let a = self.st.load(&format!("{p}.alpha"), &[1, c, 1], self.dev)?.reshape((c, 1)).map_err(err)?.exp().map_err(err)?;
        let b = self.st.load(&format!("{p}.beta"), &[1, c, 1], self.dev)?.reshape((c, 1)).map_err(err)?.exp().map_err(err)?;
        let inv = (b + 1e-9).map_err(err)?.recip().map_err(err)?;
        Ok(Snake { alpha: a, inv_beta: inv })
    }
}

impl Vae {
    pub(crate) fn load(path: &Path, cfg: &VaeConfig, dev: &Device) -> R<Vae> {
        let mut st = SafeTensors::open(path)?;
        let mut ld = Ld { st: &mut st, dev };
        let ch = cfg.channels;
        // [1] + multiples: widths from the top block down
        let mut mult = vec![1usize];
        mult.extend_from_slice(&cfg.multiples);
        let n = cfg.strides.len();
        let top = ch * mult.last().copied().unwrap_or(1);
        let conv1 = ld.conv("decoder.conv1", cfg.latent, top, 7, 1, true)?;
        let mut blocks = Vec::with_capacity(n);
        for (i, &s) in cfg.strides.iter().enumerate() {
            let cin = ch * mult.get(n - i).copied().ok_or_else(|| err("VAE widths"))?;
            let cout = ch * mult.get(n - i - 1).copied().ok_or_else(|| err("VAE widths"))?;
            let p = format!("decoder.block.{i}");
            let mut res = Vec::with_capacity(3);
            for (u, dil) in [(1, 1), (2, 3), (3, 9)] {
                let q = format!("{p}.res_unit{u}");
                res.push(Residual {
                    snake1: ld.snake(&format!("{q}.snake1"), cout)?,
                    conv1: ld.conv(&format!("{q}.conv1"), cout, cout, 7, dil, true)?,
                    snake2: ld.snake(&format!("{q}.snake2"), cout)?,
                    conv2: ld.conv(&format!("{q}.conv2"), cout, cout, 1, 1, true)?,
                });
            }
            blocks.push(Block { snake: ld.snake(&format!("{p}.snake1"), cin)?, up: ld.conv_t(&format!("{p}.conv_t1"), cin, cout, s)?, res });
        }
        let snake = ld.snake("decoder.snake1", ch)?;
        let conv2 = ld.conv("decoder.conv2", ch, cfg.audio_channels, 7, 1, false)?;
        Ok(Vae { conv1, blocks, snake, conv2, dev: dev.clone(), latent: cfg.latent })
    }

    /// One pass over latents `[latent, frames]` → `[2, frames · 1920]`.
    fn pass(&self, z: &Tensor) -> R<Tensor> {
        let mut x = self.conv1.fwd(z)?;
        for b in &self.blocks {
            x = b.up.fwd(&b.snake.fwd(&x)?)?;
            for r in &b.res {
                let y = r.conv1.fwd(&r.snake1.fwd(&x)?)?;
                let y = r.conv2.fwd(&r.snake2.fwd(&y)?)?;
                x = (x + y).map_err(err)?;
            }
        }
        self.conv2.fwd(&self.snake.fwd(&x)?)
    }

    /// Decode latents (`frames × latent`, frame-major as the DiT makes them) to stereo, window by
    /// window. `progress(done, total)` returns false to stop.
    pub(crate) fn decode(&self, latents: &[f32], frames: usize, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<(Vec<f32>, Vec<f32>)> {
        self.decode_windows(latents, frames, WINDOW, OVERLAP, progress)
    }

    pub(crate) fn decode_windows(
        &self,
        latents: &[f32],
        frames: usize,
        window: usize,
        overlap: usize,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> R<(Vec<f32>, Vec<f32>)> {
        let c = self.latent;
        if frames == 0 || latents.len() != frames.saturating_mul(c) || window == 0 {
            return Err(err("latents do not match their length"));
        }
        let hop = crate::request::HOP;
        let total = frames.checked_mul(hop).ok_or_else(|| err("too long"))?;
        let mut left = Vec::with_capacity(total);
        let mut right = Vec::with_capacity(total);
        // [latent, frames]
        let all = Tensor::from_slice(latents, (frames, c), &Device::Cpu).map_err(err)?.t().map_err(err)?.contiguous().map_err(err)?;
        let windows = frames.div_ceil(window);
        for w in 0..windows {
            let core = w * window;
            let core_len = window.min(frames - core);
            let from = core.saturating_sub(overlap);
            let to = (core + core_len + overlap).min(frames);
            let z = all.narrow(1, from, to - from).map_err(err)?.contiguous().map_err(err)?.to_device(&self.dev).map_err(err)?;
            let y = self.pass(&z)?.narrow(1, (core - from) * hop, core_len * hop).map_err(err)?;
            let y = y.to_device(&Device::Cpu).map_err(err)?.to_dtype(DType::F32).map_err(err)?;
            left.extend(y.get(0).map_err(err)?.to_vec1::<f32>().map_err(err)?);
            right.extend(y.get(1).map_err(err)?.to_vec1::<f32>().map_err(err)?);
            if !progress(w + 1, windows) {
                return Err(err("cancelled"));
            }
        }
        Ok((left, right))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::config::{parse_vae, tests::tiny_vae};

    /// Random weights of a tiny VAE decoder written to `dir/vae.safetensors`.
    pub(crate) fn fake(dir: &Path) -> (std::path::PathBuf, VaeConfig) {
        let cfg = parse_vae(&tiny_vae()).unwrap();
        let mut rng = crate::model::Noise::new(55);
        let mut ts: Vec<(String, Tensor)> = Vec::new();
        let dev = Device::Cpu;
        let mut put = |name: String, dims: &[usize], scale: f32, offset: f32| {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = rng.take(n).into_iter().map(|x| x * scale + offset).collect();
            ts.push((name, Tensor::from_vec(v, dims, &dev).unwrap()));
        };
        let ch = cfg.channels;
        let mut mult = vec![1usize];
        mult.extend_from_slice(&cfg.multiples);
        let n = cfg.strides.len();
        let top = ch * mult[n];
        let conv = |put: &mut dyn FnMut(String, &[usize], f32, f32), p: &str, cin: usize, cout: usize, k: usize, bias: bool| {
            put(format!("{p}.weight_v"), &[cout, cin, k], 1.0, 0.0);
            put(format!("{p}.weight_g"), &[cout, 1, 1], 0.1, 0.5);
            if bias {
                put(format!("{p}.bias"), &[cout], 0.05, 0.0);
            }
        };
        let snake = |put: &mut dyn FnMut(String, &[usize], f32, f32), p: &str, c: usize| {
            put(format!("{p}.alpha"), &[1, c, 1], 0.2, 0.0);
            put(format!("{p}.beta"), &[1, c, 1], 0.2, 0.0);
        };
        conv(&mut put, "decoder.conv1", cfg.latent, top, 7, true);
        for (i, &s) in cfg.strides.iter().enumerate() {
            let cin = ch * mult[n - i];
            let cout = ch * mult[n - i - 1];
            let p = format!("decoder.block.{i}");
            snake(&mut put, &format!("{p}.snake1"), cin);
            put(format!("{p}.conv_t1.weight_v"), &[cin, cout, 2 * s], 1.0, 0.0);
            put(format!("{p}.conv_t1.weight_g"), &[cin, 1, 1], 0.1, 0.5);
            put(format!("{p}.conv_t1.bias"), &[cout], 0.05, 0.0);
            for u in 1..=3 {
                let q = format!("{p}.res_unit{u}");
                snake(&mut put, &format!("{q}.snake1"), cout);
                conv(&mut put, &format!("{q}.conv1"), cout, cout, 7, true);
                snake(&mut put, &format!("{q}.snake2"), cout);
                conv(&mut put, &format!("{q}.conv2"), cout, cout, 1, true);
            }
        }
        snake(&mut put, "decoder.snake1", ch);
        conv(&mut put, "decoder.conv2", ch, 2, 7, false);
        let p = dir.join("vae.safetensors");
        crate::model::weights::write(&p, &ts);
        (p, cfg)
    }

    #[test]
    fn convolutions_match_their_definitions() {
        let dev = Device::Cpu;
        let mut rng = crate::model::Noise::new(3);
        // dilated, padded convolution
        let (cin, cout, k, dil, t) = (3usize, 2usize, 7usize, 3usize, 20usize);
        let w: Vec<f32> = rng.take(cout * cin * k);
        let x: Vec<f32> = rng.take(cin * t);
        let wt = Tensor::from_vec(w.clone(), (cout, cin, k), &dev).unwrap();
        let taps = (0..k).map(|j| wt.narrow(2, j, 1).unwrap().squeeze(2).unwrap().contiguous().unwrap()).collect();
        let c = Conv { taps, bias: None, dil };
        let got = c.fwd(&Tensor::from_vec(x.clone(), (cin, t), &dev).unwrap()).unwrap().to_vec2::<f32>().unwrap();
        let pad = (k - 1) * dil / 2;
        for o in 0..cout {
            for i in 0..t {
                let mut want = 0f32;
                for ci in 0..cin {
                    for j in 0..k {
                        let src = (i + j * dil) as isize - pad as isize;
                        if src >= 0 && (src as usize) < t {
                            want += w[(o * cin + ci) * k + j] * x[ci * t + src as usize];
                        }
                    }
                }
                assert!((got[o][i] - want).abs() < 1e-5, "conv {o} {i}");
            }
        }
        // transposed convolution, kernel 2s, padding s/2
        for s in [2usize, 4, 6, 10] {
            let (cin, cout, l) = (3usize, 2usize, 5usize);
            let kk = 2 * s;
            let w: Vec<f32> = rng.take(cin * cout * kk);
            let x: Vec<f32> = rng.take(cin * l);
            let wt = Tensor::from_vec(w.clone(), (cin, cout, kk), &dev)
                .unwrap()
                .permute((1, 2, 0))
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((cout * kk, cin))
                .unwrap();
            let ct = ConvT { w: wt, bias: Tensor::zeros((cout, 1), DType::F32, &dev).unwrap(), cout, stride: s };
            let got = ct.fwd(&Tensor::from_vec(x.clone(), (cin, l), &dev).unwrap()).unwrap().to_vec2::<f32>().unwrap();
            assert_eq!(got[0].len(), l * s);
            let p = s / 2;
            for o in 0..cout {
                for n in 0..l * s {
                    let mut want = 0f32;
                    for t in 0..l {
                        for j in 0..kk {
                            if t * s + j == n + p {
                                for ci in 0..cin {
                                    want += w[(ci * cout + o) * kk + j] * x[ci * l + t];
                                }
                            }
                        }
                    }
                    assert!((got[o][n] - want).abs() < 1e-5, "stride {s}: {o} {n}");
                }
            }
        }
    }

    #[test]
    fn windowed_decoding_equals_one_pass() {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-vae-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let (p, cfg) = fake(&d);
        let vae = Vae::load(&p, &cfg, &Device::Cpu).unwrap();
        let frames = 45;
        let z = crate::model::Noise::new(8).take(frames * cfg.latent);
        let (l1, r1) = vae.decode_windows(&z, frames, frames, 0, &mut |_, _| true).unwrap();
        assert_eq!(l1.len(), frames * crate::request::HOP);
        let (l2, r2) = vae.decode_windows(&z, frames, 10, OVERLAP, &mut |_, _| true).unwrap();
        let diff = l1.iter().zip(&l2).chain(r1.iter().zip(&r2)).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let peak = l1.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(peak > 1e-3, "the fake decoder makes sound");
        assert!(diff <= 1e-5 * peak.max(1.0), "windows differ from one pass by {diff}");
        // too little context is visibly different (the overlap is what makes it exact)
        let (l3, _) = vae.decode_windows(&z, frames, 10, 0, &mut |_, _| true).unwrap();
        assert!(l1.iter().zip(&l3).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) > diff);
        // cancelling stops, bad lengths are errors
        assert!(vae.decode_windows(&z, frames, 10, OVERLAP, &mut |_, _| false).is_err());
        assert!(vae.decode(&z[..10], frames, &mut |_, _| true).is_err());
        assert!(vae.decode(&[], 0, &mut |_, _| true).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
