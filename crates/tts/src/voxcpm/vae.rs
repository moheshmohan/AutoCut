//! The AudioVAE: a causal convolutional autoencoder between waveforms and VoxCPM's 64-channel
//! latents (25 frames a second).
//!
//! Encoder: a 7-tap convolution, then one block per stride (three residual units — Snake, a
//! depthwise 7-tap convolution dilated 1, 3 and 9, Snake, a 1×1 convolution — then Snake and a
//! strided convolution doubling the channels), then a 3-tap convolution to the latent mean.
//! Decoder: a depthwise 7-tap and a 1×1 convolution, then one block per stride (Snake, a
//! transposed convolution halving the channels, three residual units), Snake, a 7-tap convolution
//! to one channel and tanh. VoxCPM2's decoder also scales and shifts each block's input by
//! learned per-channel values for the output-rate bucket. Every convolution pads on the left
//! only, so the networks are causal with a finite receptive field: [`Vae::encode`] and
//! [`Vae::decode`] run long audio in windows with enough left context that the kept output is
//! what one pass over everything would give, and memory stays bounded.
//!
//! Signals are channel-major `Vec<f32>`; the dense convolutions are matrix products on candle
//! (CPU), the depthwise ones and Snake are plain loops spread over threads with rayon.

use rayon::prelude::*;

use candle_core::{Device, Tensor};

use super::config::{Config, Variant};
use super::{R, err};
use crate::kokoro::weights::Weights;

/// A signal `[channels × frames]`, channel-major.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sig {
    pub c: usize,
    pub t: usize,
    pub x: Vec<f32>,
}

/// Most columns of an im2col / col2im matrix built at once (bounds the scratch memory).
const COLS: usize = 8192;

struct Conv {
    /// `[cout, cin · k]` (dense) or `[c, k]` (depthwise).
    w: Tensor,
    wv: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    dil: usize,
    left: usize,
    depthwise: bool,
}

fn vecf(w: &Weights, name: &str, n: usize) -> R<Vec<f32>> {
    let t = w.get(name)?;
    let v = t.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?;
    if v.len() != n {
        return Err(err(format!("weight `{name}` has {} values, expected {n}", v.len())));
    }
    Ok(v)
}

impl Conv {
    fn load(w: &Weights, name: &str, cin: usize, cout: usize, k: usize, stride: usize, dil: usize, left: usize, depthwise: bool) -> R<Conv> {
        let per = if depthwise { 1 } else { cin };
        let t = w.shaped(&format!("{name}.weight"), &[cout, per, k])?;
        let wv = t.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?;
        let w2 = t.reshape((cout, per * k)).map_err(err)?;
        let b = vecf(w, &format!("{name}.bias"), cout)?;
        if depthwise && cin != cout {
            return Err(err(format!("{name}: a depthwise convolution keeps its channels")));
        }
        Ok(Conv { w: w2, wv, b, cin, cout, k, stride, dil, left, depthwise })
    }

    fn fwd(&self, s: &Sig) -> R<Sig> {
        if s.c != self.cin {
            return Err(err(format!("convolution expects {} channels, got {}", self.cin, s.c)));
        }
        if self.depthwise {
            return Ok(self.depthwise(s));
        }
        let span = (self.k - 1) * self.dil;
        let padded = s.t + self.left;
        if padded < span + 1 {
            return Ok(Sig { c: self.cout, t: 0, x: Vec::new() });
        }
        let tout = (padded - span - 1) / self.stride + 1;
        let mut out = vec![0f32; self.cout * tout];
        let rows = self.cin * self.k;
        let mut t0 = 0;
        while t0 < tout {
            let n = COLS.min(tout - t0);
            let y = if self.k == 1 && self.stride == 1 && self.left == 0 {
                // a 1×1 convolution reads the signal as it is
                let mut cols = vec![0f32; self.cin * n];
                for c in 0..self.cin {
                    if let (Some(dst), Some(src)) = (cols.get_mut(c * n..(c + 1) * n), s.x.get(c * s.t + t0..c * s.t + t0 + n)) {
                        dst.copy_from_slice(src);
                    }
                }
                matmul(&self.w, cols, rows, n)?
            } else {
                let mut cols = vec![0f32; rows * n];
                cols.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
                    let (c, kk) = (r / self.k, r % self.k);
                    let src = s.x.get(c * s.t..(c + 1) * s.t).unwrap_or(&[]);
                    for (j, v) in row.iter_mut().enumerate() {
                        let i = ((t0 + j) * self.stride + kk * self.dil) as isize - self.left as isize;
                        *v = if i >= 0 { src.get(i as usize).copied().unwrap_or(0.0) } else { 0.0 };
                    }
                });
                matmul(&self.w, cols, rows, n)?
            };
            for co in 0..self.cout {
                let bias = self.b.get(co).copied().unwrap_or(0.0);
                if let (Some(dst), Some(src)) = (out.get_mut(co * tout + t0..co * tout + t0 + n), y.get(co * n..(co + 1) * n)) {
                    for (d, v) in dst.iter_mut().zip(src) {
                        *d = v + bias;
                    }
                }
            }
            t0 += n;
        }
        Ok(Sig { c: self.cout, t: tout, x: out })
    }

    /// Stride 1, `left = (k − 1) · dil`: same length.
    fn depthwise(&self, s: &Sig) -> Sig {
        let t = s.t;
        let mut out = vec![0f32; s.c * t];
        out.par_chunks_mut(t.max(1)).enumerate().for_each(|(c, dst)| {
            let src = s.x.get(c * t..(c + 1) * t).unwrap_or(&[]);
            let w = self.wv.get(c * self.k..(c + 1) * self.k).unwrap_or(&[]);
            let b = self.b.get(c).copied().unwrap_or(0.0);
            for (i, d) in dst.iter_mut().enumerate() {
                let mut acc = b;
                for (kk, wk) in w.iter().enumerate() {
                    let j = (i + kk * self.dil) as isize - self.left as isize;
                    if j >= 0
                        && let Some(v) = src.get(j as usize)
                    {
                        acc += wk * v;
                    }
                }
                *d = acc;
            }
        });
        Sig { c: s.c, t, x: out }
    }
}

/// `w` `[m, k]` times `x` `[k, n]` (row-major) → `[m, n]`.
fn matmul(w: &Tensor, x: Vec<f32>, k: usize, n: usize) -> R<Vec<f32>> {
    let xt = Tensor::from_vec(x, (k, n), &Device::Cpu).map_err(err)?;
    w.matmul(&xt).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)
}

/// Snake: `x + sin²(αx) / (α + 10⁻⁹)` per channel, in place.
fn snake(s: &mut Sig, alpha: &[f32]) {
    let t = s.t.max(1);
    s.x.par_chunks_mut(t).zip(alpha.par_iter()).for_each(|(row, a)| {
        let inv = 1.0 / (a + 1e-9);
        for v in row.iter_mut() {
            let q = (a * *v).sin();
            *v += inv * q * q;
        }
    });
}

/// Transposed convolution with kernel `2 · stride`, no padding, trimmed at the end to
/// `frames · stride` samples (the causal transposed convolution).
struct ConvT {
    /// `[cout · k, cin]`.
    w: Tensor,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
}

impl ConvT {
    fn load(w: &Weights, name: &str, cin: usize, cout: usize, stride: usize) -> R<ConvT> {
        let k = 2 * stride;
        let t = w.shaped(&format!("{name}.weight"), &[cin, cout, k])?;
        let wt = t.reshape((cin, cout * k)).map_err(err)?.t().map_err(err)?.contiguous().map_err(err)?;
        Ok(ConvT { w: wt, b: vecf(w, &format!("{name}.bias"), cout)?, cin, cout, k, stride })
    }

    fn fwd(&self, s: &Sig) -> R<Sig> {
        if s.c != self.cin {
            return Err(err(format!("transposed convolution expects {} channels, got {}", self.cin, s.c)));
        }
        let tout = s.t * self.stride;
        let mut out = vec![0f32; self.cout * tout];
        let mut t0 = 0;
        while t0 < s.t {
            let n = COLS.min(s.t - t0);
            let mut cols = vec![0f32; self.cin * n];
            for c in 0..self.cin {
                if let (Some(dst), Some(src)) = (cols.get_mut(c * n..(c + 1) * n), s.x.get(c * s.t + t0..c * s.t + t0 + n)) {
                    dst.copy_from_slice(src);
                }
            }
            let y = matmul(&self.w, cols, self.cin, n)?; // [cout · k, n]
            out.par_chunks_mut(tout.max(1)).enumerate().for_each(|(co, dst)| {
                for kk in 0..self.k {
                    let Some(row) = y.get((co * self.k + kk) * n..(co * self.k + kk + 1) * n) else { continue };
                    for (j, v) in row.iter().enumerate() {
                        if let Some(d) = dst.get_mut((t0 + j) * self.stride + kk) {
                            *d += v;
                        }
                    }
                }
            });
            t0 += n;
        }
        for co in 0..self.cout {
            let bias = self.b.get(co).copied().unwrap_or(0.0);
            if let Some(row) = out.get_mut(co * tout..(co + 1) * tout) {
                row.iter_mut().for_each(|v| *v += bias);
            }
        }
        Ok(Sig { c: self.cout, t: tout, x: out })
    }
}

struct Residual {
    a1: Vec<f32>,
    dw: Conv,
    a2: Vec<f32>,
    pw: Conv,
}

impl Residual {
    fn load(w: &Weights, name: &str, c: usize, dil: usize) -> R<Residual> {
        Ok(Residual {
            a1: vecf(w, &format!("{name}.block.0.alpha"), c)?,
            dw: Conv::load(w, &format!("{name}.block.1"), c, c, 7, 1, dil, 6 * dil, true)?,
            a2: vecf(w, &format!("{name}.block.2.alpha"), c)?,
            pw: Conv::load(w, &format!("{name}.block.3"), c, c, 1, 1, 1, 0, false)?,
        })
    }
    fn fwd(&self, s: &Sig) -> R<Sig> {
        let mut y = s.clone();
        snake(&mut y, &self.a1);
        let mut y = self.dw.fwd(&y)?;
        snake(&mut y, &self.a2);
        let mut y = self.pw.fwd(&y)?;
        y.x.par_iter_mut().zip(s.x.par_iter()).for_each(|(a, b)| *a += b);
        Ok(y)
    }
}

struct EncBlock {
    res: Vec<Residual>,
    alpha: Vec<f32>,
    down: Conv,
}

struct DecBlock {
    cond: Option<(Vec<f32>, Vec<f32>)>,
    alpha: Vec<f32>,
    up: ConvT,
    res: Vec<Residual>,
}

pub(crate) struct Vae {
    enc_first: Conv,
    enc: Vec<EncBlock>,
    mu: Conv,
    dec_dw: Conv,
    dec_pw: Conv,
    dec: Vec<DecBlock>,
    out_alpha: Vec<f32>,
    out: Conv,
    pub enc_hop: usize,
    pub dec_hop: usize,
    pub latent: usize,
    /// Left context of a window, latent frames (above the receptive fields).
    enc_ctx: usize,
    dec_ctx: usize,
}

/// Frames produced per window.
const WINDOW: usize = 64;

/// How far back (input samples) the encoder's output frame reaches, an upper bound.
fn encoder_reach(rates: &[usize], v2: bool) -> usize {
    let mut reach = 6; // the first 7-tap convolution
    let mut d = 1usize;
    for s in rates {
        let left = 2 * s.div_ceil(2) - if v2 { s % 2 } else { 0 };
        reach += 78 * d + (2 * s + left) * d; // three dilated residual units, the strided convolution
        d *= s;
    }
    reach + 2 * d // the 3-tap convolution to the mean
}

/// How far back (latent frames) a decoded sample reaches, an upper bound.
fn decoder_reach(rates: &[usize]) -> f64 {
    let mut reach = 6.0; // the depthwise 7-tap convolution
    let mut m = 1.0;
    for s in rates {
        let s = *s as f64;
        reach += 2.0 / m + 78.0 / (m * s); // the transposed convolution, three residual units
        m *= s;
    }
    reach + 6.0 / m
}

impl Vae {
    pub(crate) fn load(w: &Weights, cfg: &Config) -> R<Vae> {
        let v = &cfg.vae;
        let v2 = cfg.variant == Variant::V2;
        let mut d = v.encoder_dim;
        let enc_first = Conv::load(w, "encoder.block.0", 1, d, 7, 1, 1, 6, false)?;
        let mut enc = Vec::new();
        for (i, s) in v.encoder_rates.iter().enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            let res = [1, 3, 9].iter().enumerate().map(|(j, dil)| Residual::load(w, &format!("{p}.block.{j}"), d, *dil)).collect::<R<Vec<_>>>()?;
            let alpha = vecf(w, &format!("{p}.block.3.alpha"), d)?;
            // the causal padding of a strided convolution: 2⌈s/2⌉, less s mod 2 in VoxCPM2
            let left = 2 * s.div_ceil(2) - if v2 { s % 2 } else { 0 };
            let down = Conv::load(w, &format!("{p}.block.4"), d, 2 * d, 2 * s, *s, 1, left, false)?;
            enc.push(EncBlock { res, alpha, down });
            d *= 2;
        }
        let mu = Conv::load(w, "encoder.fc_mu", d, v.latent_dim, 3, 1, 1, 2, false)?;
        let l = v.latent_dim;
        let dec_dw = Conv::load(w, "decoder.model.0", l, l, 7, 1, 1, 6, true)?;
        let dec_pw = Conv::load(w, "decoder.model.1", l, v.decoder_dim, 1, 1, 1, 0, false)?;
        let bucket = cfg.rate_bucket();
        let buckets = cfg.rate_buckets();
        let mut c = v.decoder_dim;
        let mut dec = Vec::new();
        for (i, s) in v.decoder_rates.iter().enumerate() {
            let idx = i + 2;
            let p = format!("decoder.model.{idx}");
            let cond = match bucket {
                Some(b) => {
                    let scale = w.shaped(&format!("decoder.sr_cond_model.{idx}.scale_embed.weight"), &[buckets, c])?;
                    let bias = w.shaped(&format!("decoder.sr_cond_model.{idx}.bias_embed.weight"), &[buckets, c])?;
                    let row = |t: Tensor| -> R<Vec<f32>> { t.get(b).map_err(err)?.to_vec1::<f32>().map_err(err) };
                    Some((row(scale)?, row(bias)?))
                }
                None => None,
            };
            let alpha = vecf(w, &format!("{p}.block.0.alpha"), c)?;
            let up = ConvT::load(w, &format!("{p}.block.1"), c, c / 2, *s)?;
            let res = [1, 3, 9].iter().enumerate().map(|(j, dil)| Residual::load(w, &format!("{p}.block.{}", j + 2), c / 2, *dil)).collect::<R<Vec<_>>>()?;
            dec.push(DecBlock { cond, alpha, up, res });
            c /= 2;
        }
        let n = v.decoder_rates.len();
        let out_alpha = vecf(w, &format!("decoder.model.{}.alpha", n + 2), c)?;
        let out = Conv::load(w, &format!("decoder.model.{}", n + 3), c, 1, 7, 1, 1, 6, false)?;
        let enc_hop = cfg.encode_hop();
        let enc_ctx = encoder_reach(&v.encoder_rates, v2).div_ceil(enc_hop.max(1)) + 2;
        let dec_ctx = decoder_reach(&v.decoder_rates).ceil() as usize + 2;
        Ok(Vae { enc_first, enc, mu, dec_dw, dec_pw, dec, out_alpha, out, enc_hop, dec_hop: cfg.decode_hop(), latent: l, enc_ctx, dec_ctx })
    }

    fn encode_once(&self, audio: &[f32]) -> R<Sig> {
        let mut s = self.enc_first.fwd(&Sig { c: 1, t: audio.len(), x: audio.to_vec() })?;
        for b in &self.enc {
            for r in &b.res {
                s = r.fwd(&s)?;
            }
            snake(&mut s, &b.alpha);
            s = b.down.fwd(&s)?;
        }
        self.mu.fwd(&s)
    }

    fn decode_once(&self, z: &Sig) -> R<Vec<f32>> {
        let mut s = self.dec_pw.fwd(&self.dec_dw.fwd(z)?)?;
        for b in &self.dec {
            if let Some((scale, bias)) = &b.cond {
                let t = s.t.max(1);
                s.x.par_chunks_mut(t).zip(scale.par_iter().zip(bias.par_iter())).for_each(|(row, (a, c))| row.iter_mut().for_each(|v| *v = *v * a + c));
            }
            snake(&mut s, &b.alpha);
            s = b.up.fwd(&s)?;
            for r in &b.res {
                s = r.fwd(&s)?;
            }
        }
        snake(&mut s, &self.out_alpha);
        let y = self.out.fwd(&s)?;
        Ok(y.x.into_iter().map(f32::tanh).collect())
    }

    /// Latents `[latent × frames]` of `audio` (at the encoder's rate), right-padded with silence
    /// to whole frames.
    pub(crate) fn encode(&self, audio: &[f32]) -> R<Sig> {
        let hop = self.enc_hop;
        let frames = audio.len().div_ceil(hop);
        let mut padded = audio.to_vec();
        padded.resize(frames * hop, 0.0);
        let mut out = vec![0f32; self.latent * frames];
        let mut f0 = 0;
        while f0 < frames {
            let n = WINDOW.min(frames - f0);
            let ctx = self.enc_ctx.min(f0);
            let piece = padded.get((f0 - ctx) * hop..(f0 + n) * hop).ok_or_else(|| err("encoder window out of range"))?;
            let z = self.encode_once(piece)?;
            if z.t != ctx + n || z.c != self.latent {
                return Err(err("the audio encoder produced an unexpected length"));
            }
            for c in 0..self.latent {
                if let (Some(dst), Some(src)) = (out.get_mut(c * frames + f0..c * frames + f0 + n), z.x.get(c * z.t + ctx..c * z.t + ctx + n)) {
                    dst.copy_from_slice(src);
                }
            }
            f0 += n;
        }
        Ok(Sig { c: self.latent, t: frames, x: out })
    }

    /// The waveform (at the output rate) of latents `[latent × frames]`.
    pub(crate) fn decode(&self, z: &Sig) -> R<Vec<f32>> {
        if z.c != self.latent {
            return Err(err("latents have the wrong number of channels"));
        }
        let hop = self.dec_hop;
        let mut out = Vec::with_capacity(z.t * hop);
        let mut f0 = 0;
        while f0 < z.t {
            let n = WINDOW.min(z.t - f0);
            let ctx = self.dec_ctx.min(f0);
            let w = ctx + n;
            let mut x = vec![0f32; self.latent * w];
            for c in 0..self.latent {
                if let (Some(dst), Some(src)) = (x.get_mut(c * w..(c + 1) * w), z.x.get(c * z.t + f0 - ctx..c * z.t + f0 + n)) {
                    dst.copy_from_slice(src);
                }
            }
            let y = self.decode_once(&Sig { c: self.latent, t: w, x })?;
            if y.len() != w * hop {
                return Err(err("the audio decoder produced an unexpected length"));
            }
            out.extend_from_slice(y.get(ctx * hop..).unwrap_or(&[]));
            f0 += n;
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Random AudioVAE weights for `cfg` (already weight-norm-folded names), like the checkpoint's.
    pub(crate) fn fake(cfg: &Config, seed: u64) -> HashMap<String, Tensor> {
        let mut rng = crate::voxcpm::Noise::new(seed);
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        let mut t = |name: String, dims: &[usize], scale: f32, offset: f32| {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = rng.take(n).into_iter().map(|x| x * scale + offset).collect();
            m.insert(name, Tensor::from_vec(v, dims, &dev).unwrap());
        };
        let v = &cfg.vae;
        let conv = |t: &mut dyn FnMut(String, &[usize], f32, f32), name: &str, cout: usize, per: usize, k: usize| {
            t(format!("{name}.weight"), &[cout, per, k], 1.0 / ((per * k) as f32).sqrt(), 0.0);
            t(format!("{name}.bias"), &[cout], 0.01, 0.0);
        };
        let mut d = v.encoder_dim;
        conv(&mut t, "encoder.block.0", d, 1, 7);
        for (i, s) in v.encoder_rates.iter().enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            for j in 0..3 {
                t(format!("{p}.block.{j}.block.0.alpha"), &[1, d, 1], 0.1, 1.0);
                conv(&mut t, &format!("{p}.block.{j}.block.1"), d, 1, 7);
                t(format!("{p}.block.{j}.block.2.alpha"), &[1, d, 1], 0.1, 1.0);
                conv(&mut t, &format!("{p}.block.{j}.block.3"), d, d, 1);
            }
            t(format!("{p}.block.3.alpha"), &[1, d, 1], 0.1, 1.0);
            conv(&mut t, &format!("{p}.block.4"), 2 * d, d, 2 * s);
            d *= 2;
        }
        conv(&mut t, "encoder.fc_mu", v.latent_dim, d, 3);
        let l = v.latent_dim;
        conv(&mut t, "decoder.model.0", l, 1, 7);
        conv(&mut t, "decoder.model.1", v.decoder_dim, l, 1);
        let mut c = v.decoder_dim;
        for (i, s) in v.decoder_rates.iter().enumerate() {
            let p = format!("decoder.model.{}", i + 2);
            if cfg.variant == Variant::V2 {
                t(format!("decoder.sr_cond_model.{}.scale_embed.weight", i + 2), &[cfg.rate_buckets(), c], 0.1, 1.0);
                t(format!("decoder.sr_cond_model.{}.bias_embed.weight", i + 2), &[cfg.rate_buckets(), c], 0.1, 0.0);
            }
            t(format!("{p}.block.0.alpha"), &[1, c, 1], 0.1, 1.0);
            t(format!("{p}.block.1.weight"), &[c, c / 2, 2 * s], 1.0 / (c as f32).sqrt(), 0.0);
            t(format!("{p}.block.1.bias"), &[c / 2], 0.01, 0.0);
            for j in 0..3 {
                let q = format!("{p}.block.{}", j + 2);
                t(format!("{q}.block.0.alpha"), &[1, c / 2, 1], 0.1, 1.0);
                conv(&mut t, &format!("{q}.block.1"), c / 2, 1, 7);
                t(format!("{q}.block.2.alpha"), &[1, c / 2, 1], 0.1, 1.0);
                conv(&mut t, &format!("{q}.block.3"), c / 2, c / 2, 1);
            }
            c /= 2;
        }
        let n = v.decoder_rates.len();
        t(format!("decoder.model.{}.alpha", n + 2), &[1, c, 1], 0.1, 1.0);
        conv(&mut t, &format!("decoder.model.{}", n + 3), 1, c, 7);
        m
    }

    pub(crate) fn tiny(v2: bool) -> (Config, Vae) {
        let cfg = crate::voxcpm::config::parse(crate::voxcpm::config::tests::tiny(v2).as_bytes()).unwrap();
        let w = Weights::from_map(fake(&cfg, 9));
        let vae = Vae::load(&w, &cfg).unwrap();
        (cfg, vae)
    }

    #[test]
    fn convolutions_match_direct_sums() {
        let dev = Device::Cpu;
        // a strided causal conv: cin 2, cout 3, k 4, stride 2, left 2
        let w: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
        let b = vec![0.1f32, -0.2, 0.3];
        let mut ws = Weights::from_map(HashMap::new());
        ws.insert("c.weight", Tensor::from_vec(w.clone(), (3, 2, 4), &dev).unwrap());
        ws.insert("c.bias", Tensor::from_vec(b.clone(), 3, &dev).unwrap());
        let conv = Conv::load(&ws, "c", 2, 3, 4, 2, 1, 2, false).unwrap();
        let x: Vec<f32> = (0..20).map(|i| (i as f32 * 0.21).cos()).collect();
        let s = Sig { c: 2, t: 10, x: x.clone() };
        let y = conv.fwd(&s).unwrap();
        assert_eq!(y.t, (10 + 2 - 4) / 2 + 1);
        for co in 0..3 {
            for t in 0..y.t {
                let mut acc = b[co];
                for ci in 0..2 {
                    for k in 0..4 {
                        let i = (t * 2 + k) as isize - 2;
                        if (0..10).contains(&i) {
                            acc += w[(co * 2 + ci) * 4 + k] * x[ci * 10 + i as usize];
                        }
                    }
                }
                assert!((y.x[co * y.t + t] - acc).abs() < 1e-5);
            }
        }
        // transposed: cin 2, cout 1, stride 3 (k 6) → 3 samples per frame, causal
        ws.insert("u.weight", Tensor::from_vec((0..12).map(|i| i as f32 * 0.1).collect::<Vec<f32>>(), (2, 1, 6), &dev).unwrap());
        ws.insert("u.bias", Tensor::from_vec(vec![0.5f32], 1, &dev).unwrap());
        let up = ConvT::load(&ws, "u", 2, 1, 3).unwrap();
        let y = up.fwd(&s).unwrap();
        assert_eq!(y.t, 30);
        for o in 0..30 {
            let mut acc = 0.5;
            for t in 0..10 {
                for k in 0..6 {
                    if t * 3 + k == o {
                        acc += (0..2).map(|ci| (ci * 6 + k) as f32 * 0.1 * x[ci * 10 + t]).sum::<f32>();
                    }
                }
            }
            assert!((y.x[o] - acc).abs() < 1e-4, "{o}");
        }
    }

    #[test]
    fn windowed_runs_match_one_pass_and_are_causal() {
        for v2 in [false, true] {
            let (cfg, vae) = tiny(v2);
            // long enough for several windows
            let frames = WINDOW * 2 + 17;
            let audio: Vec<f32> = (0..frames * vae.enc_hop).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
            let z = vae.encode(&audio).unwrap();
            let once = vae.encode_once(&audio).unwrap();
            assert_eq!((z.c, z.t), (cfg.feat, frames));
            assert_eq!(once.t, frames);
            let d = z.x.iter().zip(&once.x).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(d < 1e-4, "encode windows differ by {d} (v2 {v2})");
            let y = vae.decode(&z).unwrap();
            let y1 = vae.decode_once(&z).unwrap();
            assert_eq!(y.len(), frames * vae.dec_hop);
            let d = y.iter().zip(&y1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(d < 1e-4, "decode windows differ by {d} (v2 {v2})");
            assert!(y.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
            // causal: changing the last frame leaves earlier samples alone
            let mut z2 = z.clone();
            for c in 0..z2.c {
                z2.x[c * z2.t + frames - 1] += 1.0;
            }
            let y2 = vae.decode(&z2).unwrap();
            let keep = (frames - 1) * vae.dec_hop;
            assert_eq!(&y2[..keep], &y[..keep]);
            assert_ne!(&y2[keep..], &y[keep..]);
        }
    }

    #[test]
    fn receptive_fields_of_the_released_models() {
        // VoxCPM1.5: 44.1 kHz, rates 2·3·6·7·7 (hop 1764); VoxCPM2: 16 kHz, 2·5·8·8 (hop 640)
        assert!(encoder_reach(&[2, 3, 6, 7, 7], false).div_ceil(1764) <= 24);
        assert!(encoder_reach(&[2, 5, 8, 8], true).div_ceil(640) <= 24);
        assert!(decoder_reach(&[7, 7, 6, 3, 2]) < 24.0);
        assert!(decoder_reach(&[8, 6, 5, 2, 2, 2]) < 24.0);
    }

    #[test]
    fn empty_and_short_inputs() {
        let (_, vae) = tiny(false);
        let z = vae.encode(&[]).unwrap();
        assert_eq!(z.t, 0);
        assert!(vae.decode(&z).unwrap().is_empty());
        let z = vae.encode(&[0.1, 0.2]).unwrap();
        assert_eq!(z.t, 1, "padded to one frame");
        assert!(vae.decode(&Sig { c: 3, t: 1, x: vec![0.0; 3] }).is_err(), "wrong channels");
    }
}
