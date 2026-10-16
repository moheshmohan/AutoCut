//! The OpenVoice V2 tone-colour converter: a VITS synthesizer without a text encoder. Shapes are
//! `[1, channels, time]` unless noted; everything runs in f32 on the CPU.
//!
//! - `ref_enc`: linear spectrogram → layer norm → six 3×3/2 convolutions → GRU → 256-d tone
//!   colour (the speaker embedding).
//! - `enc_q`: posterior encoder (WaveNet stack) spectrogram → latent `z` (192 channels).
//! - `flow`: four mean-only affine couplings with channel flips, conditioned on the tone colour;
//!   run forward with the source colour and backward with the target colour.
//! - `dec`: HiFi-GAN generator, ×256 up-sampling (8·8·2·2) to 22.05 kHz.
//!
//! `zero_g` (the V2 config): the encoder and decoder see a zero speaker vector, so the tone
//! colour lives only in the flow.

use candle_core::{Device, Tensor};

use crate::TtsError;
use crate::kokoro::layers::{Conv, ConvT, LayerNorm, Linear, R, leaky};
use crate::kokoro::weights::{Weights, err};

/// Spectrogram bins (n_fft 1024).
pub(crate) const SPEC: usize = 513;
/// Latent channels.
const INTER: usize = 192;
/// Tone-colour (speaker embedding) size.
pub(crate) const GIN: usize = 256;
/// Samples per spectrogram frame (the decoder's total up-sampling).
pub(crate) const HOP: usize = 256;

fn sigmoid(x: &Tensor) -> R<Tensor> {
    candle_nn::ops::sigmoid(x).map_err(err)
}

/// WaveNet-style stack with gated tanh·sigmoid units and a global condition (dilation 1).
struct Wn {
    in_layers: Vec<Conv>,
    res_skip: Vec<Conv>,
    cond: Conv,
    hidden: usize,
}

impl Wn {
    fn load(wt: &Weights, name: &str, layers: usize, kernel: usize, hidden: usize) -> R<Wn> {
        let pad = (kernel - 1) / 2;
        let mut in_layers = Vec::with_capacity(layers);
        let mut res_skip = Vec::with_capacity(layers);
        for i in 0..layers {
            in_layers.push(Conv::load(wt, &format!("{name}.in_layers.{i}"), pad, 1, 1, 1)?);
            res_skip.push(Conv::load(wt, &format!("{name}.res_skip_layers.{i}"), 0, 1, 1, 1)?);
        }
        Ok(Wn { in_layers, res_skip, cond: Conv::load(wt, &format!("{name}.cond_layer"), 0, 1, 1, 1)?, hidden })
    }

    fn fwd(&self, x: &Tensor, g: &Tensor) -> R<Tensor> {
        let h = self.hidden;
        let gc = self.cond.fwd(g)?; // [1, 2·h·layers, 1]
        let mut x = x.clone();
        let mut out = x.zeros_like().map_err(err)?;
        let n = self.in_layers.len();
        for (i, (inl, rs)) in self.in_layers.iter().zip(&self.res_skip).enumerate() {
            let x_in = inl.fwd(&x)?;
            let g_l = gc.narrow(1, i * 2 * h, 2 * h).map_err(err)?;
            let a = x_in.broadcast_add(&g_l).map_err(err)?;
            let t = a.narrow(1, 0, h).map_err(err)?.tanh().map_err(err)?;
            let s = sigmoid(&a.narrow(1, h, h).map_err(err)?)?;
            let acts = (t * s).map_err(err)?;
            let r = rs.fwd(&acts)?;
            if i + 1 < n {
                x = (x + r.narrow(1, 0, h).map_err(err)?).map_err(err)?;
                out = (out + r.narrow(1, h, h).map_err(err)?).map_err(err)?;
            } else {
                out = (out + r).map_err(err)?;
            }
        }
        Ok(out)
    }
}

/// Spectrogram → latent: `z = m + ε·τ·exp(logs)`.
struct Posterior {
    pre: Conv,
    enc: Wn,
    proj: Conv,
}

impl Posterior {
    fn load(wt: &Weights) -> R<Posterior> {
        Ok(Posterior {
            pre: Conv::load(wt, "enc_q.pre", 0, 1, 1, 1)?,
            enc: Wn::load(wt, "enc_q.enc", 16, 5, INTER)?,
            proj: Conv::load(wt, "enc_q.proj", 0, 1, 1, 1)?,
        })
    }

    fn fwd(&self, spec: &Tensor, g: &Tensor, tau: f64, noise: &Tensor) -> R<Tensor> {
        let x = self.pre.fwd(spec)?;
        let x = self.enc.fwd(&x, g)?;
        let stats = self.proj.fwd(&x)?;
        let m = stats.narrow(1, 0, INTER).map_err(err)?;
        let logs = stats.narrow(1, INTER, INTER).map_err(err)?;
        let spread = (logs.exp().map_err(err)? * tau).map_err(err)?;
        (m + (noise * spread).map_err(err)?).map_err(err)
    }
}

/// One mean-only affine coupling: the first half conditions a shift of the second.
struct Coupling {
    pre: Conv,
    enc: Wn,
    post: Conv,
}

impl Coupling {
    fn load(wt: &Weights, k: usize) -> R<Coupling> {
        let n = format!("flow.flows.{k}");
        Ok(Coupling {
            pre: Conv::load(wt, &format!("{n}.pre"), 0, 1, 1, 1)?,
            enc: Wn::load(wt, &format!("{n}.enc"), 4, 5, INTER)?,
            post: Conv::load(wt, &format!("{n}.post"), 0, 1, 1, 1)?,
        })
    }

    fn fwd(&self, x: &Tensor, g: &Tensor, reverse: bool) -> R<Tensor> {
        let half = INTER / 2;
        let x0 = x.narrow(1, 0, half).map_err(err)?;
        let x1 = x.narrow(1, half, half).map_err(err)?;
        let h = self.pre.fwd(&x0)?;
        let h = self.enc.fwd(&h, g)?;
        let m = self.post.fwd(&h)?;
        let x1 = if reverse { (x1 - m).map_err(err)? } else { (m + x1).map_err(err)? };
        Tensor::cat(&[&x0, &x1], 1).map_err(err)
    }
}

/// Four couplings, each followed by a channel flip (the checkpoint's flows 0, 2, 4, 6).
struct Flow {
    layers: Vec<Coupling>,
    flip: Tensor,
}

impl Flow {
    fn load(wt: &Weights, dev: &Device) -> R<Flow> {
        let layers = [0, 2, 4, 6].iter().map(|k| Coupling::load(wt, *k)).collect::<R<Vec<_>>>()?;
        let idx: Vec<u32> = (0..INTER as u32).rev().collect();
        Ok(Flow { layers, flip: Tensor::from_vec(idx, INTER, dev).map_err(err)? })
    }

    fn flip(&self, x: &Tensor) -> R<Tensor> {
        x.index_select(&self.flip, 1).map_err(err)
    }

    fn fwd(&self, x: &Tensor, g: &Tensor) -> R<Tensor> {
        let mut x = x.clone();
        for c in &self.layers {
            x = c.fwd(&x, g, false)?;
            x = self.flip(&x)?;
        }
        Ok(x)
    }

    fn rev(&self, x: &Tensor, g: &Tensor) -> R<Tensor> {
        let mut x = x.clone();
        for c in self.layers.iter().rev() {
            x = self.flip(&x)?;
            x = c.fwd(&x, g, true)?;
        }
        Ok(x)
    }
}

/// HiFi-GAN residual block (type 1): dilated conv pairs with leaky ReLU and skip.
struct ResBlock {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
}

impl ResBlock {
    fn load(wt: &Weights, name: &str, kernel: usize) -> R<ResBlock> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        for (m, d) in [1usize, 3, 5].iter().enumerate() {
            convs1.push(Conv::load(wt, &format!("{name}.convs1.{m}"), (kernel * d - d) / 2, 1, *d, 1)?);
            convs2.push(Conv::load(wt, &format!("{name}.convs2.{m}"), (kernel - 1) / 2, 1, 1, 1)?);
        }
        Ok(ResBlock { convs1, convs2 })
    }

    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let mut x = x.clone();
        for (c1, c2) in self.convs1.iter().zip(&self.convs2) {
            let t = c1.fwd(&leaky(&x, 0.1)?)?;
            let t = c2.fwd(&leaky(&t, 0.1)?)?;
            x = (t + x).map_err(err)?;
        }
        Ok(x)
    }
}

struct Generator {
    pre: Conv,
    cond: Conv,
    ups: Vec<ConvT>,
    res: Vec<ResBlock>,
    post: Conv,
}

const UP_RATES: [usize; 4] = [8, 8, 2, 2];
const UP_KERNELS: [usize; 4] = [16, 16, 4, 4];
const RES_KERNELS: [usize; 3] = [3, 7, 11];

impl Generator {
    fn load(wt: &Weights) -> R<Generator> {
        let mut ups = Vec::new();
        let mut res = Vec::new();
        for (i, (u, k)) in UP_RATES.iter().zip(UP_KERNELS).enumerate() {
            ups.push(ConvT::load(wt, &format!("dec.ups.{i}"), (k - u) / 2, 0, *u, 1)?);
            for (j, rk) in RES_KERNELS.iter().enumerate() {
                res.push(ResBlock::load(wt, &format!("dec.resblocks.{}", i * RES_KERNELS.len() + j), *rk)?);
            }
        }
        Ok(Generator {
            pre: Conv::load(wt, "dec.conv_pre", 3, 1, 1, 1)?,
            cond: Conv::load(wt, "dec.cond", 0, 1, 1, 1)?,
            ups,
            res,
            post: Conv::load(wt, "dec.conv_post", 3, 1, 1, 1)?,
        })
    }

    fn fwd(&self, z: &Tensor, g: &Tensor) -> R<Tensor> {
        let mut x = self.pre.fwd(z)?.broadcast_add(&self.cond.fwd(g)?).map_err(err)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = up.fwd(&leaky(&x, 0.1)?)?;
            let mut acc: Option<Tensor> = None;
            for rb in self.res.iter().skip(i * RES_KERNELS.len()).take(RES_KERNELS.len()) {
                let y = rb.fwd(&x)?;
                acc = Some(match acc {
                    Some(a) => (a + y).map_err(err)?,
                    None => y,
                });
            }
            x = (acc.ok_or_else(|| err("generator has no residual blocks"))? / RES_KERNELS.len() as f64).map_err(err)?;
        }
        // VITS: the last activation is torch's default leaky slope
        let x = self.post.fwd(&leaky(&x, 0.01)?)?;
        x.tanh().map_err(err)
    }
}

/// Spectrogram frames → tone colour `[1, 256]`.
struct RefEnc {
    norm: LayerNorm,
    convs: Vec<(Tensor, Tensor)>,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    proj: Linear,
}

/// GRU hidden size.
const GRU: usize = 128;

impl RefEnc {
    fn load(wt: &Weights) -> R<RefEnc> {
        let mut convs = Vec::new();
        for i in 0..6 {
            let w = wt.get(&format!("ref_enc.convs.{i}.weight"))?;
            if w.rank() != 4 {
                return Err(err("ref_enc: conv weights must be 4-D"));
            }
            convs.push((w, wt.get(&format!("ref_enc.convs.{i}.bias"))?));
        }
        Ok(RefEnc {
            norm: LayerNorm::load(wt, "ref_enc.layernorm.weight", "ref_enc.layernorm.bias", 1e-5)?,
            convs,
            w_ih: wt.shaped("ref_enc.gru.weight_ih_l0", &[3 * GRU, 1152])?.t().map_err(err)?,
            w_hh: wt.shaped("ref_enc.gru.weight_hh_l0", &[3 * GRU, GRU])?.t().map_err(err)?,
            b_ih: wt.shaped("ref_enc.gru.bias_ih_l0", &[3 * GRU])?,
            b_hh: wt.shaped("ref_enc.gru.bias_hh_l0", &[3 * GRU])?,
            proj: Linear::load(wt, "ref_enc.proj")?,
        })
    }

    /// `spec`: `[1, 513, T]`.
    fn fwd(&self, spec: &Tensor) -> R<Tensor> {
        let x = spec.transpose(1, 2).map_err(err)?; // [1, T, 513]
        let x = self.norm.fwd(&x)?;
        let mut x = x.unsqueeze(1).map_err(err)?; // [1, 1, T, 513]
        for (w, b) in &self.convs {
            let c = b.dims1().map_err(err)?;
            x = x.conv2d(w, 1, 2, 1, 1).map_err(err)?.broadcast_add(&b.reshape((1, c, 1, 1)).map_err(err)?).map_err(err)?.relu().map_err(err)?;
        }
        // [1, 128, T', F'] → [T', 128·F']
        let (_, c, t, f) = x.dims4().map_err(err)?;
        if t == 0 {
            return Err(TtsError::Invalid("the reference audio is too short".into()));
        }
        let seq = x.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?.reshape((t, c * f)).map_err(err)?;
        let gi = seq.matmul(&self.w_ih).map_err(err)?.broadcast_add(&self.b_ih).map_err(err)?; // [T', 384]
        let mut h = Tensor::zeros((1, GRU), candle_core::DType::F32, spec.device()).map_err(err)?;
        for step in 0..t {
            let gi_t = gi.narrow(0, step, 1).map_err(err)?;
            let gh = h.matmul(&self.w_hh).map_err(err)?.broadcast_add(&self.b_hh).map_err(err)?;
            let r = sigmoid(&(gi_t.narrow(1, 0, GRU).map_err(err)? + gh.narrow(1, 0, GRU).map_err(err)?).map_err(err)?)?;
            let z = sigmoid(&(gi_t.narrow(1, GRU, GRU).map_err(err)? + gh.narrow(1, GRU, GRU).map_err(err)?).map_err(err)?)?;
            let n = (gi_t.narrow(1, 2 * GRU, GRU).map_err(err)? + (r * gh.narrow(1, 2 * GRU, GRU).map_err(err)?).map_err(err)?)
                .map_err(err)?
                .tanh()
                .map_err(err)?;
            // h' = (1 − z)·n + z·h
            h = ((n.clone() - (&z * &n).map_err(err)?).map_err(err)? + (z * h).map_err(err)?).map_err(err)?;
        }
        self.proj.fwd(&h)
    }
}

/// The loaded converter.
pub(crate) struct Model {
    ref_enc: RefEnc,
    enc_q: Posterior,
    flow: Flow,
    dec: Generator,
    dev: Device,
}

impl Model {
    pub(crate) fn load(wt: &Weights) -> R<Model> {
        let dev = Device::Cpu;
        Ok(Model { ref_enc: RefEnc::load(wt)?, enc_q: Posterior::load(wt)?, flow: Flow::load(wt, &dev)?, dec: Generator::load(wt)?, dev })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.dev
    }

    /// Tone colour of one spectrogram `[1, 513, T]` → 256 floats.
    pub(crate) fn tone_colour(&self, spec: &Tensor) -> R<Vec<f32>> {
        self.ref_enc.fwd(spec)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)
    }

    /// Convert a spectrogram from the `src` tone colour to `tgt`; `noise` is `[1, 192, T]`.
    pub(crate) fn convert(&self, spec: &Tensor, src: &[f32], tgt: &[f32], tau: f64, noise: &Tensor) -> R<Vec<f32>> {
        let colour = |v: &[f32]| -> R<Tensor> {
            if v.len() != GIN || v.iter().any(|x| !x.is_finite()) {
                return Err(TtsError::Invalid("a tone colour must be 256 finite numbers".into()));
            }
            Tensor::from_slice(v, (1, GIN, 1), &self.dev).map_err(err)
        };
        let (g_src, g_tgt) = (colour(src)?, colour(tgt)?);
        let zero = g_src.zeros_like().map_err(err)?; // zero_g: encoder and decoder see no speaker
        let z = self.enc_q.fwd(spec, &zero, tau, noise)?;
        let z_p = self.flow.fwd(&z, &g_src)?;
        let z_hat = self.flow.rev(&z_p, &g_tgt)?;
        let y = self.dec.fwd(&z_hat, &zero)?;
        y.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)
    }
}
