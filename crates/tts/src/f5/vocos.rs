//! The Vocos vocoder (charactr/vocos-mel-24khz, MIT; IndicF5 ships its weights unchanged): log
//! mel → waveform at 24 kHz.
//!
//! Backbone: a 7-tap convolution (100 → 512), LayerNorm, eight ConvNeXt blocks (depthwise 7-tap
//! convolution, LayerNorm, GELU MLP 512 → 1536 → 512, a learned per-channel scale, residual),
//! a final LayerNorm. Head: a linear layer to 1026 values per frame — log magnitude and phase
//! of 513 bins — magnitudes `exp`'d and capped at 100, then the inverse STFT (`center`
//! padding, see [`super::mel::istft`]).

use candle_core::{D, DType, Device, Tensor};

use super::config::Config;
use super::dit::depthwise;
use super::mel::{BINS, istft};
use crate::voxcpm::{Linear, Loader, R, err};

const LN_EPS: f32 = 1e-6;
/// Largest magnitude the head lets through (the reference's safeguard).
const MAX_MAG: f32 = 100.0;

struct Block {
    dw_w: Tensor,
    dw_b: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1: Linear,
    pw2: Linear,
    gamma: Tensor,
}

pub(crate) struct Vocos {
    embed_w: Tensor,
    embed_b: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    blocks: Vec<Block>,
    final_w: Tensor,
    final_b: Tensor,
    out: Linear,
    dev: Device,
}

fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> R<Tensor> {
    candle_nn::ops::layer_norm(&x.contiguous().map_err(err)?, w, b, LN_EPS).map_err(err)
}

impl Vocos {
    pub(crate) fn load(ld: &mut Loader, cfg: &Config) -> R<Vocos> {
        let p = |n: &str| format!("{}{n}", cfg.vocoder);
        let (d, f) = (cfg.voc_dim, cfg.voc_ff);
        let mut blocks = Vec::with_capacity(cfg.voc_blocks);
        for i in 0..cfg.voc_blocks {
            let b = |n: &str| p(&format!("backbone.convnext.{i}.{n}"));
            blocks.push(Block {
                dw_w: ld.tensor(&b("dwconv.weight"), &[d, 1, 7])?.squeeze(1).map_err(err)?,
                dw_b: ld.tensor(&b("dwconv.bias"), &[d])?,
                ln_w: ld.tensor(&b("norm.weight"), &[d])?,
                ln_b: ld.tensor(&b("norm.bias"), &[d])?,
                pw1: ld.linear(&b("pwconv1"), d, f, true)?,
                pw2: ld.linear(&b("pwconv2"), f, d, true)?,
                gamma: ld.tensor(&b("gamma"), &[d])?,
            });
        }
        Ok(Vocos {
            embed_w: ld.tensor(&p("backbone.embed.weight"), &[d, super::MELS, 7])?,
            embed_b: ld.tensor(&p("backbone.embed.bias"), &[d])?,
            ln_w: ld.tensor(&p("backbone.norm.weight"), &[d])?,
            ln_b: ld.tensor(&p("backbone.norm.bias"), &[d])?,
            blocks,
            final_w: ld.tensor(&p("backbone.final_layer_norm.weight"), &[d])?,
            final_b: ld.tensor(&p("backbone.final_layer_norm.bias"), &[d])?,
            out: ld.linear(&p("head.out"), d, 2 * BINS, true)?,
            dev: ld.dev.clone(),
        })
    }

    /// The waveform of `frames` log-mel frames (`[frames][MELS]` row-major).
    pub(crate) fn decode(&self, mel: &[f32], frames: usize) -> R<Vec<f32>> {
        if frames < 2 || mel.len() != frames * super::MELS {
            return Err(err("the vocoder needs at least two whole mel frames"));
        }
        // [1, MELS, frames] for the input convolution
        let x = Tensor::from_vec(mel.to_vec(), (1, frames, super::MELS), &self.dev).map_err(err)?.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?;
        let x = x.conv1d(&self.embed_w, 3, 1, 1, 1).map_err(err)?.broadcast_add(&self.embed_b.reshape((1, (), 1)).map_err(err)?).map_err(err)?;
        let mut h = layer_norm(&x.transpose(1, 2).map_err(err)?, &self.ln_w, &self.ln_b)?; // [1, frames, d]
        for b in &self.blocks {
            let y = depthwise(&h, &b.dw_w, &b.dw_b)?;
            let y = layer_norm(&y, &b.ln_w, &b.ln_b)?;
            let y = b.pw2.fwd(&b.pw1.fwd(&y)?.gelu_erf().map_err(err)?)?;
            h = h.add(&y.broadcast_mul(&b.gamma).map_err(err)?).map_err(err)?;
        }
        let h = layer_norm(&h, &self.final_w, &self.final_b)?;
        let o = self.out.fwd(&h)?.squeeze(0).map_err(err)?; // [frames, 2·BINS]
        let mag = o.narrow(D::Minus1, 0, BINS).map_err(err)?.exp().map_err(err)?.clamp(0f32, MAX_MAG).map_err(err)?;
        let phase = o.narrow(D::Minus1, BINS, BINS).map_err(err)?;
        let to_vec = |t: Tensor| -> R<Vec<f32>> {
            t.to_dtype(DType::F32).map_err(err)?.to_device(&Device::Cpu).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)
        };
        let (mag, phase) = (to_vec(mag)?, to_vec(phase)?);
        if mag.iter().chain(&phase).any(|v| !v.is_finite()) {
            return Err(err("the vocoder produced non-finite values"));
        }
        Ok(istft(&mag, &phase, frames))
    }
}
