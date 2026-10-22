//! ECAPA-TDNN speaker embeddings on candle: 192 values per stretch of speech, close (by cosine)
//! for the same voice and far apart for different ones.
//!
//! The network is B. Desplanques, J. Thienpondt and K. Demuynck, "ECAPA-TDNN: Emphasized Channel
//! Attention, Propagation and Aggregation in TDNN Based Speaker Verification" (Interspeech
//! 2020), with the weights SpeechBrain trained on VoxCeleb 1+2 and released under Apache-2.0
//! ([`crate::models::ECAPA_MODEL`]: `embedding_model.ckpt`, read with candle's restricted
//! PyTorch reader). The layer layout follows SpeechBrain's released configuration
//! (channels 1024, 1024, 1024, 1024, 3072; kernels 5, 3, 3, 3, 1; dilations 1–4; attention 128;
//! 192 outputs):
//!
//! 1. the [`crate::fbank`] features (80 bands);
//! 2. a TDNN block (convolution, ReLU, batch norm) of kernel 5, then three SE-Res2Net blocks
//!    (1×1 TDNN, a Res2Net of 8 groups of 128 channels with kernel 3 and dilation 2, 3 or 4, 1×1
//!    TDNN, squeeze-and-excitation of 128, the residual added); convolutions pad by reflection;
//! 3. the three blocks' outputs concatenated (3072 channels) and a 1×1 TDNN block;
//! 4. attentive statistics pooling with the clip's mean and deviation as global context: the
//!    weighted mean and standard deviation (6144 values);
//! 5. batch norm and a 1×1 convolution to 192.
//!
//! Batch norms use their running statistics (inference).

use std::path::Path;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::SpeechError;

/// Embedding size.
pub const DIM: usize = 192;
/// Shortest audio embedded, samples (a quarter second): less is padded by repetition.
const MIN_SAMPLES: usize = 4000;

fn merr(e: candle_core::Error) -> SpeechError {
    SpeechError::Model(format!("ecapa: {e}"))
}

/// Batch norm folded into a scale and shift per channel, shaped `(1, C, 1)`.
struct Norm {
    scale: Tensor,
    shift: Tensor,
}

impl Norm {
    fn load(vb: VarBuilder, c: usize) -> candle_core::Result<Self> {
        let w = vb.get(c, "weight")?;
        let b = vb.get(c, "bias")?;
        let mean = vb.get(c, "running_mean")?;
        let var = vb.get(c, "running_var")?;
        let scale = (w / (var + 1e-5)?.sqrt()?)?;
        let shift = (b - (mean * &scale)?)?;
        Ok(Self { scale: scale.reshape((1, c, 1))?, shift: shift.reshape((1, c, 1))? })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)
    }
}

/// A 1-D convolution with "same" reflection padding.
struct Conv {
    w: Tensor,
    b: Tensor,
    k: usize,
    dilation: usize,
}

impl Conv {
    fn load(vb: VarBuilder, cin: usize, cout: usize, k: usize, dilation: usize) -> candle_core::Result<Self> {
        Ok(Self { w: vb.get((cout, cin, k), "weight")?, b: vb.get(cout, "bias")?.reshape((1, cout, 1))?, k, dilation })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let pad = self.dilation * (self.k - 1) / 2;
        let x = if pad > 0 { reflect_pad(x, pad)? } else { x.clone() };
        x.conv1d(&self.w, 0, 1, self.dilation, 1)?.broadcast_add(&self.b)
    }
}

/// Reflection padding of `pad` frames on both sides of `(b, c, t)` (the edge frame not repeated;
/// a clip shorter than the padding repeats its frames).
fn reflect_pad(x: &Tensor, pad: usize) -> candle_core::Result<Tensor> {
    // a Res2Net group of a batch is a strided view
    let x = &x.contiguous()?;
    let t = x.dim(2)?;
    if t <= 1 {
        let idx: Vec<u32> = vec![0; t + 2 * pad];
        return x.index_select(&Tensor::new(idx.as_slice(), x.device())?, 2);
    }
    let period = 2 * (t - 1);
    let idx: Vec<u32> = (0..t + 2 * pad)
        .map(|i| {
            let j = (i as isize - pad as isize).rem_euclid(period as isize) as usize;
            (if j >= t { period - j } else { j }) as u32
        })
        .collect();
    x.index_select(&Tensor::new(idx.as_slice(), x.device())?, 2)
}

/// Convolution, ReLU, batch norm.
struct Tdnn {
    conv: Conv,
    norm: Norm,
}

impl Tdnn {
    fn load(vb: VarBuilder, cin: usize, cout: usize, k: usize, dilation: usize) -> candle_core::Result<Self> {
        Ok(Self { conv: Conv::load(vb.pp("conv").pp("conv"), cin, cout, k, dilation)?, norm: Norm::load(vb.pp("norm").pp("norm"), cout)? })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.norm.forward(&self.conv.forward(x)?.relu()?)
    }
}

const SCALE: usize = 8;

struct SeRes2Net {
    tdnn1: Tdnn,
    res2: Vec<Tdnn>,
    tdnn2: Tdnn,
    se1: Conv,
    se2: Conv,
}

impl SeRes2Net {
    fn load(vb: VarBuilder, c: usize, dilation: usize) -> candle_core::Result<Self> {
        let g = c / SCALE;
        Ok(Self {
            tdnn1: Tdnn::load(vb.pp("tdnn1"), c, c, 1, 1)?,
            res2: (0..SCALE - 1)
                .map(|i| Tdnn::load(vb.pp("res2net_block").pp("blocks").pp(i.to_string()), g, g, 3, dilation))
                .collect::<candle_core::Result<_>>()?,
            tdnn2: Tdnn::load(vb.pp("tdnn2"), c, c, 1, 1)?,
            se1: Conv::load(vb.pp("se_block").pp("conv1").pp("conv"), c, 128, 1, 1)?,
            se2: Conv::load(vb.pp("se_block").pp("conv2").pp("conv"), 128, c, 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let h = self.tdnn1.forward(x)?;
        let parts = h.chunk(SCALE, 1)?;
        let mut ys: Vec<Tensor> = Vec::with_capacity(SCALE);
        let mut prev: Option<Tensor> = None;
        for (i, p) in parts.iter().enumerate() {
            let y = match (i, &prev) {
                (0, _) => p.clone(),
                (1, _) | (_, None) => self.res2[i - 1].forward(p)?,
                (_, Some(pv)) => self.res2[i - 1].forward(&(p + pv)?)?,
            };
            if i > 0 {
                prev = Some(y.clone());
            }
            ys.push(y);
        }
        let h = self.tdnn2.forward(&Tensor::cat(&ys, 1)?)?;
        // squeeze and excitation
        let s = h.mean_keepdim(2)?;
        let s = candle_nn::ops::sigmoid(&self.se2.forward(&self.se1.forward(&s)?.relu()?)?)?;
        h.broadcast_mul(&s)? + residual
    }
}

/// The ECAPA-TDNN speaker model.
pub struct Ecapa {
    block0: Tdnn,
    blocks: Vec<SeRes2Net>,
    mfa: Tdnn,
    asp_tdnn: Tdnn,
    asp_conv: Conv,
    asp_bn: Norm,
    fc: Conv,
    device: Device,
}

impl Ecapa {
    /// Load `embedding_model.ckpt` (SpeechBrain's release) on the GPU when `gpu` and there is one
    /// (Metal on macOS; `FILMCRAFT_SPEECH_CPU` keeps the CPU), else the CPU.
    pub fn load(path: &Path, gpu: bool) -> Result<Self, SpeechError> {
        let device = crate::whisper::device(gpu);
        let vb = VarBuilder::from_pth(path, DType::F32, &device).map_err(merr)?;
        Self::from_vb(vb, device).map_err(merr)
    }

    fn from_vb(vb: VarBuilder, device: Device) -> candle_core::Result<Self> {
        let c = 1024;
        let bv = vb.pp("blocks");
        Ok(Self {
            block0: Tdnn::load(bv.pp("0"), crate::fbank::N_MELS, c, 5, 1)?,
            blocks: (1..=3).map(|i| SeRes2Net::load(bv.pp(i.to_string()), c, i + 1)).collect::<candle_core::Result<_>>()?,
            mfa: Tdnn::load(vb.pp("mfa"), 3 * c, 3 * c, 1, 1)?,
            asp_tdnn: Tdnn::load(vb.pp("asp").pp("tdnn"), 9 * c, 128, 1, 1)?,
            asp_conv: Conv::load(vb.pp("asp").pp("conv").pp("conv"), 128, 3 * c, 1, 1)?,
            asp_bn: Norm::load(vb.pp("asp_bn").pp("norm"), 6 * c)?,
            fc: Conv::load(vb.pp("fc").pp("conv"), 6 * c, DIM, 1, 1)?,
            device,
        })
    }

    /// The network on features `(b, 80, t)` → `(b, 192)`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.block0.forward(x)?;
        let mut outs = Vec::with_capacity(3);
        for b in &self.blocks {
            x = b.forward(&x)?;
            outs.push(x.clone());
        }
        let x = self.mfa.forward(&Tensor::cat(&outs, 1)?)?;
        // attentive statistics pooling with global context
        let t = x.dim(2)?;
        let (mean, std) = stats(&x, None)?;
        let ctx = Tensor::cat(&[x.clone(), mean.unsqueeze(2)?.repeat((1, 1, t))?, std.unsqueeze(2)?.repeat((1, 1, t))?], 1)?;
        let attn = self.asp_conv.forward(&self.asp_tdnn.forward(&ctx)?.tanh()?)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let (mean, std) = stats(&x, Some(&attn))?;
        let pooled = Tensor::cat(&[mean, std], 1)?.unsqueeze(2)?;
        let y = self.fc.forward(&self.asp_bn.forward(&pooled)?)?;
        y.squeeze(2)
    }

    /// The embeddings of several clips of mono 16 kHz audio (each at least a quarter second,
    /// shorter ones repeated; non-finite samples are silence). Clips of equal length are batched.
    pub fn embed_many(&self, clips: &[&[f32]]) -> Result<Vec<Vec<f32>>, SpeechError> {
        let mut out: Vec<Option<Vec<f32>>> = vec![None; clips.len()];
        // group by length so a batch is one tensor
        let mut order: Vec<usize> = (0..clips.len()).collect();
        order.sort_by_key(|&i| clips[i].len());
        for group in order.chunk_by(|a, b| clips[*a].len() == clips[*b].len()) {
            for batch in group.chunks(16) {
                let mut feats = Vec::new();
                let mut frames = 0;
                for &i in batch {
                    let x = padded(clips[i]);
                    let (f, v) = crate::fbank::fbank(&x);
                    frames = f;
                    feats.push(v);
                }
                let n = batch.len();
                let flat: Vec<f32> = feats.concat();
                let x = Tensor::from_vec(flat, (n, frames, crate::fbank::N_MELS), &self.device).and_then(|t| t.transpose(1, 2)?.contiguous()).map_err(merr)?;
                let y = self.forward(&x).and_then(|y| y.to_dtype(DType::F32)?.to_vec2::<f32>()).map_err(merr)?;
                for (k, &i) in batch.iter().enumerate() {
                    out[i] = y.get(k).cloned();
                }
            }
        }
        out.into_iter().map(|e| e.ok_or_else(|| SpeechError::Model("ecapa: an embedding is missing".into()))).collect()
    }

    /// The embedding of one clip.
    pub fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, SpeechError> {
        self.embed_many(&[audio])?.pop().ok_or_else(|| SpeechError::Model("ecapa: no embedding".into()))
    }
}

/// `audio` at least [`MIN_SAMPLES`] long (repeated; silence when empty), at most
/// [`crate::fbank::MAX_SAMPLES`].
fn padded(audio: &[f32]) -> Vec<f32> {
    let audio = audio.get(..audio.len().min(crate::fbank::MAX_SAMPLES)).unwrap_or(&[]);
    if audio.len() >= MIN_SAMPLES {
        return audio.to_vec();
    }
    if audio.is_empty() {
        return vec![0.0; MIN_SAMPLES];
    }
    audio.iter().copied().cycle().take(MIN_SAMPLES).collect()
}

/// The (weighted) mean and standard deviation over time of `(b, c, t)`: `(b, c)` each. Without
/// weights every frame counts the same.
fn stats(x: &Tensor, w: Option<&Tensor>) -> candle_core::Result<(Tensor, Tensor)> {
    let t = x.dim(2)?;
    let w = match w {
        Some(w) => w.clone(),
        None => (Tensor::ones_like(x)? / t.max(1) as f64)?,
    };
    let mean = (x * &w)?.sum(2)?;
    let d = x.broadcast_sub(&mean.unsqueeze(2)?)?;
    let var = (w * d.sqr()?)?.sum(2)?;
    let std = var.clamp(1e-12, f64::MAX)?.sqrt()?;
    Ok((mean, std))
}

/// The cosine similarity of two embeddings (0 when either is empty, zero or not finite).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let c = dot / (na * nb);
    if c.is_finite() { c.clamp(-1.0, 1.0) } else { 0.0 }
}

/// ECAPA from `models_dir`, loaded once per process on the GPU where there is one (`None`: not
/// downloaded or unreadable).
pub fn load_cached(models_dir: &Path) -> Option<std::sync::Arc<Ecapa>> {
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};
    static CACHE: OnceLock<Mutex<Option<(std::path::PathBuf, Arc<Ecapa>)>>> = OnceLock::new();
    let path = crate::models::support_file(models_dir, crate::models::ECAPA_MODEL)?;
    let mut g = CACHE.get_or_init(Default::default).lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((p, m)) = g.as_ref()
        && *p == path
    {
        return Some(m.clone());
    }
    let m = Arc::new(Ecapa::load(&path, true).or_else(|_| Ecapa::load(&path, false)).ok()?);
    *g = Some((path, m.clone()));
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_is_safe() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[0.0; 3], &[1.0; 3]), 0.0);
        assert_eq!(cosine(&[f32::NAN; 3], &[1.0; 3]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn reflection_padding() {
        let x = Tensor::new(&[[[1f32, 2., 3., 4.]]], &Device::Cpu).unwrap();
        let y: Vec<f32> = reflect_pad(&x, 2).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(y, vec![3., 2., 1., 2., 3., 4., 3., 2.]);
        let one = Tensor::new(&[[[5f32]]], &Device::Cpu).unwrap();
        let y: Vec<f32> = reflect_pad(&one, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(y, vec![5., 5., 5.]);
    }

    #[test]
    fn short_clips_are_padded() {
        assert_eq!(padded(&[]).len(), MIN_SAMPLES);
        assert_eq!(padded(&[1.0, 2.0]).len(), MIN_SAMPLES);
        assert_eq!(padded(&vec![0.5; 9000]).len(), 9000);
    }

    /// The real model, when downloaded (`FILMCRAFT_MODELS_DIR`): one voice is closer to itself
    /// than to another (synthetic voices with different pitch and formants).
    #[test]
    fn the_model_tells_voices_apart_when_present() {
        let Some(dir) = std::env::var_os("FILMCRAFT_MODELS_DIR").map(std::path::PathBuf::from) else { return };
        let Some(path) = crate::models::support_file(&dir, crate::models::ECAPA_MODEL) else { return };
        let m = Ecapa::load(&path, false).unwrap();
        let a1 = crate::diarize::tests::voice(110.0, [(600.0, 90.0), (1100.0, 120.0)], 2.0, 1);
        let a2 = crate::diarize::tests::voice(110.0, [(600.0, 90.0), (1100.0, 120.0)], 2.0, 7);
        let b = crate::diarize::tests::voice(230.0, [(400.0, 80.0), (2600.0, 150.0)], 2.0, 3);
        let e = m.embed_many(&[&a1, &a2, &b]).unwrap();
        assert!(e.iter().all(|v| v.len() == DIM && v.iter().all(|x| x.is_finite())));
        // a batch embeds each clip as alone
        let alone = m.embed(&b).unwrap();
        assert!(cosine(&alone, &e[2]) > 0.9999, "{}", cosine(&alone, &e[2]));
        let same = cosine(&e[0], &e[1]);
        let diff = cosine(&e[0], &e[2]);
        assert!(same > diff + 0.1, "same {same}, different {diff}");
        // hostile input
        assert_eq!(m.embed(&[]).unwrap().len(), DIM);
        assert!(m.embed(&[f32::NAN; 100]).unwrap().iter().all(|x| x.is_finite()));
    }
}
