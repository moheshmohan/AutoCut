//! VoxCPM zero-shot voice cloning (feature `voxcpm`): pure-Rust inference on candle.
//!
//! VoxCPM (OpenBMB, Apache-2.0) is a tokenizer-free text-to-speech model that speaks directly in
//! the voice of a short reference clip. Text tokens and the clip's continuous audio latents go
//! through a MiniCPM-4 language model (the text-semantic LM, TSLM); a smaller residual acoustic
//! LM (RALM) refines its state; a local diffusion transformer (LocDiT) turns the two into the
//! next *patch* of latents (four 25 Hz frames, 160 ms) by flow matching with classifier-free
//! guidance; a local encoder (LocEnc) feeds the patch back in; a stop head ends the utterance;
//! the AudioVAE decoder turns the latents into a waveform. "Ultimate cloning" puts the reference
//! clip and its exact transcript in front of the text, so the model *continues* the speaker:
//! timbre, accent, rhythm and recording colour all carry over.
//!
//! Ported from the Apache-2.0 reference implementation (github.com/OpenBMB/VoxCPM, `src/voxcpm`,
//! commit cce58a5) and the released `config.json` files. One code path runs both generations
//! ([`Variant`]): VoxCPM1.5 (0.8 B parameters, 44.1 kHz in and out) and VoxCPM2 (2.3 B,
//! 16 kHz in, 48 kHz out, plus an isolated reference prefix). Weights are downloaded on first
//! use (`crate::catalog`) and never bundled.
//!
//! - [`config`]: `config.json`, checked against what this code implements.
//! - [`weights`]: the safetensors reader (seek per tensor, bf16 → f32) and the AudioVAE checkpoint.
//! - [`tokenizer`]: the text BPE from `tokenizer.json`.
//! - [`text`]: English normalisation (numbers, years, symbols) and sentence chunking.
//! - `minicpm`: the MiniCPM-4 transformer stack (RMSNorm, LongRoPE, grouped-query attention with a
//!   KV cache, SiLU MLP) the four networks share.
//! - `local`: LocEnc, LocDiT and the Euler flow-matching solver with CFG-Zero★ guidance.
//! - `vae`: the causal AudioVAE (Snake activations, depthwise residual units), run in windows
//!   so memory stays bounded however long the audio.
//! - `model`: prompt layout, prefill, the autoregressive patch loop with its stop head and caps.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod config;
mod local;
mod minicpm;
mod model;
pub mod text;
pub mod tokenizer;
mod vae;
pub mod weights;

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::TtsError;

pub use config::Variant;
pub use model::{Prompt, VoxCpm};

pub(crate) type R<T> = Result<T, TtsError>;

pub(crate) fn err(e: impl std::fmt::Display) -> TtsError {
    TtsError::Model(e.to_string())
}

/// Files of a VoxCPM release this code reads.
pub const FILES: [&str; 4] = ["config.json", "tokenizer.json", "model.safetensors", "audiovae.pth"];

/// Default classifier-free guidance (the reference's).
pub const DEFAULT_CFG: f32 = 2.0;
/// Guidance range accepted.
pub const CFG_RANGE: (f32, f32) = (1.0, 5.0);
/// Default flow-matching steps (the reference's).
pub const DEFAULT_TIMESTEPS: usize = 10;
/// Step range accepted.
pub const TIMESTEP_RANGE: (usize, usize) = (2, 50);

/// How one utterance is generated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Classifier-free guidance scale, [`CFG_RANGE`].
    pub cfg: f32,
    /// Flow-matching (Euler) steps per patch, [`TIMESTEP_RANGE`].
    pub timesteps: usize,
    /// Seed of the diffusion noise: the same seed, text and prompt give the same audio.
    pub seed: u64,
    /// Regenerate (next seed) when an utterance runs far longer than its text, at most this often.
    pub retries: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { cfg: DEFAULT_CFG, timesteps: DEFAULT_TIMESTEPS, seed: 1, retries: 2 }
    }
}

impl Settings {
    pub fn check(&self) -> R<()> {
        if !(self.cfg.is_finite() && (CFG_RANGE.0..=CFG_RANGE.1).contains(&self.cfg)) {
            return Err(TtsError::Invalid(format!("guidance must be between {} and {}", CFG_RANGE.0, CFG_RANGE.1)));
        }
        if !(TIMESTEP_RANGE.0..=TIMESTEP_RANGE.1).contains(&self.timesteps) {
            return Err(TtsError::Invalid(format!("steps must be between {} and {}", TIMESTEP_RANGE.0, TIMESTEP_RANGE.1)));
        }
        if self.retries > 5 {
            return Err(TtsError::Invalid("at most 5 retries".into()));
        }
        Ok(())
    }
}

/// The device inference runs on: Metal on Apple GPUs with the `metal` feature (when one is
/// there), else the CPU.
pub fn device() -> Device {
    #[cfg(feature = "metal")]
    {
        if std::env::var_os("FILMCRAFT_VOXCPM_CPU").is_none()
            && let Ok(d) = Device::new_metal(0)
        {
            return d;
        }
    }
    Device::Cpu
}

/// A linear layer `y = x Wᵀ + b` over the last dimension.
pub(crate) struct Linear {
    w: Tensor,
    b: Option<Tensor>,
    out: usize,
}

impl Linear {
    /// Several bias-free layers over the same input as one (outputs concatenated).
    pub(crate) fn stack(parts: &[Linear]) -> R<Linear> {
        if parts.iter().any(|p| p.b.is_some()) {
            return Err(err("only bias-free layers are stacked"));
        }
        let ws: Vec<&Tensor> = parts.iter().map(|p| &p.w).collect();
        Ok(Linear { w: Tensor::cat(&ws, 0).map_err(err)?, b: None, out: parts.iter().map(|p| p.out).sum() })
    }

    pub(crate) fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let dims = x.dims().to_vec();
        let (last, lead) = dims.split_last().ok_or_else(|| err("linear: scalar input"))?;
        let rows: usize = lead.iter().product();
        let mut x2 = x.reshape((rows, *last)).map_err(err)?;
        if x2.dtype() != self.w.dtype() {
            x2 = x2.to_dtype(self.w.dtype()).map_err(err)?;
        }
        let mut y = x2.matmul(&self.w.t().map_err(err)?).map_err(err)?;
        if y.dtype() != DType::F32 {
            y = y.to_dtype(DType::F32).map_err(err)?;
        }
        if let Some(b) = &self.b {
            y = y.broadcast_add(b).map_err(err)?;
        }
        let mut shape = lead.to_vec();
        shape.push(self.out);
        y.reshape(shape).map_err(err)
    }
}

/// Reads named weights from `model.safetensors` with their expected shapes.
pub(crate) struct Loader<'a> {
    pub st: &'a mut weights::SafeTensors,
    pub dev: &'a Device,
    /// Storage type of the linear layers' weights (activations stay f32).
    pub wdtype: DType,
}

impl Loader<'_> {
    pub(crate) fn tensor(&mut self, name: &str, shape: &[usize]) -> R<Tensor> {
        self.st.load(name, shape, self.dev)
    }

    pub(crate) fn linear(&mut self, name: &str, inp: usize, out: usize, bias: bool) -> R<Linear> {
        let w = if self.wdtype == DType::F32 {
            self.tensor(&format!("{name}.weight"), &[out, inp])?
        } else {
            // converted on the CPU, one tensor at a time, before it goes to the device
            self.st.load(&format!("{name}.weight"), &[out, inp], &Device::Cpu)?.to_dtype(self.wdtype).map_err(err)?.to_device(self.dev).map_err(err)?
        };
        let b = if bias { Some(self.tensor(&format!("{name}.bias"), &[out])?) } else { None };
        Ok(Linear { w, b, out })
    }
}

/// `dir` holds every file of a release.
pub fn complete(dir: &Path) -> bool {
    FILES.iter().all(|f| dir.join(f).is_file())
}

/// A deterministic standard-normal source (SplitMix64 + Box–Muller, as the other voices use).
pub(crate) struct Noise(crate::kokoro::dsp::Rng);

impl Noise {
    pub(crate) fn new(seed: u64) -> Noise {
        Noise(crate::kokoro::dsp::Rng::new(seed))
    }
    pub(crate) fn take(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.0.normal() as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_checked() {
        assert!(Settings::default().check().is_ok());
        for s in [
            Settings { cfg: f32::NAN, ..Default::default() },
            Settings { cfg: 0.5, ..Default::default() },
            Settings { cfg: 9.0, ..Default::default() },
            Settings { timesteps: 0, ..Default::default() },
            Settings { timesteps: 1000, ..Default::default() },
            Settings { retries: 100, ..Default::default() },
        ] {
            assert!(s.check().is_err(), "{s:?}");
        }
    }

    #[test]
    fn noise_is_seeded() {
        let a = Noise::new(7).take(64);
        assert_eq!(a, Noise::new(7).take(64));
        assert_ne!(a, Noise::new(8).take(64));
        let mean = a.iter().sum::<f32>() / 64.0;
        assert!(mean.abs() < 0.5, "{mean}");
    }

    #[test]
    fn linear_flattens_leading_dims() {
        let dev = Device::Cpu;
        let l = Linear {
            w: Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &dev).unwrap(),
            b: Some(Tensor::from_vec(vec![0f32, 0., 10.], 3, &dev).unwrap()),
            out: 3,
        };
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &dev).unwrap();
        let y = l.fwd(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 3]);
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![1., 2., 13., 3., 4., 17.]);
    }
}
