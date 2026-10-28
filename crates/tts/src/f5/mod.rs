//! IndicF5 voice cloning (feature `indicf5`): pure-Rust inference on candle.
//!
//! IndicF5 (AI4Bharat, MIT) is F5-TTS fine-tuned on 1417 hours of eleven Indian languages,
//! Malayalam among them. F5-TTS is non-autoregressive: the mel spectrogram of the whole
//! utterance is generated at once by flow matching, *infilling* after a reference clip — the
//! clip's mel and its transcript go in front of the text, so the voice continues in the clip's
//! speaker. The Vocos vocoder (charactr, MIT) turns the mel into 24 kHz audio. Weights are
//! downloaded on first use (`crate::catalog::INDICF5`, one 1.40 GB file holding both) and never
//! bundled.
//!
//! Ported from the MIT reference inference code shipped with the release (`f5_tts/model/cfm.py`,
//! `backbones/dit.py`, `modules.py`, `infer/utils_infer.py`), x-transformers' rotary embedding
//! and the `vocos` package (MIT).
//!
//! - [`config`]: the architecture, read from the tensor shapes and checked.
//! - [`vocab`]: the character vocabulary and the reference's tokenization.
//! - [`mel`]: the mel front end (STFT, HTK filterbank, log) and the vocoder's inverse STFT.
//! - `dit`: the diffusion transformer.
//! - `vocos`: the vocoder.
//! - `model`: references, durations and the sampler.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod config;
mod dit;
pub mod mel;
pub(crate) mod model;
pub mod vocab;
mod vocos;

use candle_core::Device;

use crate::TtsError;

pub use model::{F5, REFERENCE_SECONDS, Reference, spoken_len};

/// Audio rate in and out.
pub const RATE: u32 = 24_000;
/// Mel hop, FFT size and bands.
pub const HOP: usize = 256;
pub const N_FFT: usize = 1024;
pub const MELS: usize = 100;
/// Files of a release this code reads.
pub const FILES: [&str; 2] = ["model.safetensors", "vocab.txt"];

/// How one utterance is generated (the reference's defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Euler steps (the reference's `nfe_step`), 2–64.
    pub steps: usize,
    /// Classifier-free guidance, 0–5.
    pub cfg: f32,
    /// Sway sampling coefficient, −1–1 (−1 spends the steps early, on the coarse shape).
    pub sway: f32,
    /// Speaking speed, 0.25–4 (the duration is divided by it).
    pub speed: f32,
    /// Seed of the noise.
    pub seed: u64,
    /// Scale of the initial noise, 0.05–2 (Matcha-TTS's `temperature`): below 1 the sampler
    /// starts nearer the mean and reads flatter and steadier, above 1 livelier and riskier. 1 is
    /// the reference's sampler exactly (the noise is not touched).
    pub temperature: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { steps: 32, cfg: 2.0, sway: -1.0, speed: 1.0, seed: 1, temperature: 1.0 }
    }
}

impl Settings {
    pub fn check(&self) -> Result<(), TtsError> {
        if !(2..=64).contains(&self.steps) {
            return Err(TtsError::Invalid("steps must be between 2 and 64".into()));
        }
        if !(self.cfg.is_finite() && (0.0..=5.0).contains(&self.cfg)) {
            return Err(TtsError::Invalid("guidance must be between 0 and 5".into()));
        }
        if !(self.sway.is_finite() && (-1.0..=1.0).contains(&self.sway)) {
            return Err(TtsError::Invalid("sway must be between -1 and 1".into()));
        }
        if !(self.speed.is_finite() && (0.25..=4.0).contains(&self.speed)) {
            return Err(TtsError::Invalid("speed must be between 0.25 and 4".into()));
        }
        if !(self.temperature.is_finite() && (0.05..=2.0).contains(&self.temperature)) {
            return Err(TtsError::Invalid("temperature must be between 0.05 and 2".into()));
        }
        Ok(())
    }
}

/// The device inference runs on: as VoxCPM's ([`crate::voxcpm::device`]: Metal on macOS, else
/// the CPU); `FILMCRAFT_F5_CPU` forces the CPU.
pub fn device() -> Device {
    if std::env::var_os("FILMCRAFT_F5_CPU").is_some() {
        return Device::Cpu;
    }
    crate::voxcpm::device()
}

/// `dir` holds every file of a release.
pub fn complete(dir: &std::path::Path) -> bool {
    FILES.iter().all(|f| dir.join(f).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_checked() {
        assert!(Settings::default().check().is_ok());
        for s in [
            Settings { steps: 1, ..Default::default() },
            Settings { steps: 1000, ..Default::default() },
            Settings { cfg: f32::NAN, ..Default::default() },
            Settings { cfg: 9.0, ..Default::default() },
            Settings { sway: -3.0, ..Default::default() },
            Settings { speed: 0.0, ..Default::default() },
            Settings { speed: f32::INFINITY, ..Default::default() },
            Settings { temperature: 0.0, ..Default::default() },
            Settings { temperature: f32::NAN, ..Default::default() },
            Settings { temperature: 3.0, ..Default::default() },
        ] {
            assert!(s.check().is_err(), "{s:?}");
        }
    }
}
